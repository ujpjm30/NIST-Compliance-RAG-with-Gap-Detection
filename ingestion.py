"""
Corpus build + FAISS index for the NIST 800-53 RAG pipeline.

Default baseline: one chunk per entry in the local Rev 5 snapshot, including
withdrawn entries. The optional chars strategy splits the same text without
discarding its tail; it writes a separate index for comparison.

faiss and sentence_transformers are imported lazily inside run_ingestion() so
build_chunks() can be imported and tested with pandas alone.
"""
import os
import sys
import pickle
import re
import argparse
import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field, replace
from typing import List

# macOS wheels may load different OpenMP runtimes (PyTorch, FAISS, sklearn).
# Configure them before importing native libraries to avoid worker crashes.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data/NIST_SP-800-53_rev5_catalog_load.csv"
INDEX_DIR = PROJECT_DIR / "faiss_index_full_baseline"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 64          # was 1, which made encoding needlessly slow
INCLUDE_DISCUSSION = True  # NIST's "Discussion" text carries the intent, not just the requirement


@dataclass
class Chunk:
    chunk_id: str
    control_id: str
    parent_id: str          # 'AC-2(1)' -> 'AC-2'; base controls point at themselves
    is_enhancement: bool
    family: str
    title: str
    related: List[str] = field(default_factory=list)  # NIST's own cross-reference graph
    text: str = ""
    char_start: int = 0
    char_end: int = 0
    chunk_index: int = 0
    chunk_count: int = 1
    is_withdrawn: bool = False
    withdrawal_notice: str = ""


def parse_identifier(identifier: str):
    """'AC-2(1)' -> ('AC', 'AC-2', True).  'AC-2' -> ('AC', 'AC-2', False)."""
    family = identifier.split("-")[0].strip()
    m = re.match(r"^([A-Za-z]{2}-\d+)\(\s*\d+\s*\)$", identifier.strip())
    if m:
        return family, m.group(1), True
    return family, identifier.strip(), False


def parse_related(value, valid_ids: set[str] | None = None) -> List[str]:
    if pd.isna(value):
        return []

    raw = str(value).strip()
    if raw.casefold() in {"", "[none]", "none", "nan"}:
        return []

    related = []
    for part in raw.split(","):
        control_id = re.sub(r"\s+", "", part).rstrip(".").upper()
        if not control_id:
            continue
        if not re.fullmatch(r"[A-Z]{2}-\d+(?:\(\d+\))?", control_id):
            raise ValueError(f"Invalid related control ID: {part!r}")
        if valid_ids is not None and control_id not in valid_ids:
            raise ValueError(f"Unknown related control ID: {control_id}")
        if control_id not in related:
            related.append(control_id)
    return related


def split_text_spans(text: str, size: int, overlap: int):
    """Yield lossless source offsets; prefer paragraph/sentence/word boundaries."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size.")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            minimum = start + max(overlap + 1, size // 2)
            for separator in ("\n\n", "\n", ". ", "; ", " "):
                boundary = text.rfind(separator, minimum, end)
                if boundary >= minimum:
                    end = boundary + len(separator)
                    break
        yield start, end
        if end == len(text):
            break
        start = end - overlap


def split_control(chunk: Chunk, chunk_size: int = 500, overlap: int = 50):
    """Repeat the ID/title in every chunk; chunk_size includes that prefix."""
    prefix = f"{chunk.control_id} {chunk.title}\n"
    if not chunk.text.startswith(prefix):
        raise ValueError(f"Missing control heading: {chunk.control_id}")
    budget = chunk_size - len(prefix)
    if budget <= 0 or not 0 <= overlap < budget:
        raise ValueError(
            f"Chunk size {chunk_size} leaves insufficient room after the heading "
            f"of {chunk.control_id}; increase size or reduce overlap."
        )
    body = chunk.text[len(prefix):]
    spans = list(split_text_spans(body, budget, overlap)) or [(0, 0)]
    return [
        replace(
            chunk,
            chunk_id=f"{chunk.control_id}::chunk-{number + 1}",
            text=prefix + body[start:end],
            char_start=len(prefix) + start,
            char_end=len(prefix) + end,
            chunk_index=number,
            chunk_count=len(spans),
        )
        for number, (start, end) in enumerate(spans)
    ]


def build_chunks(csv_path=DATA_PATH, *, strategy="full", chunk_size=500, overlap=50) -> List[Chunk]:
    if strategy not in {"full", "chars"}:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[df["identifier"].notna()]
    valid_ids = set(df["identifier"].astype(str).str.strip())

    chunks = []
    for _, row in df.iterrows():
        identifier = str(row["identifier"]).strip()
        family, parent_id, is_enh = parse_identifier(identifier)

        body = "" if pd.isna(row.get("control_text")) else str(row["control_text"]).strip()
        discussion = "" if pd.isna(row.get("discussion")) else str(row["discussion"]).strip()

        # The control ID and title go into the embedded text on purpose: a query
        # that names a control ID should be able to match on it.
        parts = [f"{identifier} {str(row['name']).strip()}", body]
        if INCLUDE_DISCUSSION and discussion:
            parts.append(f"Discussion: {discussion}")

        text = "\n".join(p for p in parts if p)
        withdrawal = re.search(r"(?im)^\[?Withdrawn\s*:[^\n]*", text)
        chunk = Chunk(
            chunk_id=identifier,
            control_id=identifier,
            parent_id=parent_id,
            is_enhancement=is_enh,
            family=family,
            title=str(row["name"]).strip(),
            related=parse_related(row.get("related"), valid_ids),
            text=text,
            char_end=len(text),
            is_withdrawn=withdrawal is not None,
            withdrawal_notice=withdrawal.group(0) if withdrawal else "",
        )
        if strategy == "chars":
            chunks.extend(split_control(chunk, chunk_size, overlap))
        else:
            chunks.append(chunk)
    return chunks


def run_ingestion(*, csv_path=DATA_PATH, strategy="full", chunk_size=500,
                  overlap=50, index_dir=None):
    if index_dir is None:
        index_dir = (INDEX_DIR if strategy == "full" else
                     PROJECT_DIR / f"faiss_index_chars_{chunk_size}_overlap_{overlap}")
    index_dir = Path(index_dir).resolve()
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        settings = (strategy, chunk_size if strategy == "chars" else None,
                    overlap if strategy == "chars" else None)
        if settings != (previous.get("strategy"), previous.get("chunk_size"),
                        previous.get("overlap")):
            raise ValueError("This directory holds another strategy; choose a new --index-dir.")
    elif strategy != "full" and (index_dir / "index.bin").exists():
        raise ValueError("Existing index has no strategy manifest; choose a new --index-dir.")

    print(f"[INFO] Loading CSV: {csv_path}")
    chunks = build_chunks(csv_path, strategy=strategy, chunk_size=chunk_size, overlap=overlap)

    import numpy as np
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    if sys.platform == "darwin":
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)

    controls = {c.control_id: c for c in chunks}
    n_enh = sum(c.is_enhancement for c in controls.values())
    print(f"[INFO] {len(chunks)} chunks from {len(controls)} entries "
          f"({len(controls) - n_enh} base controls, {n_enh} enhancements, "
          f"{len({c.family for c in chunks})} families); strategy={strategy}")

    print("[INFO] Loading model in CPU mode...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH
    lengths = model.tokenizer(
        [c.text for c in chunks], truncation=False, padding=False,
        return_length=True, verbose=False,
    )["length"]
    truncated = sum(length > MAX_SEQ_LENGTH for length in lengths)
    print(f"[INFO] {truncated}/{len(chunks)} chunks exceed the {MAX_SEQ_LENGTH}-token limit.")

    print("[INFO] Encoding starting...")
    embeddings = model.encode(
        [c.text for c in chunks],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings).astype("float32")

    print("[INFO] Building FAISS index...")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.bin"))
    with (index_dir / "metadata.pkl").open("wb") as f:
        pickle.dump([asdict(c) for c in chunks], f)
    manifest = {
        "metadata_schema_version": 2,
        "created_at_ns": time.time_ns(),
        "corpus_path": str(Path(csv_path).resolve()),
        "corpus_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "strategy": strategy,
        "chunk_size": chunk_size if strategy == "chars" else None,
        "overlap": overlap if strategy == "chars" else None,
        "control_count": len(controls), "chunk_count": len(chunks),
        "embedding_model": MODEL_NAME, "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE, "include_discussion": INCLUDE_DISCUSSION,
        "max_input_tokens": max(lengths), "truncated_chunk_count": truncated,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Ingestion complete: {index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a full-control or lossless character-chunk index.")
    parser.add_argument("--strategy", choices=("full", "chars"), default="full")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--index-dir", type=Path)
    args = parser.parse_args()
    run_ingestion(csv_path=args.csv, strategy=args.strategy, chunk_size=args.chunk_size,
                  overlap=args.overlap, index_dir=args.index_dir)
