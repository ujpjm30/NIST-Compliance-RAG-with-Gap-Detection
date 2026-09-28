import os
import sys
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, field, fields

# Match ingestion's macOS workaround before loading native libraries.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "faiss_index_full_baseline"

@dataclass
class RetrievedChunk:
    chunk_id: str
    control_id: str
    family: str
    title: str
    text: str
    score: float
    parent_id: str = ""
    is_enhancement: bool = False
    related: list = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    chunk_index: int = 0
    chunk_count: int = 1
    is_withdrawn: bool = False
    withdrawal_notice: str = ""

    @classmethod
    def from_metadata(cls, meta: dict, score: float) -> "RetrievedChunk":
        """Build from a stored metadata dict, ignoring keys this class doesn't have.

        Without this filter, adding a field in ingestion.py breaks retrieval with
        'unexpected keyword argument'. The index and the reader can now evolve apart.
        """
        allowed = {f.name for f in fields(cls)} - {"score"}
        return cls(**{k: v for k, v in meta.items() if k in allowed}, score=score)


class NISTRetriever:
    def __init__(self, index_dir=DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir).resolve()
        for name in ("index.bin", "metadata.pkl"):
            if not (self.index_dir / name).is_file():
                raise FileNotFoundError(
                    f"Missing {self.index_dir / name}. Build this index with ingestion.py first."
                )
        manifest_path = self.index_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if sys.platform == "darwin":
            torch.set_num_threads(1)
            faiss.omp_set_num_threads(1)
        settings = self.manifest or {}
        self.model = SentenceTransformer(settings.get("embedding_model", "all-MiniLM-L6-v2"), device="cpu")
        self.model.max_seq_length = settings.get("max_seq_length", 512)

        self.index = faiss.read_index(str(self.index_dir / "index.bin"))
        with (self.index_dir / "metadata.pkl").open("rb") as f:
            self.metadata = pickle.load(f)

        if any("is_withdrawn" not in meta for meta in self.metadata):
            raise ValueError("Index lacks withdrawal metadata. Rebuild it with ingestion.py using its original strategy.")

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index has {self.index.ntotal} vectors but metadata has "
                f"{len(self.metadata)} entries. Re-run ingestion.py."
            )
        print(f"[INFO] Retriever ready. {self.index.ntotal} chunks indexed.")

    def retrieve(self, query: str, k: int = 5, *, include_withdrawn: bool = False) -> list:
        if k <= 0:
            raise ValueError("k must be positive.")
        q_vec = self.model.encode([query], device="cpu").astype("float32")
        faiss.normalize_L2(q_vec)
        # This small flat index permits filtering the complete ranked list, so
        # withdrawn hits never consume slots intended for active requirements.
        scores, indices = self.index.search(q_vec, self.index.ntotal)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if self.metadata[idx]["is_withdrawn"] and not include_withdrawn:
                continue
            results.append(
                RetrievedChunk.from_metadata(self.metadata[idx], round(float(score), 4))
            )
            if len(results) == k:
                break
        return results
