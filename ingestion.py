import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict

DATA_PATH = "data/NIST_SP-800-53_rev5_catalog_load.csv"
INDEX_DIR = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2" 
SELECTED_FAMILIES = {"AC": 6, "AU": 5, "CM": 5, "IA": 5, "IR": 4, "SC": 5}

@dataclass
class Chunk:
    chunk_id: str
    control_id: str
    title: str
    family: str
    text: str

def run_ingestion():
    print(f"[INFO] Loading CSV: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    
    chunks = []
    for fam, count in SELECTED_FAMILIES.items():
        subset = df[df['identifier'].str.startswith(fam, na=False)].head(count)
        for _, row in subset.iterrows():
            chunks.append(Chunk(
                chunk_id=str(row['identifier']),
                control_id=str(row['identifier']),
                title=str(row['name']),
                family=fam,
                text=str(row['control_text'])[:500]
            ))
    
    print(f"[INFO] {len(chunks)} chunks prepared.")


    print("[INFO] Loading model in CPU mode...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    model.max_seq_length = 512

    print("[INFO] Encoding starting...")
    texts = [c.text for c in chunks]

    embeddings = model.encode(texts, batch_size=1, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("[INFO] Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.bin"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump([asdict(c) for c in chunks], f)
    
    print("[SUCCESS] Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()