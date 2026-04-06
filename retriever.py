import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    chunk_id: str
    control_id: str
    title: str
    family: str
    text: str
    score: float

class NISTRetriever:
    def __init__(self, index_dir="faiss_index"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        
        index_path = os.path.join(index_dir, "index.bin")
        meta_path = os.path.join(index_dir, "metadata.pkl")
        
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Retriever ready.")

    def retrieve(self, query: str, k: int = 5):
        q_vec = self.model.encode([query], device="cpu").astype("float32")
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0: continue
            meta = self.metadata[idx]
            results.append(RetrievedChunk(**meta, score=round(float(score), 4)))
        return results