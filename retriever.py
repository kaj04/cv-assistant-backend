# retriever.py
import json
import os
from typing import List, Dict, Any
import numpy as np

# IMPORTANT:
# Must match the model used in build_index.py.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Singleton cache to avoid reloading model and vectors on every request ----
_RETRIEVER_SINGLETON = None

class Retriever:
    """
    Semantic retriever for CV Assistant.

    - Loads vectors.json once.
    - Loads SentenceTransformer model once.
    - Uses float32 and normalized vectors to reduce RAM and CPU.
    """

    def __init__(self, vectors_path: str = "vectors.json"):
        from sentence_transformers import SentenceTransformer

        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Cannot find {vectors_path}. Run build_index.py first.")

        with open(vectors_path, "r", encoding="utf-8") as f:
            self.index: List[Dict[str, Any]] = json.load(f)

        # Matrix of float32 embeddings
        embs = np.asarray([entry["embedding"] for entry in self.index], dtype=np.float32)

        # Normalize (cosine = dot)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        self.embeddings_matrix = (embs / norms).astype(np.float32)

        # Cache text/metadata to avoid repeated lookups
        self.texts = [e["text"] for e in self.index]
        self.sources = [e.get("source", "") for e in self.index]
        self.chunk_ids = [e.get("chunk_id", "") for e in self.index]

        # CPU-only model
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    def _embed_text(self, text: str) -> np.ndarray:
        # Returns normalized float32 vector
        vec = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q = self._embed_text(query)  
        sims = self.embeddings_matrix @ q  
        idx = np.argsort(-sims)[:top_k]

        out: List[Dict[str, Any]] = []
        for i in idx:
            out.append({
                "text": self.texts[i],
                "source": self.sources[i],
                "chunk_id": self.chunk_ids[i],
                "score": float(sims[i]),
            })
        return out


def get_retriever(vectors_path: str = "vectors.json") -> Retriever:
    global _RETRIEVER_SINGLETON
    if _RETRIEVER_SINGLETON is None:
        _RETRIEVER_SINGLETON = Retriever(vectors_path=vectors_path)
    return _RETRIEVER_SINGLETON
