import json
import os
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

# IMPORTANT:
# This MUST match the model you used in build_index.py,
# or the cosine similarity won't make semantic sense.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    """
    Semantic retriever for Francesco Colasurdo's CV Assistant.

    Responsibilities:
    - Load the vector index (vectors.json) produced by build_index.py.
    - Embed an incoming user question using the SAME embedding model.
    - Compute cosine similarity between the question embedding and each chunk embedding.
    - Return the top_k most relevant text chunks as context for the agent.
    """

    def __init__(self, vectors_path: str = "vectors.json"):
        # Load embedding model once
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Load vector index
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(
                f"Cannot find {vectors_path}. Run build_index.py first."
            )

        with open(vectors_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        # Prebuild the matrix of all chunk embeddings for fast similarity scoring
        self.embeddings_matrix = np.array(
            [entry["embedding"] for entry in self.index],
            dtype=np.float32,
        )

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Convert a question string into an embedding vector using the same
        SentenceTransformer model that was used offline during indexing.
        """
        emb = self.model.encode([text], show_progress_bar=False)
        return np.array(emb[0], dtype=np.float32)

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity = (A · B) / (||A|| * ||B||)

        Interpretation:
        - 1.0   => very semantically similar
        - ~0.0  => unrelated
        - -1.0  => opposite (rare in practice for embeddings)
        """
        dot = float(np.dot(vec_a, vec_b))
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, object]]:
        """
        Given a natural-language query (e.g. "Describe your thesis work"),
        return the top_k most relevant knowledge chunks.

        Each returned chunk is a dict:
        {
          "text": "...chunk of my real experience...",
          "source": "thesis.md",
          "chunk_id": "thesis.md#2",
          "score": 0.82  # cosine similarity
        }

        The agent will feed these chunks into the LLM. This is the "R" in RAG.
        """

        # 1. Embed the user's question
        query_vec = self._embed_text(query)

        # 2. Score every chunk using cosine similarity
        scored_chunks = []
        for idx, entry in enumerate(self.index):
            chunk_vec = self.embeddings_matrix[idx]
            score = self._cosine_similarity(query_vec, chunk_vec)

            scored_chunks.append(
                {
                    "text": entry["text"],
                    "source": entry.get("source", ""),
                    "chunk_id": entry.get("chunk_id", ""),
                    "score": score,
                }
            )

        # 3. Sort by descending semantic similarity
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        # 4. Return only the best matches
        return scored_chunks[:top_k]
