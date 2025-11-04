import os

# retriever.py
import json
from typing import List, Dict, Any
import numpy as np

# Optional TF-IDF imports are guarded to avoid hard failures if artifacts are missing
from pathlib import Path
try:
    import joblib
    from scipy.sparse import load_npz
    from sklearn.preprocessing import normalize as _sk_normalize
except Exception:
    joblib = None
    load_npz = None
    _sk_normalize = None

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
    - (Optional) Loads TF-IDF artifacts and performs hybrid fusion (RRF) without
      changing the public API or output format.
    """

    def __init__(self, vectors_path: str = "vectors.json"):
        from sentence_transformers import SentenceTransformer

        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Cannot find {vectors_path}. Run build_index.py first.")

        with open(vectors_path, "r", encoding="utf-8") as f:
            self.index: List[Dict[str, Any]] = json.load(f)

        # Matrix of float32 embeddings (dense)
        embs = np.asarray([entry["embedding"] for entry in self.index], dtype=np.float32)

        # Normalize (cosine = dot)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        self.embeddings_matrix = (embs / norms).astype(np.float32)

        # Cache text/metadata to avoid repeated lookups
        self.texts = [e["text"] for e in self.index]
        self.sources = [e.get("source", "") for e in self.index]
        self.chunk_ids = [e.get("chunk_id", "") for e in self.index]

        # Fast map: chunk_id -> dense index position
        self._id2idx = {cid: i for i, cid in enumerate(self.chunk_ids) if cid}

        # CPU-only model (must match build_index.py)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

        # ---- Optional TF-IDF state (loaded if artifacts exist) ----
        self._tfidf_available = False
        self._tfidf_doc_ids: List[str] = []
        self._tfidf_X_norm = None
        self._tfidf_vectorizer = None

        self._maybe_load_tfidf_artifacts()

    def _maybe_load_tfidf_artifacts(self) -> None:
        """
        Loads TF-IDF artifacts if present. If anything is missing or dependencies are absent,
        it silently disables the sparse branch (fallback to dense-only).
        """
        try:
            if joblib is None or load_npz is None or _sk_normalize is None:
                # Dependencies not available: skip TF-IDF
                return

            meta_path = Path("tfidf_meta.json")
            mat_path = Path("tfidf_matrix.npz")
            vec_path = Path("tfidf_vectorizer.pkl")
            if not (meta_path.exists() and mat_path.exists() and vec_path.exists()):
                return

            # Load metadata
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._tfidf_doc_ids = meta.get("doc_ids", [])

            # Load matrix and pre-normalize rows for cosine
            X = load_npz(mat_path)  # CSR matrix [n_docs x vocab]
            self._tfidf_X_norm = _sk_normalize(X, copy=False)

            # Load vectorizer
            self._tfidf_vectorizer = joblib.load(vec_path)

            # Sanity: ensure TF-IDF doc_ids exist in dense space (best effort)
            # (If a doc_id is missing, it will simply be skipped at search time.)
            self._tfidf_available = True
        except Exception:
            # Any failure keeps the system in dense-only mode
            self._tfidf_available = False
            self._tfidf_doc_ids = []
            self._tfidf_X_norm = None
            self._tfidf_vectorizer = None

    def _embed_text(self, text: str) -> np.ndarray:
        # Returns normalized float32 vector
        vec = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)

    def _dense_search_ids(self, query_vec: np.ndarray, pre_k: int = 50) -> List[str]:
        """
        Returns a list of doc_ids (chunk_ids) ranked by dense cosine similarity.
        """
        sims = self.embeddings_matrix @ query_vec  # dense cosine
        idx = np.argsort(-sims)[:pre_k]
        return [self.chunk_ids[i] for i in idx]

    def _sparse_search_ids(self, query: str, pre_k: int = 50) -> List[str]:
        """
        Returns a list of doc_ids (chunk_ids) ranked by TF-IDF cosine similarity.
        If TF-IDF is not available, returns an empty list.
        """
        if not self._tfidf_available or not query:
            return []

        q = self._tfidf_vectorizer.transform([query])
        q = _sk_normalize(q)  # cosine in TF-IDF space
        scores = (q @ self._tfidf_X_norm.T).toarray().ravel()
        if scores.size == 0:
            return []

        idx = np.argsort(-scores)[:pre_k]
        out_ids: List[str] = []
        for i in idx:
            # Map TF-IDF row i -> its doc_id, then ensure it exists in dense index
            doc_id = self._tfidf_doc_ids[i] if i < len(self._tfidf_doc_ids) else None
            if doc_id and doc_id in self._id2idx:
                out_ids.append(doc_id)
        return out_ids

    @staticmethod
    def _rrf_fuse(orderings: Dict[str, List[str]], k: int = 60) -> List[str]:
        """
        Reciprocal Rank Fusion over doc_id lists.
        Returns a fused list of doc_ids sorted by fused score.
        """
        from collections import defaultdict
        scores = defaultdict(float)
        for _, ranked in orderings.items():
            for rank, doc_id in enumerate(ranked, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # 1) Dense ranking (as before)
        q = self._embed_text(query)
        sims = self.embeddings_matrix @ q
        dense_idx = np.argsort(-sims)  # keep full order once for scoring
        dense_order_ids = [self.chunk_ids[i] for i in dense_idx[:50]]

        # 2) Optional sparse ranking (TF-IDF)
        sparse_order_ids = self._sparse_search_ids(query, pre_k=50)

        # 3) Fusion → list of final doc_ids
        if sparse_order_ids:
            fused_ids = self._rrf_fuse({"dense": dense_order_ids, "sparse": sparse_order_ids}, k=60)
            final_ids = fused_ids[:top_k]
        else:
            final_ids = dense_order_ids[:top_k]

        # 4) Build output in the exact same shape as before.
        #    Score remains the dense cosine score for compatibility.
        out: List[Dict[str, Any]] = []
        for doc_id in final_ids:
            i = self._id2idx.get(doc_id)
            if i is None:
                continue
            out.append({
                "text": self.texts[i],
                "source": self.sources[i],
                "chunk_id": self.chunk_ids[i],
                "score": float(sims[i]),  # keep dense cosine score for stability
            })
        return out


def get_retriever(vectors_path: str = "vectors.json") -> Retriever:
    global _RETRIEVER_SINGLETON
    if _RETRIEVER_SINGLETON is None:
        _RETRIEVER_SINGLETON = Retriever(vectors_path=vectors_path)
    return _RETRIEVER_SINGLETON
