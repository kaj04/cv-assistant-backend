# tfidf_index.py
#
# Loads the TF-IDF artifacts and exposes a simple search() method.
# It is intentionally minimal to avoid coupling with the rest of the codebase.

from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
import joblib

TFIDF_META = Path("tfidf_meta.json")
TFIDF_MATRIX = Path("tfidf_matrix.npz")
TFIDF_VECTORIZER = Path("tfidf_vectorizer.pkl")


class TfidfSearcher:
    """Lightweight TF-IDF searcher built on top of the saved artifacts."""

    def __init__(self) -> None:
        self.available = (
            TFIDF_META.exists() and
            TFIDF_MATRIX.exists() and
            TFIDF_VECTORIZER.exists()
        )
        if not self.available:
            self.doc_ids = []
            self._X_norm = None
            self._vectorizer = None
            return

        meta = json.loads(TFIDF_META.read_text(encoding="utf-8"))
        self.doc_ids = meta.get("doc_ids", [])
        X = load_npz(TFIDF_MATRIX)  # CSR matrix [n_docs x vocab]
        self._X_norm = normalize(X, copy=False)  # pre-normalized for cosine
        self._vectorizer = joblib.load(TFIDF_VECTORIZER)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Returns a list of (doc_id, score) sorted by descending similarity."""
        if not self.available or not query:
            return []
        q = self._vectorizer.transform([query])
        q = normalize(q)  # cosine in TF-IDF space
        scores = (q @ self._X_norm.T).toarray().ravel()
        if scores.size == 0:
            return []
        idx = np.argsort(-scores)[:top_k]
        out: List[Tuple[str, float]] = []
        for i in idx:
            s = float(scores[i])
            if s > 0:
                out.append((self.doc_ids[i], s))
        return out
