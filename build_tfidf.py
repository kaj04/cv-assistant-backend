#!/usr/bin/env python3
# build_tfidf.py
#
# Builds a TF-IDF index in parallel to vectors.json (no DB required).
# It reads the already-chunked texts from vectors.json to guarantee 1:1
# alignment (same chunk_id, same text) with dense embeddings.
#
# Output files:
# - tfidf_meta.json        (light metadata, row order, and params)
# - tfidf_matrix.npz       (sparse CSR matrix)
# - tfidf_vectorizer.pkl   (persisted vectorizer to avoid refit at boot)

import json
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORS_JSON = Path("vectors.json")
TFIDF_META = Path("tfidf_meta.json")
TFIDF_MATRIX = Path("tfidf_matrix.npz")
TFIDF_VECTORIZER = Path("tfidf_vectorizer.pkl")


def _load_chunks_from_vectors(vectors_path: Path) -> List[Dict[str, Any]]:
    """Reads chunks from vectors.json (list or {'chunks': [...]}) and returns them."""
    data = json.loads(vectors_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "chunks" in data:
        chunks = data["chunks"]
    elif isinstance(data, list):
        chunks = data
    else:
        raise ValueError("vectors.json format not recognized (expected list or dict['chunks']).")
    return chunks


def _extract_doc_id_and_text(chunks: List[Dict[str, Any]]):
    """Extracts (doc_ids, texts) from chunks, being defensive on key names."""
    doc_ids, texts = [], []
    for i, ch in enumerate(chunks):
        # Defensive: supports common key variants
        doc_id = ch.get("id") or ch.get("doc_id") or ch.get("chunk_id") or str(i)
        text = ch.get("text") or ch.get("content") or ch.get("chunk") or ""
        if not isinstance(text, str):
            text = str(text)
        doc_ids.append(str(doc_id))
        texts.append(text.strip())
    return doc_ids, texts


def main():
    # Validates input availability
    if not VECTORS_JSON.exists():
        raise SystemExit("vectors.json not found. Run build_index.py first.")

    # Loads chunks and extracts corpus
    chunks = _load_chunks_from_vectors(VECTORS_JSON)
    doc_ids, corpus = _extract_doc_id_and_text(chunks)
    if not corpus:
        raise SystemExit("No text found in vectors.json chunks.")

    # Builds a simple, robust TF-IDF index.
    # ngram_range (1,2) helps with compound names; English stopwords fit your content.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,  # fine for small/medium corpora
    )
    X: csr_matrix = vectorizer.fit_transform(corpus)

    # Persists matrix and light metadata (keeps the matrix out of JSON to avoid huge files)
    save_npz(TFIDF_MATRIX, X)
    meta = {
        "schema_version": 1,
        "doc_ids": doc_ids,            # row order of X
        "num_docs": int(X.shape[0]),
        "vocabulary_size": int(X.shape[1]),
        "ngram_range": [1, 2],
        "stop_words": "english",
        "max_df": 0.9,
        "min_df": 1,
        "source": "vectors.json",
        "note": "Parallel TF-IDF index for hybrid retrieval.",
    }
    TFIDF_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Persists the vectorizer to avoid refitting at runtime
    joblib.dump(vectorizer, TFIDF_VECTORIZER)

    print(f"OK: saved {TFIDF_META}, {TFIDF_MATRIX}, {TFIDF_VECTORIZER} "
          f"(docs={X.shape[0]}, vocab={X.shape[1]})")


if __name__ == "__main__":
    main()
