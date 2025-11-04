import os
import glob
import json
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer


def load_documents(data_dir: str = "data") -> List[Dict[str, str]]:
    """
    Reads all .md files from the data/ directory and returns a list of
    {"source": file_name, "text": content}.

    Technical rationale:
    - This function acts as the "ingestion layer": it defines the single entry point
    for declaring what constitutes the candidate's official knowledge base.
    -If a file needs to be excluded in the future (e.g., contains private information),
    the change should be made here.
    """
    documents = []

    pattern = os.path.join(data_dir, "*.md")
    for filepath in glob.glob(pattern):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # basic normalization: remove triple newlines and unnecessary empty lines
        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()

        documents.append({
            "source": os.path.basename(filepath),
            "text": cleaned,
        })

    return documents


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50
) -> List[str]:
    """
    Splits the text into overlapping chunks.

    Technical notes:
    - max_tokens here is "about words", not actual LLM tokens.
      It's fine for MVP because you only need reasonably sized blocks.
    - overlap_tokens is used to avoid losing context between chunks.

    Returns a list of strings (each string = chunk).
    """

    # Simple split in "words" to estimate tokens.
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + max_tokens
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunks.append(chunk_text)

        # Apply overlap:
        # The next chunk starts "overlap_tokens" words before the end
        # of the previous chunk, so we don't split important concepts.
        start = end - overlap_tokens

        if start < 0:
            start = 0

        if start >= len(words):
            break

    return chunks


def build_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    For each loaded document, generates chunks and adds metadata.

    Returns a list of dictionaries like:
    {
      "source": "thesis_project.md",
      "chunk_id": "thesis_project.md#0",
      "text": "During my internship at NTT Data I..."
    }

    Technical rationale:
    - Having chunk_id allows in the retriever and API response to say where the information comes from.
      This is useful for transparency towards the recruiter and for your own debugging.
    """

    all_chunks = []

    for doc in documents:
        source = doc["source"]
        text = doc["text"]

        chunks = chunk_text(text)

        for i, ch in enumerate(chunks):
            all_chunks.append({
                "source": source,
                "chunk_id": f"{source}#{i}",
                "text": ch
            })

    return all_chunks


def embed_chunks(
    chunks: List[Dict[str, str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[Dict[str, object]]:
    """
    Generates the vector embedding for each chunk.

    - Load an embedding model like all-MiniLM-L6-v2
      (small, fast, perfect for MVP).
    - For each chunk["text"], we get a numerical vector.
    - Save that vector together with the text and metadata.

    Returns: list of dictionaries
    {
      "source": "thesis_project.md",
      "chunk_id": "thesis_project.md#0",
      "text": "...",
      "embedding": [0.012, -0.034, ...]  # list of floats
    }

    Technical rationale:
    - This output is exactly what you will save in vectors.json
      and that will be queried by the retriever using cosine similarity.
    """

    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract only the text from the chunks
    chunk_texts = [c["text"] for c in chunks]

    print("[INFO] Computing embeddings for all chunks...")
    embeddings = model.encode(chunk_texts, show_progress_bar=True)

    # Combine embeddings + metadata
    vector_entries = []
    for c, emb in zip(chunks, embeddings):
        vector_entries.append({
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "embedding": emb.tolist()
        })

    return vector_entries


def save_vectors(vectors: List[Dict[str, object]], output_path: str = "vectors.json"):
    """
    Saves the list of vectors and metadata in a single JSON file.

    Motivazione tecnica:
    - This file is a 'mini vector store'.
    - The retriever will load it for RAG.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(vectors)} vector entries to {output_path}")


def main():
    """
    Executes the entire end-to-end pipeline:
    1. Loads the markdown documents in ./data
    2. Splits them into chunks
    3. Calculates the embeddings
    4. Saves everything in vectors.json
    """

    print("[STEP 1] Loading markdown documents from ./data ...")
    docs = load_documents(data_dir="data")
    print(f"[INFO] Loaded {len(docs)} documents")

    print("[STEP 2] Chunking documents ...")
    chunks = build_chunks(docs)
    print(f"[INFO] Generated {len(chunks)} chunks")

    print("[STEP 3] Embedding chunks ...")
    vectors = embed_chunks(chunks)
    print(f"[INFO] Generated {len(vectors)} embedded vector entries")

    print("[STEP 4] Saving vectors.json ...")
    save_vectors(vectors, output_path="vectors.json")

    print("[DONE] Index build complete. You can now use vectors.json in your retriever.")


if __name__ == "__main__":
    main()
