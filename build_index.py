import os
import glob
import json
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer


def load_documents(data_dir: str = "data") -> List[Dict[str, str]]:
    """
    Legge tutti i file .md nella cartella data/ e restituisce una lista di
    {"source": nome_file, "text": contenuto}.

    Motivazione tecnica:
    - Questa funzione è lo "ingestion layer": è il punto unico dove dichiariamo
      qual è la knowledge base ufficiale del candidato.
    - Se un domani vuoi escludere un file (es. info troppo private), lo fai qui.
    """
    documents = []

    pattern = os.path.join(data_dir, "*.md")
    for filepath in glob.glob(pattern):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # normalizzazione base: togliamo spazi doppi e righe vuote inutili
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
    Spezza il testo in chunk sovrapposti.

    Nota tecnica:
    - max_tokens qui è "circa parole", non veri token LLM.
      Va bene per MVP perché ti basta avere blocchi di dimensione ragionevole.
    - overlap_tokens serve per non perdere contesto tra un chunk e l'altro.

    Ritorna una lista di stringhe (ogni stringa = chunk).
    """

    # Semplice split in "parole" per stimare i token.
    # In futuro potresti sostituire questo con un vero tokenizer del modello.
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + max_tokens
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunks.append(chunk_text)

        # Applichiamo l'overlap:
        # Il prossimo chunk ricomincia "overlap_tokens" parole prima della fine
        # del chunk precedente, così non spezzi concetti importanti.
        start = end - overlap_tokens

        if start < 0:
            start = 0

        if start >= len(words):
            break

    return chunks


def build_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Per ogni documento caricato, genera i chunk e aggiunge metadati.

    Ritorna una lista di dizionari tipo:
    {
      "source": "thesis_project.md",
      "chunk_id": "thesis_project.md#0",
      "text": "During my internship at NTT Data I..."
    }

    Motivazione tecnica:
    - Avere chunk_id ti permette domani (nel retriever e nella risposta API)
      di dire da dove arriva l'informazione. Questo è utile per trasparenza
      verso il recruiter e per debug tuo.
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
    Genera l'embedding vettoriale per ogni chunk.

    - Carichiamo un modello di embedding tipo all-MiniLM-L6-v2
      (piccolo, veloce, perfetto per MVP).
    - Per ogni chunk["text"], otteniamo un vettore numerico.
    - Salviamo quel vettore assieme al testo e ai metadati.

    Ritorno: lista di dict
    {
      "source": "thesis_project.md",
      "chunk_id": "thesis_project.md#0",
      "text": "...",
      "embedding": [0.012, -0.034, ...]  # lista di float
    }

    Motivazione tecnica:
    - Questo output è esattamente quello che salverai in vectors.json
      e che verrà interrogato domani dal retriever usando cosine similarity.
    """

    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Estraggo solo il testo dei chunk
    chunk_texts = [c["text"] for c in chunks]

    print("[INFO] Computing embeddings for all chunks...")
    embeddings = model.encode(chunk_texts, show_progress_bar=True)

    # Combina embeddings + metadati
    vector_entries = []
    for c, emb in zip(chunks, embeddings):
        vector_entries.append({
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "embedding": emb.tolist()  # numpy -> lista Python serializzabile in JSON
        })

    return vector_entries


def save_vectors(vectors: List[Dict[str, object]], output_path: str = "vectors.json"):
    """
    Salva la lista di vettori e metadati in un file JSON unico.

    Motivazione tecnica:
    - Questo file è il tuo 'mini vector store'.
    - Domani il retriever lo caricherà per fare RAG.
    - È leggibile da umani e semplice da versionare su GitHub.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(vectors)} vector entries to {output_path}")


def main():
    """
    Esegue l'intera pipeline end-to-end:
    1. Carica i documenti markdown in ./data
    2. Li spezza in chunk
    3. Calcola gli embeddings
    4. Salva tutto in vectors.json
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