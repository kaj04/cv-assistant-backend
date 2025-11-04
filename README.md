# 🧠 Ask My CV — Personal RAG Assistant

This repository contains the backend for **“Ask My CV”**, an AI assistant that answers questions about me — *as if I were responding in first person* — by retrieving information directly from my real CV and experience.

> 🚀 **Live:** Deployed on [Render](https://render.com) and connected to my personal website, where visitors and recruiters can chat with the assistant.

---

## 🏗️ How It Works

1. **Knowledge Base**  
   All my professional information (experience, education, skills, values, projects) lives as markdown files under `/data/*.md`.

2. **Index Building** — `build_index.py`  
   - Loads every markdown file  
   - Splits content into overlapping chunks  
   - Generates dense embeddings with **Sentence-Transformers**  
   - Saves them to `vectors.json`

3. **TF-IDF Index (Hybrid Search)** — `build_tfidf.py`  
   - Reads the same chunks from `vectors.json`  
   - Builds a **TF-IDF matrix** (`tfidf_matrix.npz`)  
   - Enables **hybrid retrieval** by combining keyword and semantic similarity

4. **Retriever** — `retriever.py`  
   - Embeds the user query with the same model used for the index  
   - Retrieves the top chunks using cosine similarity  
   - (If TF-IDF files are present) performs **hybrid fusion** via **Reciprocal Rank Fusion (RRF)**  
   - Returns the top-K most relevant chunks for the final answer

5. **Agent & API**  
   - `agent.py`: crafts a structured prompt (“speak as Francesco, do not invent”)  
   - `server.py`: FastAPI app exposing `POST /api/chat`, used by the website widget

---

## ⚙️ Tech Stack

| Layer | Technologies |
|:------|:--------------|
| Backend | Python · FastAPI |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| RAG Logic | Dense + TF-IDF Hybrid Retrieval |
| Orchestration | [datapizza-ai](https://github.com/datapizza-labs/datapizza-ai) |
| Deployment | Render (auto-deploy from GitHub) |
| Frontend | Hugo static site + custom JS chat widget |

---

## 🎯 Goal

Enable visitors and recruiters to ask natural questions like:

> “What did you build at NTT Data?”  
> “What are your strongest skills?”  
> “What motivates you?”

…and receive honest, first-person answers drawn directly from my verified experience and personal data — *never hallucinated*.

---

## 🧩 Repository Structure

