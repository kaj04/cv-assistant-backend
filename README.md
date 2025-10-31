# cv-assistant-backend
RAG-based personal AI assistant ("Ask my CV") for answering questions about my profile using retrieval over my own data.

# Ask My CV — Personal RAG Assistant

This backend powers "Ask My CV", an AI assistant that answers questions about me (Francesco Colasurdo) as if I were answering in first person.

## How it works
1. Content about me (experience, skills, education, values, projects) lives in `/data/*.md`.
2. `build_index.py`:
   - loads all `.md`,
   - chunks them,
   - generates embeddings with `sentence-transformers`,
   - writes `vectors.json`.
3. `retriever.py`:
   - given a user question,
   - embeds the question,
   - finds the most relevant chunks using cosine similarity.
4. `agent.py`:
   - builds a final answer with a system prompt ("speak as Francesco, don't invent"),
   - uses datapizza-ai to orchestrate tools.
5. `server.py`:
   - FastAPI app exposing POST /api/chat
   - this is what the website frontend calls.

## Tech stack
- Python (FastAPI)
- RAG pipeline (embeddings + cosine similarity, no DB)
- sentence-transformers
- datapizza-ai orchestration
- Hugo static site + JS widget for chat

## Goal
Let recruiters or visitors to my website ask:
- "What did you build at NTT Data?"
- "What are your strongest skills?"
- "What motivates you?"
and get an answer that is accurate, honest, and sounds like me.
