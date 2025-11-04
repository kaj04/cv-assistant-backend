import os
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

# ---- Env ----
load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY not found. Set it before running the server.")

class SourceItem(BaseModel):
    source: str
    chunk_id: Union[int, str]
    score: float

class ChatQuery(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceItem]] = None

# ---- Dynamic import of the agent after load_dotenv ----
try:
    from agent import answer_question 
except Exception as e:
    print("[WARN] agent.py not available or with errors:", e)
    def answer_question(q: str):
        raise RuntimeError("agent.answer_question not available")

# ---- App ----
app = FastAPI()
app.state.embedder = None

@app.on_event("startup")
def load_model_once():
    if app.state.embedder is None:
        app.state.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---- CORS (GitHub Pages) ----
# Your site origin: https://kaj04.github.io  (no path)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https://kaj04\.github\.io$",  # exact match of the origin
    allow_credentials=False,                            # no cookies/sessions
    allow_methods=["GET", "POST", "OPTIONS"],           # explicit
    allow_headers=["*"],                                # e.g. "content-type"
    max_age=86400,                                      # preflight cache (optional)
)

@app.options("/api/chat")
def options_chat():
    return Response(status_code=200)

# ---- Routes ----
@app.get("/")
def root():
    return {"ok": True, "service": "ask-my-cv"}

@app.get("/healthz")
def health():
    return {"status": "healthy"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(q: ChatQuery):
    try:
        result = answer_question(q.question)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
