import os
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# ---- Env ----
load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY non trovata. Impostarla prima di prod.")

# ---- Modelli (UNICA definizione) ----
class SourceItem(BaseModel):
    source: str
    chunk_id: Union[int, str]
    score: float

class ChatQuery(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceItem]] = None

# ---- Importa l'agente dopo il load_dotenv ----
try:
    from agent import answer_question  # deve esistere
except Exception as e:
    print("[WARN] agent.py non disponibile o con errori:", e)
    def answer_question(q: str):
        raise RuntimeError("agent.answer_question non disponibile")

# ---- App ----
app = FastAPI(title="Ask My CV - Backend", version="1.0.0")

origins = [
    "http://localhost:1313",
    "http://localhost:3000",
    "https://<TUO-USERNAME>.github.io",
    "https://<TUO-DOMINIO>",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "service": "ask-my-cv"}

@app.get("/healthz")
def health():
    return {"status": "healthy"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(q: ChatQuery):
    try:
        result = answer_question(q.question)  # <-- dict con answer/sources/question
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),  # lista di SourceItem
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
