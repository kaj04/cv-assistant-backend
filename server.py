import os
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

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
app = FastAPI()
app.state.embedder = None

@app.on_event("startup")
def load_model_once():
    if app.state.embedder is None:
        # modello leggero; caricato una sola volta
        app.state.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---- CORS (GitHub Pages) ----
# Origin del tuo sito: https://kaj04.github.io  (niente path)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kaj04.github.io"],
    allow_credentials=False,                   # non usi cookie/sessioni
    allow_methods=["GET", "POST", "OPTIONS"],  # espliciti per il preflight
    allow_headers=["*"],                       # o ["content-type", "authorization"]
)

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
        result = answer_question(q.question)  # dict con answer/sources
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
