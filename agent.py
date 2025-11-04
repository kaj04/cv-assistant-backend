import os
from typing import List, Dict

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

print("[DBG] OPENAI_API_KEY prefix:", (os.getenv("OPENAI_API_KEY") or "")[:12])

from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool
# Rich tracing disabled for RAM on Render/local.
# from datapizza.tracing import ContextTracing

from retriever import get_retriever

SYSTEM_PROMPT = """
You are the personal AI assistant of Francesco Colasurdo.

You talk with recruiters, students, and visitors who are exploring Francesco’s website.
Your purpose is to give them a concise and authentic preview of his background, projects, and skills — based strictly on his real documents.

Rules: 
- Always speak in first person singular (e.g. “I designed…”, “I implemented…”).
- Use ONLY the information retrieved from the candidate context tool (the RAG context). 
- If something is not present or not yet published, say:
  "I haven’t publicly shared that information yet."
- Never invent dates, roles, or technologies not explicitly mentioned.
- Keep the tone natural, confident, and approachable — like a 21-year-old master's student in the Netherlands who’s curious, proactive, and professional.
- Be brief (max 2-3 sentences): this chat is a casual preview, not a full interview. 
- Mirror the user’s language: reply in English if they write in English, reply in Italian if they write in Italian.
- Grammar and style must always be clean and polished.
""".strip()


@tool(
    name="get_candidate_context",
    description="Retrieve verified information about Francesco's actual skills, thesis work, and project responsibilities."
)
def get_candidate_context(question: str) -> str:
    """
    Retrieve relevant knowledge chunks from vectors.json via cached Retriever.
    """
    retriever = get_retriever(vectors_path="vectors.json")
    top_chunks = retriever.retrieve(question, top_k=3)

    parts: List[str] = []
    for c in top_chunks:
        parts.append(
            f"[SOURCE: {c['source']} / {c['chunk_id']} | SCORE: {c['score']:.4f}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def build_llm_client() -> OpenAIClient:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY in environment. Set it before running the agent."
        )
    return OpenAIClient(api_key=api_key, model="gpt-4o-mini")


def build_agent() -> Agent:
    client = build_llm_client()
    agent = Agent(
        name="cv-assistant",
        client=client,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_candidate_context],
    )
    return agent


def answer_question(user_question: str) -> Dict[str, object]:
    agent = build_agent()

    # senza tracing "ricco"
    agent_response = agent.run(user_question)
    final_answer_text = getattr(agent_response, "text", str(agent_response))

    retriever = get_retriever(vectors_path="vectors.json")
    top_chunks = retriever.retrieve(user_question, top_k=3)

    sources = [
        {"source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]}
        for c in top_chunks
    ]

    return {"answer": final_answer_text, "sources": sources, "question": user_question}
