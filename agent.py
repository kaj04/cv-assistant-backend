import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)  # carica .env ovunque giri l'import

print("[DBG] OPENAI_API_KEY prefix:",
      (os.getenv("OPENAI_API_KEY") or "")[:12])  # stampa 'sk-...' troncato
from typing import List, Dict
from retriever import Retriever

# Datapizza AI:
# - Agent: orchestrates LLM + tool calls
# - OpenAIClient: wraps OpenAI's chat models
# - @tool: exposes Python functions the agent can call
# - ContextTracing: captures spans / token usage / cost observability
#   so this can be treated like a production agent. 
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool
from datapizza.tracing import ContextTracing


SYSTEM_PROMPT = """
You are the personal AI assistant of Francesco Colasurdo.

You answer questions from recruiters and visitors about my background,
my thesis work, my AI projects, my technical skills, and my hands-on experience.

Rules:
- Always speak in first person singular ("I designed...", "I implemented...").
- Use ONLY the information you are given via the candidate context tool.
- If something is not present in the retrieved context, say:
  "I have not publicly shared that information yet."
- Be concise, professional, and factual.
- Maximum 6 sentences.
- Do not invent dates, roles, responsibilities, or technologies that are not explicitly mentioned.
- Mirror the language of the user's question: reply in English if the question is in English, reply in Italian if the question is in Italian.
""".strip()


@tool(
    name="get_candidate_context",
    description="Retrieve verified information about Francesco's actual skills, thesis work, and project responsibilities."
)
def get_candidate_context(question: str) -> str:
    """
    Tool exposed to the agent.
    Given the recruiter's question, retrieve the most relevant knowledge chunks
    from Francesco's private index (vectors.json) via the Retriever.

    The output of this tool becomes the trusted CONTEXT for the LLM.
    """

    retriever = Retriever(vectors_path="vectors.json")
    top_chunks = retriever.retrieve(question, top_k=4)

    parts: List[str] = []
    for c in top_chunks:
        parts.append(
            f"[SOURCE: {c['source']} / {c['chunk_id']} | SCORE: {c['score']:.4f}]\n"
            f"{c['text']}"
        )

    context_block = "\n\n".join(parts)
    return context_block


def build_llm_client() -> OpenAIClient:
    """
    Build the OpenAI-backed LLM client used by the agent.

    Requirements:
    - You must have an active OpenAI API key in the environment as OPENAI_API_KEY.
    - Your OpenAI account must have billing enabled / prepaid credits. 
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY in environment. "
            "Set it before running the agent."
        )

    # gpt-4o-mini: cheaper, good quality for recruiter-style Q&A.
    client = OpenAIClient(
        api_key=api_key,
        model="gpt-4o-mini",
    )
    return client


def build_agent() -> Agent:
    """
    Create the production agent for the CV Assistant.

    - system_prompt: enforces voice, style, and anti-hallucination rules.
    - tools: the agent can call get_candidate_context() to ground itself
             in verified facts from Francesco's documents.
    - client: the LLM backend (OpenAI via OpenAIClient).
    """
    client = build_llm_client()

    agent = Agent(
        name="cv-assistant",
        client=client,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_candidate_context],
    )

    return agent


def answer_question(user_question: str) -> Dict[str, object]:
    """
    High-level entrypoint.
    This is what FastAPI will call (e.g. POST /api/chat).

    Steps:
    1. Create the agent.
    2. Run the agent inside a tracing span ("candidate_query").
       ContextTracing collects timing, token usage, etc.,
       which is important for cost monitoring and debugging. 
    3. Extract the generated answer (agent_response.text).
    4. Re-run retrieval locally to include source metadata in the HTTP response.
    """

    agent = build_agent()

    with ContextTracing().trace("candidate_query"):
        agent_response = agent.run(user_question)

    final_answer_text = getattr(agent_response, "text", str(agent_response))

    retriever = Retriever(vectors_path="vectors.json")
    top_chunks = retriever.retrieve(user_question, top_k=4)

    sources = [
        {
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "score": c["score"],
        }
        for c in top_chunks
    ]

    return {
        "answer": final_answer_text,
        "sources": sources,
        "question": user_question,
    }
