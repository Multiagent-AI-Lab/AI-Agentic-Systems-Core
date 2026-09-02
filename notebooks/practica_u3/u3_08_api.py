# u3_08_api.py — ejecutar con: uvicorn u3_08_api:app --reload --port 8000
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import asyncio

# ── Importar y recriar el pipeline completo ──────────────────
from typing import Annotated, TypedDict

import chromadb
import httpx
from chromadb.utils import embedding_functions
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from fastapi import FastAPI, HTTPException
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.multiagent_core.skills.evaluation.output_scorer import score_heuristic
from src.multiagent_core.skills.memory.episodic_retriever import (
    add_episode,
    format_for_context,
    retrieve,
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# LLM para search y analysis
if OPENROUTER_API_KEY:
    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://github.com/Multiagent-AI-Lab/AI-Agentic-Systems-Core",
            "X-Title": "AI Agentic Systems Core - U3_08",
        },
    )
    crew_llm = f"openrouter/{OPENROUTER_MODEL}"
elif GOOGLE_API_KEY:
    from langchain_google_genai import ChatGoogleGenerativeAI

    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0
    )
    crew_llm = "gemini/gemini-2.0-flash"
else:
    raise RuntimeError("Se requiere OPENROUTER_API_KEY o GOOGLE_API_KEY en .env")

# ChromaDB
chroma_client = chromadb.EphemeralClient()
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection("papers_api", embedding_function=default_ef)


@tool
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Busca papers en arXiv."""
    import xml.etree.ElementTree as ET

    try:
        r = httpx.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "max_results": max_results,
                "sortBy": "relevance",
            },
            timeout=15,
        )
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        return [
            {
                "title": e.find("atom:title", ns).text.strip(),
                "abstract": e.find("atom:summary", ns).text.strip()[:500],
                "url": e.find("atom:id", ns).text.strip(),
                "authors": [a.find("atom:name", ns).text for a in e.findall("atom:author", ns)][:3],
            }
            for e in root.findall("atom:entry", ns)
        ]
    except Exception as ex:  # noqa: BLE001 -- sin cambios de logica, migrado del origen
        return [{"error": str(ex), "title": "Error", "abstract": "", "url": "", "authors": []}]


@tool
def index_papers(papers: list) -> str:
    """Indexa papers en ChromaDB."""
    docs, metas, ids = [], [], []
    for i, p in enumerate(papers):
        if p.get("error"):
            continue
        docs.append(f"{p.get('title','')}. {p.get('abstract','')}")
        metas.append(
            {
                "title": p.get("title", ""),
                "url": p.get("url", ""),
                "authors": ", ".join(p.get("authors", [])),
            }
        )
        ids.append(f"paper_{i}_{abs(hash(p.get('url', str(i)))) % 100000}")
    if not docs:
        return "Sin papers validos."
    collection.add(documents=docs, metadatas=metas, ids=ids)
    return f"{len(docs)} papers indexados."


@tool
def semantic_search(query: str, n_results: int = 3) -> str:
    """Busca semánticamente en ChromaDB."""
    count = collection.count()
    if count == 0:
        return "Sin papers indexados."
    res = collection.query(query_texts=[query], n_results=min(n_results, count))
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return "\n\n".join(f"• {m.get('title','')}\n  {d[:200]}" for d, m in zip(docs, metas))


# State
class ResearchState(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    search_results: list[dict]
    episode_context: str
    analysis: str
    draft_report: str
    final_report: str
    review_score: float
    review_cycles: int
    max_review_cycles: int
    tokens_used: int
    estimated_cost_usd: float
    budget_usd: float
    github_issue_url: str | None
    status: str


def initial_state(topic: str, budget_usd: float = 0.50) -> ResearchState:
    return ResearchState(
        topic=topic,
        messages=[],
        search_results=[],
        episode_context="",
        analysis="",
        draft_report="",
        final_report="",
        review_score=0.0,
        review_cycles=0,
        max_review_cycles=3,
        tokens_used=0,
        estimated_cost_usd=0.0,
        budget_usd=budget_usd,
        github_issue_url=None,
        status="running",
    )


search_tools = [search_arxiv, index_papers, semantic_search]


def search_agent_node(state):
    if state["status"] == "budget_exceeded":
        return {"status": "budget_exceeded"}
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un investigador. Busca papers con search_arxiv, indexa con index_papers, verifica con semantic_search.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    executor = AgentExecutor(
        agent=create_tool_calling_agent(_llm, search_tools, prompt),
        tools=search_tools,
        max_iterations=5,
        verbose=False,
    )
    result = executor.invoke({"input": f"Busca papers sobre: {state['topic']}"})
    papers = collection.query(query_texts=[state["topic"]], n_results=5)["metadatas"][0]
    return {
        "search_results": papers,
        "messages": [HumanMessage(content=f"Búsqueda: {result['output'][:200]}")],
    }


def analysis_agent_node(state):
    if state["status"] == "budget_exceeded":
        return {"status": "budget_exceeded"}
    past = retrieve(query=state["topic"], user_id="research_system_api", top_k=3)
    ep_ctx = format_for_context(past)
    summary = "\n".join(f"- {p.get('title','Paper')}" for p in state.get("search_results", [])[:5])
    resp = _llm.invoke(
        f"Analiza papers sobre '{state['topic']}'.\nPapers:\n{summary}\n{ep_ctx}\nGener análisis estructurado."
    )
    add_episode(
        content=f"Análisis '{state['topic']}': {resp.content[:300]}",
        user_id="research_system_api",
        metadata={"topic": state["topic"]},
    )
    return {"analysis": resp.content, "episode_context": ep_ctx}


class _SSTool(BaseTool):
    name: str = "semantic_search_tool"
    description: str = "Busca en papers indexados"

    class _In(BaseModel):
        query: str = Field(description="Query")

    args_schema: type[BaseModel] = _In

    def _run(self, query: str) -> str:
        res = collection.query(query_texts=[query], n_results=2)
        return (
            "\n---\n".join(res["documents"][0])
            if res["documents"] and res["documents"][0]
            else "Sin resultados."
        )


writer_agent = Agent(
    role="Scientific Writer",
    goal="Escribir reportes científicos claros",
    backstory="Escritor científico senior.",
    tools=[_SSTool()],
    verbose=False,
    llm=crew_llm,
)


def writing_agent_node(state):
    if state["status"] == "budget_exceeded":
        return {"status": "budget_exceeded"}
    task = Task(
        description=f"Escribe reporte sobre '{state['topic']}':\n{state['analysis'][:1500]}\nFormato Markdown, 600-800 palabras.",
        expected_output="Reporte científico en Markdown de 600-800 palabras",
        agent=writer_agent,
    )
    result = Crew(
        agents=[writer_agent], tasks=[task], process=Process.sequential, verbose=False
    ).kickoff()
    return {"draft_report": result.raw if hasattr(result, "raw") else str(result)}


REPORT_CRITERIA = {"completeness": 0.35, "structure": 0.25, "actionability": 0.25, "length": 0.15}


def review_agent_node(state):
    sr = score_heuristic(state["draft_report"], criteria=REPORT_CRITERIA, passing_threshold=0.65)
    score = round(sr.score * 100, 1)
    if sr.passed or state["review_cycles"] >= state["max_review_cycles"] - 1:
        return {
            "final_report": state["draft_report"],
            "review_score": score,
            "review_cycles": state["review_cycles"] + 1,
        }
    return {
        "analysis": state["analysis"] + f"\n\n[Feedback]: {sr.feedback}",
        "review_score": score,
        "review_cycles": state["review_cycles"] + 1,
    }


def should_revise(state):
    return (
        "publish"
        if (
            state.get("final_report")
            or state["review_cycles"] >= state["max_review_cycles"]
            or state["status"] == "budget_exceeded"
        )
        else "rewrite"
    )


def publish_agent_node(state):
    import tempfile
    from pathlib import Path

    report = state.get("final_report") or state.get("draft_report", "Sin reporte")
    out = Path(tempfile.gettempdir()) / f"report_{state['topic'][:20].replace(' ','_')}.md"
    out.write_text(report, encoding="utf-8")
    return {"status": "completed", "github_issue_url": str(out)}


workflow = StateGraph(ResearchState)
workflow.add_node("search", search_agent_node)
workflow.add_node("analyze", analysis_agent_node)
workflow.add_node("write", writing_agent_node)
workflow.add_node("review", review_agent_node)
workflow.add_node("publish", publish_agent_node)
workflow.set_entry_point("search")
workflow.add_edge("search", "analyze")
workflow.add_edge("analyze", "write")
workflow.add_edge("write", "review")
workflow.add_conditional_edges("review", should_revise, {"publish": "publish", "rewrite": "write"})
workflow.add_edge("publish", END)
_memory = MemorySaver()
_app = workflow.compile(checkpointer=_memory)

# ── FastAPI ──────────────────────────────────────────────────
app = FastAPI(
    title="Research Intelligence Platform API",
    description="Sistema multi-agente de investigación automatizada (U3_08)",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    topic: str
    budget_usd: float = 0.50


class ResearchResponse(BaseModel):
    final_report: str
    review_score: float
    status: str
    github_url: str | None


@app.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    if len(request.topic.strip()) < 3:
        raise HTTPException(status_code=400, detail="El topic debe tener al menos 3 caracteres.")
    state = initial_state(request.topic, request.budget_usd)
    config = {"configurable": {"thread_id": f"api_{request.topic[:20]}"}}
    loop = asyncio.get_event_loop()
    final = await loop.run_in_executor(None, lambda: _app.invoke(state, config))
    return ResearchResponse(
        final_report=final.get("final_report") or final.get("draft_report") or "",
        review_score=final.get("review_score", 0),
        status=final.get("status", "unknown"),
        github_url=final.get("github_issue_url"),
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "research_intelligence_platform", "version": "1.0.0"}
