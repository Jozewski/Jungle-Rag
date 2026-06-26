"""
The Survival Guide — RAG API
FastAPI backend: retrieves relevant knowledge chunks and generates
a grounded answer via Groq (free Llama3 inference).
"""
from dotenv import load_dotenv
load_dotenv()
import os
import json
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

import rag


# ---------------------------------------------------------------------------
# Startup: build the vector index once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] Building knowledge index…")
    rag.build_index()
    print("[Server] Index ready. Starting server.")
    yield


app = FastAPI(title="Survival Guide RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Groq client (free tier — llama-3.1-8b-instant is fast and capable)
# ---------------------------------------------------------------------------

def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable is not set. "
                   "Get a free key at https://console.groq.com",
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    source: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    question: str


# ---------------------------------------------------------------------------
# System prompt — enforces grounded answers only
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Survival Guide, an expert wilderness and emergency survival assistant.

CRITICAL RULES:
1. Answer ONLY using the information provided in the <context> block below.
2. If the context does not contain enough information to answer the question, say exactly:
   "I don't have enough information in my knowledge base to answer that."
3. Do NOT add information from your general knowledge beyond what is in the context.
4. Be specific, practical, and actionable.
5. Keep answers concise but complete — use bullet points for steps or lists.
6. Always cite the relevant guide section (e.g., "According to the shelter guide...")
   when helpful.
"""


def build_prompt(question: str, chunks: List[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_clean = chunk["source"].replace("_", " ").replace(".md", "").replace(".txt", "").replace(".json", "").title()
        context_parts.append(f"[Source {i}: {source_clean}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Retrieve relevant chunks
    chunks = rag.retrieve(question, top_k=5)

    if not chunks:
        raise HTTPException(status_code=500, detail="Could not retrieve context from knowledge base.")

    # 2. Build grounded prompt
    prompt = build_prompt(question, chunks)

    # 3. Generate answer via Groq
    client = get_groq_client()
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,       # low temp = more factual, less hallucination
            max_tokens=800,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {str(e)}")

    # 4. Format sources for display
    sources = [
        SourceChunk(
            source=c["source"].replace("_", " ").replace(".md", "").replace(".txt", "").replace(".json", "").title(),
            text=c["text"][:300] + "…" if len(c["text"]) > 300 else c["text"],
            score=c["score"],
        )
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources, question=question)


@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "Survival Guide RAG is running"}


@app.get("/api/topics")
async def topics():
    """Return a list of available knowledge base topics."""
    from pathlib import Path
    kb_dir = Path(__file__).parent / "knowledge_base"
    files = sorted(kb_dir.glob("*"))
    topics_list = []
    for f in files:
        if f.suffix.lower() in {".md", ".txt", ".json"}:
            name = f.stem.replace("_", " ").title()
            topics_list.append({"name": name, "file": f.name})
    return {"topics": topics_list}


# ---------------------------------------------------------------------------
# Static files (frontend)
# ---------------------------------------------------------------------------

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(static_dir, "index.html"))
