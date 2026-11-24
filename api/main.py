"""FastAPI backend for the RAG web app.

    python serve.py                     # http://127.0.0.1:8000
    uvicorn api.main:app --reload       # same, with auto-reload while editing

Endpoints (interactive docs at /docs) - all need a signed-in user (viewer role):
    GET  /api/me       who am I, which role
    GET  /api/health   is the knowledge base ready? how many chunks?
    GET  /api/files    ingested files, types and sections (for the filter menus)
    POST /api/ask      question (+ filters) -> cited answer, sources, search scores
    POST /api/search   question (+ filters) -> ranked chunks only, no LLM call

Sign-in: GitHub OAuth, see api/auth.py (/auth/login, /auth/callback, /auth/logout).

The web page in web/ is served from /.
"""
from __future__ import annotations

import logging
import os
import secrets
import threading
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from ingestion.errors import IngestionError
from ingestion.settings import Settings
from retrieval import Filters, RAGPipeline
from retrieval.errors import EmptyDatabaseError, GenerationError, QuestionError, RetrievalError
from retrieval.step1_prepare_question import prepare_question
from retrieval.step2_hybrid_search import hybrid_search
from retrieval.step3_select_sources import select_sources

from . import auth
from .auth import audit, require

log = logging.getLogger("api")
WEB_DIR = Path(__file__).parent.parent / "web"
FAKE_MODE = os.getenv("RAG_FAKE_MODE", "").lower() in ("1", "true", "yes")  # offline testing only


# ===================================================================== state
class State:
    """Holds the one RAGPipeline shared by all requests. If it can't start
    (empty database, missing API key) the error is kept and every request
    tries again, so the server recovers on its own after `python ingest.py`."""

    def __init__(self):
        self.rag: RAGPipeline | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def pipeline(self) -> RAGPipeline:
        with self._lock:
            if self.rag is None:
                try:
                    self.rag = RAGPipeline(Settings(), fake=FAKE_MODE)
                    self.error = None
                except (IngestionError, RetrievalError, ValueError) as e:
                    self.error = str(e)
                    raise HTTPException(status_code=503, detail=self.error) from e
        self.rag.refresh_index_if_needed()  # pick up chunks ingested while running
        return self.rag


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.pipeline()
        log.info("Knowledge base ready: %d chunks", state.rag.store.count())
    except HTTPException:
        log.warning("Not ready yet: %s", state.error)
    yield


app = FastAPI(title="RAG Knowledge Assistant", version="1.0", lifespan=lifespan)

# Signed session cookie (HttpOnly; SameSite=Lax blocks cross-site form posts).
# Without RAG_SESSION_SECRET a throw-away secret is used and every API call
# answers 503 "not configured" until it is set.
app.add_middleware(
    SessionMiddleware,
    secret_key=auth.config.session_secret or secrets.token_urlsafe(32),
    session_cookie="rag_session",
    max_age=auth.config.session_seconds,
    same_site="lax",
    https_only=auth.config.public_url.startswith("https://"),
)
app.include_router(auth.router)


# ==================================================================== models
class FilterIn(BaseModel):
    file_name: str | None = None
    file_type: str | None = Field(None, pattern="^(pdf|docx|txt)$")
    section: str | None = Field(None, max_length=200)
    page: int | None = Field(None, ge=1)
    tables_only: bool = False
    exclude_flagged: bool = False

    def to_filters(self) -> Filters:
        return Filters(file_name=self.file_name or None, file_type=self.file_type or None,
                       section=(self.section or "").strip() or None, page=self.page,
                       tables_only=self.tables_only, exclude_flagged=self.exclude_flagged)


class QuestionIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    filters: FilterIn = Field(default_factory=FilterIn)
    top_k: int | None = Field(None, ge=1, le=10)
    include_search: bool = True  # also return every search hit with its scores


def hit_out(hit) -> dict:
    m = hit.metadata
    return {"file_name": m.get("file_name"), "section": m.get("section"), "page": m.get("page_start"),
            "similarity": float(hit.similarity), "keyword_score": float(hit.keyword_score),
            "semantic_rank": hit.vector_rank, "keyword_rank": hit.keyword_rank,
            "fused_score": round(float(hit.fused_score), 5), "relevant": bool(hit.relevant),
            "has_table": bool(m.get("has_table", False)), "has_ocr": bool(m.get("has_ocr", False)),
            "quality_flags": m.get("quality_flags", "none"), "preview": hit.text[:300]}


def source_out(source, cited: list[int]) -> dict:
    return {"number": source.number, "file_name": source.file_name, "section": source.section,
            "page": source.page, "similarity": float(source.similarity),
            "keyword_score": float(source.keyword_score), "cited": source.number in cited,
            "text": source.text}


# ================================================================= endpoints
@app.get("/api/health")
def health(user: dict = Depends(require("viewer"))):
    try:
        rag = state.pipeline()
    except HTTPException:
        return JSONResponse(status_code=503, content={"ready": False, "error": state.error})
    return {"ready": True, "chunks": rag.store.count(), "embedder": rag.embedder.name,
            "llm": getattr(rag.llm, "name", "?"), "collection": rag.settings.collection_name,
            "top_k": rag.settings.top_k, "fake_mode": FAKE_MODE}


@app.get("/api/files")
def files(user: dict = Depends(require("viewer"))):
    rag = state.pipeline()
    metadatas = rag.store.collection.get(include=["metadatas"])["metadatas"]
    chunks: Counter = Counter()
    info: dict = {}
    sections: dict = defaultdict(set)
    for m in metadatas:
        name = m.get("file_name", "?")
        chunks[name] += 1
        info[name] = m.get("file_type")
        if m.get("section"):
            sections[name].add(m["section"])
    return [{"file_name": n, "file_type": info[n], "chunks": chunks[n],
             "sections": sorted(sections[n])} for n in sorted(chunks)]


@app.post("/api/ask")
def ask(body: QuestionIn, user: dict = Depends(require("viewer"))):
    rag = state.pipeline()
    audit("ask", user["login"], question=body.question[:300], filters=body.filters.to_filters().describe())
    try:
        answer, hits = rag.ask_with_details(body.question, body.filters.to_filters(), body.top_k)
    except QuestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EmptyDatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except GenerationError as e:
        # The answer failed, but retrieval worked: send the passages so the user still gets help
        relevant = [h for h in getattr(e, "hits", []) if h.relevant][: body.top_k or rag.settings.top_k]
        return JSONResponse(status_code=502, content={
            "detail": f"Could not generate an answer: {e}",
            "passages": [hit_out(h) for h in relevant]})

    return {
        "question": answer.question,
        "answer": answer.text,
        "found": answer.found,
        "cited": answer.cited,
        "sources": [source_out(s, answer.cited) for s in answer.sources],
        "warnings": answer.warnings,
        "filters": answer.filters,
        "seconds": answer.seconds,
        "search": [hit_out(h) for h in hits[:15]] if body.include_search else [],
    }


@app.post("/api/search")
def search(body: QuestionIn, user: dict = Depends(require("viewer"))):
    """Retrieval only (steps 1-3): useful to see why a chunk was or wasn't chosen."""
    rag = state.pipeline()
    audit("search", user["login"], question=body.question[:300])
    try:
        query = prepare_question(body.question, rag.embedder, rag.settings)
    except QuestionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    hits = hybrid_search(query, body.filters.to_filters(), rag.store, rag.index, rag.settings)
    sources = select_sources(hits, rag.settings)
    return {"question": query.text, "keywords": query.keywords,
            "selected": len(sources), "hits": [hit_out(h) for h in hits[:20]]}


# The web page (mounted last so /api/* routes win)
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
