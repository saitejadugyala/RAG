"""Tests for the FastAPI backend (offline: fake embeddings + fake LLM)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from ingestion import IngestionPipeline, Settings
from retrieval import RAGPipeline
from retrieval.errors import GenerationError
from retrieval.models import NO_INFO_ANSWER

@pytest.fixture(scope="module")
def rag(tmp_path_factory, rag_docs):
    folder = tmp_path_factory.mktemp("api")
    assert IngestionPipeline(Settings(data_dir=folder), fake_embeddings=True).run([rag_docs])["success"] == 3
    return RAGPipeline(Settings(data_dir=folder), fake=True)


@pytest.fixture(autouse=True)
def sign_in_off(monkeypatch):
    """These tests are about the endpoints, not sign-in (see test_auth.py)."""
    import api.auth as auth
    monkeypatch.setattr(auth, "config", auth.AuthConfig(Settings(auth_mode="off")))


@pytest.fixture
def client(rag, monkeypatch):
    state = api_main.State()
    state.rag = rag
    monkeypatch.setattr(api_main, "state", state)
    rag.llm = type(rag.llm)()  # fresh fake LLM for every test
    with TestClient(api_main.app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ready"] is True and body["chunks"] > 0


def test_files_list_for_the_filter_menus(client):
    files = {f["file_name"]: f for f in client.get("/api/files").json()}
    assert set(files) == {"rag_ingestion_guide.pdf", "rag_retrieval_notes.txt", "rag_generation_guide.docx"}
    assert "5. Chunking strategies" in files["rag_ingestion_guide.pdf"]["sections"]


def test_ask_returns_cited_answer_sources_and_search(client):
    r = client.post("/api/ask", json={"question": "What are the usual BM25 parameters k1 and b?"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] and body["cited"]
    assert 1 <= len(body["sources"]) <= 3
    assert body["sources"][0]["file_name"] == "rag_retrieval_notes.txt"
    assert {"similarity", "keyword_score", "semantic_rank", "relevant"} <= set(body["search"][0])


def test_unrelated_question_gets_no_information(client):
    body = client.post("/api/ask", json={"question": "What is the capital of France?"}).json()
    assert body["found"] is False and body["answer"] == NO_INFO_ANSWER and body["sources"] == []


def test_filters_are_applied(client):
    body = client.post("/api/ask", json={"question": "chunk size tokens overlap",
                                         "filters": {"file_type": "txt"}}).json()
    assert all(h["file_name"].endswith(".txt") for h in body["search"])


def test_top_k_per_request(client):
    body = client.post("/api/ask", json={"question": "BM25 keyword search rare words parameters",
                                         "top_k": 1}).json()
    assert len(body["sources"]) <= 1


def test_search_endpoint_makes_no_llm_call(client, rag, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the LLM must not be called by /api/search")

    monkeypatch.setattr(rag.llm, "complete", explode)
    body = client.post("/api/search", json={"question": "Reciprocal Rank Fusion k = 60"}).json()
    assert body["hits"] and "reciprocal" in body["keywords"]


@pytest.mark.parametrize("payload", [{"question": "hi"}, {"question": "x" * 2001}, {},
                                     {"question": "valid question", "filters": {"file_type": "exe"}},
                                     {"question": "valid question", "top_k": 50}])
def test_bad_requests_are_rejected(client, payload):
    assert client.post("/api/ask", json=payload).status_code == 422


def test_llm_failure_returns_502_with_passages(client, rag, monkeypatch):
    def fail(*args, **kwargs):
        raise GenerationError("model unavailable")

    monkeypatch.setattr(rag.llm, "complete", fail)
    r = client.post("/api/ask", json={"question": "What are the usual BM25 parameters k1 and b?"})
    assert r.status_code == 502
    assert "model unavailable" in r.json()["detail"] and r.json()["passages"]


def test_not_ready_returns_503(monkeypatch, tmp_path):
    state = api_main.State()
    monkeypatch.setattr(api_main, "state", state)

    def not_ready():
        state.error = "The vector database is empty. Run `python ingest.py` first."
        raise api_main.HTTPException(status_code=503, detail=state.error)

    monkeypatch.setattr(state, "pipeline", not_ready)
    with TestClient(api_main.app) as c:
        assert c.get("/api/health").status_code == 503
        assert c.post("/api/ask", json={"question": "anything at all"}).status_code == 503


def test_web_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "RAG Knowledge Assistant" in r.text
    assert client.get("/app.js").status_code == 200 and client.get("/style.css").status_code == 200


def test_replaced_file_is_picked_up_without_restart(tmp_path):
    """Replace a document with an edited version that has the SAME number of chunks:
    the running pipeline must search the new text, not the old one."""
    docs = tmp_path / "docs"
    docs.mkdir()
    note = docs / "note.txt"
    note.write_text("Release notes\n\nThe nightly backup starts at 02:00 and finishes before dawn.\n")
    # one tiny file = every word looks common to BM25, so use a low keyword cut-off here
    settings = Settings(data_dir=tmp_path / "data", min_keyword_score=0.1)
    IngestionPipeline(settings, fake_embeddings=True).run([docs])
    rag = RAGPipeline(settings, fake=True)
    assert rag.retrieve("nightly backup 02:00")[2]

    note.write_text("Release notes\n\nMaintenance happens at 04:30 each Sunday for two hours.\n")
    IngestionPipeline(settings, fake_embeddings=True).run([docs])  # same file name, new content
    assert rag.store.count() == 1 and rag.refresh_index_if_needed() is True
    sources = rag.retrieve("Sunday maintenance 04:30")[2]
    assert sources and "04:30" in sources[0].text
    assert not rag.retrieve("nightly backup 02:00")[2]  # old text is gone


def test_numpy_scores_from_chromadb_can_be_sent_as_json(rag, monkeypatch):
    """Regression: ChromaDB returns numpy numbers. When a numpy similarity PASSED the
    cut-off, `relevant` became numpy.bool and /api/ask crashed with a 500.
    (Fake embeddings never pass 0.30, so use a cut-off every chunk passes.)"""
    import dataclasses
    import json as json_module

    strict = RAGPipeline.__new__(RAGPipeline)
    strict.__dict__.update(rag.__dict__)
    # few candidates per search -> chunks found ONLY by keyword search are in the response;
    # their similarity is computed from ChromaDB's numpy vectors (the crashing case)
    strict.settings = dataclasses.replace(rag.settings, min_similarity=-1.0, min_relative_score=0,
                                          candidates=3, top_k=3)
    state = api_main.State()
    state.rag = strict
    monkeypatch.setattr(api_main, "state", state)
    with TestClient(api_main.app) as c:
        r = c.post("/api/ask", json={"question": "What are the usual BM25 parameters k1 and b?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert any(h["semantic_rank"] is None for h in body["search"]), "no keyword-only hit to test"
    assert all(type(h["relevant"]) is bool for h in body["search"])
    json_module.dumps(body)  # everything is plain JSON


def test_select_sources_returns_plain_python_types():
    import numpy as np

    from retrieval.models import Hit
    from retrieval.step3_select_sources import select_sources

    hits = [Hit("a", "a", {"file_name": "a"}, similarity=np.float64(0.7), keyword_score=np.float64(5.0))]
    select_sources(hits, Settings())
    assert type(hits[0].relevant) is bool
