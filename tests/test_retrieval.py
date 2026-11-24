"""Tests for the retrieval + generation pipeline (offline: fake embeddings + fake LLM).

Fake embeddings carry no meaning, so in these tests relevance comes from the
keyword (BM25) side of the hybrid search; the semantic side is tested for
plumbing (filters, fusion, cosine) rather than for meaning.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import IngestionPipeline, Settings
from retrieval import Filters, RAGPipeline
from retrieval.errors import EmptyDatabaseError, GenerationError
from retrieval.models import NO_INFO_ANSWER
from retrieval.step2_hybrid_search import chroma_where

SCRIPTS = Path(__file__).parent.parent / "scripts"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """These tests ask about the sample company documents (scripts/make_docs.py),
    generated into a temporary folder so they don't depend on what is in docs/."""
    import sys

    sys.path.insert(0, str(SCRIPTS))
    import make_docs

    docs = tmp_path_factory.mktemp("company_docs")
    make_docs.main(docs)
    folder = tmp_path_factory.mktemp("rag")
    summary = IngestionPipeline(Settings(data_dir=folder), fake_embeddings=True).run([docs])
    assert summary["success"] == 7
    return folder


@pytest.fixture
def rag(data_dir):
    return RAGPipeline(Settings(data_dir=data_dir), fake=True)


class ScriptedLLM:
    """Returns a fixed reply and remembers whether it was called."""

    name = "scripted"

    def __init__(self, reply: dict | str):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return self.reply


# ----------------------------------------------------------- answers + citations
def test_answer_comes_with_a_valid_citation(rag):
    answer = rag.ask("How many days of annual leave per year?")
    assert answer.found
    assert "annual leave" in answer.text.lower()  # the fake LLM copies a source sentence
    assert answer.cited and all(1 <= n <= len(answer.sources) for n in answer.cited)
    assert len(answer.sources) <= 3
    cited = answer.sources[answer.cited[0] - 1]
    assert cited.file_name == "employee_handbook.docx"


def test_unrelated_question_says_no_information_without_calling_the_llm(rag):
    rag.llm = ScriptedLLM({"found": True, "answer": "Paris [1]", "citations": [1]})
    answer = rag.ask("What is the capital of France?")
    assert not answer.found
    assert answer.text == NO_INFO_ANSWER
    assert answer.sources == []
    assert rag.llm.calls == 0  # nothing relevant -> the LLM is never asked


def test_llm_saying_not_found_gives_no_information(rag):
    rag.llm = ScriptedLLM({"found": False, "answer": "", "citations": []})
    answer = rag.ask("How many days of annual leave per year?")
    assert rag.llm.calls == 1
    assert not answer.found and answer.text == NO_INFO_ANSWER


def test_citations_to_missing_sources_are_removed(rag):
    rag.llm = ScriptedLLM({"found": True, "answer": "Annual leave is 25 days [1][7].",
                           "citations": [1, 7]})
    answer = rag.ask("How many days of annual leave per year?")
    assert "[7]" not in answer.text and "[1]" in answer.text
    assert answer.cited == [1]
    assert any("don't exist" in w for w in answer.warnings)


def test_broken_llm_reply_raises_generation_error(rag):
    rag.llm = ScriptedLLM("this is not json")
    with pytest.raises(GenerationError):
        rag.ask("How many days of annual leave per year?")


# ------------------------------------------------------------------ hybrid search
def test_keyword_search_finds_exact_terms(rag):
    _, hits, sources = rag.retrieve("What is the helpdesk extension 4400?")
    assert sources, "an exact number should be found by keyword search"
    best = hits[0]
    assert best.keyword_rank == 1 and best.metadata["file_name"] == "employee_handbook.docx"
    assert "4400" in best.text


def test_fused_ranking_and_scores_are_filled(rag):
    _, hits, _ = rag.retrieve("refund for annual plans")
    assert hits == sorted(hits, key=lambda h: h.fused_score, reverse=True)
    for h in hits:
        assert -1.0 <= h.similarity <= 1.0  # every hit has a cosine similarity
        assert h.vector_rank is not None or h.keyword_rank is not None


def test_top_k_limits_the_sources(data_dir):
    rag = RAGPipeline(Settings(data_dir=data_dir, top_k=1, min_similarity=-1), fake=True)
    _, _, sources = rag.retrieve("plan price dashboards")
    assert len(sources) == 1


# --------------------------------------------------------------- metadata filters
def test_file_type_filter(rag):
    _, hits, _ = rag.retrieve("plan price dashboards", Filters(file_type="txt"))
    assert hits and all(h.metadata["file_type"] == "txt" for h in hits)


def test_file_name_and_tables_only_filter(rag):
    _, hits, _ = rag.retrieve("plan price", Filters(file_name="product_manual.pdf", tables_only=True))
    assert hits and all(h.metadata["file_name"] == "product_manual.pdf" and h.metadata["has_table"]
                        for h in hits)


def test_page_filter(rag):
    _, hits, _ = rag.retrieve("troubleshooting dashboard slow", Filters(page=3))
    assert hits and all(h.metadata["page_start"] <= 3 <= h.metadata["page_end"] for h in hits)


def test_section_filter(rag):
    _, hits, _ = rag.retrieve("limits", Filters(section="expenses"))
    assert hits and all("expenses" in h.metadata["section"].lower() for h in hits)


def test_filter_that_matches_nothing_gives_no_information(rag):
    answer = rag.ask("annual leave", Filters(file_name="does_not_exist.pdf"))
    assert not answer.found and answer.text == NO_INFO_ANSWER


def test_chroma_where_clause():
    assert chroma_where(Filters()) is None
    assert chroma_where(Filters(file_type="PDF")) == {"file_type": "pdf"}
    assert chroma_where(Filters(file_type="pdf", tables_only=True)) == {
        "$and": [{"file_type": "pdf"}, {"has_table": True}]}


# ------------------------------------------------------------------ edge cases
def test_empty_database_is_reported(tmp_path):
    with pytest.raises(EmptyDatabaseError):
        RAGPipeline(Settings(data_dir=tmp_path), fake=True)


def test_questions_are_logged(rag, data_dir):
    rag.ask("How many days of annual leave per year?")
    log = (data_dir / "reports" / "questions.jsonl").read_text().strip().splitlines()
    assert json.loads(log[-1])["question"] == "How many days of annual leave per year?"


# ------------------------------------------------------------ relative cut-off
def _hit(name, similarity, keyword_score=0.0):
    from retrieval.models import Hit
    return Hit(id=name, text=name, metadata={"file_name": name},
               similarity=similarity, keyword_score=keyword_score)


def test_relative_cutoff_drops_weak_sources():
    from retrieval.step3_select_sources import select_sources

    # the real run: 0.66 and 0.55 are about leave, 0.30 is "Remote Work"
    hits = [_hit("leave table", 0.66), _hit("leave policy", 0.55), _hit("remote work", 0.30)]
    sources = select_sources(hits, Settings())  # limit = 0.66 x 0.6 = 0.396
    assert [s.file_name for s in sources] == ["leave table", "leave policy"]


def test_relative_cutoff_keeps_strong_keyword_matches():
    from retrieval.step3_select_sources import select_sources

    hits = [_hit("meaning match", 0.70), _hit("exact code 4400", 0.31, keyword_score=6.5)]
    names = [s.file_name for s in select_sources(hits, Settings())]
    assert names == ["meaning match", "exact code 4400"]


def test_relative_cutoff_adapts_to_low_scoring_questions():
    from retrieval.step3_select_sources import select_sources

    # a vague question: everything scores lower, but 0.38 is still close to the best
    hits = [_hit("a", 0.45), _hit("b", 0.38), _hit("c", 0.31)]  # limit = 0.27
    assert len(select_sources(hits, Settings())) == 3


def test_relative_cutoff_can_be_turned_off():
    from retrieval.step3_select_sources import select_sources

    hits = [_hit("a", 0.66), _hit("b", 0.30)]
    assert len(select_sources(hits, Settings(min_relative_score=0))) == 2
