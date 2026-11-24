"""Tests for the evaluation: every metric is checked on hand-made inputs whose
correct score is known, plus one offline end-to-end run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.dataset import EvalCase, load_dataset
from evaluation.report import write_reports
from evaluation.runner import check_thresholds, run_evaluation, summarise
from evaluation.step1_retrieval_metrics import facts_found, retrieval_metrics
from evaluation.step2_generation_metrics import generation_metrics
from ingestion import IngestionPipeline, Settings
from retrieval.models import NO_INFO_ANSWER, Answer, Hit, Source

DATASET = Path(__file__).parent.parent / "eval" / "dataset.jsonl"

CASE = EvalCase(id="leave", question="How many days of annual leave?", answerable=True,
                expected_sources=[{"file": "handbook.docx", "contains": "Annual leave | 25"}],
                must_include=[["25"], ["days", "day"]], reference="25 days per year.")
NO_ANSWER_CASE = EvalCase(id="none", question="CEO salary?", answerable=False)


def source(number, text, file="handbook.docx"):
    return Source(number=number, text=text, file_name=file, section=None, page=None,
                  similarity=0.5, keyword_score=0.0)


def hit(text, file="handbook.docx", relevant=True):
    return Hit(id=text, text=text, metadata={"file_name": file}, similarity=0.5, relevant=relevant)


# --------------------------------------------------------------- the dataset
def test_project_dataset_is_valid():
    cases = load_dataset(DATASET)
    assert len(cases) >= 20
    assert any(not c.answerable for c in cases)


def test_dataset_problems_are_reported(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "question": "q?", "answerable": True}) + "\n")
    with pytest.raises(ValueError, match="expected_sources"):
        load_dataset(bad)


# ------------------------------------------------------------ retrieval metrics
def test_facts_found_accepts_any_spelling():
    assert facts_found([["25"], ["five", "5"]], "Twenty-five... no: 25 days, up to 5") == 1.0
    assert facts_found([["25"], ["march"]], "25 days") == 0.5


def test_correct_source_in_second_place():
    sources = [source(1, "Remote work rules"), source(2, "| Annual leave | 25 | Yes |")]
    m = retrieval_metrics(CASE, sources, [hit(s.text) for s in sources])
    assert m["hit"] == 1.0 and m["reciprocal_rank"] == 0.5 and m["precision"] == 0.5
    assert m["context_recall"] == 0.5  # "25" is there, "days" isn't
    assert m["diagnosis"] == "ok"


def test_miss_diagnosis():
    wanted = "| Annual leave | 25 | Yes |"
    other = [source(1, "Remote work rules")]
    assert retrieval_metrics(CASE, other, [hit("x"), hit(wanted)])["diagnosis"] == "ranked_below_top_k"
    assert retrieval_metrics(CASE, other, [hit(wanted, relevant=False)])["diagnosis"] == "dropped_by_relevance"
    assert retrieval_metrics(CASE, other, [hit("x")])["diagnosis"] == "not_found_by_search"
    assert retrieval_metrics(CASE, other, [])["hit"] == 0.0


def test_same_text_in_wrong_file_does_not_count():
    sources = [source(1, "| Annual leave | 25 | Yes |", file="other.pdf")]
    assert retrieval_metrics(CASE, sources, [])["hit"] == 0.0


def test_unanswerable_retrieval():
    assert retrieval_metrics(NO_ANSWER_CASE, [], [])["no_sources"] == 1.0
    assert retrieval_metrics(NO_ANSWER_CASE, [source(1, "x")], [])["no_sources"] == 0.0


# ----------------------------------------------------------- generation metrics
def answer(text, found=True, cited=(1,), sources=None):
    return Answer(question="q", text=text, found=found, cited=list(cited),
                  sources=sources or [source(1, "| Annual leave | 25 | Yes |"), source(2, "Remote")])


def test_good_answer_scores_full_marks():
    m = generation_metrics(CASE, answer("You get 25 days [1]."))
    assert m == {"answered": 1.0, "fact_recall": 1.0, "citation_valid": 1.0, "citation_precision": 1.0}


def test_wrong_citation_lowers_citation_precision():
    m = generation_metrics(CASE, answer("You get 25 days [2].", cited=(2,)))
    assert m["citation_precision"] == 0.0 and m["fact_recall"] == 1.0


def test_false_refusal():
    m = generation_metrics(CASE, answer(NO_INFO_ANSWER, found=False, cited=()))
    assert m["answered"] == 0.0 and m["fact_recall"] == 0.0


def test_refusal_on_unanswerable_question():
    assert generation_metrics(NO_ANSWER_CASE, answer(NO_INFO_ANSWER, found=False))["correct_refusal"] == 1.0
    assert generation_metrics(NO_ANSWER_CASE, answer("The CEO earns a lot [1]."))["correct_refusal"] == 0.0


# ----------------------------------------------------- summary and thresholds
def test_summary_averages_and_thresholds():
    rows = [
        {"id": "a", "question": "q", "answerable": True, "tags": ["t"], "seconds": 1.0,
         "retrieval": {"hit": 1.0, "reciprocal_rank": 1.0, "precision": 1.0, "context_recall": 1.0,
                       "diagnosis": "ok"},
         "generation": {"answered": 1.0, "fact_recall": 1.0, "citation_valid": 1.0,
                        "citation_precision": 1.0}},
        {"id": "b", "question": "q", "answerable": True, "tags": ["t"], "seconds": 3.0,
         "retrieval": {"hit": 0.0, "reciprocal_rank": 0.0, "precision": 0.0, "context_recall": 0.0,
                       "diagnosis": "not_found_by_search"},
         "generation": {"answered": 0.0, "fact_recall": 0.0, "citation_valid": 0.0,
                        "citation_precision": None}},
    ]
    s = summarise(rows, [])
    assert s["metrics"]["retrieval.hit_rate"] == 0.5
    assert s["metrics"]["generation.false_refusal_rate"] == 0.5
    assert s["diagnosis"] == {"not_found_by_search": 1, "ok": 1}
    assert [f["id"] for f in s["failures"]] == ["b"]
    failed = check_thresholds(s["metrics"])
    assert any("hit_rate" in f for f in failed) and any("false_refusal_rate" in f for f in failed)


# ------------------------------------------------------- end-to-end (offline)
def test_offline_evaluation_run_writes_reports(tmp_path, rag_docs):
    settings = Settings(data_dir=tmp_path)
    IngestionPipeline(settings, fake_embeddings=True).run([rag_docs])
    cases = load_dataset(DATASET)
    summary = run_evaluation(cases, settings, fake=True, progress=lambda _: None)

    assert summary["questions"] == len(cases) and summary["errors"] == 0
    for name in ("retrieval.hit_rate", "retrieval.mrr", "generation.fact_recall",
                 "generation.refusal_accuracy", "generation.citation_precision"):
        assert 0.0 <= summary["metrics"][name] <= 1.0
    json_file, md_file = write_reports(summary, [], None, settings.reports_dir)
    assert json.loads(json_file.read_text())["passed"] is True
    assert "## Metrics" in md_file.read_text()


def test_retrieval_only_run_makes_no_generation_metrics(tmp_path, rag_docs):
    settings = Settings(data_dir=tmp_path)
    IngestionPipeline(settings, fake_embeddings=True).run([rag_docs / "rag_retrieval_notes.txt"])
    cases = [c for c in load_dataset(DATASET) if c.id.startswith("ret-")]
    summary = run_evaluation(cases, settings, fake=True, retrieval_only=True, progress=lambda _: None)
    assert "retrieval.hit_rate" in summary["metrics"]
    assert not any(k.startswith("generation.") for k in summary["metrics"])
