"""Runs every test question through the real pipeline and scores it.

For each question:
    retrieve (steps 1-3)  ─► retrieval metrics
    answer   (step 4)     ─► generation metrics
Then averages everything, breaks it down by tag, compares with the previous
run, and checks the pass/fail thresholds.
"""
from __future__ import annotations

import json
import statistics
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ingestion.settings import Settings
from retrieval import RAGPipeline
from retrieval.models import Answer

from .dataset import EvalCase
from .step1_retrieval_metrics import retrieval_metrics
from .step2_generation_metrics import generation_metrics

# Minimum scores for the run to PASS (exit code 0). Tune them to your needs.
THRESHOLDS = {
    "retrieval.hit_rate": 0.85,
    "retrieval.mrr": 0.70,
    "retrieval.context_recall": 0.80,
    "generation.fact_recall": 0.80,
    "generation.false_refusal_rate": 0.15,  # maximum, lower is better
    "generation.refusal_accuracy": 0.80,
    "generation.citation_valid_rate": 0.90,
}
LOWER_IS_BETTER = {"generation.false_refusal_rate"}


def run_evaluation(cases: list[EvalCase], settings: Settings, *, fake: bool = False,
                   retrieval_only: bool = False, progress=print) -> dict:
    rag = RAGPipeline(settings, fake=fake)

    results = []
    for number, case in enumerate(cases, start=1):
        progress(f"[{number}/{len(cases)}] {case.id}")
        results.append(_evaluate_case(case, rag, retrieval_only))

    summary = summarise(results, cases)
    summary.update(
        run_id=datetime.now().strftime("%Y%m%d-%H%M%S"),
        settings={
            "llm_model": "none" if retrieval_only else rag.llm.name,
            "embedding_model": rag.embedder.name,
            "collection": settings.collection_name,
            "top_k": settings.top_k,
            "min_similarity": settings.min_similarity,
            "min_relative_score": settings.min_relative_score,
            "min_keyword_score": settings.min_keyword_score,
            "chunk_strategy": settings.chunk_strategy,
        },
        questions=len(cases),
        results=results,
    )
    return summary


def _evaluate_case(case: EvalCase, rag: RAGPipeline, retrieval_only: bool) -> dict:
    row = {"id": case.id, "question": case.question, "answerable": case.answerable,
           "tags": case.tags}
    started = time.time()
    try:
        if retrieval_only:
            rag.retrieve(case.question)
        else:
            answer: Answer = rag.ask(case.question)
            row["answer"] = answer.text
            row["cited"] = answer.cited
        row["sources"] = [f"[{s.number}] {s.label()} (sim {s.similarity:.2f}, kw {s.keyword_score:.1f})"
                          for s in rag.last_sources]
        row["retrieval"] = retrieval_metrics(case, rag.last_sources, rag.last_hits)
        if not retrieval_only:
            row["generation"] = generation_metrics(case, answer)
    except Exception as e:  # one failing question must not stop the evaluation
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc(limit=3)
    row["seconds"] = round(time.time() - started, 2)
    return row


# ======================================================================= summary
def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def summarise(results: list[dict], cases: list[EvalCase]) -> dict:
    ok = [r for r in results if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]

    def get(rows, section, key):
        return [r.get(section, {}).get(key) for r in rows if section in r]

    metrics = {
        "retrieval.hit_rate": _mean(get(answerable, "retrieval", "hit")),
        "retrieval.mrr": _mean(get(answerable, "retrieval", "reciprocal_rank")),
        "retrieval.precision": _mean(get(answerable, "retrieval", "precision")),
        "retrieval.context_recall": _mean(get(answerable, "retrieval", "context_recall")),
        "retrieval.unanswerable_blocked": _mean(get(unanswerable, "retrieval", "no_sources")),
    }
    if any("generation" in r for r in ok):
        answered = get(answerable, "generation", "answered")
        metrics.update({
            "generation.answer_rate": _mean(answered),
            "generation.false_refusal_rate": round(1 - _mean(answered), 3) if answered else None,
            "generation.fact_recall": _mean(get(answerable, "generation", "fact_recall")),
            "generation.citation_valid_rate": _mean(
                [r["generation"]["citation_valid"] for r in answerable
                 if r.get("generation", {}).get("answered")]),
            "generation.citation_precision": _mean(get(answerable, "generation", "citation_precision")),
            "generation.refusal_accuracy": _mean(get(unanswerable, "generation", "correct_refusal")),
        })

    seconds = sorted(r["seconds"] for r in results)
    by_tag = defaultdict(list)
    for r in ok:
        for tag in r["tags"]:
            by_tag[tag].append(r)

    return {
        "metrics": metrics,
        "latency": {"mean_s": _mean(seconds),
                    "p95_s": seconds[min(len(seconds) - 1, int(len(seconds) * 0.95))] if seconds else None},
        "errors": len(results) - len(ok),
        "diagnosis": dict(sorted(_count(r["retrieval"].get("diagnosis") for r in answerable).items())),
        "by_tag": {
            tag: {
                "questions": len(rows),
                "hit_rate": _mean(r["retrieval"].get("hit") for r in rows if r["answerable"]),
                "fact_recall": _mean(r.get("generation", {}).get("fact_recall")
                                     for r in rows if r["answerable"]),
                "refusal_accuracy": _mean(r.get("generation", {}).get("correct_refusal")
                                          for r in rows if not r["answerable"]),
            }
            for tag, rows in sorted(by_tag.items())
        },
        "failures": _failures(results),
    }


def _count(values) -> dict:
    counts: dict = defaultdict(int)
    for v in values:
        if v is not None:
            counts[v] += 1
    return dict(counts)


def _failures(results: list[dict]) -> list[dict]:
    """Questions worth a look, with the reason in plain words."""
    out = []
    for r in results:
        reasons = []
        if "error" in r:
            reasons.append(r["error"])
        else:
            ret, gen = r.get("retrieval", {}), r.get("generation", {})
            if r["answerable"]:
                if ret.get("hit") == 0:
                    reasons.append(f"retrieval: correct source not sent ({ret.get('diagnosis')})")
                if gen and gen.get("answered") == 0:
                    reasons.append("generation: answered 'no information' (false refusal)")
                elif gen and gen.get("fact_recall", 1) < 1:
                    reasons.append(f"generation: missing key facts (recall {gen['fact_recall']:.2f})")
                if gen.get("answered") and gen.get("citation_valid") == 0:
                    reasons.append("generation: no valid citation")
            elif gen and gen.get("correct_refusal") == 0:
                reasons.append("generation: answered a question the documents can't answer")
        if reasons:
            out.append({"id": r["id"], "question": r["question"], "reasons": reasons,
                        "answer": r.get("answer"), "sources": r.get("sources")})
    return out


# ================================================================ thresholds
def check_thresholds(metrics: dict) -> list[str]:
    failed = []
    for name, limit in THRESHOLDS.items():
        value = metrics.get(name)
        if value is None:
            continue
        bad = value > limit if name in LOWER_IS_BETTER else value < limit
        if bad:
            sign = "≤" if name in LOWER_IS_BETTER else "≥"
            failed.append(f"{name} = {value:.3f} (needs {sign} {limit})")
    return failed


def previous_report(reports_dir: Path, current_run: str) -> dict | None:
    files = sorted(reports_dir.glob("eval-*.json"))
    files = [f for f in files if current_run not in f.name]
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return None
