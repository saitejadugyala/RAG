"""STEP 2 - Generation metrics: is the answer right, and does it cite properly?

These are rule-based (no LLM needed, free and repeatable).

Answerable questions:
  answered            1 if the system gave an answer (not "no information")
  fact recall         share of key facts (must_include) present in the answer
  citation valid      1 if the answer cites at least one source, all numbers real
  citation precision  share of cited sources that really contain the answer

Unanswerable questions:
  correct refusal     1 if the reply was "no information"
"""
from __future__ import annotations

from .dataset import EvalCase
from .step1_retrieval_metrics import facts_found, is_expected


def generation_metrics(case: EvalCase, answer) -> dict:
    if not case.answerable:
        return {"correct_refusal": 0.0 if answer.found else 1.0}

    if not answer.found:
        return {"answered": 0.0, "fact_recall": 0.0, "citation_valid": 0.0,
                "citation_precision": None}

    numbers = {s.number: s for s in answer.sources}
    cited = [numbers[n] for n in answer.cited if n in numbers]
    correct_cited = [s for s in cited if is_expected(s.text, s.file_name, case)]
    return {
        "answered": 1.0,
        "fact_recall": facts_found(case.must_include, answer.text),
        "citation_valid": 1.0 if cited and len(cited) == len(answer.cited) else 0.0,
        "citation_precision": len(correct_cited) / len(cited) if cited else 0.0,
    }
