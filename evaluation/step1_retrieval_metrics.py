"""STEP 1 - Retrieval metrics: did search find the chunk that holds the answer?

Per question (answerable questions only):
  hit@k            1 if a correct source is among the top-k sources sent to the LLM
  reciprocal rank  1 / position of the first correct source (1st = 1.0, 2nd = 0.5, ...)
  precision@k      share of the sent sources that are correct
  context recall   share of the key facts that appear somewhere in the sent sources
                   (if a fact isn't in the context, the LLM can't possibly answer it)
  diagnosis        WHY a miss happened:
                     ok                       correct source was sent to the LLM
                     ranked_below_top_k       found and relevant, but not in the top k
                     dropped_by_relevance     found by search, removed by the score cut-offs
                     not_found_by_search      neither search found it

Unanswerable questions:
  no_sources       1 if the relevance check let nothing through (ideal: the LLM
                   isn't even asked)
"""
from __future__ import annotations

import re

from .dataset import EvalCase


def normalise(text: str) -> str:
    text = text.lower().replace("’", "'").replace(" ", " ")
    return re.sub(r"\s+", " ", text)


def facts_found(must_include: list[list[str]], text: str) -> float:
    """Share of fact groups with at least one accepted spelling present in text."""
    if not must_include:
        return 1.0
    haystack = normalise(text)
    found = sum(any(normalise(option) in haystack for option in group) for group in must_include)
    return found / len(must_include)


def is_expected(text: str, file_name: str, case: EvalCase) -> bool:
    return any(file_name == e["file"] and normalise(e["contains"]) in normalise(text)
               for e in case.expected_sources)


def retrieval_metrics(case: EvalCase, sources: list, hits: list) -> dict:
    if not case.answerable:
        return {"no_sources": 1.0 if not sources else 0.0, "sources_sent": len(sources)}

    correct = [is_expected(s.text, s.file_name, case) for s in sources]
    first = next((i for i, ok in enumerate(correct, start=1) if ok), None)

    matching_hits = [h for h in hits if is_expected(h.text, h.metadata.get("file_name", ""), case)]
    if first:
        diagnosis = "ok"
    elif any(h.relevant for h in matching_hits):
        diagnosis = "ranked_below_top_k"
    elif matching_hits:
        diagnosis = "dropped_by_relevance"
    else:
        diagnosis = "not_found_by_search"

    best = matching_hits[0] if matching_hits else None
    return {
        "hit": 1.0 if first else 0.0,
        "reciprocal_rank": 1 / first if first else 0.0,
        "precision": sum(correct) / len(sources) if sources else 0.0,
        "context_recall": facts_found(case.must_include, "\n".join(s.text for s in sources)),
        "diagnosis": diagnosis,
        "search_rank": hits.index(best) + 1 if best else None,
        "best_similarity": best.similarity if best else None,
        "best_keyword_score": best.keyword_score if best else None,
        "sources_sent": len(sources),
    }
