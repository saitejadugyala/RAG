"""STEP 3 - Keep only relevant chunks and pick the top 3.

Search always returns *something*, even for a question the documents can't
answer. So every hit must pass two relevance checks first:

  a. fixed cut-off      cosine similarity >= min_similarity      (default 0.30)
                        OR keyword score  >= min_keyword_score   (default 4.0)

  b. relative cut-off   cosine similarity >= min_relative_score x the best
                        similarity found for this question       (default 0.6)

  Example: best = 0.66  ->  relative limit = 0.66 x 0.6 = 0.40
           0.55 kept, 0.30 dropped (it passed the fixed cut-off, but is far
           weaker than the best match, so it would only add noise)

The relative cut-off adapts to each question: for a vague question where
everything scores lower, the limit drops with it.

Strong keyword matches (an exact code, phone extension or name) are exempt
from the relative cut-off, because their meaning-based score is often low.

If NO hit passes, the pipeline answers "no information" without calling the
LLM at all - it can't invent an answer from unrelated text.
"""
from __future__ import annotations

from ingestion.settings import Settings

from .models import Hit, Source


def select_sources(hits: list[Hit], settings: Settings) -> list[Source]:
    # a. fixed cut-off
    for hit in hits:
        # bool(): a numpy comparison gives numpy.bool, which can't be sent as JSON
        hit.relevant = bool(hit.similarity >= settings.min_similarity
                            or _strong_keyword_match(hit, settings))

    # b. relative cut-off, measured against the best similarity that passed (a)
    passed = [h for h in hits if h.relevant]
    best_similarity = max((h.similarity for h in passed), default=0.0)
    if best_similarity > 0:
        limit = best_similarity * settings.min_relative_score
        for hit in passed:
            if hit.similarity < limit and not _strong_keyword_match(hit, settings):
                hit.relevant = False

    best = [h for h in hits if h.relevant][: settings.top_k]  # hits are already in fused order
    return [
        Source(
            number=i,
            text=h.text,
            file_name=h.metadata.get("file_name", "unknown"),
            section=h.metadata.get("section"),
            page=h.metadata.get("page_start"),
            similarity=h.similarity,
            keyword_score=h.keyword_score,
        )
        for i, h in enumerate(best, start=1)
    ]


def _strong_keyword_match(hit: Hit, settings: Settings) -> bool:
    return bool(hit.keyword_score >= settings.min_keyword_score)
