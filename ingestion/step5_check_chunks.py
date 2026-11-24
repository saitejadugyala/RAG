"""STEP 5 - Check every chunk's quality.

  REJECT  the chunk is useless or would break embedding -> not stored
  FLAG    the chunk is kept, with a note in its `quality_flags` metadata,
          so you can filter or down-rank it at search time

Rejected and flagged chunks are also listed in data/reports/ for review.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

from .models import Chunk
from .settings import Settings

WORD = re.compile(r"[A-Za-z]{2,}")


def check_chunks(chunks: list[Chunk], settings: Settings) -> dict:
    seen: set[str] = set()
    counts: Counter = Counter()

    for chunk in chunks:
        reason = _reject_reason(chunk, settings, seen, only_chunk=len(chunks) == 1)
        if reason:
            chunk.rejected = True
            chunk.flags.append(f"reject:{reason}")
        else:
            chunk.flags += [f"flag:{f}" for f in _quality_flags(chunk, settings)]
        counts.update(chunk.flags)
        chunk.metadata["quality_flags"] = ",".join(chunk.flags) or "none"

    return {
        "total": len(chunks),
        "ok": sum(not c.rejected and not c.flags for c in chunks),
        "flagged": sum(not c.rejected and bool(c.flags) for c in chunks),
        "rejected": sum(c.rejected for c in chunks),
        "reasons": dict(counts),
    }


def _reject_reason(chunk: Chunk, settings: Settings, seen: set[str], only_chunk: bool) -> str | None:
    text = chunk.text.strip()
    is_table = chunk.metadata.get("has_table", False)

    if not text:
        return "empty"
    if chunk.tokens > settings.embedding_max_tokens:
        return "too_long_for_embedding"
    if chunk.tokens < max(3, settings.min_chunk_tokens // 4) and not only_chunk and not is_table:
        return "too_short"
    if not WORD.search(text) and not is_table:
        return "no_words"
    if _letters_digits_ratio(text) < 0.3:
        return "mostly_symbols"

    fingerprint = hashlib.md5(re.sub(r"\W+", " ", text.lower()).encode()).hexdigest()
    if fingerprint in seen:
        return "duplicate"
    seen.add(fingerprint)
    return None


def _quality_flags(chunk: Chunk, settings: Settings) -> list[str]:
    text = chunk.text
    is_table = chunk.metadata.get("has_table", False)
    words = WORD.findall(text)
    flags = []

    confidence = chunk.metadata.get("ocr_confidence")
    if confidence is not None and confidence < 70:
        flags.append("low_ocr_confidence")
    if len(words) >= 20 and len({w.lower() for w in words}) / len(words) < 0.25:
        flags.append("repetitive")
    if chunk.tokens < settings.min_chunk_tokens * 2:
        flags.append("short")
    if chunk.tokens > settings.chunk_size * 1.1:
        flags.append("over_chunk_size")
    if not is_table and sum(c.isdigit() for c in text) > sum(c.isalpha() for c in text):
        flags.append("mostly_numbers")
    if not is_table and _letters_digits_ratio(text) < 0.6:
        flags.append("many_symbols")
    if chunk.index > 0 and text[:1].islower() and settings.chunk_strategy != "fixed":
        flags.append("starts_mid_sentence")
    return flags


def _letters_digits_ratio(text: str) -> float:
    text = re.sub(r"\s", "", text)
    return sum(c.isalnum() for c in text) / len(text) if text else 0.0
