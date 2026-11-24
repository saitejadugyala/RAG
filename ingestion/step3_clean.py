"""STEP 3 - Clean the text and remove noise.

Keeps every useful sentence and removes what hurts search quality:

  a. Fix characters      ligatures (ﬁ -> fi), invisible characters, curly quotes
  b. Remove noise lines  page numbers, "......" leaders, symbol-only lines,
                         OCR garbage
  c. Remove headers and  short lines repeated at the top/bottom of most pages
     footers
  d. Repair line breaks  "re-\\nport" -> "report", join hard-wrapped lines
  e. Remove duplicates   identical blocks (e.g. the same disclaimer twice)

A document with too little text left afterwards is rejected as empty.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

from .errors import EmptyDocumentError
from .models import Block, Document
from .settings import Settings


def clean_document(doc: Document, settings: Settings) -> Document:
    raw_chars = sum(len(b.text) for b in doc.blocks)
    headers_footers = _find_headers_footers(doc)

    cleaned: list[Block] = []
    seen: set[str] = set()
    duplicates = 0
    for block in doc.blocks:
        if block.kind == "table":
            text = _clean_table(block.text)
        else:
            text = _clean_text(block.text, block.kind, headers_footers.get(block.page, set()))
        if not text:
            continue

        key = hashlib.md5(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()
        if key in seen:  # (e) duplicate block
            duplicates += 1
            continue
        seen.add(key)
        cleaned.append(Block(text, block.kind, block.page, block.ocr_confidence))

    doc.blocks = cleaned
    clean_chars = sum(len(b.text) for b in cleaned)
    doc.stats.update(
        raw_chars=raw_chars,
        clean_chars=clean_chars,
        noise_removed_pct=round(100 * (1 - clean_chars / raw_chars), 1) if raw_chars else 0.0,
        duplicate_blocks_removed=duplicates,
    )

    if clean_chars < settings.min_document_chars:
        raise EmptyDocumentError(
            f"Only {clean_chars} characters of useful text "
            f"(minimum {settings.min_document_chars}). The file may be empty or image-only.")
    return doc


# ---------------------------------------------------------------- (a) characters
INVISIBLE = re.compile(r"[​‌‍⁠﻿­]")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PLAIN_PUNCTUATION = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": " - ", "…": "...",
})
BULLET = re.compile(r"^[•●▪■‣⁃∙*·]\s*", re.M)


def fix_characters(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # also turns ligatures like ﬁ into fi
    text = INVISIBLE.sub("", text)
    text = CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.translate(PLAIN_PUNCTUATION)


# ---------------------------------------------------------------- (b) noise lines
PAGE_NUMBER = [
    re.compile(r"^(page\s*)?\d{1,4}(\s*of\s*\d{1,4})?$", re.I),  # "3", "Page 3 of 10"
    re.compile(r"^[-–—]\s*\d{1,4}\s*[-–—]$"),  # "- 3 -"
    re.compile(r"^[ivxlcdm]{1,6}$", re.I),  # "iv"
]
DOT_LEADER = re.compile(r"\.{4,}\s*\d*$")  # "Introduction ........ 3"
SYMBOLS_ONLY = re.compile(r"^[\W_]+$")


def is_noise_line(line: str, kind: str) -> bool:
    if not line:
        return False
    if any(p.match(line) for p in PAGE_NUMBER) or DOT_LEADER.search(line) or SYMBOLS_ONLY.match(line):
        return True
    # OCR garbage: very short lines or lines that are mostly symbols, e.g. "~=|;'_"
    return kind.startswith("ocr") and (len(line) < 3 or _letters_digits_ratio(line) < 0.5)


def _letters_digits_ratio(text: str) -> float:
    text = text.replace(" ", "")
    return sum(ch.isalnum() for ch in text) / len(text) if text else 0.0


# ---------------------------------------------------------------- (c) headers/footers
EDGE = 3  # headers/footers are in the first/last 3 lines of a page


def _tidy(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).strip()


def _same_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())  # "Page 3" matches "Page 4"


def _find_headers_footers(doc: Document) -> dict[int, set[str]]:
    """Return {page: lines to remove}. A line counts as a header/footer if it
    sits at the top or bottom of at least half the pages (and at least 3)."""
    page_lines: dict[int, list[str]] = {}
    for block in doc.blocks:
        if block.page is not None and block.kind in ("text", "ocr_page"):
            lines = (_tidy(ln) for ln in fix_characters(block.text).split("\n"))
            page_lines.setdefault(block.page, []).extend(ln for ln in lines if ln)
    if len(page_lines) < 3:
        return {}

    edges = {page: lines[:EDGE] + lines[-EDGE:] for page, lines in page_lines.items()}
    counts = Counter(key for lines in edges.values()
                     for key in {_same_line(ln) for ln in lines if len(ln) <= 120})
    repeated = {key for key, n in counts.items() if n >= max(3, len(edges) // 2)}
    return {page: {ln for ln in lines if _same_line(ln) in repeated}
            for page, lines in edges.items()}


# ---------------------------------------------------------------- (d) line breaks
HYPHEN_AT_LINE_END = re.compile(r"(\w)-\n(\w)")


def repair_line_breaks(text: str) -> str:
    """Join lines that were wrapped by the PDF/OCR layout, keep real paragraphs,
    headings and list items on their own lines."""
    text = HYPHEN_AT_LINE_END.sub(r"\1\2", text)
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
        if not lines:
            continue
        joined = [lines[0]]
        for line in lines[1:]:
            previous = joined[-1]
            new_item = (line.startswith(("- ", "#", "|"))
                        or re.match(r"^(\d+[.)]|[a-z][.)])\s", line)
                        or previous.startswith("#"))
            short_sentence = previous.endswith((".", ":", "?", "!", ";")) and len(previous) < 60
            if new_item or (short_sentence and line[:1].isupper()):
                joined.append(line)
            else:
                joined[-1] = previous + " " + line
        paragraphs.append("\n".join(joined))
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------- putting it together
def _clean_text(text: str, kind: str, headers_footers: set[str]) -> str:
    text = BULLET.sub("- ", fix_characters(text))
    kept = []
    for line in text.split("\n"):
        line = _tidy(line)
        if line in headers_footers or is_noise_line(line, kind):
            continue
        kept.append(line)
    text = repair_line_breaks("\n".join(kept))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clean_table(text: str) -> str:
    lines = [_tidy(ln) for ln in fix_characters(text).split("\n")]
    return "\n".join(ln for ln in lines if ln)
