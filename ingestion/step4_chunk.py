"""STEP 4 - Split the document into chunks.

Sizes are counted in tokens with the same tokenizer the OpenAI embedding
model uses (cl100k_base), so chunks never exceed what the model accepts.

Strategies (settings.chunk_strategy):
  structure  (default) keep heading sections and tables separate, then use
             recursive splitting inside each section when content is too large
  recursive  split by paragraph -> line -> sentence -> word until each piece
             fits, then pack pieces into chunks with overlap
  sentence   pack whole sentences; never cuts a sentence in half
  fixed      fixed-size token windows with overlap (fast, ignores structure)

Structure chunking works on the document's MAIN section level: the highest heading
level that repeats (page topics in a slide-like PDF, chapters in a Word file with one
title). A new chunk group starts at every main heading; smaller sub-headings inside a
section stay together and are packed up to the chunk size, so a section made of many
short labelled paragraphs ("Core idea", "Example", ...) doesn't become many tiny chunks.

For every strategy:
  * a table becomes its own chunk; a big table is split by rows and the header
    row is repeated in every piece
  * tiny leftover pieces are merged into the previous chunk
  * each chunk keeps its file, page range, section heading and content types
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from .errors import ChunkingError
from .models import Chunk, Document
from .settings import Settings

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
JOINER_TOKENS = 2  # chunks join pieces with "\n\n"


@lru_cache(maxsize=1)
def _tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, disallowed_special=()))


@dataclass
class Piece:
    """A paragraph (or part of one) with where it came from."""

    text: str
    tokens: int
    kind: str
    page: int | None
    section: str | None
    ocr_confidence: float | None = None
    subsection: str | None = None  # a smaller heading inside the section (structure mode)

    def copy_with(self, text: str) -> "Piece":
        return Piece(text, count_tokens(text), self.kind, self.page, self.section,
                     self.ocr_confidence, self.subsection)


# ===================================================================== main
def chunk_document(doc: Document, settings: Settings) -> list[Chunk]:
    try:
        groups = _split_into_pieces(doc, structure_aware=settings.chunk_strategy == "structure")
        chunks_as_pieces: list[list[Piece]] = []
        for is_table, pieces in groups:
            if is_table:
                chunks_as_pieces += [[p] for p in _split_table(pieces[0], settings.chunk_size)]
            else:
                chunks_as_pieces += _chunk_text(pieces, settings)
        chunks_as_pieces = _merge_tiny(chunks_as_pieces, settings)
    except Exception as e:
        raise ChunkingError(f"Chunking failed: {e}") from e
    if not chunks_as_pieces:
        raise ChunkingError("No chunks were produced")

    chunks = [_make_chunk(doc, i, pieces, settings) for i, pieces in enumerate(chunks_as_pieces)]
    for chunk in chunks:
        chunk.metadata["total_chunks"] = len(chunks)
    return chunks


def _chunk_text(pieces: list[Piece], settings: Settings) -> list[list[Piece]]:
    size, overlap = settings.chunk_size, settings.chunk_overlap
    if settings.chunk_strategy == "fixed":
        return _fixed_windows(pieces, size, overlap)
    if settings.chunk_strategy == "sentence":
        return _pack(_into_sentences(pieces, size), size, overlap)
    if settings.chunk_strategy == "structure":
        return _pack_subsections(pieces, size, overlap)
    return _pack(_shrink_to_fit(pieces, size), size, overlap)


def _pack_subsections(pieces: list[Piece], size: int, overlap: int) -> list[list[Piece]]:
    """Structure mode, inside one main section: keep every sub-section WHOLE.
    Small sub-sections are combined up to `size`; a sub-section is only split
    (with overlap) when it is bigger than a chunk on its own. So short labelled
    notes end up together, and Q&A-style sections stay one answer per chunk."""
    units: list[list[Piece]] = []
    for piece in pieces:
        if not units or (HEADING.match(piece.text) and piece is not pieces[0]):
            units.append([])
        units[-1].append(piece)

    chunks: list[list[Piece]] = []
    current: list[Piece] = []
    used = 0
    for unit in units:
        tokens = sum(p.tokens + JOINER_TOKENS for p in unit)
        if tokens > size:  # too big on its own: split it, then carry on fresh
            if current:
                chunks.append(current)
                current, used = [], 0
            chunks += _pack(_shrink_to_fit(unit, size), size, overlap)
            continue
        if current and used + tokens > size:
            chunks.append(current)
            current, used = [], 0
        current += unit
        used += tokens
    if current:
        chunks.append(current)
    return chunks


# ===================================================== document -> pieces
HEADING = re.compile(r"^(#{1,6})\s+(.*)")


def _paragraphs(doc: Document):
    for block in doc.blocks:
        if block.kind == "table":
            yield block, None
            continue
        for text in re.split(r"\n\s*\n", block.text):
            if text.strip():
                yield block, text.strip()


def main_section_level(doc: Document) -> int | None:
    """The heading level that divides this document into sections: the highest
    level used more than once. A single "# Title" doesn't count as a section level."""
    levels = [len(m.group(1)) for _, text in _paragraphs(doc)
              if text and (m := HEADING.match(text))]
    repeated = [lvl for lvl in set(levels) if levels.count(lvl) > 1]
    return min(repeated) if repeated else (min(levels) if levels else None)


def _split_into_pieces(doc: Document, *, structure_aware: bool = False) -> list[tuple[bool, list[Piece]]]:
    """Return groups in document order: (True, [table]) or (False, [paragraphs]).
    In structure mode a new group starts at every MAIN section heading."""
    level = main_section_level(doc) if structure_aware else None
    groups: list[tuple[bool, list[Piece]]] = []
    paragraphs: list[Piece] = []
    section = subsection = None
    for block, text in _paragraphs(doc):
        if text is None:  # a table
            if paragraphs:
                groups.append((False, paragraphs))
                paragraphs = []
            groups.append((True, [Piece(block.text, count_tokens(block.text), "table",
                                        block.page, section, None, subsection)]))
            continue
        heading = HEADING.match(text)
        if heading:
            name = heading.group(2).split("\n")[0].strip()
            if not structure_aware:
                section = name
            elif level is None or len(heading.group(1)) <= level:  # a main section starts
                if paragraphs:
                    groups.append((False, paragraphs))
                    paragraphs = []
                section, subsection = name, None
            else:  # a sub-heading: stays in the current section's group
                subsection = name
        paragraphs.append(Piece(text, count_tokens(text), block.kind, block.page, section,
                                block.ocr_confidence, subsection))
    if paragraphs:
        groups.append((False, paragraphs))
    return groups


# ============================================== recursive splitting
SPLIT_ORDER = ["\n\n", "\n", SENTENCE_END, " "]


def split_recursively(text: str, size: int, separators: list = SPLIT_ORDER) -> list[str]:
    """Split text on the first separator that works; split again any part
    that is still too big using the next separator. Last resort: cut tokens."""
    if count_tokens(text) <= size:
        return [text]
    if not separators:
        ids = _tokenizer().encode(text, disallowed_special=())
        return [_tokenizer().decode(ids[i:i + size]) for i in range(0, len(ids), size)]

    separator, rest = separators[0], separators[1:]
    parts = separator.split(text) if isinstance(separator, re.Pattern) else text.split(separator)
    if len(parts) == 1:
        return split_recursively(text, size, rest)

    glue = " " if isinstance(separator, re.Pattern) else separator
    result, current = [], ""
    for part in (p.strip() for p in parts):
        if not part:
            continue
        candidate = f"{current}{glue}{part}" if current else part
        if count_tokens(candidate) <= size:
            current = candidate
            continue
        if current:
            result.append(current)
        if count_tokens(part) > size:
            result += split_recursively(part, size, rest)
            current = ""
        else:
            current = part
    if current:
        result.append(current)
    return result


def _shrink_to_fit(pieces: list[Piece], size: int) -> list[Piece]:
    out = []
    for p in pieces:
        if p.tokens <= size:
            out.append(p)
        else:
            out += [p.copy_with(t) for t in split_recursively(p.text, size, SPLIT_ORDER[1:])]
    return out


def _into_sentences(pieces: list[Piece], size: int) -> list[Piece]:
    out = []
    for p in pieces:
        for sentence in SENTENCE_END.split(p.text):
            if sentence.strip():
                out += [p.copy_with(t) for t in split_recursively(sentence.strip(), size, [" "])]
    return out


# ============================================== packing with overlap
def _pack(pieces: list[Piece], size: int, overlap: int) -> list[list[Piece]]:
    """Fill each chunk with pieces up to `size` tokens. The next chunk starts
    with the last pieces of the previous one (up to `overlap` tokens), so
    context that crosses a chunk boundary is not lost."""
    chunks: list[list[Piece]] = []
    current: list[Piece] = []
    used = 0
    for piece in pieces:
        if current and used + piece.tokens + JOINER_TOKENS > size:
            # never end a chunk with a heading whose text is in the next chunk
            dangling = current.pop() if len(current) > 1 and _heading_only(current[-1]) else None
            chunks.append(current)
            current, used = _overlap_tail(current, overlap)
            if dangling:
                current.append(dangling)
                used += dangling.tokens + JOINER_TOKENS
            if used + piece.tokens + JOINER_TOKENS > size:  # overlap would not leave room
                current, used = ([dangling], dangling.tokens + JOINER_TOKENS) if dangling else ([], 0)
        current.append(piece)
        used += piece.tokens + JOINER_TOKENS
    if current:
        chunks.append(current)
    return chunks


def _heading_only(piece: Piece) -> bool:
    return bool(HEADING.match(piece.text)) and "\n" not in piece.text


def _overlap_tail(pieces: list[Piece], overlap: int) -> tuple[list[Piece], int]:
    tail, used = [], 0
    for piece in reversed(pieces):
        if used + piece.tokens > overlap:
            break
        tail.insert(0, piece)
        used += piece.tokens + JOINER_TOKENS
    return tail, used


def _fixed_windows(pieces: list[Piece], size: int, overlap: int) -> list[list[Piece]]:
    """Fixed token windows, done page by page so page numbers stay correct."""
    by_page: dict[int | None, list[Piece]] = {}
    for p in pieces:
        by_page.setdefault(p.page, []).append(p)

    chunks = []
    for page_pieces in by_page.values():
        ids = _tokenizer().encode("\n\n".join(p.text for p in page_pieces), disallowed_special=())
        for start in range(0, len(ids), size - overlap):
            window = _tokenizer().decode(ids[start:start + size])
            chunks.append([page_pieces[0].copy_with(window)])
            if start + size >= len(ids):
                break
    return chunks


# ============================================== tables and tiny pieces
def _split_table(table: Piece, size: int) -> list[Piece]:
    """Split a big Markdown table by rows, repeating the header in each part."""
    if table.tokens <= size:
        return [table]
    lines = table.text.split("\n")
    has_header = len(lines) > 2 and lines[1].startswith("| ---")
    header = lines[:2] if has_header else []
    rows = lines[2:] if has_header else lines

    parts, current = [], []
    for row in rows:
        if current and count_tokens("\n".join(header + current + [row])) > size:
            parts.append(table.copy_with("\n".join(header + current)))
            current = []
        current.append(row)
    if current:
        parts.append(table.copy_with("\n".join(header + current)))
    return parts


def _merge_tiny(chunks: list[list[Piece]], settings: Settings) -> list[list[Piece]]:
    """Fold tiny text chunks (a caption, one OCR line) into the previous text
    chunk instead of storing them alone or losing them."""
    def tokens(chunk):
        return sum(p.tokens + JOINER_TOKENS for p in chunk)

    def is_table(chunk):
        return chunk[0].kind == "table"

    merged: list[list[Piece]] = []
    for chunk in chunks:
        if not is_table(chunk) and tokens(chunk) < settings.min_chunk_tokens:
            # look back at most two chunks (i.e. past one table)
            target = next((c for c in reversed(merged[-2:])
                           if not is_table(c) and tokens(c) + tokens(chunk) <= settings.chunk_size
                           and (settings.chunk_strategy != "structure"
                                or c[0].section == chunk[0].section)),
                          None)
            if target is not None:
                target.extend(chunk)
                continue
        merged.append(chunk)

    # Anything still tiny (e.g. a title with no text under it) joins the NEXT
    # chunk instead - a heading belongs with what follows it, even a table
    # (the table chunk then also says which section it belongs to).
    result: list[list[Piece]] = []
    carry: list[Piece] = []
    for chunk in merged:
        if carry and tokens(carry) + tokens(chunk) <= settings.chunk_size:
            chunk = carry + chunk
            carry = []
        elif carry:
            result.append(carry)
            carry = []
        if not is_table(chunk) and tokens(chunk) < settings.min_chunk_tokens:
            carry = chunk
        else:
            result.append(chunk)
    if carry:
        result.append(carry)
    return result


# ============================================== chunk + metadata
def _make_chunk(doc: Document, index: int, pieces: list[Piece], settings: Settings) -> Chunk:
    text = "\n\n".join(p.text for p in pieces).strip()
    kinds = sorted({p.kind for p in pieces})
    pages = sorted({p.page for p in pieces if p.page is not None})
    # The chunk's section is the one most of its text belongs to
    section_tokens: dict[str, int] = {}
    for p in pieces:
        if p.section:
            section_tokens[p.section] = section_tokens.get(p.section, 0) + p.tokens
    section = max(section_tokens, key=section_tokens.get) if section_tokens else None
    confidences = [p.ocr_confidence for p in pieces if p.ocr_confidence is not None]

    metadata = {
        "source": doc.path,
        "file_name": doc.name,
        "file_type": doc.file_type,
        "file_hash": doc.fingerprint,
        "chunk_index": index,
        "chunk_strategy": settings.chunk_strategy,
        "content_types": ",".join(kinds),
        "has_table": "table" in kinds,
        "has_ocr": any(k.startswith("ocr") for k in kinds),
    }
    if pages:
        metadata["page_start"], metadata["page_end"] = pages[0], pages[-1]
    if section:
        metadata["section"] = section[:200]
    subsections = list(dict.fromkeys(p.subsection for p in pieces if p.subsection))
    if subsections:
        metadata["subsections"] = " | ".join(subsections)[:500]
    if doc.title:
        metadata["doc_title"] = doc.title[:200]
    if confidences:
        metadata["ocr_confidence"] = min(confidences)

    # Same file + same text => same id, so re-ingesting overwrites instead of duplicating
    chunk_id = hashlib.sha256(f"{doc.fingerprint}:{index}:{text}".encode()).hexdigest()[:32]
    return Chunk(id=chunk_id, text=text, index=index, tokens=count_tokens(text), metadata=metadata)
