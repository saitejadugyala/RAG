"""The data that moves from one step to the next.

    file ──step 2──► Document (list of Blocks) ──step 4──► list of Chunks
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Block:
    """One piece of content pulled out of a file."""

    text: str
    kind: str = "text"  # "text" | "table" | "ocr_image" | "ocr_page"
    page: int | None = None  # page number (PDF only)
    ocr_confidence: float | None = None  # set for OCR blocks (0-100)


@dataclass
class Document:
    """Everything extracted from one file."""

    path: str
    name: str
    file_type: str  # "pdf" | "docx" | "txt"
    fingerprint: str  # SHA-256 of the file, used to detect changes
    blocks: list[Block] = field(default_factory=list)
    title: str | None = None
    warnings: list[str] = field(default_factory=list)  # problems that did not stop the file
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of text that gets embedded and stored."""

    id: str
    text: str
    index: int
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)  # quality notes from step 5
    rejected: bool = False  # True => not embedded or stored
    embedding: list[float] | None = None


@dataclass
class FileResult:
    """What happened to one file. Collected into the run report."""

    file: str
    status: str = "failed"  # "success" | "skipped" | "failed" | "quarantined"
    failed_step: str | None = None
    error: str | None = None
    chunks: int = 0
    stored: int = 0
    rejected: int = 0
    flagged: int = 0
    ocr_blocks: int = 0
    tables: int = 0
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
