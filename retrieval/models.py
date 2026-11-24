"""The data that moves through the retrieval pipeline.

    question ─► Query ─► list of Hits ─► top 3 Sources ─► Answer (with citations)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

NO_INFO_ANSWER = "I couldn't find any information related to this topic in the documents."


@dataclass
class Filters:
    """Metadata filters - only chunks matching ALL the given values are searched."""

    file_name: str | None = None  # e.g. "employee_handbook.docx"
    file_type: str | None = None  # "pdf" | "docx" | "txt"
    section: str | None = None  # text the section heading must contain
    page: int | None = None  # PDF page number
    tables_only: bool = False  # only chunks that contain a table
    exclude_flagged: bool = False  # skip chunks with quality flags (e.g. low OCR)

    def is_empty(self) -> bool:
        return self == Filters()

    def describe(self) -> str:
        parts = [f"{k}={v}" for k, v in asdict(self).items() if v not in (None, False)]
        return ", ".join(parts) or "none"


@dataclass
class Query:
    """The user's question, prepared for both kinds of search."""

    text: str
    vector: list[float]  # for semantic (meaning) search
    keywords: list[str]  # for keyword (exact word) search


@dataclass
class Hit:
    """One chunk found by the search, with every score that led to it."""

    id: str
    text: str
    metadata: dict[str, Any]
    similarity: float = 0.0  # cosine similarity to the question (0-1)
    keyword_score: float = 0.0  # BM25 score (0 = no shared keywords)
    vector_rank: int | None = None  # position in the semantic results (1 = best)
    keyword_rank: int | None = None  # position in the keyword results
    fused_score: float = 0.0  # reciprocal rank fusion of both ranks
    relevant: bool = False  # passed the relevance check


@dataclass
class Source:
    """A chunk that is sent to the LLM, numbered so the answer can cite it."""

    number: int  # [1], [2], [3]
    text: str
    file_name: str
    section: str | None
    page: int | None
    similarity: float
    keyword_score: float

    def label(self) -> str:
        where = [self.file_name]
        if self.section:
            where.append(f'section "{self.section}"')
        if self.page:
            where.append(f"page {self.page}")
        return " · ".join(where)


@dataclass
class Answer:
    question: str
    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)  # sources sent to the LLM
    cited: list[int] = field(default_factory=list)  # source numbers used in the answer
    filters: str = "none"
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
