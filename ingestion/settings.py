"""All settings in one place.

Every value has a sensible default and can be overridden in `.env`
(see `.env.example`), e.g.  RAG_CHUNK_SIZE=400
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# Raised when the way documents are chunked changes, so the next ingest.py run
# re-chunks every file automatically. 3 = structure chunking packs whole
# sub-sections; PDF lines stored as separate blocks are joined into paragraphs.
CHUNKING_VERSION = 3

KNOWN_NAMES: set[str] = set()  # every RAG_* name this file reads


def env(name: str, default) -> str:
    KNOWN_NAMES.add(f"RAG_{name}")
    return os.getenv(f"RAG_{name}", str(default))


def unknown_env_names() -> list[str]:
    """RAG_* variables that are set but not used - usually a typo or an old name.
    They would otherwise be ignored silently."""
    return sorted(n for n in os.environ if n.startswith("RAG_") and n not in KNOWN_NAMES)


@dataclass
class Settings:
    # ---- Folders --------------------------------------------------------------
    docs_dir: Path = Path(env("DOCS_DIR", "docs"))  # where your documents live
    data_dir: Path = Path(env("DATA_DIR", "data"))  # vector DB, reports, logs, quarantine

    # ---- Step 1: file checks -----------------------------------------------------
    allowed_types: tuple[str, ...] = (".pdf", ".txt", ".docx")
    max_file_mb: float = float(env("MAX_FILE_MB", 100))

    # ---- Step 2: extraction + OCR ------------------------------------------------
    ocr_enabled: bool = env("OCR_ENABLED", "true").lower() == "true"
    ocr_language: str = env("OCR_LANG", "eng")
    ocr_dpi: int = int(env("OCR_DPI", 300))  # resolution used to render scanned pages
    ocr_min_confidence: float = float(env("OCR_MIN_CONF", 40))  # 0-100
    min_image_px: int = int(env("MIN_IMAGE_PX", 80))  # smaller images are icons: skip
    scanned_page_chars: int = int(env("SCANNED_PAGE_CHARS", 30))  # less text => scanned

    # ---- Step 3: cleaning ------------------------------------------------------------
    min_document_chars: int = int(env("MIN_DOC_CHARS", 50))  # less => document is empty

    # ---- Step 4: chunking ------------------------------------------------------------
    chunk_strategy: str = env("CHUNK_STRATEGY", "structure")  # structure | recursive | sentence | fixed
    chunk_size: int = int(env("CHUNK_SIZE", 500))  # tokens
    chunk_overlap: int = int(env("CHUNK_OVERLAP", 75))  # tokens
    min_chunk_tokens: int = int(env("MIN_CHUNK_TOKENS", 20))

    # ---- Step 6: embeddings (OpenAI, 1536 dimensions) --------------------------------
    embedding_model: str = env("EMBED_MODEL", "text-embedding-3-small")
    embedding_dimensions: int = int(env("EMBED_DIMS", 1536))
    embedding_batch_size: int = int(env("EMBED_BATCH", 100))
    embedding_max_tokens: int = 8191  # OpenAI limit per input
    max_retries: int = int(env("MAX_RETRIES", 5))

    # ---- Step 7: vector store (ChromaDB) ---------------------------------------------
    collection_name: str = env("COLLECTION", "documents")

    # ---- Retrieval (ask.py) ---------------------------------------------------------
    top_k: int = int(env("TOP_K", 3))  # chunks sent to the LLM
    candidates: int = int(env("CANDIDATES", 20))  # taken from each search before fusing
    rrf_k: int = int(env("RRF_K", 60))  # reciprocal rank fusion constant
    min_similarity: float = float(env("MIN_SIMILARITY", 0.30))  # cosine, 0-1
    min_keyword_score: float = float(env("MIN_KEYWORD_SCORE", 4.0))  # BM25
    min_relative_score: float = float(env("MIN_RELATIVE_SCORE", 0.6))  # x best similarity

    # ---- Generation (ask.py) --------------------------------------------------------
    llm_model: str = env("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: str = env("LLM_TEMPERATURE", "0")  # empty = model default
    llm_max_tokens: int = int(env("LLM_MAX_TOKENS", 600))

    # ---- Web app login (serve.py) - GitHub OAuth ----------------------------------
    auth_mode: str = env("AUTH", "github")  # github | off (off = no login, this computer only)
    github_client_id: str = field(default=env("GITHUB_CLIENT_ID", ""), repr=False)
    github_client_secret: str = field(default=env("GITHUB_CLIENT_SECRET", ""), repr=False)
    session_secret: str = field(default=env("SESSION_SECRET", ""), repr=False)
    public_url: str = env("PUBLIC_URL", "http://127.0.0.1:8000")  # where the app is opened
    session_hours: int = int(env("SESSION_HOURS", 8))
    # Roles by GitHub username, comma separated. Anyone not listed gets default_role.
    admins: str = env("ADMINS", "")
    editors: str = env("EDITORS", "")
    viewers: str = env("VIEWERS", "")
    default_role: str = env("DEFAULT_ROLE", "none")  # none (refuse) | viewer

    def __post_init__(self) -> None:
        self.docs_dir, self.data_dir = Path(self.docs_dir), Path(self.data_dir)
        if self.chunk_strategy not in ("structure", "recursive", "sentence", "fixed"):
            raise ValueError(f"Unknown chunk strategy: {self.chunk_strategy}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.auth_mode not in ("github", "off"):
            raise ValueError(f"RAG_AUTH must be 'github' or 'off', not '{self.auth_mode}'")
        if self.default_role not in ("none", "viewer"):
            raise ValueError(f"RAG_DEFAULT_ROLE must be 'none' or 'viewer', not '{self.default_role}'")
        if not 0 <= self.min_relative_score <= 1:
            raise ValueError("min_relative_score must be between 0 and 1 (0 turns it off)")
        if self.top_k < 1 or self.candidates < self.top_k:
            raise ValueError("top_k must be >= 1 and candidates >= top_k")

    def processing_signature(self) -> str:
        """The settings that change what gets stored. If any of them changes,
        files are re-ingested even when the file itself is unchanged."""
        return (f"v{CHUNKING_VERSION}|{self.chunk_strategy}|size={self.chunk_size}|overlap={self.chunk_overlap}"
                f"|min={self.min_chunk_tokens}|ocr={self.ocr_enabled}"
                f"|dims={self.embedding_dimensions}")

    # ---- Paths derived from data_dir --------------------------------------------------
    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "quarantine"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def manifest_file(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def log_file(self) -> Path:
        return self.data_dir / "ingest.log"

    def create_folders(self) -> None:
        for folder in (self.data_dir, self.chroma_dir, self.quarantine_dir, self.reports_dir):
            folder.mkdir(parents=True, exist_ok=True)
