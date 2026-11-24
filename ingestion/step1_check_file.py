"""STEP 1 - Check the file before doing any work on it.

Catches problems early so later steps never see a file they can't handle:
missing or unreadable files, wrong type, empty or huge files, and files
whose content doesn't match their extension (renamed or corrupted).
"""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

from .errors import FileCheckError
from .settings import Settings


def check_file(path: Path, settings: Settings) -> str:
    """Return the file type ("pdf", "docx" or "txt"), or raise FileCheckError."""
    if not path.is_file():
        raise FileCheckError(f"File not found: {path}")
    if not os.access(path, os.R_OK):
        raise FileCheckError("File is not readable (check permissions)")
    if path.name.startswith("~$"):
        raise FileCheckError("Office temporary/lock file, not a document")

    extension = path.suffix.lower()
    if extension not in settings.allowed_types:
        raise FileCheckError(
            f"Unsupported file type '{extension}'. Allowed: {', '.join(settings.allowed_types)}")

    size = path.stat().st_size
    if size == 0:
        raise FileCheckError("File is empty (0 bytes)")
    if size > settings.max_file_mb * 1024 * 1024:
        raise FileCheckError(f"File is {size / 1e6:.1f} MB; the limit is {settings.max_file_mb} MB")

    with path.open("rb") as f:
        start = f.read(8192)

    if extension == ".pdf":
        _check_pdf(start)
        return "pdf"
    if extension == ".docx":
        _check_docx(path, start)
        return "docx"
    _check_txt(start)
    return "txt"


def file_fingerprint(path: Path) -> str:
    """SHA-256 of the file. If it hasn't changed, we can skip re-ingesting it."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
def _check_pdf(start: bytes) -> None:
    if b"%PDF-" not in start[:1024]:
        raise FileCheckError("Named .pdf but the content is not a PDF")


def _check_docx(path: Path, start: bytes) -> None:
    if not start.startswith(b"PK\x03\x04"):  # a .docx is a zip archive
        raise FileCheckError("Named .docx but the content is not a DOCX "
                             "(old .doc files are not supported)")
    try:
        with zipfile.ZipFile(path) as z:
            if "word/document.xml" not in z.namelist():
                raise FileCheckError("Zip archive is not a Word document")
            if z.testzip():
                raise FileCheckError("DOCX is corrupted")
    except zipfile.BadZipFile as e:
        raise FileCheckError(f"DOCX is corrupted: {e}") from e


def _check_txt(start: bytes) -> None:
    has_utf16_bom = start.startswith((b"\xff\xfe", b"\xfe\xff"))
    if b"\x00" in start and not has_utf16_bom:
        raise FileCheckError("Looks like a binary file, not plain text")
