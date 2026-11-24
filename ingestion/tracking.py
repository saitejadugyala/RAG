"""Bookkeeping for failure handling and re-runs.

  Manifest      data/manifest.json - one entry per file: fingerprint, status,
                settings used, chunk count. A file is skipped next time only if
                neither the file nor those settings changed. Written
                atomically after every file, so a crash never corrupts it.
  Quarantine    data/quarantine/ - a COPY of every file that can't be ingested,
                plus <name>.error.json saying which step failed and why.
  Reports       data/reports/run-<id>.json (summary of the run) and
                data/reports/chunks-<id>.jsonl (rejected/flagged chunks to review)
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .models import Chunk
from .settings import Settings

log = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.files: dict[str, dict] = {}
        if path.exists():
            try:
                self.files = json.loads(path.read_text())
            except json.JSONDecodeError:
                shutil.copy2(path, path.with_suffix(".corrupt.json"))
                log.error("manifest.json was corrupt; saved a copy and started fresh")

    def is_unchanged(self, file: str, fingerprint: str, embedder: str, signature: str) -> bool:
        """True if the file, the embedding model AND the chunk/OCR settings are
        all the same as the last successful run - then there is nothing to redo."""
        entry = self.files.get(file, {})
        return (entry.get("status") == "success"
                and entry.get("fingerprint") == fingerprint
                and entry.get("embedder") == embedder
                and entry.get("settings") == signature)

    def record(self, file: str, **info) -> None:
        self.files.setdefault(file, {}).update(info, updated_at=now())
        self.save()

    def remove(self, file: str) -> None:
        self.files.pop(file, None)
        self.save()

    def save(self) -> None:
        # Write to a temp file, then rename: the old manifest stays intact if we crash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(handle, "w") as f:
            json.dump(self.files, f, indent=2)
        os.replace(temp, self.path)


def quarantine(path: Path, step: str, error: str, settings: Settings, run_id: str) -> None:
    """Copy a file that can't be ingested aside, with the reason. The original stays put."""
    try:
        settings.quarantine_dir.mkdir(parents=True, exist_ok=True)
        source = str(path.resolve())
        destination = settings.quarantine_dir / path.name
        error_file = settings.quarantine_dir / f"{path.name}.error.json"
        # Keep the readable legacy name unless it belongs to a different source.
        if destination.exists() or error_file.exists():
            try:
                previous = json.loads(error_file.read_text()).get("file")
            except (OSError, json.JSONDecodeError, AttributeError):
                previous = None
            if previous and str(Path(previous).resolve()) != source:
                suffix = hashlib.sha256(source.encode()).hexdigest()[:10]
                destination = settings.quarantine_dir / f"{path.stem}-{suffix}{path.suffix}"
                error_file = settings.quarantine_dir / f"{path.name}-{suffix}.error.json"
        if path.is_file():
            shutil.copy2(path, destination)
        error_file.write_text(json.dumps(
            {"file": source, "failed_step": step, "error": error, "run_id": run_id, "at": now()},
            indent=2))
    except OSError as e:
        log.error("could not quarantine %s: %s", path, e)


def log_chunks_for_review(chunks: list[Chunk], report_file: Path) -> None:
    """Append rejected and flagged chunks to the review file."""
    to_review = [c for c in chunks if c.rejected or c.flags]
    if not to_review:
        return
    with report_file.open("a") as f:
        for c in to_review:
            f.write(json.dumps({
                "file": c.metadata.get("file_name"), "chunk": c.index,
                "rejected": c.rejected, "flags": c.flags, "tokens": c.tokens,
                "page": c.metadata.get("page_start"), "preview": c.text[:200],
            }) + "\n")
