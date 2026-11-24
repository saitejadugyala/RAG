"""The pipeline: runs the 7 steps on every file and handles failures.

    1 check file -> 2 extract (+OCR) -> 3 clean -> 4 chunk
                 -> 5 check chunks -> 6 embed -> 7 store

What happens when something goes wrong:
  * one bad file never stops the run - the next file is processed
  * the file itself is bad (corrupt, empty, unsupported, nothing usable)
        -> "quarantined": copied to data/quarantine/ with the reason
  * OpenAI or ChromaDB had a temporary problem (after retries)
        -> "failed": left as is and retried automatically on the next run
  * OpenAI refused every chunk of a file (bad content, not an outage)
        -> "quarantined", so it isn't retried forever
  * a single page, image or chunk fails -> warning; the rest is still ingested
  * an unexpected bug -> "failed", full traceback in data/ingest.log
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

from .errors import EmbeddingRejectedError, IngestionError, NoValidChunksError
from .models import FileResult
from .ocr import OCR
from .settings import Settings, unknown_env_names
from .step1_check_file import check_file, file_fingerprint
from .step2_extract import extract
from .step3_clean import clean_document
from .step4_chunk import chunk_document
from .step5_check_chunks import check_chunks
from .step6_embed import embed_chunks, make_embedder
from .step7_store import VectorStore
from .tracking import Manifest, log_chunks_for_review, now, quarantine

log = logging.getLogger("ingestion")


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None, *, fake_embeddings: bool = False):
        self.settings = settings or Settings()
        self.settings.create_folders()
        for name in unknown_env_names():
            log.warning("Setting %s in .env is not used (typo or old name?) - see .env.example",
                        name)
        self.ocr = OCR(self.settings)
        self.embedder = make_embedder(self.settings, fake=fake_embeddings)
        self.store = VectorStore(self.settings, self.embedder.name)
        self.manifest = Manifest(self.settings.manifest_file)
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.review_file = self.settings.reports_dir / f"chunks-{self.run_id}.jsonl"

    # ------------------------------------------------------------------ run
    def run(self, paths: list[str | Path] | None = None, *, force: bool = False) -> dict:
        """Ingest files and/or folders (default: settings.docs_dir). Returns a summary."""
        started = time.time()
        files = find_files(paths or [self.settings.docs_dir], self.settings)
        log.info("Found %d file(s)\n", len(files))

        results = []
        for number, path in enumerate(files, start=1):
            log.info("[%d/%d] %s", number, len(files), path.name)
            results.append(self.ingest_file(path, force=force))

        summary = self._summary(results, time.time() - started)
        report_file = self.settings.reports_dir / f"run-{self.run_id}.json"
        report_file.write_text(json.dumps(summary, indent=2))
        summary["report_file"] = str(report_file)
        summary["review_file"] = str(self.review_file) if self.review_file.exists() else None
        return summary

    # ------------------------------------------------------------ one file
    def ingest_file(self, path: Path, *, force: bool = False) -> FileResult:
        started = time.time()
        file = str(path.resolve())
        result = FileResult(file=file)
        step = "1 check file"
        try:
            # STEP 1 - check the file
            file_type = check_file(path, self.settings)
            fingerprint = file_fingerprint(path)
            signature = self.settings.processing_signature()
            if not force and self.manifest.is_unchanged(file, fingerprint, self.embedder.name,
                                                        signature):
                result.status = "skipped"
                log.info("  - skipped: file and settings unchanged since the last run\n")
                return result
            log.info("  ✓ 1 checked        %s, %.1f KB", file_type, path.stat().st_size / 1024)

            # STEP 2 - extract text, tables, images (OCR)
            step = "2 extract"
            doc = extract(path, file_type, fingerprint, self.settings, self.ocr)
            result.ocr_blocks = sum(b.kind.startswith("ocr") for b in doc.blocks)
            result.tables = sum(b.kind == "table" for b in doc.blocks)
            log.info("  ✓ 2 extracted      %d text, %d tables, %d OCR",
                     len(doc.blocks) - result.ocr_blocks - result.tables, result.tables,
                     result.ocr_blocks)

            # STEP 3 - clean and remove noise
            step = "3 clean"
            doc = clean_document(doc, self.settings)
            log.info("  ✓ 3 cleaned        %d chars kept, %.0f%% noise removed",
                     doc.stats["clean_chars"], doc.stats["noise_removed_pct"])

            # STEP 4 - chunk
            step = "4 chunk"
            chunks = chunk_document(doc, self.settings)
            log.info("  ✓ 4 chunked        %d chunks (%s, %d tokens, %d overlap)", len(chunks),
                     self.settings.chunk_strategy, self.settings.chunk_size,
                     self.settings.chunk_overlap)

            # STEP 5 - check chunk quality
            step = "5 check chunks"
            quality = check_chunks(chunks, self.settings)
            if quality["rejected"] == quality["total"]:
                raise NoValidChunksError(f"all {quality['total']} chunks rejected: {quality['reasons']}")
            log.info("  ✓ 5 checked chunks %d ok, %d flagged, %d rejected",
                     quality["ok"], quality["flagged"], quality["rejected"])

            # STEP 6 - embed
            step = "6 embed"
            result.warnings += embed_chunks(chunks, self.embedder, self.settings)
            embedded = sum(c.embedding is not None for c in chunks)
            log.info("  ✓ 6 embedded       %d vectors x %d dims", embedded,
                     self.settings.embedding_dimensions)
            if not embedded:
                # Outages and bad keys already stopped above (fatal errors), so
                # reaching here means OpenAI refused each chunk -> bad content.
                raise EmbeddingRejectedError(
                    f"OpenAI refused all {quality['total'] - quality['rejected']} chunks: "
                    f"{'; '.join(result.warnings[-3:])}")

            # STEP 7 - store
            step = "7 store"
            stored = self.store.save_file_chunks(file, chunks)
            log.info("  ✓ 7 stored         %d chunks in '%s'", stored, self.settings.collection_name)

            # Done
            result.status = "success"
            result.chunks, result.stored = len(chunks), stored
            result.rejected = sum(c.rejected for c in chunks)
            result.flagged = sum(not c.rejected and bool(c.flags) for c in chunks)
            result.warnings += doc.warnings
            for warning in result.warnings:
                log.info("  ! warning: %s", warning)
            log_chunks_for_review(chunks, self.review_file)
            self.manifest.record(file, fingerprint=fingerprint, status="success",
                                 embedder=self.embedder.name, settings=signature,
                                 chunks=stored, stats=doc.stats, ingested_at=now())

        except IngestionError as e:
            step = e.step if e.step != "unknown" else step
            result.failed_step, result.error = step, str(e)
            if e.retryable:
                result.status = "failed"
                log.error("  ✗ FAILED at step %s: %s\n    (will be retried on the next run)", step, e)
            else:
                result.status = "quarantined"
                quarantine(path, step, str(e), self.settings, self.run_id)
                log.error("  ✗ QUARANTINED at step %s: %s", step, e)
            self.manifest.record(file, status=result.status, failed_step=step, error=str(e))

        except Exception as e:  # a bug, not a bad file: log it and keep going
            result.failed_step, result.error = step, f"{type(e).__name__}: {e}"
            log.error("  ✗ FAILED at step %s (unexpected): %s", step, result.error)
            log.debug(traceback.format_exc())
            self.manifest.record(file, status="failed", failed_step=step, error=result.error)

        finally:
            result.seconds = round(time.time() - started, 2)
        log.info("")
        return result

    # --------------------------------------------------------------- helpers
    def remove_deleted_files(self) -> list[str]:
        """Delete the vectors of files that no longer exist on disk."""
        removed = [f for f in list(self.manifest.files) if not Path(f).exists()]
        for file in removed:
            self.store.delete_file(file)
            self.manifest.remove(file)
        return removed

    def _summary(self, results: list[FileResult], seconds: float) -> dict:
        def count(status):
            return sum(r.status == status for r in results)

        return {
            "run_id": self.run_id,
            "seconds": round(seconds, 1),
            "embedder": self.embedder.name,
            "collection": self.settings.collection_name,
            "vectors_in_collection": self.store.count(),
            "files": len(results),
            "success": count("success"),
            "skipped": count("skipped"),
            "failed": count("failed"),
            "quarantined": count("quarantined"),
            "chunks_stored": sum(r.stored for r in results),
            "chunks_rejected": sum(r.rejected for r in results),
            "chunks_flagged": sum(r.flagged for r in results),
            "results": [r.to_dict() for r in results],
        }


def find_files(paths: list[str | Path], settings: Settings) -> list[Path]:
    """Expand folders (recursively) into files. Hidden files and our own
    data folder are ignored. Missing paths are kept so step 1 reports them."""
    data_dir = settings.data_dir.resolve()
    files: list[Path] = []
    for p in map(Path, paths):
        p = p.expanduser()
        if p.is_dir():
            files += sorted(f for f in p.rglob("*")
                            if f.is_file() and not f.name.startswith(".")
                            and data_dir not in f.resolve().parents)
        else:
            files.append(p)
    return files
