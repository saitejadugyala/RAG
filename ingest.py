"""Ingest documents into the vector database.

    python ingest.py                        # ingest everything in ./docs
    python ingest.py my_folder report.pdf   # ingest specific folders/files
    python ingest.py --force                # re-ingest files that haven't changed
    python ingest.py --strategy sentence --chunk-size 400
    python ingest.py --no-ocr
    python ingest.py --remove-deleted       # drop vectors of files deleted from disk
    python ingest.py --fake-embeddings --collection test   # offline test, no API key
"""
import argparse
import logging
import sys
from pathlib import Path

from ingestion import IngestionPipeline, Settings
from ingestion.errors import IngestionError


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="files or folders (default: docs)")
    parser.add_argument("--force", action="store_true", help="re-ingest unchanged files")
    parser.add_argument("--strategy", choices=["structure", "recursive", "sentence", "fixed"])
    parser.add_argument("--chunk-size", type=int, help="tokens per chunk")
    parser.add_argument("--chunk-overlap", type=int, help="tokens shared between chunks")
    parser.add_argument("--no-ocr", action="store_true", help="skip images and scanned pages")
    parser.add_argument("--collection", help="ChromaDB collection name")
    parser.add_argument("--fake-embeddings", action="store_true",
                        help="offline test vectors, no OpenAI calls")
    parser.add_argument("--remove-deleted", action="store_true",
                        help="remove vectors of files deleted from disk, then exit")
    return parser.parse_args()


def make_settings(args) -> Settings:
    overrides = {
        "chunk_strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "collection_name": args.collection,
        "ocr_enabled": False if args.no_ocr else None,
    }
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def setup_logging(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    screen = logging.StreamHandler(sys.stdout)
    screen.setFormatter(logging.Formatter("%(message)s"))
    screen.setLevel(logging.INFO)
    logfile = logging.FileHandler(settings.log_file)
    logfile.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.DEBUG, handlers=[screen, logfile])
    # Third-party libraries log every HTTP call; keep only their warnings.
    # (openai 3.x uses "httpx2"/"httpcore2"; older versions "httpx"/"httpcore".)
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "chromadb",
                  "urllib3", "PIL", "tenacity"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def print_summary(s: dict) -> None:
    print("=" * 60)
    print(f"Files: {s['files']}   success: {s['success']}   skipped: {s['skipped']}   "
          f"failed: {s['failed']}   quarantined: {s['quarantined']}")
    print(f"Chunks stored: {s['chunks_stored']}   flagged: {s['chunks_flagged']}   "
          f"rejected: {s['chunks_rejected']}")
    print(f"Vectors in '{s['collection']}': {s['vectors_in_collection']}   ({s['seconds']}s)")
    problems = [r for r in s["results"] if r["status"] in ("failed", "quarantined")]
    if problems:
        print("\nProblems:")
        for r in problems:
            print(f"  ✗ {Path(r['file']).name}  [{r['status']} at step {r['failed_step']}]  {r['error']}")
    print(f"\nRun report:   {s['report_file']}")
    if s["review_file"]:
        print(f"Chunk review: {s['review_file']}")


def main() -> int:
    args = parse_args()
    try:
        settings = make_settings(args)
    except ValueError as e:
        print(f"Settings error: {e}")
        return 2
    setup_logging(settings)
    print("=== RAG Ingestion Pipeline ===\n")

    try:
        pipeline = IngestionPipeline(settings, fake_embeddings=args.fake_embeddings)
    except IngestionError as e:  # missing API key, wrong collection, ...
        print(f"Cannot start: {e}")
        return 2

    if args.remove_deleted:
        removed = pipeline.remove_deleted_files()
        print(f"Removed {len(removed)} deleted file(s) from the vector database")
        return 0

    summary = pipeline.run(args.paths or None, force=args.force)
    print_summary(summary)
    return 1 if summary["failed"] else 0  # non-zero so cron/CI notice failures


if __name__ == "__main__":
    sys.exit(main())
