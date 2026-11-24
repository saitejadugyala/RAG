"""Evaluate retrieval and generation quality on the test question set.

    python evaluate.py                      # full evaluation (retrieval + answers)
    python evaluate.py --retrieval-only     # only search quality, no LLM calls at all
    python evaluate.py --tag ocr_image      # only questions with this tag
    python evaluate.py --id leave-annual    # one question
    python evaluate.py --fake               # offline test run (fake embeddings and LLMs)

Reports are written to data/reports/eval-<run>.json and .md.
Exit code: 0 = all thresholds met, 1 = some threshold failed, 2 = could not run.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from evaluation.dataset import load_dataset  # noqa: E402
from evaluation.report import print_summary, write_reports  # noqa: E402
from evaluation.runner import check_thresholds, previous_report, run_evaluation  # noqa: E402
from ingestion import Settings  # noqa: E402
from ingestion.errors import IngestionError  # noqa: E402
from retrieval.errors import RetrievalError  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="eval/dataset.jsonl")
    p.add_argument("--retrieval-only", action="store_true", help="no answer generation")
    p.add_argument("--tag", help="only questions with this tag")
    p.add_argument("--id", help="only the question with this id")
    p.add_argument("--top-k", type=int)
    p.add_argument("--collection")
    p.add_argument("--fake", action="store_true", help="offline: fake embeddings and LLM")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    try:
        cases = load_dataset(Path(args.dataset))
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 2
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    if args.id:
        cases = [c for c in cases if c.id == args.id]
    if not cases:
        print("No questions match the filter.")
        return 2

    overrides = {"collection_name": args.collection, "top_k": args.top_k}
    settings = Settings(**{k: v for k, v in overrides.items() if v is not None})
    mode = "retrieval only" if args.retrieval_only else "retrieval + generation"
    print(f"=== RAG evaluation · {len(cases)} questions · {mode} ===\n")

    try:
        summary = run_evaluation(cases, settings, fake=args.fake, retrieval_only=args.retrieval_only)
    except (IngestionError, RetrievalError) as e:
        print(f"Cannot run: {e}")
        return 2

    failed = check_thresholds(summary["metrics"])
    previous = previous_report(settings.reports_dir, summary["run_id"])
    json_file, md_file = write_reports(summary, failed, previous, settings.reports_dir)
    print_summary(summary, failed, previous)
    print(f"\nReport: {md_file}\nData:   {json_file}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
