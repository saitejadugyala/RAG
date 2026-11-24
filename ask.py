"""Ask a question about your documents and get a cited answer.

    python ask.py "How many days of annual leave do I get?"
    python ask.py                                   # interactive: ask several questions
    python ask.py "pricing" --type pdf              # only PDF chunks
    python ask.py "notice period" --file employee_handbook.docx
    python ask.py "limits" --section expenses --tables-only
    python ask.py "uptime" --page 2 --exclude-flagged
    python ask.py "..." --show-search               # also show every search score
    python ask.py "..." --json                      # machine-readable output
"""
import argparse
import json
import logging
import os
import sys
import textwrap

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from ingestion import Settings  # noqa: E402
from ingestion.errors import IngestionError  # noqa: E402
from retrieval import Filters, RAGPipeline  # noqa: E402
from retrieval.errors import RetrievalError  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?", help="leave out to ask questions one after another")
    f = p.add_argument_group("metadata filters")
    f.add_argument("--file", help="only this file, e.g. employee_handbook.docx")
    f.add_argument("--type", choices=["pdf", "docx", "txt"], help="only this file type")
    f.add_argument("--section", help="section heading must contain this text")
    f.add_argument("--page", type=int, help="only chunks from this PDF page")
    f.add_argument("--tables-only", action="store_true", help="only chunks with a table")
    f.add_argument("--exclude-flagged", action="store_true",
                   help="skip chunks with quality flags (e.g. low OCR confidence)")
    p.add_argument("--top-k", type=int, help="sources sent to the LLM (default 3)")
    p.add_argument("--show-search", action="store_true", help="print every search hit and score")
    p.add_argument("--json", action="store_true", help="print the answer as JSON")
    p.add_argument("--collection", help="ChromaDB collection name")
    p.add_argument("--fake", action="store_true", help="offline test mode (no OpenAI calls)")
    return p.parse_args()


def print_answer(answer, show_search_hits=None) -> None:
    if show_search_hits is not None:
        print("\nSearch results (fused order)   sim = cosine similarity, kw = BM25 keyword score")
        for h in show_search_hits[:10]:
            mark = "✓" if h.relevant else "·"
            ranks = f"sem #{h.vector_rank or '-'} kw #{h.keyword_rank or '-'}"
            print(f"  {mark} sim {h.similarity:.3f}  kw {h.keyword_score:5.2f}  {ranks:16s} "
                  f"{h.metadata.get('file_name')} · {str(h.metadata.get('section', '-'))[:40]}")
        print("  ✓ = passed the relevance check")

    print("\nAnswer:")
    print(textwrap.fill(answer.text, width=100, initial_indent="  ", subsequent_indent="  "))
    if answer.found:
        print("\nSources:")
        for s in answer.sources:
            used = "cited" if s.number in answer.cited else "not cited"
            print(f"  [{s.number}] {s.label()}   (similarity {s.similarity:.2f}, {used})")
    for w in answer.warnings:
        print(f"\n  ! {w}")
    if answer.filters != "none":
        print(f"\n  filters: {answer.filters}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    overrides = {"collection_name": args.collection, "top_k": args.top_k}
    try:
        settings = Settings(**{k: v for k, v in overrides.items() if v is not None})
        rag = RAGPipeline(settings, fake=args.fake)
    except (ValueError, IngestionError, RetrievalError) as e:
        print(f"Cannot start: {e}")
        return 2

    filters = Filters(file_name=args.file, file_type=args.type, section=args.section,
                      page=args.page, tables_only=args.tables_only,
                      exclude_flagged=args.exclude_flagged)

    def answer_one(question: str) -> int:
        try:
            answer = rag.ask(question, filters)
        except RetrievalError as e:
            print(f"\nCould not answer: {e}")
            # show what was found, so the user still gets something useful
            relevant = [h for h in rag.last_hits if h.relevant][: settings.top_k]
            if relevant:
                print("Most relevant passages:")
                for h in relevant:
                    print(f"  - {h.metadata.get('file_name')}: {h.text[:200]!r}")
            return 1
        if args.json:
            print(json.dumps(answer.to_dict(), indent=2))
        else:
            print_answer(answer, rag.last_hits if args.show_search else None)
        return 0

    if args.question:
        return answer_one(args.question)

    print("Ask a question about your documents (empty line or Ctrl+C to quit).")
    while True:
        try:
            question = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        answer_one(question)


if __name__ == "__main__":
    sys.exit(main())
