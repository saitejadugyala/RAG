"""Search the vector database (a quick check that ingestion worked).

    python search.py "how many days of annual leave do I get?"
    python search.py "pricing" -k 3
    python search.py "pricing" --fake-embeddings --collection test   # offline test
"""
import argparse
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from ingestion import Settings  # noqa: E402
from ingestion.errors import IngestionError  # noqa: E402
from ingestion.step6_embed import make_embedder  # noqa: E402
from ingestion.step7_store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="number of results")
    parser.add_argument("--collection")
    parser.add_argument("--fake-embeddings", action="store_true")
    args = parser.parse_args()

    settings = Settings(**({"collection_name": args.collection} if args.collection else {}))
    try:
        embedder = make_embedder(settings, fake=args.fake_embeddings)
        store = VectorStore(settings, embedder.name)
    except IngestionError as e:
        print(f"Cannot start: {e}")
        return 2
    if store.count() == 0:
        print("The vector database is empty. Run `python ingest.py` first.")
        return 1

    vector = embedder.embed([args.question])[0]
    for rank, hit in enumerate(store.search(vector, k=args.k), start=1):
        meta = hit["metadata"]
        page = f", page {meta['page_start']}" if meta.get("page_start") else ""
        print(f"\n#{rank}  score {hit['score']:.3f}  {meta['file_name']}{page}  "
              f"(chunk {meta['chunk_index']}, flags: {meta.get('quality_flags')})")
        text = hit["text"].replace("\n", " ")
        print("    " + text[:300] + ("..." if len(text) > 300 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
