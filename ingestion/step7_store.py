"""STEP 7 - Save the chunks and their vectors in ChromaDB.

  * stored on disk in data/chroma, cosine similarity
  * the collection remembers which embedding model built it, and refuses
    vectors from a different model or size (mixing them breaks search)
  * replacing a file is crash-safe: new chunks are written FIRST, then the
    old chunks of that file are deleted, so search never has a gap
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from .errors import StorageError
from .models import Chunk
from .settings import Settings

WRITE_BATCH = 500


class VectorStore:
    def __init__(self, settings: Settings, embedder_name: str):
        import chromadb

        self.settings = settings
        try:
            client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            self.collection = client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine",
                          "embedding_model": embedder_name,
                          "embedding_dimensions": settings.embedding_dimensions},
                embedding_function=None,  # we always pass our own vectors
            )
        except Exception as e:
            raise StorageError(f"Cannot open ChromaDB at {settings.chroma_dir}: {e}", fatal=True) from e

        info = self.collection.metadata or {}
        if info.get("embedding_model", embedder_name) != embedder_name:
            raise StorageError(
                f"Collection '{settings.collection_name}' was built with {info['embedding_model']}, "
                f"not {embedder_name}. Use a different --collection.", fatal=True)
        if int(info.get("embedding_dimensions", settings.embedding_dimensions)) != settings.embedding_dimensions:
            raise StorageError("Collection was built with a different vector size", fatal=True)

    def save_file_chunks(self, source: str, chunks: list[Chunk]) -> int:
        """Store a file's chunks, replacing any older version of that file."""
        new = [c for c in chunks if not c.rejected and c.embedding is not None]
        try:
            for start in range(0, len(new), WRITE_BATCH):
                self._write(new[start:start + WRITE_BATCH])
            new_ids = {c.id for c in new}
            old_ids = self.collection.get(where={"source": source}, include=[])["ids"]
            stale = [i for i in old_ids if i not in new_ids]
            if stale:
                self.collection.delete(ids=stale)
        except Exception as e:
            raise StorageError(f"ChromaDB write failed: {e}") from e
        return len(new)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def _write(self, chunks: list[Chunk]) -> None:
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[_chroma_safe(c.metadata) for c in chunks],
        )

    def delete_file(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def count(self) -> int:
        return self.collection.count()

    def search(self, vector: list[float], k: int = 5, where: dict | None = None) -> list[dict]:
        result = self.collection.query(query_embeddings=[vector], n_results=k, where=where,
                                       include=["documents", "metadatas", "distances"])
        return [{"text": text, "metadata": meta, "score": round(1 - distance, 4)}
                for text, meta, distance in zip(result["documents"][0], result["metadatas"][0],
                                                result["distances"][0])]


def _chroma_safe(metadata: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool metadata values (no None)."""
    return {k: v if isinstance(v, (str, int, float, bool)) else str(v)
            for k, v in metadata.items() if v is not None}
