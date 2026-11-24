"""STEP 2 - Hybrid search with metadata filters.

Two searches run on the same filtered set of chunks:

  semantic search   ChromaDB, cosine similarity between the question vector
                    and each chunk vector. Good at meaning and paraphrases
                    ("time off" finds "annual leave").
  keyword search    BM25 over the chunk text. Good at exact terms that
                    embeddings blur: numbers, names, codes ("extension 4400").

The two ranked lists are merged with Reciprocal Rank Fusion (RRF):

    fused_score = 1 / (rrf_k + semantic_rank) + 1 / (rrf_k + keyword_rank)

RRF only uses positions, so the two very different score scales (cosine 0-1,
BM25 0-20+) don't have to be compared directly. A chunk found by both searches
gets both terms and rises to the top.

Metadata filters (file, type, page, tables only) are applied INSIDE ChromaDB with
a `where` clause, so filtered-out chunks are never searched. Section text and
quality-flag filters are applied right after, in Python.
"""
from __future__ import annotations

import math
from collections import Counter

from ingestion.settings import Settings
from ingestion.step7_store import VectorStore

from .models import Filters, Hit, Query
from .step1_prepare_question import keywords

BM25_K1 = 1.5  # how much repeated words count
BM25_B = 0.75  # how much long chunks are penalised


def hybrid_search(query: Query, filters: Filters, store: VectorStore, index: "KeywordIndex",
                  settings: Settings) -> list[Hit]:
    where = chroma_where(filters)
    allowed_ids = _allowed_ids(store, where, filters)
    if not allowed_ids:
        return []

    n = min(settings.candidates, len(allowed_ids))
    semantic = _semantic_search(query, store, where, allowed_ids, n)
    keyword = index.search(query.keywords, allowed_ids, n)

    # ---- merge both result lists (Reciprocal Rank Fusion) ----
    hits: dict[str, Hit] = {}
    for rank, hit in enumerate(semantic, start=1):
        hit.vector_rank = rank
        hits[hit.id] = hit
    for rank, (chunk_id, score) in enumerate(keyword, start=1):
        hit = hits.get(chunk_id) or Hit(chunk_id, index.texts[chunk_id], index.metadatas[chunk_id])
        hit.keyword_rank, hit.keyword_score = rank, round(float(score), 3)
        hits[chunk_id] = hit

    # chunks found only by keywords still need their cosine similarity
    _fill_missing_similarity([h for h in hits.values() if h.vector_rank is None], query, store)

    for hit in hits.values():
        hit.fused_score = sum(1 / (settings.rrf_k + r)
                              for r in (hit.vector_rank, hit.keyword_rank) if r is not None)
    return sorted(hits.values(), key=lambda h: h.fused_score, reverse=True)


# ------------------------------------------------------------------ filters
def chroma_where(filters: Filters) -> dict | None:
    """Turn Filters into a ChromaDB `where` clause (exact-match fields only)."""
    conditions = []
    if filters.file_name:
        conditions.append({"file_name": filters.file_name})
    if filters.file_type:
        conditions.append({"file_type": filters.file_type.lower().lstrip(".")})
    if filters.page is not None:
        conditions.append({"page_start": {"$lte": filters.page}})
        conditions.append({"page_end": {"$gte": filters.page}})
    if filters.tables_only:
        conditions.append({"has_table": True})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def passes_python_filters(metadata: dict, filters: Filters) -> bool:
    """Filters ChromaDB can't express: 'section contains' and quality flags."""
    if filters.section:
        headings = f"{metadata.get('section', '')} | {metadata.get('subsections', '')}".lower()
        if filters.section.lower() not in headings:  # matches a section or a sub-heading
            return False
    if filters.exclude_flagged and metadata.get("quality_flags", "none") != "none":
        return False
    return True


def _allowed_ids(store: VectorStore, where: dict | None, filters: Filters) -> set[str]:
    found = store.collection.get(where=where, include=["metadatas"])
    return {i for i, m in zip(found["ids"], found["metadatas"]) if passes_python_filters(m, filters)}


# ------------------------------------------------------------------ semantic
def _semantic_search(query: Query, store: VectorStore, where: dict | None,
                     allowed_ids: set[str], n: int) -> list[Hit]:
    # ask for extra results when Python-side filters may drop some
    wanted = min(n * 3, store.count())
    result = store.collection.query(query_embeddings=[query.vector], n_results=wanted, where=where,
                                    include=["documents", "metadatas", "distances"])
    hits = []
    for chunk_id, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                              result["metadatas"][0], result["distances"][0]):
        if chunk_id in allowed_ids:
            # float(): ChromaDB returns numpy numbers, which can't be sent as JSON
            hits.append(Hit(chunk_id, text, meta, similarity=round(float(1 - distance), 4)))
    return hits[:n]


def _fill_missing_similarity(hits: list[Hit], query: Query, store: VectorStore) -> None:
    if not hits:
        return
    found = store.collection.get(ids=[h.id for h in hits], include=["embeddings"])
    vectors = dict(zip(found["ids"], found["embeddings"]))
    for hit in hits:
        hit.similarity = round(float(_cosine(query.vector, vectors[hit.id])), 4)


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return float(dot / norm) if norm else 0.0


# ------------------------------------------------------------------ keyword
class KeywordIndex:
    """BM25 index over every stored chunk. Built once per process from ChromaDB,
    so it always matches what was ingested."""

    def __init__(self, store: VectorStore):
        everything = store.collection.get(include=["documents", "metadatas"])
        self.texts = dict(zip(everything["ids"], everything["documents"]))
        self.metadatas = dict(zip(everything["ids"], everything["metadatas"]))
        self.terms = {i: Counter(keywords(t)) for i, t in self.texts.items()}
        self.lengths = {i: sum(c.values()) for i, c in self.terms.items()}
        self.average_length = (sum(self.lengths.values()) / len(self.lengths)) if self.lengths else 0
        document_frequency = Counter(word for c in self.terms.values() for word in c)
        total = len(self.terms)
        # rare words get a high weight, words found everywhere get almost none
        self.idf = {w: math.log((total - df + 0.5) / (df + 0.5) + 1)
                    for w, df in document_frequency.items()}

    def search(self, query_words: list[str], allowed_ids: set[str], n: int) -> list[tuple[str, float]]:
        scores = []
        for chunk_id in allowed_ids:
            score = self._bm25(query_words, chunk_id)
            if score > 0:
                scores.append((chunk_id, score))
        return sorted(scores, key=lambda s: s[1], reverse=True)[:n]

    def _bm25(self, query_words: list[str], chunk_id: str) -> float:
        counts, length = self.terms.get(chunk_id), self.lengths.get(chunk_id, 0)
        if not counts:
            return 0.0
        score = 0.0
        for word in set(query_words):
            tf = counts.get(word, 0)
            if not tf:
                continue
            norm = 1 - BM25_B + BM25_B * length / (self.average_length or 1)
            score += self.idf[word] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm)
        return score
