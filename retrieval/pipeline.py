"""The retrieval pipeline: question in, cited answer out.

    1 prepare question   -> vector + keywords
    2 hybrid search      -> semantic + keyword (BM25) search, metadata filters,
                            merged with Reciprocal Rank Fusion
    3 select sources     -> relevance check, keep the top 3
    4 generate answer    -> LLM answers from those sources only, with [n] citations
                            (or "no information" if nothing relevant was found)

Every question and answer is appended to data/reports/questions.jsonl.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time

from ingestion.settings import Settings
from ingestion.step6_embed import make_embedder
from ingestion.step7_store import VectorStore
from ingestion.tracking import now

from .errors import EmptyDatabaseError, GenerationError
from .models import NO_INFO_ANSWER, Answer, Filters, Hit
from .step1_prepare_question import prepare_question
from .step2_hybrid_search import KeywordIndex, hybrid_search
from .step3_select_sources import select_sources
from .step4_generate_answer import generate_answer, make_llm

log = logging.getLogger("retrieval")


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, *, fake: bool = False):
        """fake=True uses offline embeddings and an offline LLM (for tests)."""
        self.settings = settings or Settings()
        self.embedder = make_embedder(self.settings, fake=fake)
        self.store = VectorStore(self.settings, self.embedder.name)  # same model as ingestion
        if self.store.count() == 0:
            raise EmptyDatabaseError("The vector database is empty. Run `python ingest.py` first.")
        self.index = KeywordIndex(self.store)
        self.llm = make_llm(self.settings, fake=fake)
        self.last_hits: list[Hit] = []  # every search hit of the last question (--show-search)
        self.last_sources: list = []  # sources selected for the last question (evaluation)
        self._index_lock = threading.Lock()

    def retrieve(self, question: str, filters: Filters | None = None):
        """Steps 1-3 only (no LLM). Returns (query, ranked hits, top sources)."""
        filters = filters or Filters()
        query = prepare_question(question, self.embedder, self.settings)
        hits = hybrid_search(query, filters, self.store, self.index, self.settings)
        sources = select_sources(hits, self.settings)
        self.last_hits, self.last_sources = hits, sources
        return query, hits, sources

    def refresh_index_if_needed(self) -> bool:
        """Rebuild the keyword index if the stored chunks changed since it was built
        (ingest.py ran while the web server was up). Compares chunk ids, not just the
        count: a replaced file can have the same number of chunks but new ids, because
        an id is made from the file's fingerprint and the chunk text. Returns True if rebuilt."""
        with self._index_lock:
            current_ids = set(self.store.collection.get(include=[])["ids"])
            if current_ids != set(self.index.texts):
                self.index = KeywordIndex(self.store)
                return True
        return False

    def ask(self, question: str, filters: Filters | None = None) -> Answer:
        return self.ask_with_details(question, filters)[0]

    def ask_with_details(self, question: str, filters: Filters | None = None,
                         top_k: int | None = None) -> tuple[Answer, list[Hit]]:
        """Like ask(), but also returns every search hit, and top_k can be set per
        question. Safe when several web requests arrive at the same time."""
        started = time.time()
        filters = filters or Filters()
        settings = dataclasses.replace(self.settings, top_k=top_k) if top_k else self.settings
        query = prepare_question(question, self.embedder, settings)
        hits = hybrid_search(query, filters, self.store, self.index, settings)
        sources = select_sources(hits, settings)
        self.last_hits, self.last_sources = hits, sources
        log.debug("%d hits, %d relevant, %d sent to the LLM",
                  len(hits), sum(h.relevant for h in hits), len(sources))

        if sources:
            try:
                text, found, cited, warnings = generate_answer(query.text, sources, self.llm)
            except GenerationError as e:
                e.hits = hits  # so the caller can still show the passages that were found
                raise
        else:  # nothing relevant: don't call the LLM at all
            text, found, cited, warnings = NO_INFO_ANSWER, False, [], []

        answer = Answer(question=query.text, text=text, found=found,
                        sources=sources if found else [], cited=cited,
                        filters=filters.describe(), warnings=warnings,
                        seconds=round(time.time() - started, 2))
        self._log(answer, considered=len(sources))
        return answer, hits

    def _log(self, answer: Answer, considered: int) -> None:
        try:
            self.settings.reports_dir.mkdir(parents=True, exist_ok=True)
            with (self.settings.reports_dir / "questions.jsonl").open("a") as f:
                f.write(json.dumps({
                    "at": now(), "question": answer.question, "found": answer.found,
                    "answer": answer.text, "cited": answer.cited, "filters": answer.filters,
                    "sources_considered": considered,
                    "sources": [s.label() for s in answer.sources],
                    "model": getattr(self.llm, "name", "?"), "seconds": answer.seconds,
                }) + "\n")
        except OSError as e:
            log.warning("could not write questions.jsonl: %s", e)
