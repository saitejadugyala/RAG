"""STEP 6 - Turn each chunk into a vector with OpenAI.

Model: text-embedding-3-small, 1536 dimensions.

Failure handling:
  * rate limits, timeouts, connection and 5xx errors are retried with
    exponential backoff (settings.max_retries attempts)
  * a batch OpenAI rejects is retried one chunk at a time, so only the bad
    chunk is dropped instead of the whole file
  * a wrong API key or a long outage stops the file (it is retried next run)
  * every vector is checked: 1536 numbers, no NaN/Inf, not all zeros
"""
from __future__ import annotations

import hashlib
import logging
import math
import os

from tenacity import (before_sleep_log, retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

from .errors import EmbeddingError
from .models import Chunk
from .settings import Settings

log = logging.getLogger(__name__)
MAX_TOKENS_PER_REQUEST = 300_000  # OpenAI limit for one embeddings request


class _TemporaryError(Exception):
    """An OpenAI error that is worth retrying."""


class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        import openai

        if not os.getenv("OPENAI_API_KEY"):
            raise EmbeddingError("OPENAI_API_KEY is not set. Add it to .env "
                                 "(or use --fake-embeddings to test offline).", fatal=True)
        self.settings = settings
        self.name = f"openai:{settings.embedding_model}"
        self._openai = openai
        self._client = openai.OpenAI(max_retries=0, timeout=60)  # we do the retrying
        self._embed_with_retries = retry(
            retry=retry_if_exception_type(_TemporaryError),
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=stop_after_attempt(settings.max_retries),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        )(self._embed_once)

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._embed_with_retries(texts)
        except _TemporaryError as e:
            raise EmbeddingError(f"OpenAI still failing after {self.settings.max_retries} "
                                 f"attempts: {e}", fatal=True) from e

    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        o = self._openai
        try:
            response = self._client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts,
                dimensions=self.settings.embedding_dimensions,
            )
        except (o.RateLimitError, o.APITimeoutError, o.APIConnectionError,
                o.InternalServerError) as e:
            raise _TemporaryError(str(e)) from e
        except (o.AuthenticationError, o.PermissionDeniedError) as e:
            raise EmbeddingError(f"OpenAI rejected the API key: {e}", fatal=True) from e
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


class FakeEmbedder:
    """Offline vectors made from a hash of the text - for testing the pipeline
    without an API key. They carry NO meaning; never use them for real search."""

    name = "fake:hash"

    def __init__(self, settings: Settings):
        self.dimensions = settings.embedding_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = hashlib.sha256(text.encode()).digest()
            values = [(seed[i % 32] ^ (i * 31 % 251)) / 255 - 0.5 for i in range(self.dimensions)]
            length = math.sqrt(sum(v * v for v in values)) or 1.0
            vectors.append([v / length for v in values])
        return vectors


def make_embedder(settings: Settings, fake: bool = False):
    return FakeEmbedder(settings) if fake else OpenAIEmbedder(settings)


# ======================================================================
def embed_chunks(chunks: list[Chunk], embedder, settings: Settings) -> list[str]:
    """Add an embedding to every non-rejected chunk. Returns warnings."""
    warnings: list[str] = []
    for number, batch in enumerate(_batches([c for c in chunks if not c.rejected], settings)):
        try:
            vectors = embedder.embed([c.text for c in batch])
            if len(vectors) != len(batch):
                raise EmbeddingError(f"got {len(vectors)} vectors for {len(batch)} chunks")
            for chunk, vector in zip(batch, vectors):
                _check_vector(vector, settings.embedding_dimensions)
                chunk.embedding = vector
        except Exception as e:
            if isinstance(e, EmbeddingError) and e.fatal:
                raise
            warnings.append(f"batch {number} failed ({e}); retrying one chunk at a time")
            _embed_one_by_one(batch, embedder, settings, warnings)
    return warnings


def _batches(chunks: list[Chunk], settings: Settings):
    """Group chunks by count and by total tokens (both have OpenAI limits)."""
    batch, tokens = [], 0
    for chunk in chunks:
        if batch and (len(batch) >= settings.embedding_batch_size
                      or tokens + chunk.tokens > MAX_TOKENS_PER_REQUEST):
            yield batch
            batch, tokens = [], 0
        batch.append(chunk)
        tokens += chunk.tokens
    if batch:
        yield batch


def _embed_one_by_one(batch: list[Chunk], embedder, settings: Settings, warnings: list[str]) -> None:
    for chunk in batch:
        try:
            vector = embedder.embed([chunk.text])[0]
            _check_vector(vector, settings.embedding_dimensions)
            chunk.embedding = vector
        except Exception as e:
            if isinstance(e, EmbeddingError) and e.fatal:
                raise
            chunk.rejected = True
            chunk.flags.append("reject:embedding_failed")
            chunk.metadata["quality_flags"] = ",".join(chunk.flags)
            warnings.append(f"chunk {chunk.index}: embedding failed ({e})")


def _check_vector(vector: list[float], dimensions: int) -> None:
    if len(vector) != dimensions:
        raise EmbeddingError(f"vector has {len(vector)} numbers, expected {dimensions}")
    if any(math.isnan(v) or math.isinf(v) for v in vector):
        raise EmbeddingError("vector contains NaN/Inf")
    if not any(vector):
        raise EmbeddingError("vector is all zeros")
