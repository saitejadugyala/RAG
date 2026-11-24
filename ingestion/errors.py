"""Error types, one per step.

The pipeline uses them to decide what to do when a file fails:

    retryable = False  -> the file itself is the problem (corrupt, empty, ...)
                          => copy it to data/quarantine/ with the reason
    retryable = True   -> a service had a temporary problem (OpenAI, ChromaDB)
                          => mark the file "failed"; it is retried on the next run
    fatal = True       -> stop retrying inside this run (bad API key, long outage)
"""


class IngestionError(Exception):
    step = "unknown"
    retryable = False

    def __init__(self, message: str, *, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


class FileCheckError(IngestionError):
    step = "1 check file"


class ExtractionError(IngestionError):
    step = "2 extract"


class OCRError(IngestionError):
    step = "2 extract (OCR)"


class EmptyDocumentError(IngestionError):
    step = "3 clean"


class ChunkingError(IngestionError):
    step = "4 chunk"


class NoValidChunksError(IngestionError):
    step = "5 check chunks"


class EmbeddingError(IngestionError):
    step = "6 embed"
    retryable = True


class EmbeddingRejectedError(IngestionError):
    """OpenAI answered, but refused every chunk of this file one by one.
    That is a problem with the content, not a temporary outage, so retrying
    on every run would never succeed."""

    step = "6 embed"
    retryable = False


class StorageError(IngestionError):
    step = "7 store"
    retryable = True
