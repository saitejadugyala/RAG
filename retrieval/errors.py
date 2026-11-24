"""Errors the retrieval pipeline reports to the user instead of crashing."""


class RetrievalError(Exception):
    """Base class."""


class QuestionError(RetrievalError):
    """The question is empty, too long, or could not be embedded."""


class EmptyDatabaseError(RetrievalError):
    """Nothing has been ingested yet."""


class GenerationError(RetrievalError):
    """The LLM could not produce an answer (outage, bad key, bad reply)."""
