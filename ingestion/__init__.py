"""RAG ingestion pipeline.

    from ingestion import IngestionPipeline, Settings
    summary = IngestionPipeline(Settings()).run(["docs"])
"""
from .pipeline import IngestionPipeline
from .settings import Settings

__all__ = ["IngestionPipeline", "Settings"]
