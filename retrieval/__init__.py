"""RAG retrieval + generation.

    from retrieval import RAGPipeline, Filters
    answer = RAGPipeline().ask("How many days of annual leave do I get?")
    print(answer.text)
"""
from .models import Answer, Filters
from .pipeline import RAGPipeline

__all__ = ["Answer", "Filters", "RAGPipeline"]
