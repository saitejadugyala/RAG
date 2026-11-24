"""Shared test fixtures.

Tests never read docs/ directly: that folder holds YOUR documents and can change
at any time. They generate their own copy of the three RAG-concept documents
(scripts/make_rag_docs.py) in a temporary folder instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@pytest.fixture(scope="session")
def rag_docs(tmp_path_factory) -> Path:
    import make_rag_docs

    folder = tmp_path_factory.mktemp("rag_docs")
    make_rag_docs.main(folder)
    return folder
