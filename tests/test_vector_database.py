# tests/test_vector_database.py
from pathlib import Path

import pytest

from src.vector_database.vector_database import VectorDatabase


def test_chunk_text_basic():
    """
    Ensure that chunk_text splits a simple string into chunks with overlap.
    """
    text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
    chunk_size = 10
    overlap = 2

    chunks = VectorDatabase.chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # Should produce multiple chunks
    assert len(chunks) > 1
    # Each chunk must be non-empty and at most chunk_size
    for c in chunks:
        assert isinstance(c, str)
        assert 0 < len(c) <= chunk_size


def _make_vector_db() -> VectorDatabase:
    """
    Helper that builds a VectorDatabase pointing to the real pdf_database and chroma_db.
    """
    root_dir = Path(__file__).resolve().parents[1]
    pdf_root_path = root_dir / "pdf_database"
    chroma_path = root_dir / "chroma_db"

    return VectorDatabase(
        pdf_root_path=pdf_root_path,
        chroma_path=chroma_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="articles",
        chunk_size=1000,
        chunk_overlap=200,
    )


@pytest.mark.slow
def test_search_articles_returns_results():
    """
    Integration-style test: build the index and check that search_articles returns
    a non-empty list with the expected keys.
    """
    vdb = _make_vector_db()

    # Build (or rebuild) the index from the 9 PDFs
    vdb.build_index()

    results = vdb.search_articles("machine learning", top_k=3)

    assert isinstance(results, list)
    assert len(results) > 0

    for item in results:
        assert isinstance(item, dict)
        for key in ["id", "title", "area", "score"]:
            assert key in item


@pytest.mark.slow
def test_get_article_content_roundtrip():
    """
    Integration-style test: after a search, ensure get_article_content returns
    a valid article dict with non-empty content.
    """
    vdb = _make_vector_db()
    vdb.build_index()

    results = vdb.search_articles("economy", top_k=1)
    assert results, "Expected at least one result from search_articles"

    article_id = results[0]["id"]
    article = vdb.get_article_content(article_id)

    assert isinstance(article, dict)
    for key in ["id", "title", "area", "content"]:
        assert key in article

    assert article["id"] == article_id
    assert isinstance(article["content"], str)
    assert len(article["content"].strip()) > 0
