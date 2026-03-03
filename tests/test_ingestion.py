"""Tests for PDF loading, chunking, and Qdrant indexing."""

import os
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import (
    chunk_documents,
    create_index,
    get_qdrant_client,
    load_pdf,
)

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
SPEC_PDF = DOCS_DIR / "Bms-Specs.pdf"


def test_load_pdf():
    """PyPDFLoader should return a list of Documents with page metadata."""
    assert SPEC_PDF.exists(), f"Test PDF not found: {SPEC_PDF}"
    docs = load_pdf(SPEC_PDF)
    assert len(docs) > 0, "No documents loaded"
    # Each document should have page metadata
    for doc in docs:
        assert "page" in doc.metadata, f"Missing 'page' in metadata: {doc.metadata}"
        assert "source" in doc.metadata
    print(f"  Loaded {len(docs)} pages from {SPEC_PDF.name}")
    print(f"  First page preview: {docs[0].page_content[:100]}...")


def test_chunk_documents():
    """Chunking should produce chunks with doc_type metadata."""
    docs = load_pdf(SPEC_PDF)
    chunks = chunk_documents(docs, doc_type="spec")
    assert len(chunks) > 0, "No chunks created"
    assert len(chunks) >= len(docs), "Expected at least as many chunks as pages"
    for chunk in chunks:
        assert chunk.metadata["doc_type"] == "spec"
        assert "chunk_index" in chunk.metadata
    print(f"  Created {len(chunks)} chunks from {len(docs)} pages")
    print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")


def test_create_index():
    """Indexing should embed chunks and store in Qdrant."""
    docs = load_pdf(SPEC_PDF)
    chunks = chunk_documents(docs, doc_type="spec")
    store = create_index(chunks)
    assert store is not None, "Vector store not created"

    # Verify documents are in Qdrant
    client = get_qdrant_client()
    info = client.get_collection("edr_documents")
    assert info.points_count == len(chunks), (
        f"Expected {len(chunks)} points, got {info.points_count}"
    )
    print(f"  Indexed {info.points_count} chunks in Qdrant")

    # Test basic similarity search
    results = store.similarity_search("chiller requirements", k=3)
    assert len(results) > 0, "No search results returned"
    print(f"  Search for 'chiller requirements' returned {len(results)} results")
    print(f"  Top result preview: {results[0].page_content[:100]}...")


if __name__ == "__main__":
    print("Test 1: load_pdf")
    test_load_pdf()
    print("  PASSED\n")

    print("Test 2: chunk_documents")
    test_chunk_documents()
    print("  PASSED\n")

    print("Test 3: create_index")
    test_create_index()
    print("  PASSED\n")

    print("All ingestion tests passed!")
