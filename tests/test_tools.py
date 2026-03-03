"""Tests for retrieval and search tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import chunk_documents, create_index, load_pdf
from src.tools import retrieve_documents, search_standards

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
SPEC_PDF = DOCS_DIR / "Bms-Specs.pdf"


def test_retrieve_documents():
    """retrieve_documents tool should return relevant chunks."""
    # Index the spec PDF first
    docs = load_pdf(SPEC_PDF)
    chunks = chunk_documents(docs, doc_type="spec")
    create_index(chunks)

    # Test retrieval with tool invoke
    result = retrieve_documents.invoke({"query": "BMS system requirements"})
    assert "Result 1" in result, f"Expected formatted results, got: {result[:200]}"
    assert "spec" in result.lower() or "source" in result.lower()
    print(f"  retrieve_documents returned {result.count('Result')} results")
    print(f"  Preview: {result[:200]}...")

    # Test with doc_type filter
    result_filtered = retrieve_documents.invoke({
        "query": "chiller interface",
        "doc_type": "spec",
    })
    assert "Result 1" in result_filtered
    print(f"  Filtered (doc_type=spec) returned {result_filtered.count('Result')} results")

    # Test with non-existent doc_type
    result_empty = retrieve_documents.invoke({
        "query": "anything",
        "doc_type": "proposal",
    })
    assert "No results" in result_empty or "Result" in result_empty
    print(f"  Filtered (doc_type=proposal): {result_empty[:100]}")


def test_search_standards():
    """search_standards tool should return web results."""
    result = search_standards.invoke({"query": "ASHRAE 90.1 energy standard requirements"})
    assert len(result) > 0, "No search results returned"
    print(f"  search_standards returned {len(result)} chars")
    print(f"  Preview: {result[:200]}...")


if __name__ == "__main__":
    print("Test 1: retrieve_documents")
    test_retrieve_documents()
    print("  PASSED\n")

    print("Test 2: search_standards")
    test_search_standards()
    print("  PASSED\n")

    print("All tool tests passed!")
