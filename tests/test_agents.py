"""Tests for the deep agent graph."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import chunk_documents, create_index, load_pdf

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
SPEC_PDF = DOCS_DIR / "Bms-Specs.pdf"
PROPOSAL_PDF = DOCS_DIR / "BMS-Technical-Submittal (1).pdf"


def test_graph_compiles():
    """The orchestrator graph should compile without errors."""
    from src.agents import build_graph, build_topic_graph

    # Topic sub-graph
    topic_graph = build_topic_graph().compile()
    print(f"  Topic sub-graph compiled: {topic_graph}")

    # Orchestrator graph
    orch_graph = build_graph().compile()
    print(f"  Orchestrator graph compiled: {orch_graph}")


def test_full_graph_run():
    """Full end-to-end: index docs, run orchestrator graph, get report."""
    # Index both documents
    print("  Indexing spec...")
    spec_docs = load_pdf(SPEC_PDF)
    spec_chunks = chunk_documents(spec_docs, doc_type="spec")

    all_chunks = spec_chunks

    if PROPOSAL_PDF.exists():
        print("  Indexing proposal...")
        prop_docs = load_pdf(PROPOSAL_PDF)
        prop_chunks = chunk_documents(prop_docs, doc_type="proposal")
        all_chunks = spec_chunks + prop_chunks

    create_index(all_chunks)
    print(f"  Indexed {len(all_chunks)} total chunks")

    # Run the orchestrator graph
    from src.agents import compile_graph

    graph = compile_graph()
    print("  Running orchestrator graph (this will take a few minutes)...")

    result = graph.invoke({
        "document_types": {
            SPEC_PDF.name: "spec",
            PROPOSAL_PDF.name: "proposal",
        },
        "review_request": "Review the BMS proposal against the specification requirements",
        "topics": [],
        "current_topic_index": 0,
        "all_verdicts": [],
        "topic_summaries": [],
        "final_report": "",
        "messages": [],
    })

    # Check results
    topics = result.get("topics", [])
    verdicts = result.get("all_verdicts", [])
    report = result.get("final_report", "")

    print(f"\n  Topics reviewed: {topics}")
    print(f"  Total verdicts: {len(verdicts)}")
    print(f"  Report length: {len(report)} chars")

    if verdicts:
        print(f"\n  Sample verdict:")
        v = verdicts[0]
        print(f"    {v.req_id}: {v.verdict} ({v.severity})")
        print(f"    Reasoning: {v.reasoning[:100]}...")

    if report:
        print(f"\n  Report preview:\n{report[:500]}...")

    assert len(topics) > 0, "No topics planned"
    assert len(verdicts) > 0, "No verdicts produced"
    assert len(report) > 0, "No report generated"


if __name__ == "__main__":
    print("Test 1: graph_compiles")
    test_graph_compiles()
    print("  PASSED\n")

    print("Test 2: full_graph_run")
    test_full_graph_run()
    print("  PASSED\n")

    print("All agent tests passed!")
