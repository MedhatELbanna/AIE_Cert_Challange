"""Hybrid retrieval tool with parent-child (small-to-big) pattern."""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool

from src.ingestion_advanced import get_vector_store_advanced

load_dotenv(override=True)


@tool
def retrieve_documents(query: str, doc_type: str = "all", top_k: int = 3) -> str:
    """Search indexed engineering documents using hybrid BM25 + semantic search.

    Use this tool to find relevant sections from specs, proposals, or standards.
    You can call this tool multiple times with different queries to find more
    relevant information. Try different phrasings if initial results aren't sufficient.

    Args:
        query: Natural language search query describing what you're looking for.
        doc_type: Filter by document type — "spec", "proposal", or "all".
        top_k: Number of results to return (default 3).

    Returns:
        Formatted string with relevant document chunks and their sources.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    store = get_vector_store_advanced()
    if store is None:
        return "ERROR: No documents have been indexed yet. Please index documents first."

    # Build optional doc_type filter
    dtype_filter = (
        [FieldCondition(key="metadata.doc_type", match=MatchValue(value=doc_type))]
        if doc_type != "all"
        else []
    )

    # ── Phase 1: Child → Parent (precise matching) ──────────────────────
    child_filter = Filter(must=[
        FieldCondition(key="metadata.chunk_level", match=MatchValue(value="child")),
        *dtype_filter,
    ])
    child_results: list[Document] = store.similarity_search(
        query,
        k=top_k * 3,  # over-fetch children to find best parents
        filter=child_filter,
    )

    seen_parents: set[str] = set()
    parent_ids_ordered: list[str] = []
    for doc in child_results:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen_parents:
            seen_parents.add(pid)
            parent_ids_ordered.append(pid)
        if len(parent_ids_ordered) >= top_k:
            break

    # ── Phase 2: Direct parent search (broad coverage) ──────────────────
    # Fills gaps when child→parent yields too few unique sections
    # (e.g. broad structural queries like "table of contents sections headings")
    if len(parent_ids_ordered) < top_k:
        remaining = top_k - len(parent_ids_ordered)
        parent_direct_filter = Filter(must=[
            FieldCondition(key="metadata.chunk_level", match=MatchValue(value="parent")),
            *dtype_filter,
        ])
        direct_parents: list[Document] = store.similarity_search(
            query,
            k=remaining + 5,  # slight over-fetch
            filter=parent_direct_filter,
        )
        for doc in direct_parents:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen_parents:
                seen_parents.add(pid)
                parent_ids_ordered.append(pid)
            if len(parent_ids_ordered) >= top_k:
                break

    # ── Phase 3: Fetch parent chunks by parent_id ───────────────────────
    parent_results: list[Document] = []
    child_fallback_map = {doc.metadata.get("parent_id"): doc for doc in child_results}

    for pid in parent_ids_ordered:
        parent_filter = Filter(must=[
            FieldCondition(key="metadata.parent_id", match=MatchValue(value=pid)),
            FieldCondition(key="metadata.chunk_level", match=MatchValue(value="parent")),
        ])
        parents = store.similarity_search(query, k=1, filter=parent_filter)
        if parents:
            parent_results.append(parents[0])
        elif pid in child_fallback_map:
            # Fallback: return the child chunk directly if parent not found
            parent_results.append(child_fallback_map[pid])

    if not parent_results:
        return f"No results found for query: '{query}' (doc_type={doc_type}). Try rephrasing your query."

    # ── Phase 4: Format results (same format as basic pipeline) ─────────
    formatted = []
    for i, doc in enumerate(parent_results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        dtype = doc.metadata.get("doc_type", "?")
        section = doc.metadata.get("section_path", "")
        formatted.append(
            f"--- Result {i} [source: {source}, page: {page}, type: {dtype}, section: {section}] ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted)
