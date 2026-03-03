"""LangChain tools for Agentic RAG — retrieve_documents and search_standards."""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from src.ingestion import get_vector_store

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Retrieval tool — queries Qdrant with optional doc_type filtering
# ---------------------------------------------------------------------------

@tool
def retrieve_documents(query: str, doc_type: str = "all", top_k: int = 3) -> str:
    """Search indexed engineering documents using semantic similarity.

    Use this tool to find relevant sections from specs, proposals, or standards.
    You can call this tool multiple times with different queries to find more
    relevant information. Try different phrasings if initial results aren't sufficient.

    Args:
        query: Natural language search query describing what you're looking for.
        doc_type: Filter by document type — "spec", "proposal", or "all".
        top_k: Number of results to return (default 5).

    Returns:
        Formatted string with relevant document chunks and their sources.
    """
    store = get_vector_store()
    if store is None:
        return "ERROR: No documents have been indexed yet. Please index documents first."

    # Build filter for doc_type
    search_kwargs: dict = {"k": top_k}
    if doc_type != "all":
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        search_kwargs["filter"] = Filter(
            must=[FieldCondition(key="metadata.doc_type", match=MatchValue(value=doc_type))]
        )

    results: list[Document] = store.similarity_search(query, **search_kwargs)

    if not results:
        return f"No results found for query: '{query}' (doc_type={doc_type}). Try rephrasing your query."

    # Format results with source info
    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        dtype = doc.metadata.get("doc_type", "?")
        formatted.append(
            f"--- Result {i} [source: {source}, page: {page}, type: {dtype}] ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted)


# ---------------------------------------------------------------------------
# Standards search tool — Tavily web search for engineering standards
# ---------------------------------------------------------------------------

@tool
def search_standards(query: str) -> str:
    """Search the web for engineering standards information.

    Use this tool to look up details about referenced standards like
    ASHRAE 90.1, NFPA 72, SMACNA, UL listings, etc.

    Args:
        query: Search query about a specific engineering standard.

    Returns:
        Web search results with relevant standards information and source URLs.
    """
    search = TavilySearch(max_results=3)
    response = search.invoke(query)

    if not response:
        return f"No web results found for: '{query}'"

    # TavilySearch returns a dict with a "results" key
    if isinstance(response, dict):
        results = response.get("results", [])
    elif isinstance(response, list):
        results = response
    else:
        return str(response)

    if not results:
        return f"No web results found for: '{query}'"

    formatted = []
    for i, result in enumerate(results, 1):
        if isinstance(result, dict):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            formatted.append(f"--- Result {i}: {title} ---\n{content}\nURL: {url}\n")
        else:
            formatted.append(f"--- Result {i} ---\n{result}\n")

    return "\n".join(formatted) if formatted else str(response)
