"""Advanced ingestion: Azure Document Intelligence + hierarchical chunking + hybrid index."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_vector_store_advanced = None
_section_outlines: dict[str, str] = {}  # doc_type -> formatted outline

COLLECTION_NAME_ADVANCED = "edr_documents_advanced"


def get_vector_store_advanced():
    """Return the advanced vector store singleton."""
    return _vector_store_advanced


def set_vector_store_advanced(store) -> None:
    """Set the advanced vector store singleton."""
    global _vector_store_advanced
    _vector_store_advanced = store


def get_section_outline(doc_type: str = "spec") -> str:
    """Return the stored section outline for a document type.

    The outline is a formatted string of all section headings with levels,
    generated during chunk_hierarchical(). Useful for topic planning.
    """
    return _section_outlines.get(doc_type, "")


def _build_outline(sections: list, indent: int = 0) -> list[str]:
    """Recursively build a section outline from SectionNode tree."""
    lines = []
    for section in sections:
        prefix = "  " * indent
        lines.append(f"{prefix}- {section.heading} (page {section.page})")
        lines.extend(_build_outline(section.children, indent + 1))
    return lines


# ---------------------------------------------------------------------------
# Azure Document Intelligence extraction
# ---------------------------------------------------------------------------

def load_pdf_with_layout(path: str | Path):
    """Extract layout-aware structure from PDF using Azure Document Intelligence.

    Returns an AnalyzeResult with paragraphs (with roles), tables, and markdown content.
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentContentFormat
    from azure.core.credentials import AzureKeyCredential

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")
    if not endpoint or not key:
        raise ValueError("AZURE_DI_ENDPOINT and AZURE_DI_KEY must be set in environment")

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    with open(path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
    return poller.result()


# ---------------------------------------------------------------------------
# Table → markdown conversion
# ---------------------------------------------------------------------------

def _table_to_markdown(table) -> str:
    """Convert an Azure DI DocumentTable to a markdown table string."""
    grid: dict[tuple[int, int], str] = {}
    for cell in table.cells:
        grid[(cell.row_index, cell.column_index)] = cell.content or ""

    rows = []
    for r in range(table.row_count):
        row = [grid.get((r, c), "") for c in range(table.column_count)]
        rows.append("| " + " | ".join(row) + " |")
        if r == 0:
            rows.append("| " + " | ".join(["---"] * table.column_count) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Section tree data structure
# ---------------------------------------------------------------------------

@dataclass
class SectionNode:
    heading: str
    heading_level: int
    page: int
    paragraphs: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    children: list[SectionNode] = field(default_factory=list)
    section_path: str = ""


# ---------------------------------------------------------------------------
# Build section tree from Azure DI result
# ---------------------------------------------------------------------------

_NOISE_ROLES = {"pageHeader", "pageFooter", "pageNumber", "footnote"}


def _get_heading_level(heading_text: str, markdown_content: str) -> int:
    """Determine heading level by finding the heading in the markdown content.

    Looks for lines starting with # and matching the heading text.
    """
    escaped = re.escape(heading_text.strip()[:60])
    pattern = r"^(#{1,6})\s+" + escaped
    match = re.search(pattern, markdown_content, re.MULTILINE)
    if match:
        return len(match.group(1))
    return 2  # default to level 2 if not found


def _get_paragraph_page(paragraph) -> int:
    """Extract page number from a paragraph's bounding regions."""
    if paragraph.bounding_regions:
        return paragraph.bounding_regions[0].page_number
    return 1


def build_section_tree(result) -> list[SectionNode]:
    """Build a hierarchical section tree from Azure DI AnalyzeResult.

    Parses paragraphs with roles (title, sectionHeading) to build structure.
    Body text paragraphs are attached to the most recent heading.
    """
    paragraphs = result.paragraphs or []
    tables = result.tables or []
    markdown_content = result.content or ""

    # Sort paragraphs by their offset in the document
    sorted_paras = sorted(
        paragraphs,
        key=lambda p: p.spans[0].offset if p.spans else 0,
    )

    # Convert tables to markdown strings keyed by page
    table_markdowns: list[tuple[int, str]] = []
    for table in tables:
        page = table.bounding_regions[0].page_number if table.bounding_regions else 1
        table_markdowns.append((page, _table_to_markdown(table)))

    # Build tree using a stack
    root_nodes: list[SectionNode] = []
    stack: list[SectionNode] = []  # current nesting path

    # Create a default "preamble" section for content before the first heading
    preamble = SectionNode(heading="Document Preamble", heading_level=0, page=1, section_path="Preamble")

    current_section = preamble

    for para in sorted_paras:
        role = para.role
        content = (para.content or "").strip()
        page = _get_paragraph_page(para)

        if not content:
            continue

        # Skip noise
        if role in _NOISE_ROLES:
            continue

        if role in ("title", "sectionHeading"):
            level = 1 if role == "title" else _get_heading_level(content, markdown_content)

            node = SectionNode(
                heading=content,
                heading_level=level,
                page=page,
            )

            # Pop stack until we find a parent with a lower level
            while stack and stack[-1].heading_level >= level:
                stack.pop()

            if stack:
                parent = stack[-1]
                parent.children.append(node)
                node.section_path = f"{parent.section_path} > {content}"
            else:
                root_nodes.append(node)
                node.section_path = content

            stack.append(node)
            current_section = node
        else:
            # Body text — attach to current section
            current_section.paragraphs.append(content)

    # Attach preamble if it has content
    if preamble.paragraphs:
        root_nodes.insert(0, preamble)

    # Distribute tables to their nearest section by page
    _assign_tables_to_sections(root_nodes, table_markdowns)

    return root_nodes


def _assign_tables_to_sections(nodes: list[SectionNode], tables: list[tuple[int, str]]) -> None:
    """Assign tables to sections based on page proximity."""
    all_sections = _flatten_sections(nodes)
    for page, md_table in tables:
        best_section = None
        for section in all_sections:
            if section.page <= page:
                best_section = section
        if best_section:
            best_section.tables.append(md_table)
        elif all_sections:
            all_sections[0].tables.append(md_table)


def _flatten_sections(nodes: list[SectionNode]) -> list[SectionNode]:
    """Flatten nested sections into a list preserving document order."""
    result = []
    for node in nodes:
        result.append(node)
        result.extend(_flatten_sections(node.children))
    return result


# ---------------------------------------------------------------------------
# Hierarchical chunking
# ---------------------------------------------------------------------------

_MAX_PARENT_CHARS = 4000
_PARENT_SPLIT_SIZE = 2000
_PARENT_SPLIT_OVERLAP = 400
_MIN_CHILD_CHARS = 50
_MAX_CHILD_CHARS = 1500
_CHILD_SPLIT_SIZE = 1000
_CHILD_SPLIT_OVERLAP = 200


def _section_full_text(section: SectionNode) -> str:
    """Concatenate heading + all body text + tables for a section (no children)."""
    parts = [section.heading]
    parts.extend(section.paragraphs)
    parts.extend(section.tables)
    return "\n\n".join(parts)


def chunk_hierarchical(
    sections: list[SectionNode],
    doc_type: str,
    source: str = "",
) -> tuple[list[Document], list[Document]]:
    """Create parent and child chunks from a section tree.

    Also stores a section outline in the module-level _section_outlines dict
    for use by the topic planner.

    Parent chunks: full section content (heading + body + tables).
    Child chunks: individual paragraphs/requirements within sections.

    Returns (parent_chunks, child_chunks).
    """
    # Store section outline for topic planning
    outline_lines = _build_outline(sections)
    _section_outlines[doc_type] = "\n".join(outline_lines)

    parent_chunks: list[Document] = []
    child_chunks: list[Document] = []

    all_sections = _flatten_sections(sections)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHILD_SPLIT_SIZE,
        chunk_overlap=_CHILD_SPLIT_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_PARENT_SPLIT_SIZE,
        chunk_overlap=_PARENT_SPLIT_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for section in all_sections:
        parent_id = str(uuid.uuid4())
        full_text = _section_full_text(section)

        if not full_text.strip():
            continue

        base_meta = {
            "doc_type": doc_type,
            "parent_id": parent_id,
            "section_path": section.section_path,
            "heading_level": section.heading_level,
            "page": section.page,
            "source": source,
        }

        # --- Parent chunks ---
        if len(full_text) > _MAX_PARENT_CHARS:
            # Split large parent into overlapping sub-chunks
            parent_docs = parent_splitter.create_documents(
                [full_text],
                metadatas=[{**base_meta, "chunk_level": "parent"}],
            )
            for doc in parent_docs:
                doc.metadata["parent_id"] = parent_id  # same parent_id for all sub-chunks
            parent_chunks.extend(parent_docs)
        else:
            parent_chunks.append(Document(
                page_content=full_text,
                metadata={**base_meta, "chunk_level": "parent"},
            ))

        # --- Child chunks ---
        child_index = 0

        # Process paragraphs
        merged_text = ""
        for para in section.paragraphs:
            if len(para) < _MIN_CHILD_CHARS:
                merged_text = f"{merged_text}\n{para}" if merged_text else para
                continue

            if merged_text:
                # Flush merged small paragraphs
                if len(merged_text) > _MAX_CHILD_CHARS:
                    sub_docs = splitter.create_documents(
                        [merged_text],
                        metadatas=[{**base_meta, "chunk_level": "child", "chunk_index": child_index}],
                    )
                    for doc in sub_docs:
                        doc.metadata["chunk_index"] = child_index
                        child_chunks.append(doc)
                        child_index += 1
                else:
                    child_chunks.append(Document(
                        page_content=merged_text,
                        metadata={**base_meta, "chunk_level": "child", "chunk_index": child_index},
                    ))
                    child_index += 1
                merged_text = ""

            if len(para) > _MAX_CHILD_CHARS:
                sub_docs = splitter.create_documents(
                    [para],
                    metadatas=[{**base_meta, "chunk_level": "child", "chunk_index": child_index}],
                )
                for doc in sub_docs:
                    doc.metadata["chunk_index"] = child_index
                    child_chunks.append(doc)
                    child_index += 1
            else:
                child_chunks.append(Document(
                    page_content=para,
                    metadata={**base_meta, "chunk_level": "child", "chunk_index": child_index},
                ))
                child_index += 1

        # Flush remaining merged text
        if merged_text:
            child_chunks.append(Document(
                page_content=merged_text,
                metadata={**base_meta, "chunk_level": "child", "chunk_index": child_index},
            ))
            child_index += 1

        # Tables as child chunks
        for table_md in section.tables:
            child_chunks.append(Document(
                page_content=table_md,
                metadata={**base_meta, "chunk_level": "child", "chunk_index": child_index},
            ))
            child_index += 1

    return parent_chunks, child_chunks


# ---------------------------------------------------------------------------
# Hybrid indexing (BM25 + Dense + RRF via Qdrant)
# ---------------------------------------------------------------------------

def create_index_advanced(chunks: list[Document]):
    """Index chunks with both dense (OpenAI) and sparse (BM25) embeddings.

    Uses Qdrant's native hybrid retrieval with RRF fusion.
    Returns the QdrantVectorStore instance.
    """
    global _vector_store_advanced

    from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
    from qdrant_client.models import Distance, SparseVectorParams, VectorParams

    from src.ingestion import EMBEDDING_DIM, get_embeddings, get_qdrant_client

    client = get_qdrant_client()
    embeddings = get_embeddings()
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")

    # Recreate collection with both dense and sparse vector configs
    if client.collection_exists(COLLECTION_NAME_ADVANCED):
        client.delete_collection(COLLECTION_NAME_ADVANCED)

    client.create_collection(
        collection_name=COLLECTION_NAME_ADVANCED,
        vectors_config={"": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"langchain-sparse": SparseVectorParams()},
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME_ADVANCED,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=sparse,
        validate_collection_config=False,
    )
    store.add_documents(chunks)
    _vector_store_advanced = store
    return store
