"""PDF loading, chunking, and Qdrant indexing."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_vector_store: QdrantVectorStore | None = None
_qdrant_client: QdrantClient | None = None

COLLECTION_NAME = "edr_documents"
EMBEDDING_DIM = 1536  # text-embedding-3-small


def get_embeddings() -> OpenAIEmbeddings:
    """Return OpenAI embedding model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_qdrant_client() -> QdrantClient:
    """Return singleton in-memory Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(location=":memory:")
    return _qdrant_client


# ---------------------------------------------------------------------------
# PDF Loading
# ---------------------------------------------------------------------------

def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF using PyPDFLoader, returning one Document per page."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    # Ensure source metadata
    for doc in docs:
        doc.metadata.setdefault("source", path.name)
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(
    docs: list[Document],
    doc_type: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into chunks and stamp doc_type metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_type"] = doc_type
        chunk.metadata["chunk_index"] = i
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def create_index(chunks: list[Document]) -> QdrantVectorStore:
    """Embed chunks and store in Qdrant. Returns the vector store."""
    global _vector_store

    client = get_qdrant_client()
    embeddings = get_embeddings()

    # Recreate collection (idempotent for fresh index)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    store.add_documents(chunks)
    _vector_store = store
    return store


def get_vector_store() -> QdrantVectorStore | None:
    """Return the current vector store singleton."""
    return _vector_store


def set_vector_store(store: QdrantVectorStore) -> None:
    """Set the vector store singleton (used after indexing)."""
    global _vector_store
    _vector_store = store
