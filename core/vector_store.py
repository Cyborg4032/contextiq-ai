import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_DIR      = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Singletons ────────────────────────────────────────────────────────────────
_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: Chroma | None = None

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load HuggingFace embedding model once — ~50MB, no need to reload per call."""
    global _embeddings
    if _embeddings is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # better cosine similarity
        )
        print("Embedding model loaded.")
    return _embeddings


# ── Build ─────────────────────────────────────────────────────────────────────
def build_vector_store(transcript: str) -> Chroma:
    """
    Build a fresh in-memory vector store for this session's transcript.
    Not persisted to disk — each run gets a clean store so stale data
    from previous videos never bleeds into RAG answers.
    Call save_vector_store() explicitly if persistence is needed.
    """
    global _vector_store

    print("Building vector store ...")
    chunks = _splitter.split_text(transcript)
    print(f"  -> {len(chunks)} chunks from {len(transcript):,} characters")

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    _vector_store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        # No persist_directory = in-memory only, fresh every run
    )

    print("Vector store ready.")
    return _vector_store


# ── Persist (opt-in) ──────────────────────────────────────────────────────────
def save_vector_store() -> None:
    """Explicitly persist the current session's vector store to disk."""
    if _vector_store is None:
        raise RuntimeError("No vector store in memory. Run build_vector_store() first.")

    Chroma.from_documents(
        documents=list(_vector_store._collection.get()["documents"]),
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store saved to {CHROMA_DIR}/")


# ── Load ──────────────────────────────────────────────────────────────────────
def load_vector_store() -> Chroma:
    """Load a previously saved vector store from disk."""
    global _vector_store

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"No persisted vector store found at '{CHROMA_DIR}/'. "
            "Run build_vector_store() and save_vector_store() first."
        )

    print(f"Loading vector store from {CHROMA_DIR}/ ...")
    _vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    return _vector_store


# ── Retriever ─────────────────────────────────────────────────────────────────
def get_retriever(vector_store: Chroma, k: int = 4):
    """
    MMR (Maximum Marginal Relevance) instead of plain similarity —
    returns diverse results rather than near-duplicate chunks.
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
