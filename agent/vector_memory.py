"""
agent/vector_memory.py — Long-Term Vector Memory

WHY THIS EXISTS:
  The sliding window (memory.py) keeps the last 5 exchanges in RAM.
  It is fast but temporary — it disappears when the session ends.

  This module adds a SECOND memory layer that:
    - Persists to disk (survives restarts)
    - Stores every document the agent has ever generated
    - Can retrieve the most RELEVANT chunks for any new question
      using semantic similarity (meaning-based search, not keyword search)

THE THREE LAB 4 TOOLS USED HERE:

  1. all-MiniLM-L6-v2  (SentenceTransformer via HuggingFaceEmbeddings)
     ─────────────────────────────────────────────────────────────
     Converts any piece of text into a list of 384 numbers called an
     "embedding". Sentences with similar meanings produce numbers that
     are close together in mathematical space.

     "forgot my password"  →  [0.12, -0.43, 0.88, ...]
     "password recovery"   →  [0.11, -0.45, 0.85, ...]  ← very close
     "vacation policy"     →  [0.67,  0.21, 0.03, ...]  ← far away

  2. RecursiveCharacterTextSplitter
     ─────────────────────────────────────────────────────────────
     Breaks a large markdown document into smaller "chunks" so each
     chunk can get its own meaningful embedding.

     Why split? Because a 384-number fingerprint of a 5-page document
     is too vague — it blends all topics together. Splitting ensures
     each chunk is focused on ONE idea, so search is precise.

     chunk_size=500:    each chunk ≤ 500 characters
     chunk_overlap=100: 100 characters are SHARED between adjacent
                        chunks to avoid losing context at boundaries

  3. ChromaDB (via langchain_community)
     ─────────────────────────────────────────────────────────────
     A vector database. Think of it as a spreadsheet where each row
     stores:
       - the original chunk text
       - its 384-number embedding
       - metadata (which file it came from)

     persist_directory: ChromaDB writes its data to this folder on
     disk. Every time you save a document, new rows are added. The
     data survives across Python restarts.

     similarity_search(query, k=3):
       1. Embeds the query into 384 numbers
       2. Compares those numbers to every stored embedding
       3. Returns the k rows whose embeddings are closest

HOW THE TWO MEMORY LAYERS WORK TOGETHER:

  Short-term (sliding window):
    "What did the user just say in this session?"
    Keeps the last 5 exchanges in RAM for conversational context.

  Long-term (this module):
    "What documents have I ever created that are relevant to this question?"
    Searches all saved .md files on disk using semantic similarity.

  Together they give Gemini:
    - Conversational flow from the sliding window
    - Deep document knowledge from the vector store
"""

import os
import time
import logging
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Absolute path to ChromaDB storage directory.
#
# WHY ABSOLUTE?
#   os.path.dirname(__file__) → the folder this file lives in  (agent/)
#   os.path.join(..., "..")   → one level up                   (architect-agent/)
#   os.path.abspath(...)      → resolve to a full path         (/home/gihan/.../architect-agent/chroma_db)
#
#   Using an absolute path means ChromaDB always reads/writes the SAME folder
#   regardless of which directory you launch the script from.
#   Without this, running `python main.py` from ~/Desktop would create a NEW
#   empty chroma_db/ there and lose all previously indexed documents.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHROMA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "chroma_db")
)


class VectorMemory:
    """
    Long-term semantic memory backed by ChromaDB.

    This class wraps the three Lab 4 tools into one clean interface:
      - Embedding model   (all-MiniLM-L6-v2)
      - Text splitter     (RecursiveCharacterTextSplitter)
      - Vector store      (ChromaDB, persisted to disk)

    Usage:
        vm = VectorMemory()
        vm.index_document("# System Design\n...", "system-design.md")
        docs = vm.search("ecommerce API authentication")
        context_str = vm.format_context(docs)
    """

    def __init__(self, persist_dir: str = DEFAULT_CHROMA_DIR):
        """
        Args:
            persist_dir: Folder where ChromaDB stores its data on disk.
                         Created automatically if it does not exist.
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # ── 1. Embedding Model ────────────────────────────────────────────
        # HuggingFaceEmbeddings wraps all-MiniLM-L6-v2.
        # It is used in BOTH directions:
        #   - When indexing: embed each chunk before storing
        #   - When searching: embed the query before comparing
        # Using the same model both ways guarantees the math is consistent.
        self._embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},          # use CPU (no GPU required)
            encode_kwargs={"normalize_embeddings": True},  # normalise for cosine similarity
        )

        # ── 2. Text Splitter ──────────────────────────────────────────────
        # Try TokenTextSplitter (requires tiktoken) for accurate token-based chunks.
        # Fallback to RecursiveCharacterTextSplitter if tiktoken not installed.
        try:
            self._splitter = TokenTextSplitter(
                encoding_name="cl100k_base",  # GPT-4 encoding (good approximation)
                chunk_size=500,
                chunk_overlap=100,
            )
            logger.info("TokenTextSplitter initialized (token-based chunking)")
        except ImportError:
            # Fallback: character-based splitter if tiktoken missing
            logger.warning(
                "tiktoken not installed, falling back to RecursiveCharacterTextSplitter. "
                "Token budgets will be approximate. Install tiktoken for accurate chunking."
            )
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,      # characters (approximate)
                chunk_overlap=100,
                length_function=len,
            )

        # ── 3. ChromaDB Vector Store ──────────────────────────────────────
        # Lazy-loaded on first use (see _get_store()).
        # This avoids loading ChromaDB if no documents have been saved yet.
        self._store: Chroma | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def index_document(self, content: str, source_filename: str) -> int:
        """
        Split a document into chunks, embed each chunk, and store in ChromaDB.

        Call this every time a new .md file is saved.

        Pseudocode:
            1. Split content into overlapping 500-char chunks
            2. Tag every chunk with {"source": source_filename} as metadata
            3. Add all chunks to ChromaDB (it embeds them automatically)
            4. Return the number of chunks created

        Args:
            content:         The raw markdown text of the saved document.
            source_filename: The filename (e.g. "system-design.md").
                             Stored as metadata so we know where each chunk
                             came from when we retrieve it later.

        Returns:
            Number of chunks created and stored.
        """
        start_time = time.time()
        logger.info("index_document called", extra={"saved_file": source_filename})

        # Step 1 & 2: Split the document into chunks with metadata
        split_start = time.time()
        chunks: List[Document] = self._splitter.create_documents(
            texts=[content],
            metadatas=[{"source": source_filename}],
        )
        split_duration = int((time.time() - split_start) * 1000)

        if not chunks:
            logger.warning("No chunks created from document")
            return 0

        logger.debug(f"Document split into {len(chunks)} chunks", extra={"chunks": len(chunks), "duration_ms": split_duration})

        # Step 3: Store in ChromaDB
        store_start = time.time()
        store = self._get_store()
        store.add_documents(chunks)
        store_duration = int((time.time() - store_start) * 1000)

        total_duration = int((time.time() - start_time) * 1000)
        logger.info("index_document completed", extra={"saved_file": source_filename, "chunks": len(chunks), "duration_ms": total_duration})

        return len(chunks)

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Find the k most semantically relevant chunks for a query.

        Pseudocode:
            1. Embed the query into 384 numbers using all-MiniLM-L6-v2
            2. Compare those 384 numbers against every stored embedding
               using cosine similarity (measures angle between vectors)
            3. Return the k chunks with the highest similarity scores

        Args:
            query: The user's message or any text to search for.
            k:     How many chunks to return (default 3).
                   More = richer context but more tokens used.

        Returns:
            List of Document objects (each has .page_content and .metadata).
            Returns empty list if the store has no documents yet.
        """
        store = self._get_store()

        # Check if the store is empty before searching
        # (ChromaDB raises an error if you search an empty collection)
        try:
            count = store._collection.count()
        except Exception:
            count = 0

        if count == 0:
            return []

        try:
            return store.similarity_search(query, k=k)
        except Exception:
            # Silently fail — long-term memory is a bonus, not a requirement
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        distance_threshold: float = 1.3,
    ) -> List[tuple]:
        """
        Find chunks with their similarity scores, filtering by relevance threshold.

        ChromaDB returns L2 distance scores for normalized embeddings.
        Lower score = more similar.

        L2 distance → cosine similarity conversion:
            d=0.55 → cos_sim=0.85 (very similar)
            d=1.0  → cos_sim=0.5  (moderately similar)
            d=1.2  → cos_sim=0.28 (somewhat related - recommended default)
            d=1.5  → cos_sim=-0.125 (looser matching)

        Args:
            query: The user's message to search for.
            k: Maximum chunks to retrieve.
            distance_threshold: Maximum L2 distance to consider relevant.
                                Default 1.2 (cosine similarity > 0.28).

        Returns:
            List of (Document, score) tuples where score <= distance_threshold.
            Empty list if no chunks pass threshold OR ChromaDB not initialized.
        """
        start_time = time.time()
        query_preview = query[:30] + "..." if len(query) > 30 else query
        logger.debug(f"similarity_search called: '{query_preview}'")

        # Step 1: Ensure ChromaDB is initialized (lazy load)
        try:
            store = self._get_store()
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}")
            return []

        # Step 2: Check if store has any documents
        try:
            count = store._collection.count()
        except Exception as e:
            logger.warning(f"ChromaDB collection access failed: {e}")
            count = 0

        if count == 0:
            logger.debug("ChromaDB collection is empty, no RAG context available")
            return []

        logger.debug(f"Searching {count} indexed chunks")

        # Step 3: Perform similarity search with scores
        try:
            search_start = time.time()
            results = store.similarity_search_with_score(query, k=k)
            search_duration = int((time.time() - search_start) * 1000)

            # Debug logging for threshold tuning
            for doc, score in results:
                source = doc.metadata.get("source", "unknown")
                logger.debug("Chunk found", extra={"source": source, "score": round(score, 3)})

            # Filter by relevance threshold
            filtered = [(doc, score) for doc, score in results if score <= distance_threshold]

            passed = len(filtered)
            skipped = len(results) - passed

            if skipped > 0:
                logger.debug(f"Threshold filtering: {passed} passed, {skipped} skipped (threshold={distance_threshold})")

            for doc, score in filtered:
                source = doc.metadata.get("source", "unknown")
                logger.debug("Relevant chunk", extra={"source": source, "score": round(score, 3)})

            total_duration = int((time.time() - start_time) * 1000)
            logger.info("similarity_search completed", extra={"duration_ms": total_duration, "results": len(filtered), "threshold": distance_threshold})

            return filtered
        except Exception as e:
            logger.warning(f"ChromaDB similarity search failed: {e}")
            return []

    def format_context(self, docs: List[Document]) -> str:
        """
        Format retrieved chunks into a single string for the LLM prompt.

        The formatted string is injected into the prompt so Gemini can
        read relevant document content before answering.

        Example output:
            [From: system-design.md]
            ## Overview
            The ecommerce platform uses a microservices architecture...

            ---

            [From: auth-api-spec.md]
            ## POST /auth/login
            Request body: { "email": "...", "password": "..." }

        Args:
            docs: List of Document objects from search().

        Returns:
            Formatted string, or empty string if no docs were retrieved.
        """
        if not docs:
            return ""

        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown document")
            parts.append(f"[From: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def has_documents(self) -> bool:
        """Return True if at least one document has been indexed."""
        try:
            return self._get_store()._collection.count() > 0
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_store(self) -> Chroma:
        """
        Lazy-load ChromaDB.

        Why lazy? Loading ChromaDB and the embedding model takes a few
        seconds. We only pay this cost the first time a document is saved
        or searched, not at agent startup.
        """
        if self._store is None:
            # If persist_dir already has data, Chroma loads it automatically.
            # If it is empty, Chroma creates a fresh collection.
            self._store = Chroma(
                collection_name="architect_docs",   # logical grouping name
                embedding_function=self._embeddings,
                persist_directory=self.persist_dir,
            )
        return self._store


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
# One shared instance used by both tools.py (for indexing) and
# architect.py (for retrieval). This ensures both modules operate
# on the same ChromaDB collection.
_vector_memory = VectorMemory()


def get_vector_memory() -> VectorMemory:
    """Return the shared VectorMemory singleton."""
    return _vector_memory
