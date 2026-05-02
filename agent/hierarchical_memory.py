"""
agent/hierarchical_memory.py — Two-Tier Hierarchical Vector Memory

WHY THIS EXISTS:
  Standard retrieval returns small chunks (100-300 tokens) which may be
  fragmented. Hierarchical retrieval:
    1. Searches small "child" chunks (precise embedding match)
    2. Fetches large "parent" chunks (full context)
    3. Returns complete sections to the LLM

STRUCTURE:
  - Parent collection: Large chunks (full sections, ~500 tokens)
  - Child collection: Small chunks (sentences, ~100 tokens)
  - Link: Each child has parent_id metadata pointing to its parent

RETRIEVAL FLOW:
  Query: "JWT tokens"
    → Search children (precise): finds "issues JWT tokens..."
    → Lookup parent (full context): returns full Authentication section
    → LLM receives complete context, not fragmented sentence

Created: 2026-04-25 (Phase 2 from langgraph-master-plan.md)
"""

import os
import time
import logging
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from chromadb.api import CreateCollectionConfiguration
# Import from vector_memory for shared constants
from .vector_memory import DEFAULT_CHROMA_DIR, VectorMemory

logger = logging.getLogger(__name__)


class HierarchicalVectorMemory:
    """
    Two-tier vector storage for complete context retrieval.

    Usage:
        hvm = HierarchicalVectorMemory()
        hvm.index_document_hierarchical(content, source)
        results = hvm.hierarchical_search(query, k=5)
    """

    def __init__(self, persist_dir: str = DEFAULT_CHROMA_DIR):
        """
        Initialize two-tier storage with separate collections.

        Args:
            persist_dir: Folder for ChromaDB storage (same as VectorMemory)
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # Shared embeddings (same model as VectorMemory)
        self._embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Two separate ChromaDB collections
        # Parents: Large chunks (~500 tokens)
        self._parent_store: Chroma | None = None

        # Children: Small chunks (~100 tokens)
        self._child_store: Chroma | None = None

        # Splitters
        # Parent splitter: Markdown sections
        self._parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('##', 'section'),
                ('###', 'subsection'),
            ],
            strip_headers=False,
        )

        # Child splitter: Token-based for small chunks
        self._child_splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=100,
            chunk_overlap=20,
        )

        logger.info("HierarchicalVectorMemory initialized")

    def index_document_hierarchical(
        self,
        content: str,
        source_filename: str,
    ) -> dict:
        """
        Index document with parent-child structure.

        Args:
            content: Raw markdown text
            source_filename: Filename for metadata

        Returns:
            {"parents": int, "children": int, "parent_ids": list}
        """
        start_time = time.time()
        logger.info(f"index_document_hierarchical called: {source_filename}")

        try:
            # Step 1: Split into parents (sections)
            parent_chunks = self._parent_splitter.split_text(content)

            parent_ids = []
            total_children = 0

            # Step 2: For each parent, split into children and store
            for i, parent in enumerate(parent_chunks):
                # Generate unique parent ID
                section_name = parent.metadata.get('section', 'unknown')
                parent_id = f"{source_filename}:{section_name}:{i}"
                parent_ids.append(parent_id)

                # Store parent with metadata
                parent_meta = {
                    'id': parent_id,
                    'source': source_filename,
                    'section': section_name,
                    'level': 'parent',
                    'token_count': self._count_tokens(parent.page_content),
                }

                parent_store = self._get_parent_store()
                parent_store.add_documents(
                    [Document(page_content=parent.page_content, metadata=parent_meta)]
                )

                # Split parent into children
                child_texts = self._child_splitter.split_text(parent.page_content)

                # Store children with parent_id linkage
                child_docs = []
                child_metas = []
                for j, child_text in enumerate(child_texts):
                    child_docs.append(Document(page_content=child_text))
                    child_metas.append({
                        'parent_id': parent_id,
                        'source': source_filename,
                        'section': section_name,
                        'child_index': j,
                        'level': 'child',
                    })

                if child_docs:
                    child_store = self._get_child_store()
                    child_store.add_documents(child_docs, metadatas=child_metas)
                    total_children += len(child_docs)

            duration = int((time.time() - start_time) * 1000)
            logger.info("Hierarchical indexing completed", extra={
                "parents": len(parent_chunks),
                "children": total_children,
                "duration_ms": duration,
            })

            return {
                "parents": len(parent_chunks),
                "children": total_children,
                "parent_ids": parent_ids,
            }

        except Exception as e:
            logger.error(f"Hierarchical indexing failed: {e}")
            return {"parents": 0, "children": 0, "parent_ids": []}

    def hierarchical_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """
        Search children → merge to parents (returns full context).

        Args:
            query: Search query
            k: Number of parent documents to return

        Returns:
            Parent documents (full context), not child fragments
        """
        start_time = time.time()
        logger.debug(f"Hierarchical search: '{query[:50]}...'")

        try:
            # Step 1: Search child chunks (precise)
            child_store = self._get_child_store()

            if child_store._collection.count() == 0:
                return []

            child_results = child_store.similarity_search_with_score(
                query=query,
                k=k * 2,  # Fetch more to dedupe parents
            )

            if not child_results:
                return []

            # Step 2: Get unique parent IDs from child results
            parent_ids = []
            seen = set()
            for doc, score in child_results:
                pid = doc.metadata.get('parent_id')
                if pid and pid not in seen:
                    parent_ids.append(pid)
                    seen.add(pid)

            # Step 3: Fetch parent documents
            parent_docs = []
            parent_store = self._get_parent_store()

            for pid in parent_ids[:k]:
                results = parent_store.get(where={'id': pid})
                if results['documents']:
                    # Reconstruct Document
                    parent_docs.append(Document(
                        page_content=results['documents'][0],
                        metadata=results['metadatas'][0] if results['metadatas'] else {},
                    ))

            duration = int((time.time() - start_time) * 1000)
            logger.info("Hierarchical search completed", extra={
                "results": len(parent_docs),
                "duration_ms": duration,
            })

            return parent_docs

        except Exception as e:
            logger.warning(f"Hierarchical search failed: {e}")
            return []

    def has_documents(self) -> bool:
        """Return True if at least one parent document has been indexed."""
        try:
            return self._get_parent_store()._collection.count() > 0
        except Exception:
            return False

    def delete_by_source(self, source_filename: str) -> int:
        """
        Delete all chunks for a specific document from both collections.

        Args:
            source_filename: The filename to delete

        Returns:
            Total number of chunks deleted
        """
        start_time = time.time()
        logger.info(f"delete_by_source called: {source_filename}")

        try:
            parent_store = self._get_parent_store()
            child_store = self._get_child_store()

            # Delete from parents
            parent_results = parent_store.get(where={"source": source_filename})
            parent_ids = parent_results.get("ids", [])
            if parent_ids:
                parent_store.delete(ids=parent_ids)

            # Delete from children
            child_results = child_store.get(where={"source": source_filename})
            child_ids = child_results.get("ids", [])
            if child_ids:
                child_store.delete(ids=child_ids)

            total_deleted = len(parent_ids) + len(child_ids)
            duration = int((time.time() - start_time) * 1000)

            logger.info(f"Deleted {total_deleted} chunks for {source_filename}", extra={
                "parents_deleted": len(parent_ids),
                "children_deleted": len(child_ids),
                "duration_ms": duration,
            })

            return total_deleted

        except Exception as e:
            logger.error(f"Failed to delete chunks for {source_filename}: {e}")
            return -1

    def _get_parent_store(self) -> Chroma:
        """Lazy-load parent collection."""
        if self._parent_store is None:
            self._parent_store = Chroma(
                collection_name="hierarchical_parents",
                embedding_function=self._embeddings,
                persist_directory=self.persist_dir,
                collection_configuration=CreateCollectionConfiguration(
                    hnsw={"space": "cosine"}
                ),
            )
        return self._parent_store

    def _get_child_store(self) -> Chroma:
        """Lazy-load child collection."""
        if self._child_store is None:
            self._child_store = Chroma(
                collection_name="hierarchical_children",
                embedding_function=self._embeddings,
                persist_directory=self.persist_dir,
                collection_configuration=CreateCollectionConfiguration(
                    hnsw={"space": "cosine"}
                ),
            )
        return self._child_store

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text) // 4


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_hierarchical_memory = None


def get_hierarchical_memory() -> HierarchicalVectorMemory:
    """Return the shared HierarchicalVectorMemory singleton."""
    global _hierarchical_memory
    if _hierarchical_memory is None:
        _hierarchical_memory = HierarchicalVectorMemory()
    return _hierarchical_memory


def get_retriever(mode: str = "standard") -> "VectorMemory | HierarchicalVectorMemory":
    """
    Factory for retrieval strategy selection.

    NEW (Phase 2): Allows switching between standard and hierarchical retrieval.

    Args:
        mode: "standard" (MMR) or "hierarchical" (parent-child)

    Returns:
        Appropriate retriever instance
    """
    if mode == "hierarchical":
        return get_hierarchical_memory()
    # Import here to avoid circular dependency
    from .vector_memory import get_vector_memory
    return get_vector_memory()