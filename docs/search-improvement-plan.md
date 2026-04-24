# Plan: Improve Search Algorithm & Add Document Update Capability

**Created**: 2026-04-19
**Status**: Revised (AI Engineering Audit Applied)
**Scope**: `agent/vector_memory.py`, `agent/tools.py`, `agent/architect.py`

---

## Context

### Problem

The user reported two issues:
1. **Unsuitable answers**: When asking about existing documents, the agent returns irrelevant or redundant content
2. **Creates new documents instead of updating**: When asked to update a document, the agent creates a separate new document

### Root Cause Analysis

**Issue 1 - Poor Search Quality:**
- Current implementation uses `similarity_search_with_score` (pure vector similarity)
- Returns chunks that are all similar to EACH OTHER (redundant coverage)
- Uses L2 distance with threshold 1.3, but ChromaDB defaults to L2 without explicit cosine config
- No diversity consideration (MMR not used)

**Issue 2 - No Update Capability:**
- System prompt only mentions `save_document()` and `list_documents()` (line 137-141)
- No `update_document` tool exists
- No metadata filtering to search specific documents
- No logic to check if document exists before creating new one

---

## ⚠️ Critical Constraints

### ChromaDB Distance Metric Locking

**IMPORTANT**: ChromaDB **cannot change the distance metric** after a collection is created. If `chroma_db/` already exists with L2-indexed data:
- The new cosine configuration will be **ignored**
- Or ChromaDB will throw an error

**Solution**: Must clear existing database and re-index all documents when switching distance metrics.

---

## Implementation Plan

### Phase 0: Data Migration (CRITICAL - Do First)

#### 0.1 Backup and Clear Existing Database

**File**: `agent/vector_memory.py`

**Change**: Add method to safely clear and recreate the collection.

```python
def reset_collection(self, new_space: str = "cosine") -> dict:
    """
    Clear existing collection and recreate with new distance metric.
    
    WARNING: This deletes all indexed documents!
    
    Args:
        new_space: Distance metric ("cosine", "l2", "ip")
    
    Returns:
        {"success": bool, "chunks_deleted": int, "new_space": str}
    """
    store = self._get_store()
    
    # Get count before deletion
    try:
        old_count = store._collection.count()
    except Exception:
        old_count = 0
    
    # Delete the collection
    try:
        self._store = None  # Clear reference
        # ChromaDB client delete collection
        store._client.delete_collection("architect_docs")
        logger.info(f"Deleted collection 'architect_docs' ({old_count} chunks)")
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        return {"success": False, "chunks_deleted": 0, "new_space": new_space}
    
    # Recreate with new configuration
    from chromadb.api import CreateCollectionConfiguration
    self._store = Chroma(
        collection_name="architect_docs",
        embedding_function=self._embeddings,
        persist_directory=self.persist_dir,
        collection_configuration=CreateCollectionConfiguration(
            hnsw={"space": new_space}
        ),
    )
    
    logger.info(f"Created new collection with {new_space} distance")
    return {"success": True, "chunks_deleted": old_count, "new_space": new_space}
```

#### 0.2 Re-index All Documents

**File**: `agent/dispatcher.py` (or new migration script)

**Change**: Add function to re-index all existing documents.

```python
def reindex_all_documents(save_path: str) -> dict:
    """
    Re-index all .md files from output directory.
    
    Args:
        save_path: Directory containing .md files
    
    Returns:
        {"documents": int, "chunks": int, "errors": list}
    """
    import os
    from agent.vector_memory import get_vector_memory
    
    vm = get_vector_memory()
    
    if not os.path.exists(save_path):
        return {"documents": 0, "chunks": 0, "errors": ["Save path not found"]}
    
    md_files = [f for f in os.listdir(save_path) if f.endswith(".md")]
    
    total_chunks = 0
    errors = []
    
    for filename in md_files:
        filepath = os.path.join(save_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Remove header comment if present
            if content.startswith("<!--"):
                content = content.split("-->", 1)[-1].strip()
            chunks = vm.index_document(content, source_filename=filename)
            total_chunks += chunks
            logger.info(f"Indexed {filename}: {chunks} chunks")
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            logger.error(f"Failed to index {filename}: {e}")
    
    return {
        "documents": len(md_files) - len(errors),
        "chunks": total_chunks,
        "errors": errors
    }
```

---

### Phase 1: Improve Search Algorithm

#### 1.1 Configure ChromaDB with Cosine Distance

**File**: `agent/vector_memory.py`

**Change**: Configure ChromaDB with cosine distance at initialization. Also add import.

```python
# Add import at top of file (line 76):
from chromadb.api import CreateCollectionConfiguration

# Update _get_store (line 396-400):
def _get_store(self) -> Chroma:
    if self._store is None:
        self._store = Chroma(
            collection_name="architect_docs",
            embedding_function=self._embeddings,
            persist_directory=self.persist_dir,
            collection_configuration=CreateCollectionConfiguration(
                hnsw={"space": "cosine"}
            ),
        )
    return self._store
```

**Note**: This only works for NEW collections. Existing collections need Phase 0 migration.

---

#### 1.2 Add MMR Search with Score Filtering

**File**: `agent/vector_memory.py`

**Change**: Add MMR method that returns scores for token budget management.

```python
def similarity_search_mmr_with_score(
    self,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filter: dict | None = None,
) -> List[tuple[Document, float]]:
    """
    Search using MMR - balances similarity + diversity, WITH scores.
    
    WHY SCORES ARE NEEDED:
      MMR alone returns List[Document] without scores.
      We need scores for:
        1. Relevance filtering (discard chunks below threshold)
        2. Token budget management (stop when budget exhausted)
        3. Observability (log how relevant results were)
    
    Args:
        query: Search query
        k: Number of results to return
        fetch_k: Number to fetch before MMR selection
        lambda_mult: 0 = max diversity, 1 = max similarity (default 0.5)
        filter: Metadata filter (e.g., {"source": "system-design.md"})
    
    Returns:
        List of (Document, cosine_score) tuples.
        Cosine score: 0 = identical, 1 = opposite (lower = more similar)
    """
    start_time = time.time()
    logger.debug(f"MMR search: '{query[:50]}...'")
    
    try:
        store = self._get_store()
        
        # Check if collection is empty
        if store._collection.count() == 0:
            return []
        
        # Embed the query
        query_embedding = self._embeddings.embed_query(query)
        
        # Fetch candidates for MMR
        results = store.similarity_search_by_vector_with_relevance_scores(
            embedding=query_embedding,
            k=fetch_k,
            filter=filter,
        )
        
        if not results:
            return []
        
        # Apply MMR selection on the fetched candidates
        # Note: ChromaDB's max_marginal_relevance_search doesn't return scores
        # So we implement MMR selection manually and keep the scores
        
        embeddings_list = []
        docs_with_scores = []
        
        # Get embeddings for fetched documents
        for doc, score in results:
            # Retrieve the embedding for this document
            doc_results = store._collection.get(
                ids=[doc.id],
                include=["embeddings"]
            )
            if doc_results["embeddings"] and doc_results["embeddings"][0]:
                embeddings_list.append(doc_results["embeddings"][0])
                docs_with_scores.append((doc, score))
        
        if not embeddings_list:
            return []
        
        # Apply MMR algorithm
        import numpy as np
        from langchain_chroma.vectorstores import maximal_marginal_relevance
        
        mmr_indices = maximal_marginal_relevance(
            np.array(query_embedding),
            embeddings_list,
            k=min(k, len(embeddings_list)),
            lambda_mult=lambda_mult,
        )
        
        # Return selected documents with their original scores
        selected = [(docs_with_scores[i][0], docs_with_scores[i][1]) for i in mmr_indices]
        
        duration = int((time.time() - start_time) * 1000)
        logger.info(f"MMR search completed", extra={
            "duration_ms": duration,
            "results": len(selected),
            "lambda_mult": lambda_mult,
        })
        
        return selected
        
    except Exception as e:
        logger.warning(f"MMR search failed: {e}")
        return []
```

---

#### 1.3 Add Metadata Filtering Support

**File**: `agent/vector_memory.py`

**Change**: Add method to search within specific documents.

```python
def search_by_source(
    self,
    query: str,
    source_filename: str,
    k: int = 5,
) -> List[tuple[Document, float]]:
    """
    Search only within a specific document.
    
    Args:
        query: Search query
        source_filename: Filename to filter (e.g., "system-design.md")
        k: Number of results
    
    Returns:
        Chunks from only the specified document, with scores
    """
    return self.similarity_search_mmr_with_score(
        query=query,
        k=k,
        filter={"source": source_filename},
    )
```

---

#### 1.4 Add delete_by_source Method (with Error Handling)

**File**: `agent/vector_memory.py`

**Change**: Add method to delete chunks from a specific document.

```python
def delete_by_source(self, source_filename: str) -> int:
    """
    Delete all chunks from a specific document before re-indexing.
    
    Args:
        source_filename: The filename to delete (e.g., "system-design.md")
    
    Returns:
        Number of chunks deleted, or -1 on error
    """
    try:
        store = self._get_store()
        
        # Get all IDs for this source
        results = store.get(
            where={"source": source_filename},
            include=["ids"]
        )
        
        if not results["ids"]:
            logger.debug(f"No chunks found for {source_filename}")
            return 0
        
        ids_to_delete = results["ids"]
        
        # Delete by IDs
        store.delete(ids=ids_to_delete)
        
        logger.info(f"Deleted {len(ids_to_delete)} chunks for {source_filename}")
        return len(ids_to_delete)
        
    except Exception as e:
        logger.error(f"Failed to delete chunks for {source_filename}: {e}")
        return -1
```

---

#### 1.5 Update RAG Retrieval to Use MMR with Scores

**File**: `agent/architect.py`

**Change**: Modify `_build_augmented_message()` to use MMR with score filtering and token budget.

```python
def _build_augmented_message(self, user_message: str) -> str:
    """
    Three-Pass Filtering: Search + Relevance + Budget.
    
    Pass 1: MMR search (diverse candidates)
    Pass 2: Filter by cosine score threshold
    Pass 3: Add chunks until token budget exhausted
    
    Args:
        user_message: The original user input.
    
    Returns:
        Augmented message with relevant context, or original if nothing found.
    """
    if not self._vector_memory.has_documents():
        return user_message
    
    # Pass 1: MMR search with scores
    mmr_start = time.time()
    results = self._vector_memory.similarity_search_mmr_with_score(
        query=user_message,
        k=10,
        fetch_k=30,  # Fetch more candidates for MMR to choose from
        lambda_mult=0.5,  # Balanced similarity/diversity
    )
    mmr_duration = int((time.time() - mmr_start) * 1000)
    
    if not results:
        logger.debug("No MMR results found")
        return user_message
    
    # Pass 2: Relevance filtering (cosine score threshold)
    # Cosine distance: 0 = identical, 1 = opposite
    # Lower score = more similar
    COSINE_THRESHOLD = 0.7  # Keep chunks with score < 0.7 (reasonably similar)
    
    relevant_chunks = []
    for doc, score in results:
        if score < COSINE_THRESHOLD:
            source = doc.metadata.get("source", "unknown")
            logger.debug(f"Relevant chunk: {source} (score={score:.3f})")
            relevant_chunks.append((doc, score))
        else:
            logger.debug(f"Filtered out: score {score:.3f} >= threshold {COSINE_THRESHOLD}")
    
    if not relevant_chunks:
        logger.debug("No chunks passed relevance threshold")
        return user_message
    
    logger.info(f"MMR + relevance filter: {len(relevant_chunks)}/{len(results)} chunks passed")
    
    # Pass 3: Token budget filtering
    MAX_RAG_TOKENS = 2500
    SAFETY_MARGIN = 0.10
    EFFECTIVE_LIMIT = int(MAX_RAG_TOKENS * (1 - SAFETY_MARGIN))  # 2250 tokens
    
    selected_chunks = []
    total_tokens = 0
    
    def count_tokens(text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text) // 4
        except Exception:
            return len(text) // 4
    
    for doc, score in relevant_chunks:
        chunk_tokens = count_tokens(doc.page_content)
        
        if total_tokens + chunk_tokens > EFFECTIVE_LIMIT:
            logger.debug(f"Token budget exhausted at {total_tokens} tokens")
            break
        
        selected_chunks.append(doc)
        total_tokens += chunk_tokens
    
    if not selected_chunks:
        logger.debug("No chunks fit within token budget")
        return user_message
    
    # Format and return
    context_str = self._vector_memory.format_context(selected_chunks)
    
    augmented = (
        f"RELEVANT CONTEXT ({total_tokens} tokens):\n"
        f"{context_str}\n\n"
        f"USER REQUEST:\n{user_message}"
    )
    
    total_duration = int((time.time() - mmr_start) * 1000)
    logger.info(f"RAG completed", extra={
        "duration_ms": total_duration,
        "mmr_duration_ms": mmr_duration,
        "chunks_selected": len(selected_chunks),
        "chunks_rejected_by_score": len(results) - len(relevant_chunks),
        "chunks_rejected_by_budget": len(relevant_chunks) - len(selected_chunks),
        "tokens": total_tokens,
    })
    
    return augmented
```

---

### Phase 2: Add Document Update Capability

#### 2.1 Add `update_document` Tool (with Error Handling)

**File**: `agent/tools.py`

**Change**: Add new tool that updates existing documents.

```python
@tool
def update_document(filename: str, content: str) -> str:
    """
    Update an existing markdown document.
    
    Use this when the user asks to UPDATE, MODIFY, or REVISE a document
    that already exists. 
    
    IMPORTANT: Only call this for documents that already exist.
    For new documents, use save_document instead.

    Args:
        filename: The existing file name (e.g., 'system-design.md')
        content: The new/updated markdown content (full document)

    Returns:
        Confirmation with file path and re-index status
    """
    global _save_path
    start_time = time.time()
    
    logger.info(f"update_document called", extra={"filename": filename})
    
    # 1. Check if file exists
    full_path = os.path.join(_save_path, filename)
    if not os.path.exists(full_path):
        logger.warning(f"File not found: {filename}")
        return f"❌ File not found: {filename}. Use save_document to create new files."
    
    # 2. Write updated content
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"<!-- Updated by Software Architect Agent | {timestamp} -->\n\n"
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(header + content)
        logger.info(f"File written: {filename}")
    except Exception as e:
        logger.error(f"Failed to write file: {e}")
        return f"❌ Failed to write file: {str(e)}"
    
    # 3. Re-index in vector store (delete old chunks, add new)
    try:
        vm = get_vector_memory()
        deleted = vm.delete_by_source(filename)
        if deleted < 0:
            logger.warning(f"Delete failed for {filename}, proceeding with index")
        chunks = vm.index_document(content, source_filename=filename)
        logger.info(f"Re-indexed {filename}: {chunks} chunks", extra={
            "chunks_deleted": deleted,
            "chunks_added": chunks,
        })
        index_note = f" | Re-indexed ({deleted} old → {chunks} new chunks)"
    except Exception as e:
        logger.error(f"Re-indexing failed: {e}")
        index_note = f" | ⚠️ Re-indexing failed: {str(e)}"
    
    total_duration = int((time.time() - start_time) * 1000)
    logger.info(f"update_document completed", extra={
        "filename": filename,
        "duration_ms": total_duration,
    })
    
    return f"✅ Document updated: {os.path.abspath(full_path)}{index_note}"
```

---

#### 2.2 Add `get_existing_documents` Helper

**File**: `agent/tools.py`

**Change**: Add helper to check existing documents.

```python
def get_existing_documents() -> list[str]:
    """
    Return list of existing .md filenames in save_path.
    
    Used by list_documents tool and for checking if a document exists.
    """
    global _save_path
    if not os.path.exists(_save_path):
        return []
    return sorted([f for f in os.listdir(_save_path) if f.endswith(".md")])
```

---

#### 2.3 Update System Prompt

**File**: `agent/architect.py`

**Change**: Update TOOL USAGE RULES section (replace lines 137-148).

```python
TOOL USAGE RULES:
- When the user asks to CREATE, GENERATE, WRITE, or DESIGN a document → call save_document()
- When the user asks to UPDATE, MODIFY, REVISE, or CHANGE an existing document → call update_document()
- When the user asks to LIST, SHOW, or VIEW saved documents → call list_documents()
- ALWAYS call list_documents() first to check existing documents before creating a new one
- This helps you determine whether to save_document (new) or update_document (existing)
- Always confirm to the user after saving/updating (the tool returns the file path)
- Pick a descriptive, hyphenated filename (e.g., 'ecommerce-system-design', 'auth-api-spec')

DOCUMENT UPDATE BEHAVIOR:
- When updating, include the FULL updated document content (not just the new section)
- Preserve the document's overall structure unless explicitly asked to restructure
- The update_document tool will re-index the document automatically
```

---

#### 2.4 Bind update_document Tool to Agent

**File**: `agent/architect.py`

**Change**: Add update_document to tools list (line 182-186).

```python
from .tools import save_document, list_documents, update_document

# In __init__:
self._tools = [save_document, list_documents, update_document]
self._llm_with_tools = self.llm.bind_tools(self._tools)
self._tool_map = {t.name: t for t in self._tools}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `agent/vector_memory.py` | Add import, cosine config, MMR with scores, search_by_source, delete_by_source, reset_collection |
| `agent/tools.py` | Add update_document tool, get_existing_documents helper |
| `agent/architect.py` | Add import, bind tool, update system prompt, rewrite _build_augmented_message |
| `agent/dispatcher.py` | Add reindex_all_documents function |

---

## Verification Plan

### 1. Test Migration (Phase 0)

```bash
# Backup existing data
cp -r chroma_db chroma_db_backup

# Run migration
python -c "from agent.vector_memory import get_vector_memory; vm = get_vector_memory(); print(vm.reset_collection('cosine'))"

# Re-index
python -c "from agent.dispatcher import reindex_all_documents; print(reindex_all_documents('./output'))"

# Verify: Check chroma_db was recreated with cosine distance
```

### 2. Test Search Improvements

```bash
python app.py

# Test 1: MMR diversity
User: "Tell me about inventory management"
Expected: Returns chunks from BOTH documents (not all from one)

# Test 2: Score filtering
User: "Explain quantum physics"  # Irrelevant query
Expected: "No relevant context" (score threshold filters out irrelevant chunks)

# Test 3: Document-specific search
User: "Show me the database schema from inventory-management-db-schema.md"
Expected: Only chunks from that specific file
```

### 3. Test Document Updates

```bash
# Test 1: Update existing document
User: "Update inventory-management-architecture.md to add a caching section"
Expected: 
- Calls list_documents first
- Detects file exists
- Calls update_document
- Returns "✅ Document updated" with re-index status

# Test 2: Create vs Update distinction
User: "Create a new document about payment processing"
Expected: Calls save_document (not update_document)

# Test 3: Update non-existent document
User: "Update non-existent-file.md"
Expected: "❌ File not found. Use save_document to create new files."
```

---

## Implementation Order

1. **Phase 0**: Migration (backup, reset collection, re-index)
2. **Phase 1.1**: Cosine config (only works after Phase 0)
3. **Phase 1.2**: MMR with scores (critical for token budget)
4. **Phase 1.3-1.4**: Metadata filtering and delete_by_source
5. **Phase 1.5**: Update _build_augmented_message with three-pass filtering
6. **Phase 2.1-2.4**: update_document tool + system prompt + tool binding

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `chromadb` | >= 1.0.0 (current: 1.5.2) | Vector database with cosine distance support |
| `langchain-chroma` | >= 1.0.0 (current: 1.1.0) | ChromaDB integration with MMR |
| `tiktoken` | >= 0.5.0 | Token counting for budget management |
| `numpy` | (implicit via chromadb) | MMR vector operations |

---

## Observability Metrics

After implementation, these metrics will be logged:

| Metric | Description | Where |
|--------|-------------|-------|
| `mmr_duration_ms` | Time for MMR search | vector_memory.py |
| `chunks_selected` | Chunks that passed all filters | architect.py |
| `chunks_rejected_by_score` | Filtered by cosine threshold | architect.py |
| `chunks_rejected_by_budget` | Filtered by token limit | architect.py |
| `tokens` | Total RAG context tokens | architect.py |

---

## Implementation Progress

| Phase | Task | Status |
|-------|------|--------|
| 0.1 | Add reset_collection method | ✅ Done |
| 0.2 | Add reindex_all_documents function | ✅ Done |
| 1.1 | Configure ChromaDB with cosine distance | ✅ Done |
| 1.2 | Add similarity_search_mmr_with_score method | ✅ Done |
| 1.3 | Add search_by_source method | ✅ Done |
| 1.4 | Add delete_by_source method | ✅ Done |
| 1.5 | Update _build_augmented_message with three-pass | ✅ Done |
| 2.1 | Add update_document tool | ✅ Done |
| 2.2 | Add get_existing_documents helper | ✅ Done |
| 2.3 | Update system prompt | ✅ Done |
| 2.4 | Bind update_document to agent | ✅ Done |

**Legend**: ✅ Done | 🔄 In Progress | ⬜ Pending | ❌ Blocked

---

## Implementation Complete ✅

All phases have been implemented. The changes include:

1. **Phase 0 - Migration**: Added `reset_collection()` and `reindex_all_documents()` for switching from L2 to cosine distance
2. **Phase 1 - Search**: Implemented MMR search with scores, metadata filtering, and three-pass RAG filtering
3. **Phase 2 - Update**: Added `update_document` tool with proper error handling and re-indexing

### Next Steps

To apply the migration to existing data:

```bash
# Run migration (clears existing database, switches to cosine distance)
python -c "from agent.dispatcher import reset_vector_memory; print(reset_vector_memory())"

# Re-index existing documents
python -c "from agent.dispatcher import reindex_all_documents; print(reindex_all_documents())"
```

---

## Changelog

### 2026-04-21 - Semantic Search Quality Improvements

**Critical findings from production testing:**

After migration and implementation, semantic search still returned weak matches for queries like "JWT strategy". Analysis revealed five root causes:

#### Root Cause 1: Chunk Size Too Large

| Current Config | Problem |
|----------------|---------|
| `chunk_size=500` tokens | JWT mentions buried in 2453-char chunks with unrelated content |
| Only 2 chunks total | Semantic signal diluted by surrounding text |

**Test Results:**
| Chunk Size | Total Chunks | JWT-Containing | Isolation Quality |
|------------|--------------|----------------|-------------------|
| 500 | 2 | 2 | Poor (JWT mixed with overview) |
| 200 | 5 | 2 | Moderate |
| 100 | 13 | 4 | **Good (JWT in dedicated chunks)** |

**Recommended Fix:** Reduce chunk size to 200-300 tokens.

```python
# In vector_memory.py __init__
self._splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=200,      # Smaller = better semantic precision
    chunk_overlap=50,    # Overlap ensures context continuity
)
```

---

#### Root Cause 2: Query-Document Semantic Gap

**Test Results:**
| Query | Distance to JWT Chunk | Similarity |
|-------|----------------------|------------|
| "JWT strategy" | 0.608 | 39% (weak) |
| "JWT authentication strategy" | 0.357 | 64% (good) |
| "authentication service JWT tokens" | 0.268 | 73% (strong) |

**Problem:** User queries are short/vague; document uses technical language. all-MiniLM-L6-v2 is a general-purpose model, not domain-specific.

**Recommended Fix:** Implement query expansion before search.

```python
# In architect.py - add helper method

def _expand_query(query: str) -> str:
    """Expand short queries with domain-relevant terms."""
    expansion_map = {
        "jwt": "JWT authentication token security",
        "strategy": "strategy architecture design approach",
        "api": "API endpoint service gateway",
        "database": "database schema storage persistence",
        "security": "security authentication authorization",
    }
    
    expanded = query
    for key, expansion in expansion_map.items():
        if key.lower() in query.lower():
            expanded = f"{query} {expansion}"
    
    return expanded

# In _build_augmented_message():
expanded_query = _expand_query(user_message)
results = self._vector_memory.similarity_search_mmr_with_score(
    query=expanded_query,  # Use expanded query
    k=10,
    fetch_k=30,
    lambda_mult=0.5,
)
```

---

#### Root Cause 3: Distance vs Similarity Threshold Confusion

**Current code (works but confusing):**
```python
COSINE_THRESHOLD = 0.85
if score < COSINE_THRESHOLD:  # score is DISTANCE, not similarity
```

**Problem:**
- ChromaDB returns **distance** scores (0 = identical, 1 = opposite)
- Variable named `COSINE_THRESHOLD` suggests similarity, but it's actually distance
- Comments incorrectly say "scores 0.4-0.8 for similar content"
- Current threshold 0.85 distance = 15% similarity minimum

**Test verification:**
| Content Pair | Distance Score | Similarity (1 - distance) |
|--------------|----------------|---------------------------|
| Exact match "JWT authentication" | 0.07 | 93% similar |
| Similar content "JWT strategy" → JWT chunk | 0.695 | 30.5% similar |
| Dissimilar "pizza recipe" | 0.93 | 7% similar |

**Observed JWT query scores:**
| Distance | Similarity | Passes 0.85 threshold? |
|----------|------------|------------------------|
| 0.695 | 30.5% | ✓ PASS |
| 0.775 | 22.5% | ✓ PASS |

**Recommended Fix:** Keep distance threshold (it works), but rename for clarity:

```python
# In architect.py _build_augmented_message()

# ChromaDB returns COSINE DISTANCE: 0 = identical, 1 = opposite
# Lower score = more similar
# Distance threshold 0.85 = keep chunks with >15% similarity
# This is lenient but works for general-purpose embedding model
DISTANCE_THRESHOLD = 0.85

relevant_chunks = []
for doc, distance in results:
    if distance < DISTANCE_THRESHOLD:
        similarity = 1 - distance  # For logging/display
        source = doc.metadata.get("source", "unknown")
        logger.debug(f"Relevant chunk: {source} (similarity={similarity:.1%})")
        relevant_chunks.append((doc, distance))
    else:
        logger.debug(f"Filtered out: distance {distance:.3f} >= threshold {DISTANCE_THRESHOLD}")
```

**Alternative (similarity-based logic):**
```python
# Convert to similarity for intuitive thresholds
# similarity = 1 - distance
# 0.15 similarity = equivalent to 0.85 distance
SIMILARITY_THRESHOLD = 0.15  # Keep chunks with >15% similarity

for doc, distance in results:
    similarity = 1 - distance
    if similarity >= SIMILARITY_THRESHOLD:
        relevant_chunks.append((doc, distance))
```

**Why NOT use 0.30 similarity threshold:**

| Threshold | Equivalent Distance | Behavior |
|-----------|---------------------|----------|
| similarity >= 0.30 | distance < 0.70 | **Too strict** - JWT chunks filtered out |
| similarity >= 0.15 | distance < 0.85 | **Correct** - JWT chunks pass ✓ |

---

#### Root Cause 4: No Keyword Matching Boost

**Problem:** all-MiniLM-L6-v2 misses exact keyword matches when semantic context differs. Query "JWT strategy" returns distance 0.695 even when chunk contains "JWT".

**Recommended Fix:** Implement hybrid search with keyword boost.

```python
# In vector_memory.py - add new method

def keyword_boosted_search(
    self,
    query: str,
    keywords: list[str] | None = None,
    k: int = 5,
    boost_factor: float = 0.15,
) -> List[tuple]:
    """
    Semantic search with keyword presence boost.
    
    Args:
        query: Search query
        keywords: Keywords to boost (extracted from query if None)
        k: Number of results
        boost_factor: Distance reduction per keyword match (0.15 = ~15% boost)
    
    Returns:
        List of (Document, adjusted_distance) tuples
    """
    # Auto-extract keywords from query if not provided
    if keywords is None:
        # Extract significant words (skip common words)
        common_words = {"what", "is", "the", "a", "an", "how", "why", "when", "where"}
        keywords = [
            word.lower() 
            for word in query.split() 
            if word.lower() not in common_words and len(word) > 2
        ]
    
    results = self.similarity_search_mmr_with_score(query, k=k*2)
    
    boosted_results = []
    for doc, distance in results:
        # Count keyword matches in document
        keyword_match_count = sum(
            1 for kw in keywords 
            if kw.lower() in doc.page_content.lower()
        )
        
        # Apply boost: reduce distance for keyword matches
        # More matches = lower distance = higher relevance
        adjusted_distance = max(0, distance - (boost_factor * keyword_match_count))
        
        boosted_results.append((doc, adjusted_distance, distance))
    
    # Sort by adjusted distance and return top k
    boosted_results.sort(key=lambda x: x[1])
    
    # Log boosting effect
    for doc, adj_dist, orig_dist in boosted_results[:k]:
        boost_applied = orig_dist - adj_dist
        if boost_applied > 0:
            logger.debug(f"Keyword boost: {orig_dist:.3f} → {adj_dist:.3f}")
    
    return [(doc, adj_dist) for doc, adj_dist, _ in boosted_results[:k]]
```

---

#### Root Cause 5: MMR Not Effective on Small Collections

**Current config:**
```python
lambda_mult=0.5  # Balanced
fetch_k=30       # Fixed, not adaptive
```

**Problem:** With only 2 chunks, MMR can't function properly. MMR needs diverse candidates to select from.

**Recommended Fix:** Adaptive fetch_k based on collection size.

```python
# In architect.py _build_augmented_message()

# Adaptive fetch_k: ensure enough candidates for MMR
collection_count = self._vector_memory._get_store()._collection.count()
fetch_k = min(30, max(10, collection_count * 2))  # At least 2x collection size
k = min(10, max(3, collection_count // 2))        # Request reasonable portion

results = self._vector_memory.similarity_search_mmr_with_score(
    query=expanded_query,
    k=k,
    fetch_k=fetch_k,
    lambda_mult=0.7,  # Favor similarity more when collection is small
)
```

---

### Implementation Priority for Semantic Fixes

| Priority | Fix | Impact | Effort | Status |
|----------|-----|--------|--------|--------|
| **1** | Use similarity threshold (0.30) | High | Low | ⬜ Pending |
| **2** | Reduce chunk size to 200 | High | Low | ⬜ Pending |
| **3** | Query expansion | Medium | Medium | ⬜ Pending |
| **4** | Keyword boost | Medium | Medium | ⬜ Pending |
| **5** | Adaptive fetch_k | Low | Low | ⬜ Pending |

---

### Implementation Priority for Semantic Fixes (Updated 2026-04-21)

| Priority | Fix | Impact | Effort | Status |
|----------|-----|--------|--------|--------|
| **1** | Use correct threshold (keep 0.85 distance) | High | Low | ✅ Done |
| **2** | Reduce chunk size to 200 | High | Low | ⬜ Pending |
| **3** | Query expansion | Medium | Medium | ⬜ Pending |
| **4** | Keyword boost | Medium | Medium | ⬜ Pending |
| **5** | Adaptive fetch_k | Low | Low | ⬜ Pending |
| **6** | Metadata-enhanced chunking | High | Medium | ⬜ Pending |
| **7** | Semantic chunking (structure-aware) | High | Low | ⬜ Pending |
| **8** | Hierarchical retrieval (parent-child) | Medium | Medium | ⬜ Pending |
| **9** | GraphRAG (knowledge graph) | Low (deferred) | High | ❌ Deferred |

**Detailed Analysis:**
- [metadata-enhanced-retrieval.md](metadata-enhanced-retrieval.md) - Metadata strategy
- [advanced-retrieval-feasibility.md](advanced-retrieval-feasibility.md) - Semantic chunking, hierarchical retrieval, GraphRAG feasibility

---

### 2026-04-19 - AI Engineering Audit Applied

**Critical fixes:**
- Added Phase 0 for distance metric migration (ChromaDB cannot change metric after creation)
- Changed MMR to return scores for token budget management
- Removed intent detection (Phase 3.1) - rely on LLM intelligence + list_documents check

**Medium fixes:**
- Added comprehensive error handling to all ChromaDB operations
- Added observability metrics for search quality measurement
- Added logging throughout for debugging

**Minor fixes:**
- Documented lambda_mult rationale
- Noted hybrid search as future consideration