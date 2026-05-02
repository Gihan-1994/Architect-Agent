# LangGraph Integration Master Plan

**Created**: 2026-04-25
**Status**: Implementation In Progress
**Scope**: Comprehensive implementation plan integrating LangGraph Self-RAG, HITL, and all planned retrieval features

---

## Implementation Progress Tracker

**Last Updated**: 2026-04-27

### Todo List Summary

| #  | Phase   | Task                                                       | Status     |
|----|---------|------------------------------------------------------------|------------|
| 1  | Phase 1 | Implement SemanticChunkingStrategy in vector_memory.py     | ✅ done    |
| 2  | Phase 1 | Add metadata enrichment to index_document()                | ✅ done    |
| 3  | Phase 1 | Add metadata filtering methods to vector_memory.py         | ✅ done    |
| 4  | Phase 2 | Implement HierarchicalVectorMemory class                   | ✅ done    |
| 5  | Phase 2 | Add index_document_hierarchical() method                   | ✅ done    |
| 6  | Phase 2 | Add hierarchical_search() method                           | ✅ done    |
| 7  | Phase 3 | Add langgraph to requirements.txt                          | ✅ done    |
| 8  | Phase 3 | Create langgraph_architect.py with all nodes               | ✅ done    |
| 9  | Phase 3 | Implement ArchitectState with all stress test fields       | ✅ done    |
| 10 | Phase 3 | Implement _retrieve_node with error handling               | ✅ done    |
| 11 | Phase 3 | Implement _grade_node with error handling                  | ✅ done    |
| 12 | Phase 3 | Implement _agent_node with token budget                    | ✅ done    |
| 13 | Phase 3 | Implement _approval_node with timeout, None handling       | ✅ done    |
| 14 | Phase 3 | Implement _execute_node with multiple tool support         | ✅ done    |
| 15 | Phase 3 | Implement enable_grading/enable_rewrite configuration      | ✅ done    |
| 16 | Phase 4 | Update dispatcher.py for LangGraph                         | ✅ done    |
| 17 | Phase 4 | Update main.py for Terminal HITL                           | ✅ done    |
| 18 | Phase 4 | Update app.py for Web UI HITL                              | 🔄 active  |
| 19 | Phase 4 | Write 8 stress test cases                                  | ⏳ next    |

---

### Phase 1: Vector Memory Enhancement

| Task | Status | File | Description |
|------|--------|------|-------------|
| SemanticChunkingStrategy class | completed | vector_memory.py | Two-stage semantic chunking |
| Metadata enrichment | completed | vector_memory.py | Add 8-field metadata to chunks |
| search_by_section() | completed | vector_memory.py | Filter by section metadata |
| search_with_chunk_type_preference() | completed | vector_memory.py | Filter by chunk type |
| _classify_chunk_type() | completed | vector_memory.py | Classify prose/code/table/diagram |
| _count_tokens() | completed | vector_memory.py | Centralized token counting |
| get_full_document_by_source() | completed | vector_memory.py | Full document fallback |

### Phase 2: Hierarchical Retrieval

| Task | Status | File | Description |
|------|--------|------|-------------|
| HierarchicalVectorMemory class | completed | hierarchical_memory.py (NEW) | Two-tier parent-child storage |
| index_document_hierarchical() | completed | hierarchical_memory.py | Index with parent-child structure |
| hierarchical_search() | completed | hierarchical_memory.py | Search children → fetch parent |
| Retrieval factory | completed | hierarchical_memory.py | get_retriever(mode) selection |
| Module export | completed | __init__.py | Export new functions |

### Phase 3: LangGraph Self-RAG

| Task | Status | File | Description |
|------|--------|------|-------------|
| Add langgraph dependency | completed | requirements.txt | langgraph>=0.3.0 |
| Create langgraph_architect.py | in_progress | NEW file | LangGraph orchestration layer |
| ArchitectState TypedDict | pending | langgraph_architect.py | State with stress test fields |
| _retrieve_node | pending | langgraph_architect.py | Error handling + full doc fallback |
| _grade_node | pending | langgraph_architect.py | Error handling + fail-safe |
| _rewrite_node | pending | langgraph_architect.py | Query reformulation |
| _agent_node | pending | langgraph_architect.py | Token budget + multiple tools + pruning |
| _approval_node | pending | langgraph_architect.py | Timeout + None handling + Command |
| _execute_node | pending | langgraph_architect.py | Multiple tools iteration |
| Self-RAG configuration | pending | langgraph_architect.py | enable_grading/enable_rewrite params |
| SqliteSaver checkpointer | pending | langgraph_architect.py | Production persistence |

### Phase 4: Dispatcher + UI Integration

| Task | Status | File | Description |
|------|--------|------|-------------|
| Update dispatcher.py | pending | dispatcher.py | LangGraph invocation + interrupt |
| Terminal HITL (main.py) | pending | main.py | Blocking approval prompt |
| Web UI HITL (app.py) | pending | app.py | Non-blocking approval modal |
| Thread ID uniqueness | pending | dispatcher.py | Terminal vs Web unique IDs |
| Test cases (8 scenarios) | pending | tests/ | Stress test validation |

---

## Executive Summary

This plan integrates findings from all four planning documents to create a unified implementation roadmap:

| Source Document | Key Features | Integration Point |
|-----------------|--------------|-------------------|
| [search-improvement-plan.md](search-improvement-plan.md) | MMR search, cosine distance, update_document | ✅ Already implemented - foundation |
| [metadata-enhanced-retrieval.md](metadata-enhanced-retrieval.md) | 8-field metadata schema, metadata filtering | Phase 1 - enhance `index_document()` |
| [advanced-retrieval-feasibility.md](advanced-retrieval-feasibility.md) | Semantic chunking, hierarchical retrieval | Phase 1-2 - vector_memory.py |
| [langgraph-integration-feasibility.md](langgraph-integration-feasibility.md) | Self-RAG, HITL, state management | Phase 3-4 - orchestration layer |

**Total Effort**: 7-10 days across 4 phases

---

## Architecture Overview

### Layer Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4: ORCHESTRATION (LangGraph) - NEW                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ StateGraph                                                   │   │
│  │                                                             │   │
│  │ Nodes:                                                      │   │
│  │   • retrieve_node    → calls VectorMemory methods           │   │
│  │   • grade_node       → LLM validates relevance              │   │
│  │   • rewrite_node     → LLM reformulates query               │   │
│  │   • agent_node       → LLM with tools                       │   │
│  │   • approval_node    → interrupt() for HITL                 │   │
│  │   • execute_node     → tool execution                       │   │
│  │                                                             │   │
│  │ State: ArchitectState (TypedDict)                           │   │
│  │   • messages: Annotated[list, add_messages]                 │   │
│  │   • retrieved_chunks: list[Document]                        │   │
│  │   • query_iterations: int                                   │   │
│  │   • pending_tool: dict                                      │   │
│  │                                                             │   │
│  │ File: agent/langgraph_architect.py (NEW)                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          │ CALLS (orchestrates)                     │
│                          ▼                                          │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 3: LANGCHAIN AGENT (Current - Modified)                      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ SoftwareArchitectAgent (architect.py)                       │   │
│  │                                                             │   │
│  │ CURRENT:                                                    │   │
│  │   • Manual ReAct loop (lines 266-301) → DEAD CODE           │   │
│  │   • _build_augmented_message() → DEAD CODE                  │   │
│  │   • _count_rag_tokens() → DEAD CODE                         │   │
│  │                                                             │   │
│  │ AFTER LANGGRAPH:                                            │   │
│  │   • Wrapper for LangGraph graph                             │   │
│  │   • Model switching helper                                  │   │
│  │   • Token counting (delegates to LangGraph)                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          │ USES                                      │
│                          ▼                                          │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2: TOOLS (Current - Minimal Changes)                         │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Tools (tools.py)                                            │   │
│  │                                                             │   │
│  │   • save_document    → unchanged                            │   │
│  │   • update_document  → ADD interrupt() for HITL             │   │
│  │   • list_documents   → unchanged                            │   │
│  │                                                             │   │
│  │ HITL Integration:                                           │   │
│  │   interrupt() inside update_document for user approval      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          │ CALLS for indexing                        │
│                          ▼                                          │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 1: VECTOR MEMORY (Enhanced)                                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ VectorMemory (vector_memory.py)                             │   │
│  │                                                             │   │
│  │ CURRENT:                                                    │   │
│  │   • similarity_search_mmr_with_score() ✓                    │   │
│  │   • delete_by_source() ✓                                    │   │
│  │   • search_by_source() ✓                                    │   │
│  │   • index_document() → NEEDS semantic + metadata            │   │
│  │                                                             │   │
│  │ NEW IN PHASE 1-2:                                           │   │
│  │   • SemanticChunkingStrategy class                          │   │
│  │   • extract_section_from_markdown()                         │   │
│  │   • extract_document_title()                                │   │
│  │   • count_tokens() (centralized)                            │   │
│  │   • search_by_section()                                     │   │
│  │   • search_with_chunk_type_preference()                     │   │
│  │                                                             │   │
│  │ NEW IN PHASE 2:                                             │   │
│  │   • HierarchicalVectorMemory class                          │   │
│  │   • index_document_hierarchical()                           │   │
│  │   • hierarchical_search()                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Vector Memory Enhancement (Foundation)

**Duration**: 2-3 days
**Goal**: Implement semantic chunking + metadata before LangGraph (no orchestration changes)

### 1.1 Semantic Chunking Implementation

**File**: `agent/vector_memory.py`

**Change**: Replace fixed-size chunking with structure-aware chunking.

```python
# NEW CLASS: SemanticChunkingStrategy (add to vector_memory.py)

from langchain_text_splitters import MarkdownHeaderTextSplitter

class SemanticChunkingStrategy:
    """
    Two-stage semantic chunking:
    
    Stage 1: MarkdownHeaderTextSplitter splits by structure (## sections)
    Stage 2: TokenTextSplitter splits oversized sections
    
    Result: Chunks aligned with document structure, not arbitrary token limits.
    """
    
    def __init__(self, max_chunk_size: int = 300):
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'document_title'),
                ('##', 'section'),
                ('###', 'subsection'),
            ],
            strip_headers=False,
        )
        self.size_splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=max_chunk_size,
            chunk_overlap=50,
        )
    
    def split(self, content: str, source_filename: str) -> List[Document]:
        """Split markdown into semantic chunks with section metadata."""
        # Stage 1: Split by headers
        header_chunks = self.header_splitter.split_text(content)
        
        # Stage 2: Further split oversized chunks + enrich metadata
        final_chunks = []
        for chunk in header_chunks:
            # Extract metadata from chunk
            section_meta = chunk.metadata.copy()
            section_meta["source"] = source_filename
            
            if len(chunk.page_content) > self.size_splitter._chunk_size:
                sub_chunks = self.size_splitter.split_text(chunk.page_content)
                for sub in sub_chunks:
                    final_chunks.append(Document(
                        page_content=sub,
                        metadata=section_meta,
                    ))
            else:
                final_chunks.append(Document(
                    page_content=chunk.page_content,
                    metadata=section_meta,
                ))
        
        return final_chunks
```

**Integration Point**: Modify `index_document()` to use semantic splitter.

```python
# In VectorMemory.__init__ - replace splitter
self._semantic_splitter = SemanticChunkingStrategy(max_chunk_size=300)

# In index_document() - replace chunking logic
def index_document(self, content: str, source_filename: str) -> int:
    """Split document with semantic chunking, embed, and store."""
    # REMOVE: self._splitter.create_document()
    # ADD:
    chunks = self._semantic_splitter.split(content, source_filename)
    
    # Enrich with additional metadata (chunk_type, token_count, chunk_index)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_type"] = self._classify_chunk_type(chunk.page_content)
        chunk.metadata["token_count"] = self._count_tokens(chunk.page_content)
    
    # Store (unchanged)
    store = self._get_store()
    store.add_documents(chunks)
    return len(chunks)

def _classify_chunk_type(self, content: str) -> str:
    """Classify chunk as prose, code, table, or diagram."""
    if content.strip().startswith('```'):
        return "code"
    elif content.strip().startswith('|') or '---' in content[:50]:
        return "table"
    elif 'mermaid' in content.lower() or 'graph' in content.lower():
        return "diagram"
    return "prose"

def _count_tokens(self, text: str) -> int:
    """Centralized token counting."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4
```

### 1.2 Metadata Filtering Methods

**File**: `agent/vector_memory.py`

**Change**: Add metadata-filtered search methods.

```python
# ADD after search_by_source()

def search_by_section(
    self,
    query: str,
    section: str | List[str],
    k: int = 5,
) -> List[tuple]:
    """Search within specific document sections."""
    if isinstance(section, str):
        filter_dict = {"section": section}
    else:
        filter_dict = {"section": {"$or": section}}
    
    return self.similarity_search_mmr_with_score(
        query=query,
        k=k,
        filter=filter_dict,
    )

def search_with_chunk_type_preference(
    self,
    query: str,
    prefer_types: List[str] = ["prose"],
    k: int = 5,
) -> List[tuple]:
    """Search prioritizing certain chunk types."""
    return self.similarity_search_mmr_with_score(
        query=query,
        k=k,
        filter={"chunk_type": {"$or": prefer_types}},
    )

def search_by_document(
    self,
    query: str,
    source_filename: str,
    k: int = 5,
) -> List[tuple]:
    """Alias for search_by_source (consistent naming)."""
    return self.search_by_source(query, source_filename, k)
```

### 1.3 Dead Code Identification - Phase 1

**No dead code yet** - Phase 1 only enhances vector_memory.py, doesn't change architect.py.

---

## Phase 2: Hierarchical Retrieval (Optional Enhancement)

**Duration**: 1-2 days
**Goal**: Add parent-child chunk structure for better context

**Decision Point**: Implement only if retrieval quality testing shows fragmentation issues.

### 2.1 HierarchicalVectorMemory Class

**File**: `agent/vector_memory.py`

**Change**: Add new class (coexists with existing VectorMemory).

```python
class HierarchicalVectorMemory:
    """
    Two-tier vector storage for complete context retrieval.
    
    - Parent collection: Large chunks (full sections, ~500 tokens)
    - Child collection: Small chunks (sentences, ~100 tokens)
    
    Retrieval: Search children → fetch parent context.
    """
    
    def __init__(self, persist_dir: str):
        self.embeddings = HuggingFaceEmbeddings(...)
        
        # Two separate collections
        self.parent_store = Chroma(
            collection_name='parents',
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
        
        self.child_store = Chroma(
            collection_name='children',
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
    
    def index_document_hierarchical(self, content: str, source: str) -> dict:
        """Index with parent-child structure."""
        # Implementation per advanced-retrieval-feasibility.md
        ...
    
    def hierarchical_search(self, query: str, k: int = 5) -> List[Document]:
        """Search children → merge to parents."""
        child_results = self.child_store.similarity_search_with_score(query, k=k*2)
        parent_ids = [doc.metadata.get("parent_id") for doc, _ in child_results]
        return self.parent_store.get(where={"id": {"$or": parent_ids[:k]}})
```

### 2.2 Configurable Retrieval Mode

**File**: `agent/vector_memory.py` (or new `agent/retrieval_factory.py`)

**Change**: Add factory for retrieval strategy selection.

```python
def get_retriever(mode: str = "standard") -> Union[VectorMemory, HierarchicalVectorMemory]:
    """
    Factory for retrieval strategy.
    
    Args:
        mode: "standard" (MMR) or "hierarchical" (parent-child)
    
    Returns:
        Appropriate retriever instance
    """
    if mode == "hierarchical":
        return HierarchicalVectorMemory(DEFAULT_CHROMA_DIR)
    return get_vector_memory()  # Standard singleton
```

---

## Phase 3: LangGraph Self-RAG (Orchestration Layer)

**Duration**: 2-3 days
**Goal**: Replace manual ReAct loop with LangGraph graph for reflection + retry

### 3.1 New File: LangGraph Architect

**File**: `agent/langgraph_architect.py` (NEW)

**Purpose**: Define StateGraph for Self-RAG workflow.

```python
"""
agent/langgraph_architect.py - LangGraph Orchestration Layer

WHY THIS EXISTS:
  Replaces manual ReAct loop with structured graph that:
    - Validates retrieval (grade_node)
    - Reformulates queries (rewrite_node)
    - Manages state persistently (ArchitectState)
    - Enables HITL (interrupt() in approval_node)
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from .vector_memory import get_vector_memory
from .tools import save_document, update_document, list_documents
from langchain_google_genai import ChatGoogleGenerativeAI

# ── State Schema ──────────────────────────────────────────────────────

class ArchitectState(TypedDict):
    """State for LangGraph architect workflow."""
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_chunks: list[Document]
    query_iterations: int
    pending_tools: list[dict] | None  # CHANGED: Support multiple tool calls (stress test 2.5)
    current_tool_index: int  # Track which tool is being processed
    retrieval_mode: str  # "standard" or "hierarchical"
    interrupt_timestamp: float | None  # CHANGED: For interrupt timeout (stress test 1.4)
    token_count: int  # CHANGED: Track per-turn token usage (stress test - token counting)
    enable_grading: bool  # CHANGED: Self-RAG configurable (stress test 3.3)
    enable_rewrite: bool  # CHANGED: Self-RAG configurable (stress test 3.3)

# ── Grading Model ──────────────────────────────────────────────────────

class GradeDocuments(BaseModel):
    """Binary relevance grading."""
    binary_score: str = Field(description="'yes' if relevant, 'no' if not")

GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved documents.\n"
    "Documents:\n{context}\n\n"
    "Question: {question}\n\n"
    "If documents contain keywords or semantic meaning related to the question, "
    "grade 'yes'. Otherwise 'no'."
)

# ── System Prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Senior Software Architect AI..."""  # From architect.py

# ── Nodes ──────────────────────────────────────────────────────────────

class LangGraphArchitect:
    """LangGraph-based architect with Self-RAG + HITL."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        enable_grading: bool = False,  # CHANGED: Configurable Self-RAG (stress test 3.3)
        enable_rewrite: bool = False,  # CHANGED: Configurable Self-RAG (stress test 3.3)
    ):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        self.llm_with_tools = self.llm.bind_tools([save_document, update_document, list_documents])
        self.llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.vector_memory = get_vector_memory()
        
        # CHANGED: Self-RAG configuration
        self.enable_grading = enable_grading
        self.enable_rewrite = enable_rewrite
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Construct Self-RAG graph with configurable grading/rewriting."""
        builder = StateGraph(ArchitectState)
        
        # Add nodes (all nodes implemented, but routing depends on config)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade", self._grade_node)
        builder.add_node("rewrite", self._rewrite_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("approval", self._approval_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("cancel", self._cancel_node)
        builder.add_node("prune", self._prune_node)
        
        # Add edges - CHANGED: routing depends on enable_grading/enable_rewrite
        builder.add_edge(START, "retrieve")
        
        if self.enable_grading:
            builder.add_conditional_edges("retrieve", self._route_after_retrieve_with_grade)
            builder.add_conditional_edges("grade", self._route_grade_result)
        else:
            builder.add_conditional_edges("retrieve", self._route_after_retrieve_no_grade)
        
        if self.enable_rewrite:
            builder.add_edge("rewrite", "retrieve")
        else:
            builder.add_edge("rewrite", "agent")  # Skip rewrite loop
        
        # CHANGED: Removed _route_approval (stress test 2.4) - approval_node uses Command
        builder.add_conditional_edges("agent", self._route_has_tool)
        builder.add_edge("execute", "agent")  # Continue ReAct
        builder.add_edge("cancel", "prune")
        builder.add_edge("prune", END)
        
        # Compile with SqliteSaver checkpointer
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver(sqlite3.connect("checkpoints.db"))
        return builder.compile(checkpointer=checkpointer)
    
    def _retrieve_node(self, state: ArchitectState) -> dict:
        """
        RAG retrieval with MMR + threshold + metadata filtering.
        
        CHANGES from stress test:
        - 1.2: Add error handling
        - 2.1: Add full document fallback when chunks empty (no token limit)
        """
        try:
            query = state["messages"][-1].content
            vm = self.vector_memory
            
            if not vm.has_documents():
                return {"retrieved_chunks": [], "query_iterations": 0}
            
            # Use existing MMR search (from search-improvement-plan.md)
            results = vm.similarity_search_mmr_with_score(
                query=query,
                k=10,
                fetch_k=30,
                lambda_mult=0.5,
            )
            
            # Apply threshold (0.85 distance = 15% similarity minimum)
            relevant = [doc for doc, score in results if score < 0.85]
            
            # CHANGED: Full document fallback if no chunks after 2 retries (stress test 2.1)
            if not relevant and state["query_iterations"] >= 2:
                # Load entire source document regardless of token count
                source = state.get("last_query_source", "architecture.md")
                full_doc = vm.get_full_document_by_source(source)
                
                if full_doc:
                    return {
                        "retrieved_chunks": [Document(
                            page_content=full_doc,
                            metadata={"source": source, "fallback": True}
                        )],
                        "messages": state.get("messages", []) + [
                            SystemMessage(content="Note: No relevant chunks found. Using full document as context.")
                        ],
                    }
            
            return {"retrieved_chunks": relevant, "query_iterations": 0}
        
        except Exception as e:
            # CHANGED: Fail-safe fallback (stress test 1.2)
            import logging
            logging.error(f"Retrieval failed: {e}")
            return {"retrieved_chunks": [], "query_iterations": 0}
    
    def _grade_node(self, state: ArchitectState) -> Literal["agent", "rewrite"]:
        """LLM grades retrieval relevance - CHANGED: with error handling (stress test 1.2)."""
        try:
            if not state["retrieved_chunks"]:
                if state["query_iterations"] >= 2:
                    return "agent"  # Give up after 2 retries
                return "rewrite"
            
            question = state["messages"][0].content
            context = "\n".join(doc.page_content for doc in state["retrieved_chunks"])
            
            prompt = GRADE_PROMPT.format(question=question, context=context)
            result = self.llm_grader.invoke([{"role": "user", "content": prompt}])
            
            if result.binary_score == "yes":
                return "agent"
            elif state["query_iterations"] >= 2:
                return "agent"  # Max retries
            return "rewrite"
        except Exception as e:
            # CHANGED: Fail-safe fallback (stress test 1.2)
            import logging
            logging.error(f"Grading failed: {e}")
            return "agent"  # Proceed without validation
    
    def _rewrite_node(self, state: ArchitectState) -> dict:
        """LLM reformulates query for better retrieval."""
        original = state["messages"][0].content
        rewrite_prompt = f"Improve this query for document search: {original}"
        rewritten = self.llm.invoke([{"role": "user", "content": rewrite_prompt}])
        
        return {
            "messages": [HumanMessage(content=rewritten.content)],
            "query_iterations": state["query_iterations"] + 1,
        }
    
    def _agent_node(self, state: ArchitectState) -> dict:
        """
        LLM with tools, using retrieved context.
        
        CHANGES from stress test:
        - 1.2: Add error handling
        - 2.3: Add state pruning
        - 2.5: Support multiple tool calls (list)
        - Token budget: Apply budget filtering here (not in retrieve_node)
        - Token counting: Track per-turn token usage
        """
        try:
            # CHANGED: Apply token budget BEFORE LLM call (stress test - token budget)
            chunks = state["retrieved_chunks"]
            budgeted_chunks = self._apply_token_budget(chunks, max_tokens=4000)
            
            # Format context
            context = self.vector_memory.format_context(budgeted_chunks)
            
            # Build augmented message
            system_prompt = SYSTEM_PROMPT
            user_message = state['messages'][-1].content
            augmented = f"RELEVANT CONTEXT:\n{context}\n\nUSER REQUEST:\n{user_message}"
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            messages[-1] = HumanMessage(content=augmented)
            
            # CHANGED: Count tokens before LLM call (stress test - token counting)
            token_count = self._count_message_tokens([system_prompt, context, user_message])
            
            response = self.llm_with_tools.invoke(messages)
            
            # CHANGED: Support multiple tool calls (stress test 2.5)
            if response.tool_calls:
                # Queue ALL tools (not just first)
                return {
                    "messages": [response],
                    "pending_tools": response.tool_calls,  # Full list
                    "current_tool_index": 0,  # Start at first
                    "token_count": token_count,
                }
            
            # CHANGED: Prune messages if growing too large (stress test 2.3)
            messages_list = state["messages"]
            if len(messages_list) > 20:
                # Keep system prompt + last 15 messages
                messages_list = [messages_list[0]] + messages_list[-15:]
            
            return {"messages": [response], "pending_tools": None, "token_count": token_count}
        
        except Exception as e:
            # CHANGED: Fail-safe fallback (stress test 1.2)
            import logging
            logging.error(f"Agent node failed: {e}")
            return {
                "messages": [AIMessage(content=f"Error processing request: {str(e)}")],
                "pending_tools": None,
            }
    
    def _apply_token_budget(self, chunks: list[Document], max_tokens: int) -> list[Document]:
        """
        CHANGED: Keep chunks within budget, preserving MMR order (stress test - token budget).
        """
        total = 0
        kept = []
        for chunk in chunks:
            tokens = self._count_message_tokens([chunk.page_content])
            if total + tokens <= max_tokens:
                kept.append(chunk)
                total += tokens
            else:
                break
        return kept
    
    def _count_message_tokens(self, message_parts: list[str]) -> int:
        """
        CHANGED: LangGraph-native token counter (stress test - token counting).
        """
        full_message = "\n".join(message_parts)
        return self.llm.get_num_tokens(full_message)
    
    def _approval_node(self, state: ArchitectState) -> Command[Literal["execute", "cancel", END]]:
        """
        HITL: interrupt for write operations.
        
        CHANGES from stress test:
        - 1.4: Add interrupt timeout (5 minutes)
        - 2.2: Add explicit None handling
        - 2.4: Use Command for ALL routing (removed _route_approval)
        - 2.5: Support multiple tool calls (list)
        """
        import time
        import logging
        
        tools = state["pending_tools"]
        idx = state["current_tool_index"]
        
        # CHANGED: Explicit None check FIRST (stress test 2.2)
        if tools is None or idx >= len(tools):
            return Command(goto=END)  # End conversation
        
        current_tool = tools[idx]
        
        # CHANGED: Check for interrupt timeout (stress test 1.4)
        if state["interrupt_timestamp"]:
            elapsed = time.time() - state["interrupt_timestamp"]
            if elapsed > 300:  # 5 minutes
                logging.warning("Approval timed out after 5 minutes")
                return Command(
                    goto="cancel",
                    update={"messages": [AIMessage("Approval timed out after 5 minutes")]}
                )
        
        # Tools requiring approval
        if current_tool["name"] in ["update_document", "save_document"]:
            # Set interrupt timestamp
            state["interrupt_timestamp"] = time.time()
            
            # CHANGED: Show tool index for multiple tools (stress test 2.5)
            decision = interrupt({
                "action": current_tool["name"],
                "args": current_tool["args"],
                "content_preview": current_tool["args"].get("content", "")[:200] + "...",
                "message": f"Approve {current_tool['name']} (tool {idx+1} of {len(tools)})?",
                "tool_index": idx + 1,
                "total_tools": len(tools),
            })
            
            if decision:
                return Command(goto="execute")
            return Command(goto="cancel")
        
        # list_documents doesn't need approval
        return Command(goto="execute")
    
    def _execute_node(self, state: ArchitectState) -> dict:
        """
        Execute approved tool.
        
        CHANGES from stress test:
        - 1.2: Add error handling
        - 2.5: Support multiple tool calls (iterate through list)
        """
        try:
            tools = state["pending_tools"]
            idx = state["current_tool_index"]
            
            # CHANGED: Handle multiple tools (stress test 2.5)
            current_tool = tools[idx]
            
            tool_map = {
                "save_document": save_document,
                "update_document": update_document,
                "list_documents": list_documents,
            }
            
            result = tool_map[current_tool["name"]].invoke(current_tool["args"])
            
            # Move to next tool
            next_idx = idx + 1
            
            if next_idx < len(tools):
                # More tools to process - loop back to approval for next tool
                return {
                    "messages": [ToolMessage(content=f"Tool {idx+1} result: {result}", tool_call_id=current_tool["id"])],
                    "current_tool_index": next_idx,
                }
            
            # All tools done
            return {
                "messages": [ToolMessage(content=str(result), tool_call_id=current_tool["id"])],
                "pending_tools": None,
                "current_tool_index": 0,
            }
        
        except Exception as e:
            # CHANGED: Fail-safe fallback (stress test 1.2)
            import logging
            logging.error(f"Tool execution failed: {e}")
            return {
                "messages": [AIMessage(content=f"Tool execution failed: {str(e)}")],
                "pending_tools": None,
            }
    
    def _cancel_node(self, state: ArchitectState) -> dict:
        """Handle cancelled tool."""
        return {
            "messages": [AIMessage(content="Action cancelled by user.")],
            "pending_tool": None,
        }
    
    def _prune_node(self, state: ArchitectState) -> dict:
        """Sliding window behavior."""
        MAX_MESSAGES = 11
        messages = state["messages"]
        
        if len(messages) > MAX_MESSAGES:
            # Keep system + last N
            system = messages[0] if isinstance(messages[0], SystemMessage) else None
            recent = messages[-MAX_MESSAGES+1:]
            return {"messages": [system] + recent if system else recent}
        return {}
    
    def _route_after_retrieve_with_grade(self, state: ArchitectState) -> Literal["grade", "agent", "rewrite"]:
        """
        CHANGED: Route after retrieve when grading enabled (stress test - Self-RAG config).
        """
        if state["retrieved_chunks"]:
            return "grade"
        if state["query_iterations"] >= 2:
            return "agent"  # Skip grading, proceed with no/fallback context
        return "rewrite"
    
    def _route_after_retrieve_no_grade(self, state: ArchitectState) -> Literal["agent", "rewrite"]:
        """
        CHANGED: Bypass grade node when grading disabled (stress test - Self-RAG config).
        """
        if state["retrieved_chunks"]:
            return "agent"  # Skip grading, go directly to agent
        if state["query_iterations"] >= 2:
            return "agent"
        return "rewrite"
    
    def _route_grade_result(self, state: ArchitectState) -> Literal["agent", "rewrite"]:
        """Route based on grade result (internal routing in grade_node handles this)."""
        # This is a placeholder - actual routing happens inside _grade_node
        return "agent"
    
    def _route_has_tool(self, state: ArchitectState) -> Literal["approval", END]:
        """Route based on whether agent has tool calls."""
        if state["pending_tools"]:
            return "approval"
        return END
    
    # REMOVED: _route_approval (stress test 2.4)
    # approval_node now handles ALL routing via Command
    
    # ── Public API ──────────────────────────────────────────────────────
    
    def run(self, user_message: str, thread_id: str = "default") -> str | dict:
        """
        Invoke graph with user message.
        
        CHANGED from stress test:
        - 1.5: Thread ID uniqueness per UI (caller must provide unique ID)
        
        Usage:
            # Terminal: thread_id = f"terminal_{uuid.uuid4()}"
            # Web UI: thread_id = f"web_{user_id}_{session_id}"
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        
        # Handle interrupt (HITL)
        if "__interrupt__" in result:
            # Return interrupt payload for UI to handle
            return {"type": "interrupt", "payload": result["__interrupt__"]}
        
        # Return final AI message
        return result["messages"][-1].content
    
    def resume(self, decision: bool, thread_id: str = "default") -> str:
        """Resume after HITL approval."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(Command(resume=decision), config=config)
        return result["messages"][-1].content
```

### 3.2 Dead Code in architect.py After Phase 3

**File**: `agent/architect.py`

| Lines | Code | Status | Reason |
|-------|------|--------|--------|
| 266-301 | Manual ReAct loop | **DEAD** | Replaced by LangGraph graph |
| 508-660 | `_build_augmented_message()` | **DEAD** | Replaced by `_retrieve_node` + `_agent_node` |
| 442-486 | `_count_rag_tokens()` | **DEAD** | Duplicate - use vector_memory `_count_tokens()` |
| 488-502 | `_fallback_token_count()` | **DEAD** | Replaced by LangGraph state |

**Recommended Action**: Mark as deprecated or remove after Phase 3 testing.

```python
# architect.py AFTER Phase 3 - becomes wrapper

class SoftwareArchitectAgent:
    """Wrapper for LangGraph architect (backward compatibility)."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        # Delegate to LangGraph
        self._graph_agent = LangGraphArchitect(model=model)
    
    def run(self, user_message: str) -> str:
        """Delegate to LangGraph.run()."""
        return self._graph_agent.run(user_message)
    
    def switch_model(self, model: str) -> None:
        """Recreate LangGraph with new model."""
        self._graph_agent = LangGraphArchitect(model=model)
    
    def clear_memory(self) -> None:
        """LangGraph memory is per-thread, no global clear."""
        pass  # Or implement thread-specific clear
    
    # Keep token estimation for display purposes
    def get_token_estimate(self) -> int:
        """Estimate tokens in current thread."""
        # Delegate to LangGraph state
        ...
```

---

## Phase 4: Dispatcher + UI Integration

**Duration**: 1-2 days
**Goal**: Update dispatcher for LangGraph, add HITL UI handling

### 4.1 Dispatcher Changes

**File**: `agent/dispatcher.py`

**Change**: Add LangGraph invocation + interrupt handling.

```python
# In dispatcher.py - add imports
from .langgraph_architect import LangGraphArchitect

# Replace architect singleton
_langitect: Optional[LangGraphArchitect] = None

def _get_langitect() -> LangGraphArchitect:
    """Lazy-load LangGraph architect."""
    global _langitect
    if _langitect is None:
        _langitect = LangGraphArchitect()
    return _langitect

def agentic_action(user_message: str, thread_id: str = "default") -> str | dict:
    """
    Entry point - now returns str OR dict (interrupt payload).
    
    Returns:
        str: Normal response
        dict: {"type": "interrupt", "payload": {...}} for HITL
    """
    agent = _get_langitect()
    result = agent.run(user_message, thread_id)
    
    # Check for interrupt
    if isinstance(result, dict) and result.get("type") == "interrupt":
        # UI must handle this - show approval modal
        return result
    
    return result

def resume_action(decision: bool, thread_id: str = "default") -> str:
    """Resume after user approves/rejects."""
    agent = _get_langitect()
    return agent.resume(decision, thread_id)

# REMOVE: _select_agent() - only one agent now
# REMOVE: _simplify_instruction() - LangGraph handles this
# KEEP: init_architect, switch_model, reset_conversation (adapted)
# KEEP: reset_vector_memory, reindex_all_documents (unchanged)
```

### 4.2 Terminal HITL Handling (main.py)

**File**: `main.py`

**Change**: Add interrupt handling + interactive approval prompt.

**Key Difference from Web UI**: Terminal HITL **blocks** and prompts immediately in the same message loop. User responds, then execution continues. No async handling or API endpoints needed.

```python
# In main.py - add imports (questionary already installed for /models)
import questionary
from agent import agentic_action, resume_action

# Add interrupt handler function

def handle_interrupt(payload: dict) -> bool:
    """
    Show approval prompt in terminal and return user decision.
    
    Args:
        payload: The interrupt payload from LangGraph
        {"action": str, "args": dict, "content_preview": str, "message": str}
        
    Returns:
        True if approved, False if rejected
    """
    action = payload.get("action", "unknown")
    args = payload.get("args", {})
    preview = payload.get("content_preview", "")
    
    # Print interrupt details
    print("\n" + "=" * 50)
    print("⏸️  APPROVAL REQUIRED")
    print("=" * 50)
    print(f"Action: {action}")
    print(f"File: {args.get('filename', 'unknown')}")
    print(f"\nPreview:\n{preview}")
    print("=" * 50)
    
    # Use questionary for interactive prompt (arrow keys + enter)
    try:
        decision = questionary.confirm(
            "Approve this action?",
            default=False
        ).ask()
        
        if decision is None:  # User pressed Ctrl+C
            print("\nAction cancelled.")
            return False
        
        return decision
    except Exception:
        # Fallback to simple input if questionary fails
        response = input("Approve? [y/N]: ").strip().lower()
        return response in ["y", "yes"]


# Modify message handling in main() loop

async def main():
    print_banner(model)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands (/tokens, /models, etc.)
            if user_input.startswith("/"):
                handle_command(user_input)
                continue
            
            # Process message
            result = agentic_action(user_input)
            
            # CHECK FOR INTERRUPT (HITL)
            if isinstance(result, dict) and result.get("type") == "interrupt":
                # Extract interrupt payload
                interrupt_data = result["payload"][0]["value"]
                
                # Show approval prompt (blocking)
                approved = handle_interrupt(interrupt_data)
                
                # Resume with decision
                final_response = resume_action(approved)
                print(f"\nArchitect: {final_response}")
            else:
                # Normal response
                print(f"\nArchitect: {result}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


# Minimal alternative (without questionary confirm):

def handle_interrupt_simple(payload: dict) -> bool:
    """Simple approval prompt without questionary."""
    print(f"\n⏸️ Approve {payload['action']}? [y/N]: ", end="")
    decision = input().strip().lower() in ["y", "yes"]
    return decision
```

### 4.3 Web UI HITL Handling (app.py)

**File**: `app.py`

**Change**: Add interrupt handling + approval modal for Gradio interface.

**Key Difference from Terminal**: Web UI returns interrupt payload to frontend, shows modal, user clicks button, frontend calls `resume` API endpoint. This is non-blocking.

```python
# In app.py - add interrupt detection

def process_message(message: str, history: list) -> str:
    """Process message, handle interrupt for approval."""
    thread_id = get_thread_id(history)
    
    result = agentic_action(message, thread_id)
    
    # Check for interrupt (HITL)
    if isinstance(result, dict) and result.get("type") == "interrupt":
        payload = result["payload"][0]["value"]  # First interrupt
        
        # Return approval request to UI (shown in chat as message)
        return f"""
        ⏸️ **Approval Required**
        
        **Action**: {payload['action']}
        **File**: {payload['args'].get('filename', 'unknown')}
        
        **Preview**:
        ```
        {payload['content_preview']}
        ```
        
        Please respond with:
        - `approve` to proceed
        - `reject` to cancel
        """
    
    return result


def handle_approval_response(response: str, history: list) -> str:
    """Handle user approval response (typed in chat)."""
    thread_id = get_thread_id(history)
    decision = response.lower() in ["approve", "yes", "ok", "y"]
    return resume_action(decision, thread_id)


# Gradio interface modification

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Message")
    
    # Track pending approval state
    pending_approval = gr.State(False)
    
    def respond(message, chat_history, pending):
        # If pending approval, treat message as approval response
        if pending:
            result = handle_approval_response(message, chat_history)
            return result, chat_history + [[message, result]], False
        
        # Normal message processing
        result = process_message(message, chat_history)
        
        # Check if result is interrupt
        is_interrupt = isinstance(result, dict) and result.get("type") == "interrupt"
        
        if is_interrupt:
            # Show approval request, set pending state
            approval_msg = format_interrupt_for_ui(result)
            return approval_msg, chat_history + [[message, approval_msg]], True
        
        return result, chat_history + [[message, result]], False
    
    msg.submit(respond, [msg, chatbot, pending_approval], [chatbot, chatbot, pending_approval])
```

### 4.4 HITL Comparison: Terminal vs Web UI

| Aspect                    | Terminal (main.py)                      | Web UI (app.py)                              |
|---------------------------|-----------------------------------------|----------------------------------------------|
| **Interrupt display**     | Print + `questionary.confirm()`         | Special response in chat                     |
| **User input**            | Keyboard `y/N` or arrow keys            | Type "approve" or click button               |
| **Resume flow**           | `resume_action()` in same loop          | `resume_action()` after user response        |
| **Blocking**              | Yes (waits for input)                   | No (returns, user responds later)            |
| **Session persistence**   | Thread ID in memory                     | Thread ID in Gradio State                    |
| **Complexity**            | Lower (~30 lines)                       | Higher (~50 lines + state)                   |

### 4.5 Dispatcher Changes (Shared by Both)

**File**: `agent/dispatcher.py`

**Change**: Add LangGraph invocation + interrupt handling + resume function.

```python
# In dispatcher.py - add imports
from .langgraph_architect import LangGraphArchitect

# Replace architect singleton
_langitect: Optional[LangGraphArchitect] = None

def _get_langitect() -> LangGraphArchitect:
    """Lazy-load LangGraph architect."""
    global _langitect
    if _langitect is None:
        _langitect = LangGraphArchitect()
    return _langitect

def agentic_action(user_message: str, thread_id: str = "default") -> str | dict:
    """
    Entry point - now returns str OR dict (interrupt payload).
    
    Used by BOTH:
      - main.py (terminal) - handles interrupt inline
      - app.py (web UI) - returns interrupt to frontend
    
    Returns:
        str: Normal response
        dict: {"type": "interrupt", "payload": [...]} for HITL
    """
    agent = _get_langitect()
    result = agent.run(user_message, thread_id)
    
    # Check for interrupt - return to UI layer for handling
    if isinstance(result, dict) and result.get("type") == "interrupt":
        return result
    
    return result

def resume_action(decision: bool, thread_id: str = "default") -> str:
    """
    Resume after user approves/rejects.
    
    Called by:
      - main.py: after handle_interrupt() returns decision
      - app.py: after user types "approve" or "reject"
    """
    agent = _get_langitect()
    return agent.resume(decision, thread_id)

# REMOVE: _select_agent() - only one agent now
# REMOVE: _simplify_instruction() - LangGraph handles this
# KEEP: init_architect, switch_model, reset_conversation (adapted)
# KEEP: reset_vector_memory, reindex_all_documents (unchanged)
```

### 4.6 Dead Code in dispatcher.py After Phase 4

| Lines | Code | Status | Reason |
|-------|------|--------|--------|
| 75-102 | `_select_agent()` | **DEAD** | Single agent, no routing needed |
| 109-149 | `_simplify_instruction()` | **DEAD** | LangGraph handles prompt cleaning |

---

## File Change Matrix

### Summary of All File Changes

| File | Phase | Change Type | Lines Added | Lines Removed | Lines Modified |
|------|-------|-------------|-------------|---------------|----------------|
| `agent/vector_memory.py` | 1 | Enhance | +150 | -20 | ~50 |
| `agent/vector_memory.py` | 2 | Add class | +200 | 0 | 0 |
| `agent/langgraph_architect.py` | 3 | **NEW** | +300 | 0 | 0 |
| `agent/architect.py` | 3 | Deprecate | +30 | ~250 | ~10 |
| `agent/dispatcher.py` | 4 | Refactor | +50 | ~100 | ~20 |
| `agent/tools.py` | 4 | Enhance | +30 | 0 | ~10 |
| `main.py` | 4 | Enhance (Terminal HITL) | +40 | 0 | ~15 |
| `app.py` | 4 | Enhance (Web UI HITL) | +50 | 0 | ~20 |

**Note**: Both `main.py` and `app.py` are updated for HITL support. They use the same dispatcher functions (`agentic_action`, `resume_action`) but handle interrupts differently based on UI type.

### Detailed Change Log

#### agent/vector_memory.py

| Section | Current | After Phase 1 | After Phase 2 |
|---------|---------|---------------|---------------|
| Imports | Line 78-88 | Add: `MarkdownHeaderTextSplitter` | Add: `HierarchicalVectorMemory` class |
| `__init__` | Line 127-173 | Replace `TokenTextSplitter` with `SemanticChunkingStrategy` | No change |
| `index_document` | Line 179-226 | **REPLACE**: Use semantic splitter + metadata enrichment | Add: `index_document_hierarchical()` in new class |
| `similarity_search_mmr_with_score` | Line 383-487 | Keep unchanged | Keep unchanged |
| `search_by_source` | Line 489-513 | Keep unchanged | Keep unchanged |
| NEW methods | None | Add: `search_by_section`, `search_with_chunk_type_preference`, `_classify_chunk_type`, `_count_tokens` | Add: `HierarchicalVectorMemory` class |

#### agent/architect.py

| Section | Current | After Phase 3 |
|---------|---------|---------------|
| `run()` | Line 213-321 | **DEAD** - Delegate to `LangGraphArchitect.run()` |
| `_build_augmented_message()` | Line 508-660 | **DEAD** - Moved to `_retrieve_node` |
| `_count_rag_tokens()` | Line 442-486 | **DEAD** - Duplicate in vector_memory |
| `_fallback_token_count()` | Line 488-502 | **DEAD** - No longer needed |
| ReAct loop | Line 266-301 | **DEAD** - Replaced by LangGraph graph |
| `switch_model()` | Line 323-340 | Keep - delegate to LangGraph |
| `get_exact_token_count()` | Line 351-440 | Keep or adapt - useful for display |

#### agent/dispatcher.py

| Section | Current | After Phase 4 |
|---------|---------|---------------|
| `_get_architect()` | Line 59-68 | Replace with `_get_langitect()` |
| `_select_agent()` | Line 75-102 | **DEAD** - Remove |
| `_simplify_instruction()` | Line 109-149 | **DEAD** - Remove |
| `agentic_action()` | Line 156-280 | **REFACTOR**: Add interrupt handling |
| `reset_vector_memory()` | Line 363-464 | Keep unchanged |
| `reindex_all_documents()` | Line 467-551 | Keep unchanged |

---

## Duplicate Code Resolution

### Identified Duplicates

| Duplicate | File 1 | File 2 | Resolution |
|-----------|--------|--------|------------|
| Token counting | `architect.py:583-592` | `vector_memory.py` (new) | Use vector_memory version, remove from architect |
| Context formatting | `architect.py` (implicit) | `vector_memory.py:265-297` | Keep vector_memory, architect calls it |
| Message trimming | `memory.py:143-152` | LangGraph `_prune_node` | Keep both during transition, then choose |

### Recommended Consolidation

1. **Token counting**: Single `_count_tokens()` in `vector_memory.py`
2. **Context formatting**: Keep `format_context()` in `vector_memory.py`
3. **Message pruning**: Keep `SlidingWindowMemory._trim()` for backward compat, add LangGraph `_prune_node` for new flow

---

## Implementation Order (Dependency Graph)

```
Phase 1 (Foundation - No Breaking Changes)
├── 1.1 Semantic chunking in vector_memory.py
├── 1.2 Metadata enrichment in vector_memory.py  
├── 1.3 Metadata filtering methods in vector_memory.py
└── 1.4 Test retrieval quality improvement

Phase 2 (Optional Enhancement)
├── 2.1 HierarchicalVectorMemory class
├── 2.2 Retrieval factory (if hierarchical needed)
└── 2.3 Test hierarchical vs standard

Phase 3 (Orchestration - Breaking Changes)
├── 3.1 Create langgraph_architect.py
├── 3.2 Test Self-RAG loop
├── 3.3 Mark architect.py code as deprecated
└── 3.4 Create wrapper in architect.py

Phase 4 (Integration)
├── 4.1 Update dispatcher.py
├── 4.2 Add interrupt handling to dispatcher
├── 4.3 Update app.py for HITL modal
├── 4.4 Test full flow
└── 4.5 Remove dead code
```

---

## Testing Strategy

### Phase 1 Tests

```python
# Test semantic chunking
vm = get_vector_memory()
vm.index_document("# Title\n## Section\nContent...", "test.md")
chunks = vm.search("section", k=5)

# Verify metadata
assert all("section" in c.metadata for c in chunks)
assert all("chunk_type" in c.metadata for c in chunks)
```

### Phase 3 Tests

```python
# Test Self-RAG
agent = LangGraphArchitect()
result = agent.run("What is the JWT strategy?")

# Test interrupt (HITL)
result = agent.run("Update system-design.md with caching")
assert result["type"] == "interrupt"

# Test resume
final = agent.resume(True)
assert "updated" in final.lower()
```

### Phase 4 Tests

```python
# Test dispatcher integration
result = agentic_action("Create API spec for auth")
assert isinstance(result, str)

result = agentic_action("Update existing doc")
assert result["type"] == "interrupt"

# Test resume
final = resume_action(True)
assert "updated" in final.lower()
```

---

## Rollback Plan

If LangGraph causes issues, rollback strategy:

| Phase | Rollback Action |
|-------|-----------------|
| Phase 1 | Remove semantic splitter, restore `TokenTextSplitter` |
| Phase 2 | Remove `HierarchicalVectorMemory`, use standard |
| Phase 3 | Restore `architect.py` from git, remove `langgraph_architect.py` |
| Phase 4 | Restore `dispatcher.py` from git, remove HITL UI |

**Git branches**: Create `feature/langgraph` branch, merge only after Phase 4 testing.

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Retrieval precision | 30-50% | 60-80% | Grade node "yes" rate |
| Query retries | 0 | < 2 avg | `query_iterations` count |
| Context completeness | Fragmented | Full sections | User feedback on answer quality |
| False updates | Frequent | Zero | Document created vs updated ratio |
| HITL usage | None | All write ops | Interrupt count per session |

---

## Summary

| Question | Answer |
|----------|--------|
| Total effort? | 7-10 days across 4 phases |
| Breaking changes? | Phase 3 only (LangGraph replaces ReAct) |
| Can rollback? | Yes - each phase is independent |
| First priority? | Phase 1 (semantic chunking + metadata) |
| When to implement HITL? | Phase 4 (after LangGraph tested) |
| What becomes dead code? | `_build_augmented_message`, ReAct loop, `_select_agent` |
| Which UI gets HITL? | **Both** - Terminal (main.py) + Web UI (app.py) |

---

## Dual UI Support Architecture

### Why Both UIs Are Maintained

The project supports **two interfaces** that will both receive HITL capability:

| Interface           | File      | Current Use               | Future Use                 |
|---------------------|-----------|---------------------------|----------------------------|
| **Terminal (CLI)**  | `main.py` | Primary development       | HITL blocking prompt       |
| **Web UI (Gradio)** | `app.py`  | Browser-based interaction | HITL non-blocking modal    |

### Shared Dispatcher Layer

Both UIs use the **same dispatcher functions**, making implementation straightforward:

```mermaid
graph TB
    subgraph UI Layer
        T[Terminal main.py<br/>BLOCKING HITL<br/>print prompt + questionary<br/>resume inline]
        W[Web UI app.py<br/>NON-BLOCKING HITL<br/>return interrupt payload<br/>call resume separately]
    end
    
    subgraph Dispatcher Layer
        D[dispatcher.py<br/>agentic_action returns str or interrupt<br/>resume_action resumes after decision]
    end
    
    T --> D
    W --> D
│                      ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Dispatcher Layer (agent/dispatcher.py)              │   │
│  │                                                     │   │
│  │ agentic_action() → returns str or interrupt dict   │   │
│  │ resume_action()  → resumes after user decision     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Order for Dual UI

Phase 4 now includes both UI updates (in sequence):

```
Phase 4 (Integration)
├── 4.1 Update dispatcher.py (shared layer)
├── 4.2 Add interrupt handling to dispatcher
├── 4.3 Update main.py for Terminal HITL (blocking prompt)
├── 4.4 Update app.py for Web UI HITL (non-blocking modal)
├── 4.5 Test both UI flows
└── 4.6 Remove dead code
```

### Terminal HITL (Primary During Development)

The terminal interface is typically used during development. HITL is simpler here:

- **Blocking**: User sees prompt immediately, must respond to continue
- **Single loop**: No state management, decision handled inline
- **Questionary**: Arrow-key selection (already installed for `/models`)

### Web UI HITL (Production Use)

The Gradio web UI may be used for production or shared access:

- **Non-blocking**: Interrupt returns to frontend, user responds later
- **State management**: Gradio `State` tracks pending approval
- **Chat-based**: User types "approve" or "reject" in chat

---

**Recommended execution**: Implement Phase 1 first (no breaking changes), test retrieval quality, then decide on Phase 2-4 based on results. Both UIs will receive HITL in Phase 4.

---

## Stress Test Test Cases (From Stress Test Analysis)

**Added**: 2026-04-25
**Source**: Stress test analysis identified 8 required test scenarios

### Test 1: Interrupt Timeout (Stress Test 1.4)

```python
def test_interrupt_timeout():
    """User doesn't respond for 5+ minutes - should auto-cancel."""
    architect = LangGraphArchitect()

    # Trigger approval interrupt
    result = architect.invoke({
        "messages": [HumanMessage("Save document")],
        "pending_tools": [{"name": "save_document", "args": {...}}],
    })

    # Simulate timeout by setting old timestamp
    state = architect.get_state(result.config)
    state.values["interrupt_timestamp"] = time.time() - 310  # 310 seconds ago

    # Resume should cancel due to timeout
    resume_result = architect.invoke(Command(resume=True), config=result.config)

    assert resume_result["messages"][-1].content == "Approval timed out after 5 minutes"
```

### Test 2: Resume with Wrong Thread ID (Stress Test 2)

```python
def test_resume_wrong_thread_id():
    """Resume with different thread_id - should fail or start fresh."""
    architect = LangGraphArchitect()

    # Create interrupt with thread_id "user_123"
    result = architect.invoke({...}, config={"thread_id": "user_123"})

    # Try to resume with wrong thread_id
    wrong_config = {"thread_id": "user_456"}

    try:
        resume_result = architect.invoke(Command(resume=True), config=wrong_config)
        # Should either fail or return new conversation
        assert resume_result["messages"][-1].content != result["messages"][-1].content
    except Exception as e:
        # Expected: cannot resume with wrong thread
        assert "thread" in str(e).lower() or "not found" in str(e).lower()
```

### Test 3: Grade Node LLM Failure (Stress Test 1.2)

```python
def test_grade_node_llm_failure():
    """Grade LLM times out, rate limits, or returns malformed response."""
    architect = LangGraphArchitect(enable_grading=True)

    # Mock LLM to raise exception
    with patch.object(architect.llm_grader, 'invoke', side_effect=TimeoutError()):
        result = architect.invoke({
            "messages": [HumanMessage("What is authentication?")],
            "retrieved_chunks": [...],
        })

        # Should fallback to "agent" path (fail-safe)
        assert "authentication" in result["messages"][-1].content

    # Test malformed response
    with patch.object(architect.llm_grader, 'invoke', return_value="invalid"):
        result = architect.invoke({...})
        assert result is not None  # Should handle gracefully
```

### Test 4: Rewrite Produces Identical Query (Stress Test 4)

```python
def test_rewrite_identical_query():
    """Rewrite returns same query - should not loop infinitely."""
    architect = LangGraphArchitect(enable_rewrite=True)

    original_query = "What is authentication?"

    # Mock rewrite to return identical query
    with patch.object(architect.llm_rewriter, 'invoke', return_value=original_query):
        result = architect.invoke({
            "messages": [HumanMessage(original_query)],
            "query_iterations": 0,
        })

        # Should stop after max iterations (2)
        assert result["query_iterations"] <= 2
        assert result["messages"][-1].content is not None
```

### Test 5: Multiple Tool Calls (Stress Test 2.5)

```python
def test_multiple_tool_calls():
    """LLM requests 2+ tools simultaneously - all should execute."""
    architect = LangGraphArchitect()

    # Mock LLM to return multiple tools
    mock_response = AIMessage(
        content="",
        tool_calls=[
            {"name": "list_documents", "args": {}, "id": "call_1"},
            {"name": "save_document", "args": {"filename": "test.md", "content": "..."}, "id": "call_2"},
        ]
    )

    with patch.object(architect.llm, 'invoke', return_value=mock_response):
        # First approval for list_documents (no approval needed)
        result = architect.invoke({
            "messages": [HumanMessage("List and save documents")],
        })

        # Should queue both tools
        assert len(result["pending_tools"]) == 2

        # Resume approval for save_document
        resume_result = architect.invoke(Command(resume=True), config=result.config)

        # Both tools should have executed
        assert "list_documents" in resume_result["messages"][-1].content
        assert "save_document" in resume_result["messages"][-1].content
```

### Test 6: State Persistence After Restart (Stress Test 6)

```python
def test_state_persistence_restart():
    """Restart server during HITL interrupt - should resume."""
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    
    architect = LangGraphArchitect(checkpointer=SqliteSaver(sqlite3.connect("checkpoints.db")))

    # Trigger interrupt
    result = architect.invoke({
        "messages": [HumanMessage("Save document")],
        "pending_tools": [{"name": "save_document", "args": {...}}],
    })

    thread_id = result.config["configurable"]["thread_id"]

    # Simulate restart: create new architect instance
    new_architect = LangGraphArchitect(checkpointer=SqliteSaver(sqlite3.connect("checkpoints.db")))

    # Resume with same thread_id
    resume_result = new_architect.invoke(
        Command(resume=True),
        config={"configurable": {"thread_id": thread_id}}
    )

    # Should successfully resume
    assert resume_result["messages"][-1].content is not None
```

### Test 7: Concurrent Sessions Same Thread ID (Stress Test 1.5)

```python
def test_concurrent_sessions_collision():
    """Two users with same thread_id - should not interfere."""
    architect = LangGraphArchitect()

    # User A starts conversation
    result_a = architect.invoke(
        {"messages": [HumanMessage("User A query")]},
        config={"thread_id": "collision_test"}
    )

    # User B tries same thread_id (should be prevented or isolated)
    result_b = architect.invoke(
        {"messages": [HumanMessage("User B query")]},
        config={"thread_id": "collision_test"}
    )

    # Results should be isolated or error raised
    # Option 1: Error raised
    # Option 2: User B overwrites User A (should be prevented)
    # Option 3: Each user gets unique state despite same ID

    # Recommended: raise error or auto-generate unique ID
    # Implementation note: Caller must provide unique thread_id per session
```

### Test 8: Token Budget Exhaustion (Stress Test - Token Budget)

```python
def test_token_budget_exhaustion():
    """Long conversation with many chunks - budget should prevent overflow."""
    architect = LangGraphArchitect()

    # Create many chunks (over budget)
    large_chunks = [Document(page_content=f"Chunk {i} with lots of content...")
                    for i in range(20)]  # ~8000 tokens total

    result = architect.invoke({
        "messages": [HumanMessage("What is architecture?")],
        "retrieved_chunks": large_chunks,
    })

    # Should only use budgeted amount (4000 tokens)
    assert result["token_count"] <= 4000

    # LLM response should still be valid
    assert result["messages"][-1].content is not None
```

---

## Stress Test Changes Summary

**Document updated**: 2026-04-25

| Section | Change | Stress Test Issue |
|---------|--------|-------------------|
| State Schema | Add pending_tools (list), interrupt_timestamp, token_count, enable_grading/rewrite | 1.4, 2.5, Token counting, Self-RAG |
| `__init__` | Add enable_grading/enable_rewrite parameters | 3.3 |
| `_build_graph` | Use SqliteSaver, conditional routing based on grading/rewrite | 1.1, Self-RAG config |
| `_retrieve_node` | Add error handling, full document fallback | 1.2, 2.1 |
| `_grade_node` | Add error handling with fail-safe | 1.2 |
| `_agent_node` | Add token budget, multiple tools, error handling, state pruning | 1.2, 2.3, 2.5, Token budget |
| `_approval_node` | Add timeout, None handling, multiple tools, Command routing | 1.4, 2.2, 2.4, 2.5 |
| `_execute_node` | Add error handling, multiple tools iteration | 1.2, 2.5 |
| Routing functions | Add `_route_after_retrieve_with_grade`, `_route_after_retrieve_no_grade` | Self-RAG config |
| `_route_approval` | REMOVED - approval_node uses Command | 2.4 |
| `run()` | Add thread_id uniqueness note | 1.5 |
| Test Cases | Add 8 comprehensive test scenarios | All stress test issues |
