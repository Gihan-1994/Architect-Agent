# LangGraph Integration Master Plan

**Created**: 2026-04-25
**Status**: Planning Complete
**Scope**: Comprehensive implementation plan integrating LangGraph Self-RAG, HITL, and all planned retrieval features

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
    pending_tool: dict | None  # {"name": str, "args": dict, "id": str}
    retrieval_mode: str  # "standard" or "hierarchical"

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
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        self.llm_with_tools = self.llm.bind_tools([save_document, update_document, list_documents])
        self.llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.vector_memory = get_vector_memory()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Construct Self-RAG graph."""
        builder = StateGraph(ArchitectState)
        
        # Add nodes
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade", self._grade_node)
        builder.add_node("rewrite", self._rewrite_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("approval", self._approval_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("cancel", self._cancel_node)
        builder.add_node("prune", self._prune_node)
        
        # Add edges
        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges("retrieve", self._route_after_retrieve)
        builder.add_edge("rewrite", "retrieve")  # Retry loop
        builder.add_edge("grade", "agent")
        builder.add_edge("agent", "approval")
        builder.add_conditional_edges("approval", self._route_approval)
        builder.add_edge("execute", "agent")  # Continue ReAct
        builder.add_edge("cancel", "prune")
        builder.add_edge("prune", END)
        
        # Compile with checkpointer (required for HITL)
        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)
    
    def _retrieve_node(self, state: ArchitectState) -> dict:
        """RAG retrieval with MMR + threshold + metadata filtering."""
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
        
        return {"retrieved_chunks": relevant, "query_iterations": 0}
    
    def _grade_node(self, state: ArchitectState) -> Literal["agent", "rewrite"]:
        """LLM grades retrieval relevance."""
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
        """LLM with tools, using retrieved context."""
        # Format context
        context = self.vector_memory.format_context(state["retrieved_chunks"])
        
        # Build augmented message
        augmented = f"RELEVANT CONTEXT:\n{context}\n\nUSER REQUEST:\n{state['messages'][-1].content}"
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        messages[-1] = HumanMessage(content=augmented)
        
        response = self.llm_with_tools.invoke(messages)
        
        if response.tool_calls:
            tc = response.tool_calls[0]
            return {
                "messages": [response],
                "pending_tool": {"name": tc["name"], "args": tc["args"], "id": tc["id"]},
            }
        return {"messages": [response], "pending_tool": None}
    
    def _approval_node(self, state: ArchitectState) -> Command[Literal["execute", "cancel"]]:
        """HITL: interrupt for write operations."""
        tool = state["pending_tool"]
        
        if tool and tool["name"] in ["update_document", "save_document"]:
            decision = interrupt({
                "action": tool["name"],
                "args": tool["args"],
                "content_preview": tool["args"].get("content", "")[:200] + "...",
                "message": f"Approve {tool['name']}?",
            })
            
            if decision:
                return Command(goto="execute")
            return Command(goto="cancel")
        
        # list_documents doesn't need approval
        return Command(goto="execute")
    
    def _execute_node(self, state: ArchitectState) -> dict:
        """Execute approved tool."""
        tool = state["pending_tool"]
        tool_map = {
            "save_document": save_document,
            "update_document": update_document,
            "list_documents": list_documents,
        }
        
        result = tool_map[tool["name"]].invoke(tool["args"])
        
        return {
            "messages": [ToolMessage(content=str(result), tool_call_id=tool["id"])],
            "pending_tool": None,
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
    
    def _route_after_retrieve(self, state: ArchitectState) -> Literal["grade", "agent"]:
        """Route after retrieve: grade if chunks found, else rewrite."""
        if state["retrieved_chunks"]:
            return "grade"
        if state["query_iterations"] >= 2:
            return "agent"  # Skip grading, proceed with no context
        return "rewrite"
    
    def _route_approval(self, state: ArchitectState) -> Literal["execute", "cancel", "prune"]:
        """Route after approval check."""
        if state["pending_tool"] is None:
            return "prune"  # No tool, end conversation
        return "execute"  # Will be overridden by approval_node Command
    
    # ── Public API ──────────────────────────────────────────────────────
    
    def run(self, user_message: str, thread_id: str = "default") -> str:
        """Invoke graph with user message."""
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

### 4.2 Gradio UI Changes

**File**: `app.py`

**Change**: Add interrupt handling + approval modal.

```python
# In app.py - add interrupt detection

def process_message(message: str, history: list) -> str:
    """Process message, handle interrupt for approval."""
    thread_id = get_thread_id(history)
    
    result = agentic_action(message, thread_id)
    
    # Check for interrupt (HITL)
    if isinstance(result, dict) and result.get("type") == "interrupt":
        payload = result["payload"][0]["value"]  # First interrupt
        
        # Return approval request to UI
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
    """Handle user approval response."""
    thread_id = get_thread_id(history)
    decision = response.lower() in ["approve", "yes", "ok"]
    return resume_action(decision, thread_id)
```

### 4.3 Dead Code in dispatcher.py After Phase 4

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
| `app.py` | 4 | Enhance | +50 | 0 | ~20 |

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

**Recommended execution**: Implement Phase 1 first (no breaking changes), test retrieval quality, then decide on Phase 2-4 based on results.