# LangGraph Integration Feasibility Analysis

**Created**: 2026-04-25
**Scope**: Evaluate LangGraph integration for Self-Reflective RAG and Human-in-the-Loop features

---

## Executive Summary

| Feature | Feasibility | Recommendation | Effort |
|---------|-------------|----------------|--------|
| **Self-Reflective RAG** | ✅ HIGH | Implement with StateGraph | 2-3 days |
| **Human-in-the-Loop** | ✅ HIGH | Use `interrupt()` for tool approval | 1-2 days |
| **Stateful Memory Integration** | ✅ MEDIUM | Replace sliding window with LangGraph state | 1 day |

**Overall**: LangGraph 1.0.10 is already installed and provides all required capabilities. Migration from manual ReAct loop is straightforward.

---

## Current Architecture Analysis

### Manual ReAct Loop (architect.py:266-301)

```python
# Current implementation
tool_iterations = 0
while response.tool_calls:
    tool_iterations += 1
    messages.append(response)
    for tool_call in response.tool_calls:
        tool_result = self._tool_map.get(tool_name).invoke(tool_args)
        messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
    response = self._llm_with_tools.invoke(messages)
```

**Limitations:**
1. No reflection/validation after tool execution
2. No ability to pause for user approval
3. Manual state management (sliding window separate from execution)
4. No automatic retry on irrelevant retrieval

### Current Memory System

| Component | Implementation | LangGraph Equivalent |
|-----------|----------------|---------------------|
| Sliding window | `SlidingWindowMemory` class | StateGraph `messages` field + `add_messages` reducer |
| Vector retrieval | `_build_augmented_message()` | Separate `retrieve` node |
| Tool execution | Inline in ReAct loop | `tool_node` with `interrupt()` support |

---

## Feature 1: Self-Reflective RAG (Self-RAG)

### Concept

```
┌─────────────────────────────────────────────────────────────────┐
│  Self-RAG Graph Flow                                            │
│                                                                 │
│  START ──► retrieve ──► grade_documents ──┬──► generate_answer  │
│                         │                  │                    │
│                         │                  └──► rewrite_query   │
│                         │                         │             │
│                         └──► (all irrelevant?) ───┘             │
│                                                                 │
│  grade_documents:                                                │
│    - LLM grades each chunk: "yes" / "no" relevance              │
│    - If any relevant → generate_answer                          │
│    - If all irrelevant → rewrite_query → retrieve again         │
│                                                                 │
│  hallucination_check (optional):                                │
│    - After generate, verify answer grounded in context          │
│    - If not grounded → regenerate with stricter prompt          │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph Implementation Pattern

Based on Context7 documentation, the pattern is:

```python
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

# ── State Schema ─────────────────────────────────────────────────────

class SelfRAGState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_docs: list[Document]  # Retrieved chunks
    relevant_docs: list[Document]   # Graded as relevant
    query_iterations: int           # Track retry count
    hallucination_score: str | None # "yes"/"no" from generation check

# ── Document Grading (from Context7) ────────────────────────────────

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, "
    "grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant."
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def grade_documents(state: SelfRAGState) -> Literal["generate", "rewrite"]:
    """Determine whether retrieved documents are relevant."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = llm.with_structured_output(GradeDocuments).invoke([{"role": "user", "content": prompt}])

    if response.binary_score == "yes":
        return "generate"
    else:
        return "rewrite"

# ── Query Rewriting ──────────────────────────────────────────────────

REWRITE_PROMPT = (
    "You are a query rewriter. The original query returned irrelevant results.\n"
    "Original query: {query}\n"
    "Improve the query to better match relevant documents. Focus on key concepts."
)

def rewrite_query(state: SelfRAGState) -> dict:
    """Rewrite the query for better retrieval."""
    original_query = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(query=original_query)
    rewritten = llm.invoke([{"role": "user", "content": prompt}])
    return {
        "messages": [HumanMessage(content=rewritten.content)],
        "query_iterations": state["query_iterations"] + 1,
    }

# ── Graph Construction ───────────────────────────────────────────────

builder = StateGraph(SelfRAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_documents)
builder.add_node("generate", generate_node)
builder.add_node("rewrite", rewrite_query)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", grade_documents)
builder.add_edge("generate", END)
builder.add_edge("rewrite", "retrieve")  # Retry loop

graph = builder.compile()
```

### Integration with Current Vector Memory

The existing `_build_augmented_message()` logic becomes a `retrieve` node:

```python
def retrieve_node(state: SelfRAGState) -> dict:
    """Retrieve relevant chunks from ChromaDB using MMR."""
    query = state["messages"][-1].content

    # Use existing VectorMemory MMR search
    vm = get_vector_memory()
    results = vm.similarity_search_mmr_with_score(query, k=10, fetch_k=30)

    # Apply existing threshold filter
    relevant = [(doc, score) for doc, score in results if score < 0.85]

    return {
        "retrieved_docs": [doc for doc, _ in relevant],
        "messages": [ToolMessage(content=vm.format_context([doc for doc, _ in relevant]), tool_call_id="retrieve")]
    }
```

---

## Feature 2: Human-in-the-Loop (HITL)

### Concept

```
┌─────────────────────────────────────────────────────────────────┐
│  HITL Workflow                                                  │
│                                                                 │
│  START ──► agent ──► tool_decision ──┬──► (no tool) ──► END     │
│                   │                  │                          │
│                   │                  └──► approve_tool           │
│                   │                         │                   │
│                   │         interrupt() ───► [PAUSE]            │
│                   │                         │                   │
│                   │         user response   │                   │
│                   │         Command(resume) │                   │
│                   │                         ▼                   │
│                   │                  approved? ──┬──► execute    │
│                   │                              │              │
│                   │                              └──► cancel     │
│                   │                                     │       │
│                   └────────────────────────────────────── END   │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph `interrupt()` Pattern

From Context7 documentation:

```python
from langgraph.types import interrupt, Command

@tool
def update_document(filename: str, content: str) -> str:
    """Update an existing document with user approval."""

    # Pause before execution - payload surfaces in result["__interrupt__"]
    response = interrupt({
        "action": "update_document",
        "filename": filename,
        "content_preview": content[:200] + "...",  # Show preview
        "message": "Approve updating this document?",
    })

    if response.get("action") == "approve":
        # Actually execute the update
        # ... existing update logic ...
        return f"Document updated: {filename}"
    else:
        return "Update cancelled by user"
```

### Approval Node Implementation

Alternative: dedicated approval node before tool execution:

```python
class ApprovalState(TypedDict):
    tool_name: str
    tool_args: dict
    status: Literal["pending", "approved", "rejected"] | None

def approval_node(state: ApprovalState) -> Command[Literal["execute", "cancel"]]:
    """Request user approval before tool execution."""
    decision = interrupt({
        "question": f"Approve {state['tool_name']}?",
        "tool_args": state["tool_args"],
    })

    if decision:
        return Command(goto="execute")
    else:
        return Command(goto="cancel")
```

### Checkpointer Requirement

HITL requires a checkpointer to persist state during pause:

```python
from langgraph.checkpoint.memory import MemorySaver
# OR for production:
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = MemorySaver()  # In-memory for testing
graph = builder.compile(checkpointer=checkpointer)

# Invoke with thread_id for state persistence
config = {"configurable": {"thread_id": "session-123"}}
result = graph.invoke(input, config=config)

# When paused, result contains __interrupt__
print(result["__interrupt__"])  # -> [Interrupt(value={...})]

# Resume with approval
resumed = graph.invoke(Command(resume=True), config=config)
```

---

## Feature 3: Stateful Memory Integration

### State Schema Design

Replace `SlidingWindowMemory` with LangGraph state:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, BaseMessage

class ArchitectState(TypedDict):
    # Conversation memory (replaces SlidingWindowMemory)
    messages: Annotated[list[AnyMessage], add_messages]

    # RAG context
    retrieved_chunks: list[Document]
    rag_tokens_used: int

    # Tool execution tracking
    pending_tool: str | None
    pending_tool_args: dict | None
    tool_iterations: int

    # Session metadata
    thread_id: str
    user_id: str | None
```

### Migration from SlidingWindowMemory

| Current | LangGraph Equivalent |
|---------|---------------------|
| `memory.add_user_message()` | Node returns `{"messages": [HumanMessage(...)]}` |
| `memory.add_ai_message()` | Node returns `{"messages": [AIMessage(...)]}` |
| `memory.get_messages()` | Access `state["messages"]` directly |
| `_trim()` sliding window | Use `add_messages` reducer (handles append automatically) |

**Note**: LangGraph's `add_messages` reducer doesn't auto-trim. Add a `prune_messages` node:

```python
MAX_MESSAGES = 11  # System + 5 exchanges (10 messages)

def prune_messages(state: ArchitectState) -> dict:
    """Keep only recent messages (sliding window behavior)."""
    messages = state["messages"]
    if len(messages) > MAX_MESSAGES:
        # Keep system message + last N messages
        system = messages[0] if isinstance(messages[0], SystemMessage) else None
        recent = messages[-MAX_MESSAGES+1:] if system else messages[-MAX_MESSAGES:]
        pruned = [system] + recent if system else recent
        return {"messages": pruned}
    return {}
```

---

## Proposed Architecture

### Graph Structure

```python
# agent/langgraph_architect.py

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

class ArchitectState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved_chunks: list[Document]
    query_iterations: int
    pending_tool: dict | None  # {"name": str, "args": dict}
    tool_approved: bool | None

# ── Nodes ────────────────────────────────────────────────────────────

def retrieve_node(state: ArchitectState) -> dict:
    """RAG retrieval with MMR + threshold filtering."""
    query = state["messages"][-1].content
    vm = get_vector_memory()

    if not vm.has_documents():
        return {"retrieved_chunks": []}

    results = vm.similarity_search_mmr_with_score(query, k=10, fetch_k=30)
    relevant = [doc for doc, score in results if score < 0.85]

    return {"retrieved_chunks": relevant, "query_iterations": 0}

def grade_retrieval_node(state: ArchitectState) -> Literal["agent", "rewrite"]:
    """Grade retrieved chunks for relevance."""
    if not state["retrieved_chunks"]:
        return "rewrite"

    # Use LLM to grade overall relevance
    question = state["messages"][0].content
    context = "\n".join(doc.page_content for doc in state["retrieved_chunks"])

    response = llm.with_structured_output(GradeDocuments).invoke([
        {"role": "user", "content": GRADE_PROMPT.format(question=question, context=context)}
    ])

    if response.binary_score == "yes":
        return "agent"
    elif state["query_iterations"] >= 2:
        return "agent"  # Stop retrying after 2 attempts
    else:
        return "rewrite"

def rewrite_query_node(state: ArchitectState) -> dict:
    """Rewrite query for better retrieval."""
    original = state["messages"][0].content
    rewritten = llm.invoke([
        {"role": "user", "content": REWRITE_PROMPT.format(query=original)}
    ])
    return {
        "messages": [HumanMessage(content=rewritten.content)],
        "query_iterations": state["query_iterations"] + 1,
    }

def agent_node(state: ArchitectState) -> dict:
    """LLM with tools, using retrieved context."""
    # Build augmented message (same as current _build_augmented_message)
    context = format_retrieved_chunks(state["retrieved_chunks"])
    augmented = f"RELEVANT CONTEXT:\n{context}\n\nUSER REQUEST:\n{state['messages'][-1].content}"

    messages = state["messages"]
    messages[-1] = HumanMessage(content=augmented)

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        # Store pending tool for approval
        return {
            "messages": [response],
            "pending_tool": {"name": response.tool_calls[0]["name"], "args": response.tool_calls[0]["args"]}
        }
    else:
        return {"messages": [response], "pending_tool": None}

def approval_node(state: ArchitectState) -> Command[Literal["execute_tool", "cancel_tool"]]:
    """Request user approval for sensitive tools (update_document, save_document)."""
    tool = state["pending_tool"]

    # Only require approval for write operations
    if tool["name"] in ["update_document", "save_document"]:
        decision = interrupt({
            "action": tool["name"],
            "args": tool["args"],
            "content_preview": tool["args"].get("content", "")[:200] + "...",
            "message": f"Approve {tool['name']}?",
        })
        if decision:
            return Command(goto="execute_tool")
        else:
            return Command(goto="cancel_tool")
    else:
        # list_documents doesn't need approval
        return Command(goto="execute_tool")

def execute_tool_node(state: ArchitectState) -> dict:
    """Execute approved tool."""
    tool = state["pending_tool"]
    result = tool_map[tool["name"]].invoke(tool["args"])
    return {
        "messages": [ToolMessage(content=str(result), tool_call_id=tool["id"])],
        "pending_tool": None,
    }

def cancel_tool_node(state: ArchitectState) -> dict:
    """Handle cancelled tool."""
    return {
        "messages": [AIMessage(content="Action cancelled by user.")],
        "pending_tool": None,
    }

def prune_messages_node(state: ArchitectState) -> dict:
    """Sliding window behavior."""
    # ... trim logic ...

# ── Graph Assembly ───────────────────────────────────────────────────

builder = StateGraph(ArchitectState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_retrieval_node)
builder.add_node("rewrite", rewrite_query_node)
builder.add_node("agent", agent_node)
builder.add_node("approval", approval_node)
builder.add_node("execute_tool", execute_tool_node)
builder.add_node("cancel_tool", cancel_tool_node)
builder.add_node("prune", prune_messages_node)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", grade_retrieval_node)
builder.add_edge("rewrite", "retrieve")
builder.add_edge("grade", "agent")
builder.add_edge("agent", "approval")
builder.add_conditional_edges("approval", lambda s: "execute_tool" if s["tool_approved"] else "cancel_tool")
builder.add_edge("execute_tool", "agent")  # Continue ReAct loop
builder.add_edge("cancel_tool", END)
builder.add_edge("prune", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

---

## Migration Complexity Analysis

### Code Changes Required

| File | Change | Effort |
|------|--------|--------|
| `agent/architect.py` | Replace manual ReAct with LangGraph graph | Major |
| `agent/memory.py` | Replace with LangGraph state or keep as wrapper | Minor |
| `agent/tools.py` | Add `interrupt()` to `update_document`, `save_document` | Minor |
| `agent/dispatcher.py` | Add graph invocation + resume handling | Moderate |
| **New file** `agent/langgraph_architect.py` | StateGraph definition | Major |

### Backward Compatibility

| Component | Compatibility Strategy |
|-----------|------------------------|
| Gradio UI | Add approval modal for `__interrupt__` results |
| CLI | Add interactive prompt for approval |
| API | Return interrupt payload, accept resume via new endpoint |

### Migration Steps

```
Phase 1: Self-RAG (No UI changes)
  1. Create langgraph_architect.py with retrieve → grade → generate flow
  2. Test with existing dispatcher integration
  3. Compare retrieval quality vs current

Phase 2: HITL (UI changes required)
  4. Add interrupt() to tools
  5. Update dispatcher to handle __interrupt__
  6. Update app.py Gradio UI for approval modal
  7. Add resume endpoint

Phase 3: Full Integration
  8. Replace sliding window with LangGraph state
  9. Add checkpointer for production persistence
  10. Remove old architect.py ReAct code
```

---

## Trade-offs Analysis

### Capabilities Gained

| Feature | Before | After |
|---------|--------|-------|
| Retrieval validation | None (blind trust) | LLM grading + retry |
| Query improvement | None | Automatic rewriting |
| Tool approval | None | Human approval before writes |
| State management | Manual (sliding window) | Built-in (StateGraph) |
| Persistence | RAM only | Checkpointer (disk) |
| Observability | Logging only | Built-in tracing |

### Complexity Added

| Aspect | Current | With LangGraph |
|--------|---------|----------------|
| Lines of code | ~150 (architect.py) | ~200-250 (graph + nodes) |
| Dependencies | None new | Already installed |
| Learning curve | Known patterns | New concepts (StateGraph, interrupt) |
| Debugging | Simple loop | Graph traversal |
| Testing | Unit tests | Graph integration tests |

### Performance Impact

| Metric | Current | With LangGraph |
|--------|---------|----------------|
| Retrieval latency | 50-100ms | Same (MMR unchanged) |
| Grade latency | N/A | +100-200ms (LLM call) |
| Rewrite latency | N/A | +200-300ms (LLM call) |
| Approval latency | N/A | User-dependent (pause) |
| Memory overhead | Low | Higher (checkpointer) |

**Recommendation**: Accept +300-500ms for Self-RAG quality improvement. HITL latency is user-controlled (pause).

---

## Implementation Recommendation

### Decision Matrix

| Feature | Implement? | Priority | When |
|---------|------------|----------|------|
| Self-RAG (retrieve-grade-rewrite) | ✅ YES | High | Phase 1 (Week 1) |
| HITL (interrupt for tools) | ✅ YES | Medium | Phase 2 (Week 2) |
| State replacement | ⚠️ OPTIONAL | Low | Phase 3 (if needed) |

### Minimal Implementation (Recommended)

Start with Self-RAG only, keep current memory:

```python
# Minimal LangGraph integration
builder = StateGraph(MessagesState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("generate", generate_node)
builder.add_node("rewrite", rewrite_node)

# Keep existing SlidingWindowMemory
# Pass messages from memory to graph, update memory after
```

This adds reflection without changing the UI.

### Full Implementation (Later)

Add HITL after UI supports approval modal:

```python
# Add interrupt to tools
@tool
def update_document(filename: str, content: str) -> str:
    response = interrupt({...})
    if response.get("approve"):
        # execute
```

---

## Summary

| Question | Answer |
|----------|--------|
| Can LangGraph integrate with current code? | ✅ YES - already installed, compatible with LangChain |
| Self-RAG feasible? | ✅ YES - Context7 pattern matches requirements |
| HITL feasible? | ✅ YES - `interrupt()` + `Command(resume)` pattern documented |
| Migration complexity? | Medium - 2-3 days for Self-RAG, 1-2 days for HITL |
| Recommended approach? | Phase 1: Self-RAG first (no UI changes), Phase 2: HITL with UI updates |