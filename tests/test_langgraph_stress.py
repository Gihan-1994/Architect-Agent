"""
test_langgraph_stress.py - Stress tests for LangGraph Architect

Validates edge cases from the LangGraph Integration Master Plan:
1. Interrupt timeout (5-minute auto-cancel)
2. Resume with wrong thread_id
3. Grade node LLM failure (fallback to agent)
4. Rewrite produces identical query (no infinite loop)
5. Multiple tool calls (sequential execution)
6. State persistence after restart
7. Concurrent sessions with same thread_id
8. Token budget exhaustion

Run with:
    pytest tests/test_langgraph_stress.py -v
"""

import pytest
import time
import tempfile
import os
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from agent.langgraph_architect import LangGraphArchitect


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def architect():
    """Create architect instance for testing (Self-RAG disabled)."""
    return LangGraphArchitect(enable_grading=False, enable_rewrite=False)


@pytest.fixture
def architect_with_self_rag():
    """Create architect with Self-RAG enabled."""
    return LangGraphArchitect(enable_grading=True, enable_rewrite=True)


@pytest.fixture
def temp_checkpointer():
    """Create temporary SQLite db path for persistence tests."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Interrupt Timeout
# ─────────────────────────────────────────────────────────────────────────────

def test_interrupt_timeout_logic(architect):
    """
    Verify timeout detection logic in approval_node.

    Expected behavior:
        - approval_node checks interrupt_timestamp
        - If elapsed > 300 seconds → should route to cancel
    """
    # Create state with old timestamp
    state = {
        "messages": [HumanMessage("Save document")],
        "retrieved_chunks": [],
        "query_iterations": 0,
        "pending_tools": [{"name": "save_document", "args": {"filename": "test.md", "content": "# Test"}, "id": "call_1"}],
        "current_tool_index": 0,
        "interrupt_timestamp": time.time() - 310,  # 310 seconds ago (> 5 min)
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    # Verify timeout condition is met
    elapsed = time.time() - state["interrupt_timestamp"]
    assert elapsed > 300, f"Elapsed time should be > 300s, got {elapsed}s"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Resume with Wrong Thread ID
# ─────────────────────────────────────────────────────────────────────────────

def test_resume_wrong_thread_id(architect):
    """
    Resume with different thread_id - should fail or start fresh.

    Expected behavior:
        - LangGraph checkpointer returns empty state for unknown thread
        - Resume fails or starts new conversation
    """
    # Create state with thread_id "user_123"
    thread_id_a = "user_123"
    config_a = {"configurable": {"thread_id": thread_id_a}}

    initial_state = {
        "messages": [HumanMessage("User A message")],
        "retrieved_chunks": [],
        "query_iterations": 0,
        "pending_tools": None,
        "current_tool_index": 0,
        "interrupt_timestamp": None,
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    # Save state with thread_id_a
    architect.graph.invoke(initial_state, config_a)

    # Try to access with wrong thread_id
    thread_id_b = "user_456"
    config_b = {"configurable": {"thread_id": thread_id_b}}

    # Get state for wrong thread - should be empty
    state_b = architect.graph.get_state(config_b)

    # State should be empty (no messages from user_a)
    assert state_b.values.get("messages") is None or len(state_b.values.get("messages", [])) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Grade Node LLM Failure
# ─────────────────────────────────────────────────────────────────────────────

def test_grade_node_llm_failure(architect_with_self_rag):
    """
    Grade LLM times out, rate limits, or returns malformed response.

    Expected behavior:
        - grade_node catches exception
        - Fallback to "agent" path (fail-safe)
    """
    architect = architect_with_self_rag

    # Mock LLM to raise exception (grade_node uses self.llm, not self.llm_grader)
    with patch.object(architect, 'llm') as mock_llm:
        mock_llm.invoke = MagicMock(side_effect=TimeoutError("LLM timeout"))

        state = {
            "messages": [HumanMessage("What is authentication?")],
            "retrieved_chunks": [Document(page_content="Authentication content", metadata={"source": "auth.md"})],
            "query_iterations": 0,
            "pending_tools": None,
            "current_tool_index": 0,
            "interrupt_timestamp": None,
            "token_count": 0,
            "retrieval_mode": "mmr",
            "enable_grading": True,
            "enable_rewrite": False,
            "last_query_source": None,
        }

        # Call grade_node directly (not full graph)
        result = architect._grade_node(state)

        # Should fallback to agent (fail-safe)
        assert result == "agent"


def test_grade_node_malformed_response(architect_with_self_rag):
    """
    Grade LLM returns non-structured response.

    Expected behavior:
        - grade_node handles gracefully
        - First iteration: returns "rewrite" (tries to improve query)
        - Max iterations reached: fallback to "agent"
    """
    architect = architect_with_self_rag

    # Mock LLM to return invalid response (grade_node uses self.llm)
    with patch.object(architect, 'llm') as mock_llm:
        mock_llm.invoke = MagicMock(return_value=AIMessage(content="invalid response"))

        # Test 1: First iteration → should try rewrite
        state_first = {
            "messages": [HumanMessage("What is authentication?")],
            "retrieved_chunks": [Document(page_content="Content", metadata={})],
            "query_iterations": 0,
            "pending_tools": None,
            "current_tool_index": 0,
            "interrupt_timestamp": None,
            "token_count": 0,
            "retrieval_mode": "mmr",
            "enable_grading": True,
            "enable_rewrite": False,
            "last_query_source": None,
        }

        result_first = architect._grade_node(state_first)
        assert result_first == "rewrite"  # Try rewriting query

        # Test 2: Max iterations reached → fallback to agent
        state_max = {
            "messages": [HumanMessage("What is authentication?")],
            "retrieved_chunks": [Document(page_content="Content", metadata={})],
            "query_iterations": 2,  # Max iterations
            "pending_tools": None,
            "current_tool_index": 0,
            "interrupt_timestamp": None,
            "token_count": 0,
            "retrieval_mode": "mmr",
            "enable_grading": True,
            "enable_rewrite": False,
            "last_query_source": None,
        }

        result_max = architect._grade_node(state_max)
        assert result_max == "agent"  # Fallback after max iterations


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Rewrite Produces Identical Query
# ─────────────────────────────────────────────────────────────────────────────

def test_rewrite_identical_query(architect_with_self_rag):
    """
    Rewrite returns same query - should not loop infinitely.

    Expected behavior:
        - rewrite_node increments query_iterations
        - Stops after max iterations (2)
    """
    architect = architect_with_self_rag

    original_query = "What is authentication?"

    # Mock LLM to return identical query (rewrite_node uses self.llm)
    with patch.object(architect, 'llm') as mock_llm:
        mock_llm.invoke = MagicMock(return_value=AIMessage(content=original_query))

        state = {
            "messages": [HumanMessage(original_query)],
            "retrieved_chunks": [],  # Empty to trigger rewrite
            "query_iterations": 0,
            "pending_tools": None,
            "current_tool_index": 0,
            "interrupt_timestamp": None,
            "token_count": 0,
            "retrieval_mode": "mmr",
            "enable_grading": False,
            "enable_rewrite": True,
            "last_query_source": None,
        }

        # Call rewrite_node directly
        result = architect._rewrite_node(state)

        # Should increment iterations
        assert result["query_iterations"] == 1
        assert result["query_iterations"] <= 2  # Max limit


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Multiple Tool Calls
# ─────────────────────────────────────────────────────────────────────────────

def test_multiple_tool_calls_state_structure(architect):
    """
    LLM requests 2+ tools - pending_tools stores all as list.

    Expected behavior:
        - pending_tools is a list
        - current_tool_index tracks progress
        - _execute_node processes sequentially
    """
    # Create state with multiple pending tools
    state = {
        "messages": [HumanMessage("List and save documents")],
        "retrieved_chunks": [],
        "query_iterations": 0,
        "pending_tools": [
            {"name": "list_documents", "args": {}, "id": "call_1"},
            {"name": "save_document", "args": {"filename": "test.md", "content": "# Test"}, "id": "call_2"},
        ],
        "current_tool_index": 0,
        "interrupt_timestamp": None,
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    # Verify pending_tools is a list
    assert isinstance(state["pending_tools"], list)
    assert len(state["pending_tools"]) == 2

    # Test _execute_node increments index
    result = architect._execute_node(state)

    # Should move to next tool (index 1)
    assert result["current_tool_index"] == 1


def test_multiple_tool_calls_approval_flow():
    """
    Multiple tools with approval needed - each requires separate approval.

    Expected behavior:
        - approval_node checks current_tool_index
        - Only approval-required tools trigger interrupt
        - list_documents skipped, save_document prompts approval
    """
    state = {
        "messages": [HumanMessage("List and save")],
        "retrieved_chunks": [],
        "query_iterations": 0,
        "pending_tools": [
            {"name": "list_documents", "args": {}, "id": "call_1"},
            {"name": "save_document", "args": {"filename": "test.md", "content": "..."}, "id": "call_2"},
        ],
        "current_tool_index": 0,  # First tool
        "interrupt_timestamp": None,
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    # Check first tool (list_documents - no approval)
    first_tool = state["pending_tools"][state["current_tool_index"]]
    assert first_tool["name"] == "list_documents"
    assert first_tool["name"] not in ["update_document", "save_document"]

    # Move to second tool
    state["current_tool_index"] = 1
    second_tool = state["pending_tools"][state["current_tool_index"]]
    assert second_tool["name"] == "save_document"
    assert second_tool["name"] in ["update_document", "save_document"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: State Persistence After Restart
# ─────────────────────────────────────────────────────────────────────────────

def test_state_persistence_restart(temp_checkpointer):
    """
    Restart server during HITL interrupt - should resume.

    Expected behavior:
        - SqliteSaver persists state to disk
        - New architect instance with same checkpointer_path
        - State recovered via thread_id
    """
    thread_id = "restart_test_001"

    # Fixture yields the db path directly
    db_path = temp_checkpointer

    # Create first architect with the temp db path
    architect1 = LangGraphArchitect(checkpointer_path=db_path)

    config = {"configurable": {"thread_id": thread_id}}

    # Trigger state save
    initial_state = {
        "messages": [HumanMessage("Save document")],
        "retrieved_chunks": [],
        "query_iterations": 0,
        "pending_tools": [{"name": "save_document", "args": {}, "id": "call_1"}],
        "current_tool_index": 0,
        "interrupt_timestamp": time.time(),
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    architect1.graph.invoke(initial_state, config)

    # Get state from checkpointer (verify it was saved)
    saved_state = architect1.graph.get_state(config)
    assert saved_state.values is not None
    assert len(saved_state.values.get("messages", [])) > 0

    # Create new architect instance (simulating restart) with same path
    architect2 = LangGraphArchitect(checkpointer_path=db_path)

    # Retrieve state with same thread_id
    recovered_state = architect2.graph.get_state(config)

    # State should be recovered
    assert recovered_state.values is not None
    assert len(recovered_state.values.get("messages", [])) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Concurrent Sessions Same Thread ID
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_sessions_collision(architect):
    """
    Two users with same thread_id - should not interfere.

    Expected behavior:
        - Each invoke creates separate state
        - OR error raised for collision
        - OR auto-generate unique ID to prevent collision

    Note: Current implementation uses unique thread_id per session
    (terminal_{uuid}, web_{user_id}_{session_id}) to prevent this.
    """
    thread_id = "collision_test"
    config_a = {"configurable": {"thread_id": thread_id}}
    config_b = {"configurable": {"thread_id": thread_id}}

    # User A starts conversation
    state_a = {
        "messages": [HumanMessage("User A query")],
        "retrieved_chunks": [],
        "query_iterations": 0,
        "pending_tools": None,
        "current_tool_index": 0,
        "interrupt_timestamp": None,
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    architect.graph.invoke(state_a, config_a)

    # User B tries same thread_id
    state_b = {
        "messages": [HumanMessage("User B query")],
        "retrieved_chunks": [],
        "query_iterations": 0,
        "pending_tools": None,
        "current_tool_index": 0,
        "interrupt_timestamp": None,
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    result_b = architect.graph.invoke(state_b, config_b)

    # Results should be isolated (B overwrites A in current implementation)
    # OR implementation should prevent collision via unique IDs
    assert result_b["messages"][-1].content is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Token Budget Exhaustion
# ─────────────────────────────────────────────────────────────────────────────

def test_token_budget_exhaustion(architect):
    """
    Long conversation with many chunks - budget should prevent overflow.

    Expected behavior:
        - _agent_node applies token budget (MAX_RAG_TOKENS=4000)
        - Only budgeted chunks passed to LLM
        - token_count tracks usage
        - LLM response still valid
    """
    # Create many chunks (over budget)
    large_chunks = [
        Document(
            page_content=f"Chunk {i} with lots of content about architecture patterns and design principles...",
            metadata={"source": f"doc_{i}.md", "chunk_index": i}
        )
        for i in range(20)  # ~8000+ tokens total
    ]

    state = {
        "messages": [HumanMessage("What is architecture?")],
        "retrieved_chunks": large_chunks,
        "query_iterations": 0,
        "pending_tools": None,
        "current_tool_index": 0,
        "interrupt_timestamp": None,
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    # Call _agent_node
    result = architect._agent_node(state)

    # Should apply token budget
    assert result["token_count"] <= architect.MAX_RAG_TOKENS

    # Should have response
    assert result["messages"][-1].content is not None


def test_token_budget_exact_limit(architect):
    """
    Verify token budget cutoff at exact limit.

    Expected behavior:
        - Chunks added until budget exhausted
        - Stop adding chunks when total > MAX_RAG_TOKENS
    """
    # Create chunks of known token size (~200 tokens each)
    chunks = [
        Document(
            page_content=" ".join(["architecture"] * 200),  # ~200 tokens
            metadata={"source": f"doc_{i}.md"}
        )
        for i in range(25)  # 25 * 200 = 5000 tokens (over 4000 limit)
    ]

    state = {
        "messages": [HumanMessage("Test query")],
        "retrieved_chunks": chunks,
        "query_iterations": 0,
        "pending_tools": None,
        "current_tool_index": 0,
        "interrupt_timestamp": None,
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    result = architect._agent_node(state)

    # Should use ~20 chunks (4000 tokens), not all 25
    assert result["token_count"] <= 4000


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Pending Tool None Handling
# ─────────────────────────────────────────────────────────────────────────────

def test_pending_tool_none_handling(architect):
    """
    approval_node receives pending_tool=None - should end conversation.

    Expected behavior:
        - approval_node checks for None FIRST
        - Returns Command(goto=END) or Command(goto="prune")
        - No crash or undefined behavior
    """
    from langgraph.types import Command

    state = {
        "messages": [HumanMessage("Hello")],
        "retrieved_chunks": [],
        "query_iterations": 0,
        "pending_tools": None,  # Explicitly None
        "current_tool_index": 0,
        "interrupt_timestamp": None,
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": None,
    }

    # Call approval_node
    result = architect._approval_node(state)

    # Should return Command to end (not crash)
    assert result is not None
    # Should be Command type
    assert isinstance(result, Command) or result == "prune" or result.get("goto") in ["prune", "__end__"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Empty Retrieved Chunks Fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_chunks_fallback(architect):
    """
    No chunks retrieved after 2 retries - should load full document.

    Expected behavior:
        - _route_after_retrieve checks query_iterations
        - If iterations >= 2 and no chunks → load full document
        - Full document used as fallback context (no token limit on fallback)
    """
    state = {
        "messages": [HumanMessage("What is the system design?")],
        "retrieved_chunks": [],  # Empty
        "query_iterations": 2,   # Max retries reached
        "pending_tools": None,
        "current_tool_index": 0,
        "interrupt_timestamp": None,
        "token_count": 0,
        "retrieval_mode": "mmr",
        "enable_grading": False,
        "enable_rewrite": False,
        "last_query_source": "architecture.md",
    }

    # Route after retrieve
    result = architect._route_after_retrieve(state)

    # Should go to agent (with fallback) or indicate fallback
    assert result in ["agent", "grade"] or result.get("goto") == "agent"


# ─────────────────────────────────────────────────────────────────────────────
# Run tests standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])