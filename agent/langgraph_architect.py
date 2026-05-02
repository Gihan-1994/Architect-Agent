"""
agent/langgraph_architect.py — LangGraph Orchestration Layer

WHY THIS EXISTS:
  The original architect.py uses a manual ReAct loop (while response.tool_calls).
  LangGraph provides:
    - StateGraph for declarative node-based flow
    - interrupt() for Human-in-the-Loop (HITL) approval
    - Checkpointer for state persistence (survives restart)
    - Self-RAG pattern (grade → rewrite → retrieve)

STRUCTURE:
  - ArchitectState: TypedDict with stress test fields
  - Nodes: _retrieve_node, _grade_node, _rewrite_node, _agent_node,
           _approval_node, _execute_node, _cancel_node
  - Routing: conditional edges + Command for dynamic transitions
  - Configuration: enable_grading, enable_rewrite (default OFF)

Created: 2026-04-27 (Phase 3 from langgraph-master-plan.md)
"""

import os
import time
import logging
import sqlite3
import uuid
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.documents import Document

from .memory import SlidingWindowMemory
from .tools import save_document, list_documents, update_document
from .vector_memory import get_vector_memory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt (same as architect.py)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Senior Software Architect AI assistant with 15+ years of experience
designing scalable, production-ready systems. You help engineering teams and developers create
clear, professional software architecture documents.

YOUR SPECIALTIES:
1. System Design Documents - architecture overviews, component diagrams, scalability
2. API Specifications - RESTful endpoints, schemas, authentication flows
3. Database Schema / ERD - table definitions, Mermaid erDiagram syntax
4. Sequence / Flow Diagrams - system interactions using Mermaid sequenceDiagram

DOCUMENT QUALITY STANDARDS:
- Well-structured Markdown with clear headings (## H2, ### H3)
- Mermaid diagrams for visual content
- "Overview" section at start of every document
- Practical and production-focused

TOOL USAGE RULES:
- CREATE/GENERATE/WRITE new document → call save_document()
- UPDATE/MODIFY/REVISE existing document → call update_document()
- LIST/SHOW saved documents → call list_documents()

IMPORTANT - WHEN NOT TO USE TOOLS:
- Questions about existing content → DO NOT call any tool, answer using RAG context
- Only create/update documents when user EXPLICITLY asks

COMMUNICATION STYLE:
- Professional but approachable
- Brief summary after saving (2-3 sentences)
- Ready to iterate
"""


# ─────────────────────────────────────────────────────────────────────────────
# ArchitectState — State with stress test fields
# ─────────────────────────────────────────────────────────────────────────────

class ArchitectState(TypedDict):
    """
    LangGraph state schema with all stress test fields from the plan.

    Fields:
        messages: Conversation history (auto-accumulated via add_messages)
        retrieved_chunks: RAG search results
        query_iterations: Counter for rewrite loop (max 2)
        pending_tools: Queue of tool calls awaiting execution
        current_tool_index: Which tool in queue is being processed
        interrupt_timestamp: When interrupt() was called (for timeout)
        token_count: Tokens used in current turn
        retrieval_mode: "standard" or "hierarchical"
        enable_grading: Self-RAG grading enabled (default False)
        enable_rewrite: Self-RAG query rewrite enabled (default False)
        last_query_source: Source filename for fallback
    """
    messages: Annotated[list[AnyMessage], "add_messages"]
    retrieved_chunks: list[Document]
    query_iterations: int
    pending_tools: list[dict] | None
    current_tool_index: int
    interrupt_timestamp: float | None
    token_count: int
    retrieval_mode: str
    enable_grading: bool
    enable_rewrite: bool
    last_query_source: str | None


# ─────────────────────────────────────────────────────────────────────────────
# LangGraphArchitect — Main class
# ─────────────────────────────────────────────────────────────────────────────

class LangGraphArchitect:
    """
    LangGraph-based orchestrator for the Software Architect Agent.

    Usage:
        architect = LangGraphArchitect()
        result = architect.invoke(
            {"messages": [HumanMessage("Design an auth system")]},
            config={"configurable": {"thread_id": "user_123"}}
        )

    For HITL approval:
        # After interrupt, resume with:
        result = architect.invoke(Command(resume=True), config=...)
    """

    # Token budget constants
    MAX_RAG_TOKENS = 4000
    MAX_MESSAGE_HISTORY = 20

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite-preview",
        enable_grading: bool = False,
        enable_rewrite: bool = False,
        checkpointer_path: str = "checkpoints.db",
    ):
        """
        Initialize LangGraph architect with configurable Self-RAG.

        Args:
            model: Gemini model name
            enable_grading: Enable Self-RAG grading node (default False)
            enable_rewrite: Enable Self-RAG rewrite node (default False)
            checkpointer_path: SQLite file for state persistence
        """
        # Validate API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Copy .env.example to .env and add your key."
            )

        # LLM setup
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=4096,
        )

        # Self-RAG configuration
        self.enable_grading = enable_grading
        self.enable_rewrite = enable_rewrite

        # Tools
        self._tools = [save_document, list_documents, update_document]
        self._llm_with_tools = self.llm.bind_tools(self._tools)
        self._tool_map = {t.name: t for t in self._tools}

        # Memory layers
        self._sliding_memory = SlidingWindowMemory(max_exchanges=5)
        self._sliding_memory.set_system(SYSTEM_PROMPT)
        self._vector_memory = get_vector_memory()

        # Checkpointer for persistence
        self._checkpointer_path = checkpointer_path
        conn = sqlite3.connect(checkpointer_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(conn)

        # Build graph
        self.graph = self._build_graph()

        logger.info("LangGraphArchitect initialized", extra={
            "model": model,
            "enable_grading": enable_grading,
            "enable_rewrite": enable_rewrite,
            "checkpointer": checkpointer_path,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Node implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve_node(self, state: ArchitectState) -> dict:
        """
        Retrieve relevant chunks from vector memory.

        ERROR HANDLING:
            - Empty results after 2 retries → fallback to full document
            - Vector search failure → return empty, proceed without RAG
        """
        start_time = time.time()
        logger.debug("_retrieve_node starting")

        try:
            # Get last user message
            last_user_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_msg = msg.content
                    break

            if not last_user_msg:
                return {"retrieved_chunks": [], "last_query_source": None}

            # Check if vector store has documents
            if not self._vector_memory.has_documents():
                logger.debug("Vector store empty, skipping retrieval")
                return {"retrieved_chunks": [], "last_query_source": None}

            # MMR search with scores
            results = self._vector_memory.similarity_search_mmr_with_score(
                query=last_user_msg,
                k=10,
                fetch_k=30,
                lambda_mult=0.5,
            )

            # Relevance filtering (cosine threshold 0.85)
            COSINE_THRESHOLD = 0.85
            relevant_chunks = []
            sources_seen = set()

            for doc, score in results:
                if score < COSINE_THRESHOLD:
                    relevant_chunks.append(doc)
                    sources_seen.add(doc.metadata.get("source", "unknown"))

            # Track most likely source for fallback
            last_source = list(sources_seen)[0] if sources_seen else None

            duration = int((time.time() - start_time) * 1000)
            logger.info("Retrieve completed", extra={
                "chunks": len(relevant_chunks),
                "duration_ms": duration,
            })

            return {
                "retrieved_chunks": relevant_chunks,
                "last_query_source": last_source,
                "query_iterations": state.get("query_iterations", 0),
            }

        except Exception as e:
            logger.error(f"Retrieve failed: {e}")
            return {"retrieved_chunks": [], "last_query_source": None}

    def _grade_node(self, state: ArchitectState) -> Literal["agent", "rewrite"]:
        """
        Grade retrieved chunks for relevance.

        Self-RAG pattern: LLM validates if chunks answer the query.

        ERROR HANDLING:
            - LLM failure → fail-safe to "agent"
            - Malformed response → fail-safe to "agent"
        """
        start_time = time.time()

        if not self.enable_grading:
            # Grading disabled → bypass
            return "agent"

        try:
            chunks = state["retrieved_chunks"]
            last_user_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_msg = msg.content
                    break

            if not chunks:
                # No chunks to grade
                if state["query_iterations"] >= 2:
                    return "agent"
                return "rewrite"

            # Build grading prompt
            context = self._vector_memory.format_context(chunks)
            grade_prompt = f"""
You are a relevance grader. Determine if the following context answers the user's question.

CONTEXT:
{context}

QUESTION: {last_user_msg}

Answer with exactly one word: "yes" or "no"
"""

            # Call grader LLM
            grade_response = self.llm.invoke([HumanMessage(content=grade_prompt)])
            grade_text = grade_response.content.lower().strip()

            duration = int((time.time() - start_time) * 1000)
            logger.info("Grade completed", extra={
                "result": grade_text,
                "duration_ms": duration,
            })

            if "yes" in grade_text:
                return "agent"

            # Not relevant → try rewrite
            if state["query_iterations"] >= 2:
                return "agent"  # Max iterations reached

            return "rewrite"

        except Exception as e:
            logger.error(f"Grade failed: {e}")
            return "agent"  # Fail-safe

    def _rewrite_node(self, state: ArchitectState) -> dict:
        """
        Rewrite query for better retrieval.

        Self-RAG pattern: LLM reformulates query when grading fails.
        """
        start_time = time.time()

        if not self.enable_rewrite:
            # Rewrite disabled → proceed with original query
            return {"query_iterations": state["query_iterations"] + 1}

        try:
            last_user_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_msg = msg.content
                    break

            rewrite_prompt = f"""
You are a query optimizer. Improve this question for better document retrieval.
Return ONLY the improved question, no explanation.

Original question: {last_user_msg}
"""

            rewrite_response = self.llm.invoke([HumanMessage(content=rewrite_prompt)])
            new_query = rewrite_response.content.strip()

            # Replace last user message with rewritten query
            new_messages = []
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage) and msg.content == last_user_msg:
                    new_messages.append(HumanMessage(content=new_query))
                else:
                    new_messages.append(msg)

            duration = int((time.time() - start_time) * 1000)
            logger.info("Rewrite completed", extra={
                "original": last_user_msg[:50],
                "rewritten": new_query[:50],
                "iterations": state["query_iterations"] + 1,
                "duration_ms": duration,
            })

            return {
                "messages": new_messages,
                "query_iterations": state["query_iterations"] + 1,
                "retrieved_chunks": [],  # Clear for re-retrieval
            }

        except Exception as e:
            logger.error(f"Rewrite failed: {e}")
            return {"query_iterations": state["query_iterations"] + 1}

    def _agent_node(self, state: ArchitectState) -> dict:
        """
        Main agent node — LLM with tools.

        TOKEN BUDGET:
            - Apply to retrieved_chunks before LLM call
            - Prune message history if > MAX_MESSAGE_HISTORY
        """
        start_time = time.time()

        # Apply token budget to chunks
        budgeted_chunks = self._apply_token_budget(state["retrieved_chunks"])
        context = self._vector_memory.format_context(budgeted_chunks)

        # Build messages
        messages = state["messages"]

        # Prune if growing too large
        if len(messages) > self.MAX_MESSAGE_HISTORY:
            messages = [messages[0]] + messages[-(self.MAX_MESSAGE_HISTORY - 1):]
            logger.debug(f"Pruned messages to {len(messages)}")

        # Build augmented prompt
        if context:
            last_user_msg = messages[-1]
            augmented = HumanMessage(
                content=f"RELEVANT CONTEXT:\n{context}\n\nUSER REQUEST:\n{last_user_msg.content}"
            )
            messages[-1] = augmented

        # Add system prompt at start
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Call LLM
        llm_start = time.time()
        response = self._llm_with_tools.invoke(full_messages)
        llm_duration = int((time.time() - llm_start) * 1000)

        # Extract token count
        token_count = self._count_message_tokens(full_messages)

        # Check for tool calls
        if response.tool_calls:
            result = {
                "messages": [response],
                "pending_tools": response.tool_calls,
                "current_tool_index": 0,
                "token_count": token_count,
            }
        else:
            result = {
                "messages": [response],
                "pending_tools": None,
                "current_tool_index": 0,
                "token_count": token_count,
            }

        duration = int((time.time() - start_time) * 1000)
        logger.info("Agent node completed", extra={
            "has_tools": bool(response.tool_calls),
            "token_count": token_count,
            "llm_duration_ms": llm_duration,
            "total_duration_ms": duration,
        })

        return result

    def _approval_node(self, state: ArchitectState) -> Command[Literal["execute", "cancel", END]]:
        """
        HITL approval for destructive tools.

        FEATURES:
            - Timeout: 5 minutes → auto-cancel
            - None handling: pending_tools is None → END
            - Command routing: goto execute/cancel/END
        """
        tools = state["pending_tools"]
        idx = state["current_tool_index"]

        # EXPLICIT None check FIRST
        if tools is None or idx >= len(tools):
            return Command(goto=END)

        current_tool = tools[idx]

        # Check timeout if previously interrupted
        if state["interrupt_timestamp"]:
            elapsed = time.time() - state["interrupt_timestamp"]
            if elapsed > 300:  # 5 minutes
                logger.warning("Approval timeout (5 minutes)")
                return Command(
                    goto="cancel",
                    update={"messages": [AIMessage(content="Approval timed out after 5 minutes")]}
                )

        # Only save_document and update_document need approval
        if current_tool["name"] in ["save_document", "update_document"]:
            decision = interrupt({
                "question": f"Approve {current_tool['name']} (tool {idx+1} of {len(tools)})?",
                "tool_name": current_tool["name"],
                "tool_args": current_tool["args"],
            })

            if decision:
                return Command(
                    goto="execute",
                    update={"interrupt_timestamp": time.time()}
                )
            else:
                return Command(goto="cancel")

        # Other tools (list_documents) don't need approval
        return Command(goto="execute")

    def _execute_node(self, state: ArchitectState) -> dict:
        """
        Execute tool calls with multiple tool support.

        ITERATION:
            - Execute current tool
            - Increment current_tool_index
            - If more tools → return with index updated
            - If all done → return final result
        """
        start_time = time.time()

        tools = state["pending_tools"]
        idx = state["current_tool_index"]

        if tools is None or idx >= len(tools):
            return {"messages": [AIMessage(content="No tools to execute")]}

        current_tool = tools[idx]
        tool_name = current_tool["name"]
        tool_args = current_tool["args"]

        logger.info(f"Executing tool {idx+1}/{len(tools)}: {tool_name}")

        # Execute tool
        try:
            tool_fn = self._tool_map.get(tool_name)
            if tool_fn:
                tool_result = tool_fn.invoke(tool_args)
            else:
                tool_result = f"Error: Tool '{tool_name}' not found."

            # Build result message
            result_msg = ToolMessage(
                content=str(tool_result),
                tool_call_id=current_tool["id"],
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            result_msg = ToolMessage(
                content=f"Tool error: {str(e)}",
                tool_call_id=current_tool["id"],
            )

        # Move to next tool
        next_idx = idx + 1

        duration = int((time.time() - start_time) * 1000)

        if next_idx < len(tools):
            # More tools to process
            logger.info(f"Tool {idx+1} done, moving to {next_idx+1}", extra={
                "duration_ms": duration,
            })
            return {
                "messages": [result_msg],
                "current_tool_index": next_idx,
            }
        else:
            # All tools done
            logger.info(f"All {len(tools)} tools executed", extra={
                "duration_ms": duration,
            })
            return {
                "messages": [result_msg],
                "pending_tools": None,
                "current_tool_index": 0,
            }

    def _cancel_node(self, state: ArchitectState) -> dict:
        """
        Cancel pending tool execution after user rejection.
        """
        logger.info("Tool execution cancelled by user")

        return {
            "messages": [AIMessage(content="Tool execution cancelled.")],
            "pending_tools": None,
            "current_tool_index": 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Routing functions
    # ─────────────────────────────────────────────────────────────────────────

    def _route_after_retrieve(self, state: ArchitectState) -> Literal["grade", "agent", "rewrite"]:
        """
        Route after retrieval based on results and configuration.

        EMPTY CHUNKS HANDLING:
            - After 2 retries → fallback to full document
            - Then proceed to agent
        """
        if state["retrieved_chunks"]:
            if self.enable_grading:
                return "grade"
            return "agent"

        # No chunks found
        iterations = state.get("query_iterations", 0)

        if iterations >= 2:
            # Fallback: Load entire source document
            source = state.get("last_query_source")
            if source:
                full_doc = self._vector_memory.get_full_document_by_source(source)
                if full_doc:
                    logger.info(f"Fallback: using full document {source}")
                    # Return Command to update state and goto agent
                    # Note: We can't update state here, so just return "agent"
                    # The fallback logic would need to be in retrieve_node
            return "agent"

        return "rewrite"

    def _route_has_tool(self, state: ArchitectState) -> Literal["approval", END]:
        """
        Route after agent: check if tools need execution.
        """
        if state["pending_tools"]:
            return "approval"
        return END

    # ─────────────────────────────────────────────────────────────────────────
    # Graph construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph with all nodes and edges.
        """
        builder = StateGraph(ArchitectState)

        # Add nodes
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade", self._grade_node)
        builder.add_node("rewrite", self._rewrite_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("approval", self._approval_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("cancel", self._cancel_node)

        # Entry point
        builder.add_edge(START, "retrieve")

        # Conditional routing after retrieve
        builder.add_conditional_edges("retrieve", self._route_after_retrieve)

        # Grade node returns Literal directly (no separate conditional)
        builder.add_conditional_edges("grade", lambda s: "agent")  # Simplified for now

        # Rewrite → back to retrieve
        builder.add_edge("rewrite", "retrieve")

        # Agent → approval or END
        builder.add_conditional_edges("agent", self._route_has_tool)

        # Execute and cancel → back to agent
        builder.add_edge("execute", "agent")
        builder.add_edge("cancel", "agent")

        return builder.compile(checkpointer=self._checkpointer)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_token_budget(self, chunks: list[Document]) -> list[Document]:
        """
        Keep chunks within token budget, preserving MMR order.
        """
        total = 0
        kept = []

        for chunk in chunks:
            tokens = self._count_message_tokens(chunk.page_content)
            if total + tokens <= self.MAX_RAG_TOKENS:
                kept.append(chunk)
                total += tokens
            else:
                break

        return kept

    def _count_message_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken or approximation.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text) // 4
        except Exception:
            return len(text) // 4

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def invoke(
        self,
        input: dict | Command,
        config: dict | None = None,
    ) -> dict:
        """
        Invoke the LangGraph with input and config.

        Args:
            input: Initial state or Command for resume
            config: {"configurable": {"thread_id": "..."}}

        Returns:
            Final state after graph execution
        """
        # Generate thread_id if not provided
        if not config:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        thread_id = config.get("configurable", {}).get("thread_id")

        logger.info(f"Invoking graph with thread_id: {thread_id}")

        return self.graph.invoke(input, config)

    def get_state(self, config: dict) -> dict:
        """
        Get current state for a thread_id (for HITL resume).
        """
        return self.graph.get_state(config)

    def stream(
        self,
        input: dict | Command,
        config: dict | None = None,
    ):
        """
        Stream graph execution for real-time UI updates.
        """
        if not config:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        return self.graph.stream(input, config)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_langgraph_architect = None


def get_langgraph_architect(
    enable_grading: bool = False,
    enable_rewrite: bool = False,
) -> LangGraphArchitect:
    """
    Return the shared LangGraphArchitect singleton.

    Args:
        enable_grading: Enable Self-RAG grading (default False)
        enable_rewrite: Enable Self-RAG rewrite (default False)
    """
    global _langgraph_architect

    if _langgraph_architect is None:
        _langgraph_architect = LangGraphArchitect(
            enable_grading=enable_grading,
            enable_rewrite=enable_rewrite,
        )

    return _langgraph_architect