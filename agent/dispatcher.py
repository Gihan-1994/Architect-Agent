"""
agent/dispatcher.py - The agentic_action Function

WHY THIS FILE EXISTS:
  This is the single entry point for the entire agent system.
  It acts as a coordinator / router that sits between the user-facing
  interface (CLI or web UI) and the individual specialised agents.

  Even though we only have one agent today (Software Architect), the
  dispatcher is designed so you can add more agents later (e.g. a
  Security Auditor agent, a DevOps agent, etc.) without changing any
  user-facing code.

EXECUTION FLOW:
  User input
      │
      ▼
  agentic_action(user_message)
      │
      ├─ _select_agent()         ← "which agent should handle this?"
      │        │
      │        └─ returns "architect"  (currently always)
      │
      ├─ _simplify_instruction()  ← "strip the instruction to its core"
      │        │
      │        └─ returns cleaned user_message
      │
      └─ route to SoftwareArchitectAgent.run(simplified_message)
               │
               ▼
           friendly response text  ←────────────── returned to caller

AGENT REGISTRY (future extension):
  To add a new agent, simply:
    1. Create a new agent class in agent/my_new_agent.py
    2. Add a condition in _select_agent() to detect when to use it
    3. Add the routing logic in agentic_action()
"""

import os
import time
import logging
import subprocess
import sys
import uuid
from typing import Optional
from .architect import SoftwareArchitectAgent
from .langgraph_architect import LangGraphArchitect, get_langgraph_architect
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Instance Registry
# We use a single instance per agent type (singleton pattern) so that
# conversation memory is shared across all calls in one session.
# ─────────────────────────────────────────────────────────────────────────────
_architect: Optional[SoftwareArchitectAgent] = None
_langgraph_architect: Optional[LangGraphArchitect] = None
_current_thread_id: Optional[str] = None


def _get_architect() -> SoftwareArchitectAgent:
    """
    Lazy-load the Software Architect agent.
    The agent is created on first use and reused for the rest of the session.
    This preserves conversation memory across multiple agentic_action() calls.
    """
    global _architect
    if _architect is None:
        _architect = SoftwareArchitectAgent()
    return _architect


def _get_langgraph_architect() -> LangGraphArchitect:
    """
    Lazy-load the LangGraph architect singleton.
    Returns the instance with default Self-RAG disabled.
    """
    global _langgraph_architect
    if _langgraph_architect is None:
        _langgraph_architect = get_langgraph_architect()
    return _langgraph_architect


def _get_thread_id(source: str = "terminal") -> str:
    """
    Generate unique thread_id for LangGraph state isolation.

    Thread ID collision prevention:
        - Terminal: "terminal_{uuid}"
        - Web UI: "web_{user_id}_{session_id}"

    Args:
        source: "terminal" or "web"

    Returns:
        Unique thread_id string
    """
    global _current_thread_id

    if source == "terminal":
        _current_thread_id = f"terminal_{uuid.uuid4()}"

    return _current_thread_id


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Entry Point (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def agentic_action_langgraph(
    user_message: str,
    thread_id: str | None = None,
    resume_approval: bool = False,
    approval_decision: bool | None = None,
) -> dict:
    """
    LangGraph-based entry point with HITL support (Phase 4).

    Features:
        - HITL approval via interrupt() for save_document/update_document
        - State persistence via SqliteSaver
        - Self-RAG (configurable, disabled by default)
        - Multiple tool support

    Args:
        user_message: User input text
        thread_id: Optional thread for state continuity (auto-generated if None)
        resume_approval: True if resuming after approval prompt
        approval_decision: True=approve, False=reject (only used when resume_approval=True)

    Returns:
        {
            "response": str,           # Agent's text response
            "needs_approval": bool,    # True if waiting for user approval
            "tool_name": str | None,   # Tool awaiting approval
            "thread_id": str,          # Thread ID for resume
        }
    """
    start_time = time.time()
    architect = _get_langgraph_architect()

    # Generate thread_id if not provided
    if thread_id is None:
        thread_id = _get_thread_id("terminal")

    config = {"configurable": {"thread_id": thread_id}}

    # Log entry
    msg_preview = user_message[:50] + "..." if len(user_message) > 50 else user_message
    logger.info(f"LangGraph request: '{msg_preview}'", extra={"thread_id": thread_id})

    try:
        if resume_approval:
            # Resume after user approved/rejected
            if approval_decision is None:
                # Default to approve if not specified
                approval_decision = True

            # Resume with decision
            result = architect.invoke(
                Command(resume=approval_decision),
                config
            )
        else:
            # New request - initialize state
            result = architect.invoke(
                {
                    "messages": [HumanMessage(content=user_message)],
                    "retrieved_chunks": [],
                    "query_iterations": 0,
                    "pending_tools": None,
                    "current_tool_index": 0,
                    "interrupt_timestamp": None,
                    "token_count": 0,
                    "retrieval_mode": "standard",
                    "enable_grading": False,
                    "enable_rewrite": False,
                    "last_query_source": None,
                },
                config
            )

        # Check if graph is interrupted (waiting for approval)
        state = architect.get_state(config)

        # Check for pending approval
        needs_approval = False
        tool_name = None

        if state and state.values.get("pending_tools"):
            pending = state.values["pending_tools"]
            idx = state.values.get("current_tool_index", 0)
            if pending and idx < len(pending):
                current_tool = pending[idx]
                if current_tool["name"] in ["save_document", "update_document"]:
                    needs_approval = True
                    tool_name = current_tool["name"]

        # Extract final message
        messages = result.get("messages", [])
        response_text = ""

        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                response_text = last_msg.content
            elif hasattr(last_msg, "content"):
                response_text = str(last_msg.content)

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info("LangGraph response", extra={
            "duration_ms": duration_ms,
            "needs_approval": needs_approval,
            "thread_id": thread_id,
        })

        return {
            "response": response_text,
            "needs_approval": needs_approval,
            "tool_name": tool_name,
            "thread_id": thread_id,
        }

    except Exception as e:
        logger.error(f"LangGraph error: {e}")
        return {
            "response": f"Error: {str(e)}",
            "needs_approval": False,
            "tool_name": None,
            "thread_id": thread_id,
        }


def get_interrupted_state(thread_id: str) -> dict | None:
    """
    Get the current interrupted state for a thread (for HITL resume).

    Args:
        thread_id: Thread to check

    Returns:
        State dict if interrupted, None if not
    """
    architect = _get_langgraph_architect()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = architect.get_state(config)
        return state.values if state else None
    except Exception as e:
        logger.warning(f"Could not get state for thread {thread_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Agent Selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_agent(message: str) -> str:
    """
    Determine which specialised agent should handle the user's message.

    Currently returns "architect" for all messages.

    Future extension:
        - "security" for questions about threat models, pen-testing, CVEs
        - "devops"   for CI/CD pipelines, Kubernetes, Docker configs
        - "database" for complex query optimisation questions
        You could use keyword matching or even another LLM call here to classify.

    Args:
        message: The raw user input.

    Returns:
        A string key identifying the chosen agent.
    """
    message_lower = message.lower()

    # Future keyword-based routing (commented out, ready to enable):
    # security_keywords = ["security", "vulnerability", "threat", "penetration", "cve"]
    # devops_keywords   = ["docker", "kubernetes", "ci/cd", "pipeline", "helm"]
    # if any(kw in message_lower else message_lower for kw in security_keywords):
    #     return "security"

    # Default: Software Architect handles everything for now
    return "architect"


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Instruction Simplification
# ─────────────────────────────────────────────────────────────────────────────

def _simplify_instruction(message: str) -> str:
    """
    Pre-process the user message before sending it to the agent.

    This step:
      - Strips leading/trailing whitespace
      - Removes common filler phrases that add length without meaning

    Currently this is lightweight. In a production system you could use
    a second (cheaper) LLM call here to rewrite ambiguous instructions
    into clear, structured ones before sending them to the main agent.

    Args:
        message: The raw user input.

    Returns:
        A cleaned, simplified version of the instruction.
    """
    simplified = message.strip()

    # Remove common filler phrases that waste tokens
    filler_phrases = [
        "could you please ",
        "can you please ",
        "i would like you to ",
        "i want you to ",
        "please help me ",
        "hey, ",
        "hi, ",
    ]
    simplified_lower = simplified.lower()
    for phrase in filler_phrases:
        if simplified_lower.startswith(phrase):
            simplified = simplified[len(phrase):]
            break  # Only strip one phrase from the front

    # Capitalise the first letter after stripping
    if simplified:
        simplified = simplified[0].upper() + simplified[1:]

    return simplified


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def agentic_action(user_message: str) -> str:
    """
    The primary entry point for the agent system.

    This function coordinates the entire agent workflow:
      1. Receive the user's message
      2. Validate token limit (prevent budget overrun)
      3. Select the best-suited agent based on message content
      4. Simplify the instruction (remove filler, clean up text)
      5. Route the simplified instruction to the chosen agent
      6. The agent uses its LLM + tools to process the request
      7. Return the agent's friendly response to the caller

    Args:
        user_message: The raw text input from the user (CLI or web UI).

    Returns:
        A friendly, formatted text response from the agent.

    Example:
        >>> response = agentic_action("Create a system design for a food delivery app")
        >>> print(response)
        "I've created the system design document for your food delivery app ..."
    """
    start_time = time.time()

    # Log entry (truncate message for privacy, show first 50 chars)
    msg_preview = user_message[:50] + "..." if len(user_message) > 50 else user_message
    logger.info(f"Received user message: '{msg_preview}'")

    if not user_message or not user_message.strip():
        logger.warning("Empty message received")
        return "Please enter a message. I'm here to help you create architecture documents!"

    # ── Token limit validation ──────────────────────────────────────────────────
    MAX_USER_TOKENS = 500
    SAFETY_MARGIN = 0.10
    EFFECTIVE_LIMIT = int(MAX_USER_TOKENS * (1 - SAFETY_MARGIN))

    token_count = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(user_message))
        logger.debug(f"tiktoken count: {token_count}", extra={"tokens": token_count})
    except ImportError:
        logger.warning("tiktoken not installed, using character-based estimate")
        MAX_USER_CHARS = 1500
        if len(user_message) > MAX_USER_CHARS:
            logger.warning(f"Message too long: {len(user_message)} chars")
            return (
                f"⚠️ Your message is too long ({len(user_message)} characters).\n"
                f"Maximum allowed: ~{MAX_USER_CHARS} characters.\n\n"
                f"Please:\n"
                f"  • Summarize your request in fewer words\n"
                f"  • Or describe what you want and I'll ask clarifying questions\n"
            )
        # Continue with simplified flow
        simplified = _simplify_instruction(user_message)
        agent_type = _select_agent(user_message)
        logger.debug(f"Selected agent: {agent_type}")
        if agent_type == "architect":
            agent = _get_architect()
            result = agent.run(simplified)
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Response generated", extra={"duration_ms": duration_ms, "tokens": "approx"})
            return result
        return "I'm not sure how to handle that request. Could you rephrase it?"
    except Exception as e:
        logger.warning(f"tiktoken encoding failed: {e}")
        MAX_USER_CHARS = 1500
        if len(user_message) > MAX_USER_CHARS:
            return (
                f"⚠️ Your message contains unusual content and is too long.\n"
                f"Maximum allowed: ~{MAX_USER_CHARS} characters.\n\n"
                f"Please summarize your request."
            )
        simplified = _simplify_instruction(user_message)
        agent_type = _select_agent(user_message)
        if agent_type == "architect":
            agent = _get_architect()
            return agent.run(simplified)
        return "I'm not sure how to handle that request. Could you rephrase it?"

    # Step 2: Verify with Gemini if approaching limit
    if token_count > EFFECTIVE_LIMIT:
        logger.debug(f"Approaching token limit ({token_count} > {EFFECTIVE_LIMIT})")
        try:
            from google import genai
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            exact_count = client.models.count_tokens(
                model="gemini-1.5-flash",
                contents=[{"parts": [{"text": user_message}]}]
            ).total_tokens
            token_count = exact_count
            logger.debug(f"Gemini exact count: {token_count}", extra={"tokens": token_count})
        except Exception:
            logger.debug("Gemini verification failed")

    # Step 3: Reject if over limit
    if token_count > MAX_USER_TOKENS:
        logger.warning(f"Message rejected: {token_count} tokens exceeds limit")
        return (
            f"⚠️ Your message is too long ({token_count} tokens).\n"
            f"Maximum allowed: {MAX_USER_TOKENS} tokens.\n\n"
            f"Please:\n"
            f"  • Summarize your request in fewer words\n"
            f"  • Or describe what you want and I'll ask clarifying questions\n"
        )

    # ── Normal agent workflow ──────────────────────────────────────────────────
    agent_type = _select_agent(user_message)
    simplified = _simplify_instruction(user_message)
    logger.debug(f"Instruction simplified: '{simplified[:30]}...'" if len(simplified) > 30 else f"Instruction simplified: '{simplified}'")

    if agent_type == "architect":
        agent = _get_architect()
        result = agent.run(simplified)
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Response generated", extra={"duration_ms": duration_ms, "tokens": token_count})
        return result

    duration_ms = int((time.time() - start_time) * 1000)
    logger.warning(f"Unknown agent type: {agent_type}", extra={"duration_ms": duration_ms})
    return f"I'm not sure how to handle that request. Could you rephrase it?"


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions (used by CLI and web UI)
# ─────────────────────────────────────────────────────────────────────────────

def init_architect(model: str) -> None:
    """
    Pre-initialise the architect with a specific model.
    Call this once at startup after the user has selected a model.
    """
    global _architect
    logger.info(f"Initializing architect with model", extra={"model": model})
    _architect = SoftwareArchitectAgent(model=model)
    logger.info("Architect initialized successfully")


def switch_model(model: str) -> None:
    """
    Switch the active architect to a different Gemini model.
    If the agent has not been created yet, creates it with the given model.
    Conversation memory is preserved.
    """
    global _architect
    logger.info(f"Switching model", extra={"model": model})
    if _architect is None:
        _architect = SoftwareArchitectAgent(model=model)
    else:
        _architect.switch_model(model)
    logger.info("Model switched successfully")


def reset_conversation() -> None:
    """
    Clear the current agent's conversation memory.
    The agent remains active — only the chat history is erased.
    """
    global _architect
    logger.info("Resetting conversation memory")
    if _architect is not None:
        _architect.clear_memory()
    logger.info("Conversation memory cleared")


def get_token_estimate() -> int:
    """
    Return a rough estimate of how many tokens are currently stored in memory.
    Useful for the user to see how much context the agent is holding.
    """
    global _architect
    if _architect is not None:
        return _architect.get_token_estimate()
    return 0


def get_exact_token_count() -> dict:
    """
    Get exact token counts including RAG context retrieved from ChromaDB.

    Uses Gemini's count_tokens API for accuracy.

    Returns:
        {
            "total": int,          # exact total tokens
            "conversation": int,   # sliding window only (excluding RAG)
            "rag_context": int,    # retrieved chunks (if any)
            "system_prompt": int,  # system message tokens
            "method": str          # "exact" or "approximate" (fallback)
        }
    """
    global _architect
    if _architect is not None:
        return _architect.get_exact_token_count()
    return {
        "total": 0,
        "conversation": 0,
        "rag_context": 0,
        "system_prompt": 0,
        "method": "fallback"
    }


def reset_vector_memory() -> dict:
    """
    Reset ChromaDB collection with cosine distance metric.

    WARNING: This deletes all indexed documents!

    Use case: ChromaDB cannot change distance metric after collection creation.
    To switch from L2 to cosine, you must delete and recreate the collection.

    NOTE: Runs in a subprocess to properly close SQLite connections.
    ChromaDB's SQLite connection cannot be closed within the same Python process,
    causing "readonly database" errors when recreating after deletion.

    Returns:
        {"success": bool, "chunks_deleted": int, "new_space": str}
    """
    import subprocess
    import sys

    logger.info("reset_vector_memory called (subprocess isolation)")

    # Run reset in subprocess to ensure SQLite connection is fully closed
    script = '''
import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.api import CreateCollectionConfiguration

# Use current working directory (set by subprocess.run cwd)
persist_dir = os.path.join(os.getcwd(), "chroma_db")

# Step 1: Get old count (if collection exists)
old_count = 0
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(
        collection_name="architect_docs",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    old_count = store._collection.count()
except Exception:
    pass  # Collection may not exist

# Step 2: Delete persist_directory entirely
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
os.makedirs(persist_dir, exist_ok=True)

# Step 3: Create fresh collection with cosine
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
store = Chroma(
    collection_name="architect_docs",
    embedding_function=embeddings,
    persist_directory=persist_dir,
    collection_configuration=CreateCollectionConfiguration(hnsw={"space": "cosine"}),
)

print(f"success: True, chunks_deleted: {old_count}, new_space: cosine")
'''

    try:
        # Get project root directory (dispatcher.py is in agent/ folder)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or os.getcwd()

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode == 0:
            # Parse output
            output = result.stdout.strip()
            logger.info(f"Subprocess reset output: {output}")

            # Parse chunks_deleted from output
            chunks_deleted = 0
            if "chunks_deleted:" in output:
                try:
                    chunks_deleted = int(output.split("chunks_deleted:")[1].split()[0].strip(","))
                except Exception:
                    pass

            return {"success": True, "chunks_deleted": chunks_deleted, "new_space": "cosine"}
        else:
            logger.error(f"Subprocess reset failed: {result.stderr}")
            return {"success": False, "chunks_deleted": 0, "new_space": "cosine", "error": result.stderr}

    except Exception as e:
        logger.error(f"Subprocess execution failed: {e}")
        return {"success": False, "chunks_deleted": 0, "new_space": "cosine", "error": str(e)}


def reindex_all_documents() -> dict:
    """
    Re-index all .md files from the output directory into ChromaDB.

    Use case: After resetting ChromaDB (e.g., switching distance metrics),
    this function scans all existing documents and re-indexes them.

    Returns:
        {
            "documents": int,      # number of documents indexed
            "chunks": int,         # total chunks created
            "errors": list[str],   # filenames that failed to index
            "duration_ms": int     # total time in milliseconds
        }
    """
    import os
    from .vector_memory import get_vector_memory
    from .tools import get_save_path

    start_time = time.time()
    logger.info("reindex_all_documents called")

    vm = get_vector_memory()
    save_path = get_save_path()

    if not os.path.exists(save_path):
        logger.warning(f"Save path not found: {save_path}")
        return {
            "documents": 0,
            "chunks": 0,
            "errors": ["Save path not found"],
            "duration_ms": 0
        }

    md_files = sorted([f for f in os.listdir(save_path) if f.endswith(".md")])

    if not md_files:
        logger.info("No documents found to re-index")
        return {
            "documents": 0,
            "chunks": 0,
            "errors": [],
            "duration_ms": 0
        }

    total_chunks = 0
    errors = []

    for filename in md_files:
        filepath = os.path.join(save_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove metadata header if present (it's not part of the document content)
            if content.startswith("<!--"):
                # Find the end of the comment block
                end_marker = content.find("-->")
                if end_marker != -1:
                    content = content[end_marker + 3:].strip()

            chunks = vm.index_document(content, source_filename=filename)
            total_chunks += chunks
            logger.info(f"Indexed {filename}: {chunks} chunks")
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            logger.error(f"Failed to index {filename}: {e}")

    duration_ms = int((time.time() - start_time) * 1000)

    result = {
        "documents": len(md_files) - len(errors),
        "chunks": total_chunks,
        "errors": errors,
        "duration_ms": duration_ms
    }

    logger.info("reindex completed", extra={
        "documents": result["documents"],
        "chunks": total_chunks,
        "errors": len(errors),
        "duration_ms": duration_ms
    })

    return result
