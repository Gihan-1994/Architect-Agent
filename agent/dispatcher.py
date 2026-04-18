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

import time
import logging
from typing import Optional
from .architect import SoftwareArchitectAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Instance Registry
# We use a single instance per agent type (singleton pattern) so that
# conversation memory is shared across all calls in one session.
# ─────────────────────────────────────────────────────────────────────────────
_architect: Optional[SoftwareArchitectAgent] = None


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
    import os

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
