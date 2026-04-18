"""
agent/memory.py - Sliding Window Memory

WHY THIS EXISTS:
  Every message you send to an LLM costs tokens. If we keep the entire
  conversation history, costs grow unbounded and eventually hit the model's
  context limit. The Sliding Window strategy fixes this by keeping only the
  most recent N exchanges (user + AI pairs), discarding older ones.

STRATEGY:
  ┌─────────────────────────────────────────────────────┐
  │  [SystemMessage]  ← ALWAYS kept (defines agent role) │
  │  [HumanMessage]  ← oldest kept exchange (turn N-4)  │
  │  [AIMessage]                                         │
  │  [HumanMessage]  ← ...                              │
  │  [AIMessage]                                         │
  │  [HumanMessage]  ← most recent exchange             │
  │  [AIMessage]                                         │
  └─────────────────────────────────────────────────────┘
  Anything older than max_exchanges is dropped automatically.

TOKEN ESTIMATE:
  A rough estimate is 1 token ≈ 4 characters. This is used for display
  purposes only (not billing).
"""

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from typing import List, Optional


class SlidingWindowMemory:
    """
    Conversation memory that keeps only the last N user-AI exchanges.

    This minimizes token usage while preserving recent context, which is
    usually the most relevant part of a conversation.

    Usage:
        memory = SlidingWindowMemory(max_exchanges=5)
        memory.set_system("You are a helpful assistant.")
        memory.add_user_message("Hello!")
        memory.add_ai_message("Hi there!")
        messages = memory.get_messages()  # → [system, human, ai]
    """

    def __init__(self, max_exchanges: int = 5):
        """
        Args:
            max_exchanges: How many user-AI back-and-forth pairs to retain.
                           Default 5 → keeps the last 5 questions + 5 answers
                           = 10 messages max (plus the system message).
        """
        # max_messages = pairs × 2 (one HumanMessage + one AIMessage per pair)
        self.max_messages: int = max_exchanges * 2
        self._messages: List[BaseMessage] = []      # rolling conversation history
        self._system: Optional[SystemMessage] = None  # always prepended

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def set_system(self, content: str) -> None:
        """Set the system message. Call this once at agent initialisation."""
        self._system = SystemMessage(content=content)

    def add_user_message(self, content: str) -> None:
        """Append a HumanMessage and trim the window."""
        self._messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        """
        Append an AIMessage and trim the window.

        NOTE: AI responses are truncated to placeholders to minimize
        token usage in conversation history. The user sees the full
        response in the UI, but the agent's memory only stores a summary.

        Detects failures vs successful document generation for debugging context.
        """
        # Detect if AI response indicates failure
        failure_indicators = [
            "error:",
            "failed to",
            "unable to",
            "could not",
            "exception:",
            "traceback",
            "⚠️",
        ]

        content_lower = content.lower()
        is_failure = any(indicator in content_lower for indicator in failure_indicators)

        if is_failure:
            # Preserve first 50 chars of error for debugging context
            truncated = f"[AI Action: Failed] {content[:50]}..."
        elif not content.strip():
            truncated = "[AI Action: No response]"
        else:
            # Success: extract document name if mentioned
            import re
            doc_match = re.search(r"(?:saved|saved to|written to)[: ]+(\S+\.md)", content)
            if doc_match:
                doc_name = doc_match.group(1)
                truncated = f"[AI Action: Generated {doc_name}]"
            else:
                truncated = "[AI Action: Generated document]"

        self._messages.append(AIMessage(content=truncated))
        self._trim()

    def get_messages(self) -> List[BaseMessage]:
        """
        Return the full message list to pass to the LLM.

        Format:  [SystemMessage, ...sliding window of HumanMessage/AIMessage...]
        The system message is always first so the model never loses its role.
        """
        result: List[BaseMessage] = []
        if self._system:
            result.append(self._system)
        result.extend(self._messages)
        return result

    def clear(self) -> None:
        """Wipe conversation history. System message is preserved."""
        self._messages = []

    def get_token_estimate(self) -> int:
        """
        Rough estimate of how many tokens are currently in memory.
        Formula: total characters ÷ 4  (industry rule-of-thumb).
        Used for display only.
        """
        total_chars = sum(len(msg.content) for msg in self.get_messages())
        return total_chars // 4

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _trim(self) -> None:
        """
        Drop messages that exceed the sliding window.

        We always drop from the front (oldest messages first),
        keeping the most recent max_messages items.
        """
        if len(self._messages) > self.max_messages:
            # Keep only the last max_messages entries
            self._messages = self._messages[-self.max_messages:]
