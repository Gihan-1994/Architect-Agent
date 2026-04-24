"""
agent/architect.py - Software Architect Agent (Core)

WHY THIS FILE EXISTS:
  This is the "brain" of the system. It wraps the Gemini LLM with:
    - A detailed system prompt that defines the agent's expertise
    - Tool bindings (so the LLM can save files)
    - Sliding window memory (to minimise token usage)
    - Vector memory retrieval (RAG — long-term document context)
    - A ReAct-style execution loop (Reason → Act → Observe → Respond)

UPDATED EXECUTION FLOW (with RAG):
  user_message
       │
       ▼
  memory.add_user_message()         ← save to sliding window
       │
       ▼
  _build_augmented_message()        ← NEW: RAG retrieval step
       │
       ├─ vector_memory.search(user_message, k=3)
       │        │
       │        └─ embeds question → finds closest chunks in ChromaDB
       │
       └─ if relevant chunks found:
               wrap them as context above the user message
               "RELEVANT CONTEXT:\n[chunks]\n\nUSER REQUEST:\n[msg]"
       │
       ▼
  llm_with_tools.invoke(messages)   ← LLM now sees BOTH sliding window
       │                               AND relevant document context
       ├─ No tool_calls? ──────────────────► return response text
       │
       └─ Has tool_calls?
               │
               ▼
         run each tool (save_document → also indexes in ChromaDB)
               │
               ▼
         llm_with_tools.invoke(messages + tool results)
               │
               ▼
         return final response text
       │
       ▼
  memory.add_ai_message()           ← save AI response to sliding window

WHY WE DON'T PUT RETRIEVED CONTEXT IN THE SLIDING WINDOW:
  The sliding window is precious — it holds conversational flow.
  Retrieved context can be very large (3 chunks × 500 chars = 1500 chars).
  If we added it to the sliding window, it would quickly push out actual
  conversation turns. Instead, we inject it fresh on EVERY call, so
  Gemini always has up-to-date context without polluting the window.
"""

import os
import time
import logging
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage, BaseMessage
from .memory import SlidingWindowMemory
from .tools import save_document, list_documents, update_document
from .vector_memory import get_vector_memory
from google import genai

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def list_available_models() -> List[str]:
    """
    Fetch all Gemini models that support content generation.

    Configures the genai SDK with the API key, then queries the
    Gemini API for the full model list and filters to only those
    that support the 'generateContent' method (i.e. chat-capable).

    Returns:
        List of model name strings, e.g. ["models/gemini-1.5-flash", ...]
        Falls back to [DEFAULT_MODEL] if the key is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return [DEFAULT_MODEL]
    client = genai.Client(api_key=api_key)
    return [
        m.name
        for m in client.models.list()
        if "generateContent" in (m.supported_actions or [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt: Defines the agent's expertise and behaviour
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Senior Software Architect AI assistant with 15+ years of experience
designing scalable, production-ready systems. You help engineering teams and developers create
clear, professional software architecture documents.

YOUR SPECIALTIES:
1. System Design Documents
   - High-level architecture overviews
   - Component diagrams (using Mermaid)
   - Technology stack recommendations
   - Scalability and reliability considerations
   - Security architecture

2. API Specifications
   - RESTful API endpoint documentation
   - Request / response schemas (JSON)
   - Authentication and authorisation flows
   - Error codes and handling
   - Versioning strategy

3. Database Schema / ERD
   - Relational table definitions
   - Entity-Relationship Diagrams using Mermaid erDiagram syntax
   - Indexing strategy
   - Data normalisation notes

4. Sequence / Flow Diagrams
   - System interaction flows using Mermaid sequenceDiagram syntax
   - User journey flows
   - Microservice communication patterns

DOCUMENT QUALITY STANDARDS:
- Always produce well-structured Markdown with clear headings (## H2, ### H3)
- Use Mermaid diagrams for all visual content (wrapped in ```mermaid code blocks)
- Include an "Overview" section at the start of every document
- Add a "Security Considerations" section where relevant
- Be practical and production-focused — not just theoretical
- Include concrete examples (sample JSON, table column definitions, etc.)

TOOL USAGE RULES:
- When the user asks to CREATE, GENERATE, WRITE, or DESIGN a NEW document → call save_document()
- When the user asks to UPDATE, MODIFY, REVISE, or CHANGE an existing document → call update_document()
- When the user asks to LIST, SHOW, or VIEW saved documents → call list_documents()

IMPORTANT - WHEN NOT TO USE TOOLS:
- When the user asks a QUESTION about existing content (e.g., "What's inside X?", "Summarize X", "Explain the architecture") → DO NOT call any tool! Just answer using the RAG context provided.
- You receive RELEVANT CONTEXT from previously saved documents. Use this to answer questions directly.
- Only create/update documents when the user EXPLICITLY asks you to write or modify a file.

DOCUMENT CREATION/UPDATE:
- Before creating a new document, consider if an existing one should be updated instead
- When updating, include the FULL updated document content (not just the new section)
- Preserve the document's overall structure unless explicitly asked to restructure
- Always confirm to the user after saving/updating (the tool returns the file path)
- Pick a descriptive, hyphenated filename (e.g., 'ecommerce-system-design', 'auth-api-spec')

COMMUNICATION STYLE:
- Be professional but approachable
- After saving a document, briefly summarise what was created (2-3 sentences)
- If the user's request is vague, make reasonable assumptions and note them
- Always be ready to iterate: invite the user to ask for revisions
"""


class SoftwareArchitectAgent:
    """
    The core agent that processes user requests about software architecture.

    It uses:
      - Google Gemini 1.5 Flash (free tier) as the LLM
      - LangChain tool binding for save/list file operations
      - SlidingWindowMemory to keep token usage low
      - A manual ReAct loop (no LangGraph dependency — keeps it simple)
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        # Validate key before doing anything else
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment. "
                "Please copy .env.example to .env and add your key. "
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        # ── LLM Setup ──────────────────────────────────────────────────────
        self.llm = ChatGoogleGenerativeAI(
            model=model,                      # use the passed param, not hardcoded
            google_api_key=api_key,
            temperature=0.7,          # 0 = deterministic, 1 = creative; 0.7 is balanced
            max_output_tokens=4096,   # max response length per call
        )

        # ── Tool Setup ─────────────────────────────────────────────────────
        # bind_tools() attaches the tool schemas to the LLM so it knows
        # what tools exist and how to call them
        self._tools = [save_document, list_documents, update_document]
        self._llm_with_tools = self.llm.bind_tools(self._tools)

        # Build a quick lookup: tool name → callable
        self._tool_map = {t.name: t for t in self._tools}

        # ── Memory Setup ───────────────────────────────────────────────────
        # Layer 1 (Short-term): Sliding window — keeps last 5 exchanges in RAM
        self.memory = SlidingWindowMemory(max_exchanges=5)
        self.memory.set_system(SYSTEM_PROMPT)

        # Layer 2 (Long-term): Vector store — the shared singleton from
        # vector_memory.py. The same instance is also used by tools.py
        # for indexing, so retrieval here always reflects the latest saves.
        self._vector_memory = get_vector_memory()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, user_message: str) -> str:
        """
        Process a user message and return the agent's final response.

        Updated ReAct loop (now includes RAG):
          1. Add the user message to the sliding window
          2. Search the vector store for relevant document chunks (RAG)
          3. Build an augmented message = [retrieved context] + [user message]
          4. Call the LLM with the full message history (including augmented msg)
          5. If tool calls are present, execute them and call the LLM again
          6. Repeat until the LLM gives a plain text response (no more tools)
          7. Save the AI response to the sliding window and return it

        Args:
            user_message: The raw text from the user.

        Returns:
            The agent's final text response as a string.
        """
        start_time = time.time()
        msg_preview = user_message[:30] + "..." if len(user_message) > 30 else user_message
        logger.info(f"Processing message: '{msg_preview}'")

        # Step 1: Save the ORIGINAL user message to the sliding window.
        self.memory.add_user_message(user_message)
        logger.debug("User message added to sliding window")

        # Step 2 & 3: RAG — retrieve relevant context and augment the message
        rag_start = time.time()
        augmented_message = self._build_augmented_message(user_message)
        rag_duration = int((time.time() - rag_start) * 1000)

        # Check if RAG context was added
        has_rag = "RELEVANT CONTEXT" in augmented_message
        if has_rag:
            logger.info(f"RAG retrieval completed", extra={"duration_ms": rag_duration, "rag": True})
        else:
            logger.debug(f"No RAG context found", extra={"duration_ms": rag_duration, "rag": False})

        # Step 4: Build the full message list for the LLM call.
        messages = self.memory.get_messages()
        messages[-1] = HumanMessage(content=augmented_message)
        msg_count = len(messages)
        logger.debug(f"Message list built: {msg_count} messages")

        # Step 5: Call the LLM
        llm_start = time.time()
        response = self._llm_with_tools.invoke(messages)
        llm_duration = int((time.time() - llm_start) * 1000)
        logger.info(f"LLM call completed", extra={"duration_ms": llm_duration})

        # Step 6: Tool execution loop (ReAct)
        tool_iterations = 0
        while response.tool_calls:
            tool_iterations += 1
            messages.append(response)

            # Execute every tool the LLM requested in this step
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_fn = self._tool_map.get(tool_name)

                tool_start = time.time()
                logger.info(f"Tool invoked: {tool_name}", extra={"tool": tool_name})

                if tool_fn:
                    tool_result = tool_fn.invoke(tool_args)
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."
                    logger.error(f"Tool not found: {tool_name}")

                tool_duration = int((time.time() - tool_start) * 1000)
                logger.info(f"Tool completed: {tool_name}", extra={"tool": tool_name, "duration_ms": tool_duration})

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

            # Call the LLM again with the tool results appended
            llm_start = time.time()
            response = self._llm_with_tools.invoke(messages)
            llm_duration = int((time.time() - llm_start) * 1000)
            logger.info(f"LLM call after tool", extra={"duration_ms": llm_duration, "iteration": tool_iterations})

        if tool_iterations > 0:
            logger.info(f"ReAct loop completed", extra={"iterations": tool_iterations})

        # Step 7: Extract final text from response
        content = response.content
        if isinstance(content, list):
            final_text = "\n".join(
                block["text"] for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            final_text = content

        # Step 8: Save the AI response to the sliding window memory
        self.memory.add_ai_message(final_text)

        # Final timing
        total_duration = int((time.time() - start_time) * 1000)
        logger.info(f"run() completed", extra={"duration_ms": total_duration, "rag": has_rag, "tools": tool_iterations})

        return final_text

    def switch_model(self, model: str) -> None:
        """
        Swap the underlying LLM to a different Gemini model.

        Conversation memory is preserved — only the LLM instance is replaced.
        Tools are rebound to the new LLM so tool calling keeps working.

        Args:
            model: The model name, e.g. "models/gemini-1.5-flash"
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            max_output_tokens=4096,
        )
        # Rebind tools so the new LLM instance knows about them
        self._llm_with_tools = self.llm.bind_tools(self._tools)

    def clear_memory(self) -> None:
        """Reset conversation history (system message is preserved).
        Note: the vector store is NOT cleared — long-term memory persists."""
        self.memory.clear()

    def get_token_estimate(self) -> int:
        """Return a rough estimate of how many tokens are in the sliding window."""
        return self.memory.get_token_estimate()

    def get_exact_token_count(self) -> dict:
        """
        Get exact token counts for what will be sent to Gemini.

        Uses google.genai count_tokens API for accuracy, including RAG context
        that is injected via _build_augmented_message().

        Returns:
            {
                "total": int,          # exact total tokens
                "conversation": int,   # sliding window only (excluding RAG)
                "rag_context": int,    # retrieved chunks (if any)
                "system_prompt": int,  # system message tokens
                "method": str          # "exact" or "approximate" (fallback)
            }
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return self._fallback_token_count()

        from google.genai import types

        messages = self.memory.get_messages()

        # Build Gemini Content objects from messages
        contents = []
        system_content = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System message is handled separately
                system_content = types.Part(text=msg.content)
            elif isinstance(msg, HumanMessage):
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg.content)]
                ))
            elif isinstance(msg, AIMessage):
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=msg.content)]
                ))

        client = genai.Client(api_key=api_key)

        try:
            # Count tokens for conversation contents
            result = client.models.count_tokens(
                model=self.llm.model,
                contents=contents,
            )
            conversation_total = result.total_tokens

            # Count system prompt separately
            system_tokens = 0
            if system_content:
                sys_result = client.models.count_tokens(
                    model=self.llm.model,
                    contents=[types.Content(
                        role="user",
                        parts=[system_content]
                    )]
                )
                system_tokens = sys_result.total_tokens

            # Detect RAG context in the last HumanMessage
            rag_tokens = 0
            if messages:
                last_user_msg = None
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        last_user_msg = msg
                        break

                if last_user_msg:
                    rag_tokens = self._count_rag_tokens(last_user_msg, client)

            conversation_tokens = conversation_total - rag_tokens

            return {
                "total": conversation_total + system_tokens,
                "conversation": conversation_tokens,
                "rag_context": rag_tokens,
                "system_prompt": system_tokens,
                "method": "exact"
            }

        except Exception as e:
            logger.warning(f"Gemini count_tokens failed: {e}")
            return self._fallback_token_count()

    def _count_rag_tokens(self, msg: HumanMessage, client) -> int:
        """
        Count tokens for RAG context portion in an augmented message.

        The augmented message format is:
            RELEVANT CONTEXT (X tokens):
            [chunk content...]

            USER REQUEST:
            [original user message]

        Args:
            msg: The HumanMessage to check for RAG context.
            client: Gemini client for token counting.

        Returns:
            Token count for RAG portion, or 0 if no RAG context found.
        """
        from google.genai import types

        content = msg.content

        # Check if message has RAG context header
        if "RELEVANT CONTEXT" not in content:
            return 0

        # Extract RAG portion (between header and "USER REQUEST:")
        parts = content.split("USER REQUEST:")
        if len(parts) < 2:
            return 0

        rag_content = parts[0]

        try:
            result = client.models.count_tokens(
                model=self.llm.model,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part(text=rag_content)]
                )]
            )
            return result.total_tokens
        except Exception:
            # Fallback to rough estimate
            return len(rag_content) // 4

    def _fallback_token_count(self) -> dict:
        """
        Fallback to rough estimate when Gemini API is unavailable.

        This uses the character-based estimate (chars ÷ 4) and cannot
        distinguish RAG context from conversation.
        """
        rough = self.memory.get_token_estimate()
        return {
            "total": rough,
            "conversation": rough,
            "rag_context": 0,
            "system_prompt": 0,
            "method": "approximate"
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_augmented_message(self, user_message: str) -> str:
        """
        Three-Pass Filtering: MMR Search + Relevance + Budget.

        Pass 1: MMR search (diverse candidates with scores)
        Pass 2: Filter by cosine score threshold (discard irrelevant chunks)
        Pass 3: Add chunks until token budget exhausted

        WHY MMR?
          Standard similarity search returns chunks that are all similar to
          EACH OTHER (redundant). MMR selects chunks that are:
            - Similar to the query (relevant)
            - Different from each other (diverse coverage)

        Args:
            user_message: The original user input.

        Returns:
            Augmented message with relevant context, or original if nothing found.
        """
        # Only search if the vector store has at least one indexed document
        if not self._vector_memory.has_documents():
            return user_message

        mmr_start = time.time()

        # Pass 1: MMR search with scores
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
        # 0.85 threshold: balances relevance recall vs precision
        # all-MiniLM-L6-v2 model produces scores 0.4-0.8 for similar content
        # 0.85 allows relevant chunks while filtering truly irrelevant (> 0.9)
        COSINE_THRESHOLD = 0.85

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

        logger.info(f"MMR + relevance filter: {len(relevant_chunks)}/{len(results)} chunks passed", extra={
            "mmr_duration_ms": mmr_duration,
            "chunks_rejected_by_score": len(results) - len(relevant_chunks),
        })

        # Pass 3: Token budget filtering
        MAX_RAG_TOKENS = 2500
        SAFETY_MARGIN = 0.10
        EFFECTIVE_LIMIT = int(MAX_RAG_TOKENS * (1 - SAFETY_MARGIN))  # 2250 tokens
        MAX_CHUNK_TOKENS = 600  # Expected max per chunk; warn if exceeded

        selected_chunks = []
        total_tokens = 0

        def count_tokens(text: str) -> int:
            """Count tokens with graceful fallback."""
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
            source = doc.metadata.get("source", "unknown")

            # Edge case: oversized chunk (indicates splitter issue)
            if chunk_tokens > MAX_CHUNK_TOKENS:
                logger.warning(
                    f"Oversized chunk detected: {source} ({chunk_tokens} tokens > {MAX_CHUNK_TOKENS}). "
                    f"This may indicate TokenTextSplitter configuration issue."
                )

            if total_tokens + chunk_tokens > EFFECTIVE_LIMIT:
                logger.debug(f"Token budget exhausted at {total_tokens} tokens")
                break

            logger.debug(f"Adding chunk: {source} ({chunk_tokens} tokens, score {score:.3f})")
            selected_chunks.append(doc)
            total_tokens += chunk_tokens

        if not selected_chunks:
            logger.debug("No chunks fit within token budget")
            return user_message

        # Gemini verification: only when approaching budget limit
        if total_tokens > 2000:
            logger.debug(f"Near budget limit ({total_tokens}), verifying with Gemini")
            try:
                client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                context_str = self._vector_memory.format_context(selected_chunks)
                exact_count = client.models.count_tokens(
                    model=self.llm.model,
                    contents=[{"parts": [{"text": context_str}]}]
                ).total_tokens

                while exact_count > MAX_RAG_TOKENS and selected_chunks:
                    removed = selected_chunks.pop()
                    logger.debug(f"Removed chunk to fit budget: {removed.metadata.get('source')}")
                    context_str = self._vector_memory.format_context(selected_chunks)
                    exact_count = client.models.count_tokens(
                        model=self.llm.model,
                        contents=[{"parts": [{"text": context_str}]}]
                    ).total_tokens
                total_tokens = exact_count
            except Exception as e:
                logger.debug(f"Gemini verification failed: {e}")

        # Format selected chunks
        context_str = self._vector_memory.format_context(selected_chunks)

        # Build augmented prompt
        augmented = (
            f"RELEVANT CONTEXT ({total_tokens} tokens):\n"
            f"{context_str}\n\n"
            f"USER REQUEST:\n{user_message}"
        )

        total_duration = int((time.time() - mmr_start) * 1000)
        logger.info("RAG completed", extra={
            "duration_ms": total_duration,
            "mmr_duration_ms": mmr_duration,
            "chunks_selected": len(selected_chunks),
            "chunks_rejected_by_score": len(results) - len(relevant_chunks),
            "chunks_rejected_by_budget": len(relevant_chunks) - len(selected_chunks),
            "tokens": total_tokens,
        })

        return augmented
