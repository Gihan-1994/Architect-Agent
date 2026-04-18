# Design Document: Token Budget Optimization

## Overview

This document analyzes four proposed architectural changes for token optimization in the Software Architect Agent project. Each change is evaluated against the current implementation.

---

## Change 1: Asymmetric Memory (Replace SlidingWindowMemory)

### Proposed Design

**Current Flaw**: `SlidingWindowMemory` stores last 5 full exchanges. Massive AI-generated architecture documents stay in RAM and consume the token budget.

**Proposed Solution**:
- Store exact text of last 5 `HumanMessage` inputs
- Truncate every `AIMessage` to a placeholder: `[AI Action: Generated architecture doc]`

### Current Implementation Analysis

**File**: `agent/memory.py`

```python
# Current behavior:
def add_ai_message(self, content: str) -> None:
    self._messages.append(AIMessage(content=content))  # Full AI response stored
    self._trim()
```

**Feasibility**: ✅ **Fully feasible**

**Implementation Steps**:

1. Keep `HumanMessage` storage unchanged

2. Modify `add_ai_message()` to store truncated placeholder with failure detection:

```python
def add_ai_message(self, content: str) -> None:
    """
    Store AI response as truncated placeholder.
    
    Detects failures vs successful document generation.
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
```

3. Example placeholder outputs:
   - `[AI Action: Generated system-design.md]` — successful save
   - `[AI Action: Failed] ⚠️ your message is too long...` — user input rejected
   - `[AI Action: Failed] error: tool execution...` — tool failure
   - `[AI Action: No response]` — empty AI response

### Trade-offs

| Aspect | Impact |
|--------|--------|
| Token savings | Massive (AI responses can be 2000+ tokens) |
| Conversation context | Reduced — agent loses "memory" of what it said |
| User experience | Still sees full response in UI, just not in memory |

---

## Change 2: TokenTextSplitter (Replace Character Splitter)

### Proposed Design

**Current Flaw**: `RecursiveCharacterTextSplitter(chunk_size=500)` uses characters, not tokens. Code blocks have unpredictable token densities.

**Proposed Solution**: Replace with `TokenTextSplitter(chunk_size=400, chunk_overlap=50)`.

### Current Implementation Analysis

**File**: `agent/vector_memory.py:139-143`

```python
# Current implementation:
self._splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # characters
    chunk_overlap=100,
    length_function=len,
)
```

**Dependencies Check**:

```bash
$ pip show tiktoken
tiktoken not installed  # ❌ Required dependency missing
```

**LangChain Availability**: `TokenTextSplitter` exists in `langchain_text_splitters/base.py:298`

**Feasibility**: ✅ **Feasible with dependency addition**

**Implementation Steps**:

1. Add `tiktoken` to `requirements.txt`

2. Modify `vector_memory.py` with graceful fallback if `tiktoken` not installed:

```python
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

# Try TokenTextSplitter (requires tiktoken), fallback to character-based
try:
    self._splitter = TokenTextSplitter(
        encoding_name="cl100k_base",  # GPT-4 encoding (good approximation)
        chunk_size=500,
        chunk_overlap=100,
    )
    logger.info("TokenTextSplitter initialized (token-based chunking)")
except ImportError:
    # Fallback: character-based splitter if tiktoken missing
    logger.warning(
        "tiktoken not installed, falling back to RecursiveCharacterTextSplitter. "
        "Token budgets will be approximate. Install tiktoken for accurate chunking."
    )
    self._splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # characters (approximate)
        chunk_overlap=100,
        length_function=len,
    )
```

### Important Note

`tiktoken` uses OpenAI tokenizers. Gemini uses different tokenization. However:
- `cl100k_base` (GPT-4) is a reasonable approximation for English text
- For exact Gemini token counting, use `google.genai` `count_tokens` API
- Recommendation: Use `cl100k_base` for splitting, `count_tokens` for budget validation

---

## Change 3: Two-Pass Filtering (Relevance + Budget)

### Proposed Design

**Current Flaw**: `_build_augmented_message()` blindly retrieves chunks until token limit is reached, even when chunks are irrelevant to the task.

**Proposed Solution**: Two-Pass Filtering:
- **Pass 1 (Relevance)**: Filter by similarity score threshold
- **Pass 2 (Budget)**: Add chunks until token ceiling reached

### Current Implementation Analysis

**File**: `agent/architect.py:356`

```python
# Current implementation (blindly adds chunks):
retrieved_docs = self._vector_memory.search(user_message, k=3)  # Fixed k=3, no relevance check
```

**Feasibility**: ✅ **Fully feasible**

### Required: Add `similarity_search_with_score()` to VectorMemory

**File**: `agent/vector_memory.py` — add new method after `search()`:

```python
def similarity_search_with_score(
    self, 
    query: str, 
    k: int = 10,
    distance_threshold: float = 0.55,
) -> List[tuple]:
    """
    Find chunks with their similarity scores, filtering by relevance threshold.
    
    ChromaDB returns L2 distance scores. Lower score = more similar.
    
    Args:
        query: The user's message to search for.
        k: Maximum chunks to retrieve.
        distance_threshold: Maximum distance to consider relevant.
                            Default 0.55 (moderately similar).
                            Lower = stricter (0.3 = very similar only)
                            Higher = looser (1.0 = almost anything)
    
    Returns:
        List of (Document, score) tuples where score <= distance_threshold.
        Empty list if no chunks pass threshold OR ChromaDB not initialized.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Step 1: Ensure ChromaDB is initialized (lazy load)
    try:
        store = self._get_store()
    except Exception as e:
        logger.warning(f"ChromaDB initialization failed: {e}")
        return []
    
    # Step 2: Check if store has any documents
    try:
        count = store._collection.count()
    except Exception as e:
        logger.warning(f"ChromaDB collection access failed: {e}")
        count = 0
    
    if count == 0:
        logger.debug("ChromaDB collection is empty, no RAG context available")
        return []
    
    # Step 3: Perform similarity search with scores
    try:
        # ChromaDB returns (Document, distance_score) tuples
        results = store.similarity_search_with_score(query, k=k)
        
        # Debug logging for threshold tuning
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            logger.debug(f"Chunk score: {score:.3f}, source: {source}")
        
        # Filter by relevance threshold
        filtered = [(doc, score) for doc, score in results if score <= distance_threshold]
        
        if len(filtered) < len(results):
            skipped = len(results) - len(filtered)
            logger.debug(f"Filtered out {skipped} irrelevant chunks (score > {distance_threshold})")
        
        return filtered
    except Exception as e:
        logger.warning(f"ChromaDB similarity search failed: {e}")
        return []
```

### Implementation Steps for `_build_augmented_message()`

**File**: `agent/architect.py`

```python
def _build_augmented_message(self, user_message: str) -> str:
    """
    Two-Pass Filtering: Relevance + Budget.
    
    Pass 1: Filter chunks by similarity score (skip irrelevant garbage)
    Pass 2: Add chunks until token budget reached
    """
    if not self._vector_memory.has_documents():
        return user_message
    
    # Pass 1: Fetch with scores + relevance filtering
    # ChromaDB L2 distance: lower score = more similar
    results = self._vector_memory.similarity_search_with_score(
        user_message, 
        k=10,
        distance_threshold=0.55,  # Configurable, default moderately similar
    )
    
    # No relevant chunks found — don't waste tokens on empty context block
    if not results:
        return user_message
    
    import tiktoken
    import os
    import logging
    from google import genai
    
    logger = logging.getLogger(__name__)
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Budget constants
    MAX_RAG_TOKENS = 2500
    SAFETY_MARGIN = 0.10
    EFFECTIVE_LIMIT = int(MAX_RAG_TOKENS * (1 - SAFETY_MARGIN))  # 2250 tokens
    MAX_CHUNK_TOKENS = 600  # Expected max per chunk; warn if exceeded
    
    selected_chunks = []
    total_tokens = 0
    
    # Helper function: count tokens with fallback for unusual text
    def count_tokens(text: str, enc) -> int:
        """Count tokens with graceful fallback for unusual text."""
        try:
            return len(enc.encode(text))
        except Exception as e:
            # Fallback: character-based estimate (1 token ≈ 4 chars)
            logger.warning(f"tiktoken encoding failed for unusual text: {e}")
            estimated = len(text) // 4
            logger.debug(f"Using fallback estimate: {estimated} tokens")
            return estimated
    
    # Pass 2: Budget filtering
    for doc, score in results:
        chunk_tokens = count_tokens(doc.page_content, enc)
        source = doc.metadata.get("source", "unknown")
        
        # Edge case: oversized chunk (indicates splitter issue)
        # Option C: Allow but log warning
        if chunk_tokens > MAX_CHUNK_TOKENS:
            logger.warning(
                f"Oversized chunk detected: {source} ({chunk_tokens} tokens > {MAX_CHUNK_TOKENS}). "
                f"This may indicate TokenTextSplitter configuration issue. "
                f"Chunk will be included if budget allows."
            )
        
        if total_tokens + chunk_tokens > EFFECTIVE_LIMIT:
            logger.debug(f"Budget exhausted at {total_tokens} tokens, stopping")
            break
        
        logger.debug(f"Adding chunk: {source} ({chunk_tokens} tokens, score {score:.3f})")
        selected_chunks.append(doc)
        total_tokens += chunk_tokens
    
    # No chunks fit within budget
    if not selected_chunks:
        return user_message
    
    # Gemini verification: only when approaching budget limit
    # (tiktoken approximates; Gemini tokenizer differs)
    # This triggers when ~4+ chunks of 500 tokens each are selected
    if total_tokens > 2000:
        logger.debug(f"Near budget limit ({total_tokens}), verifying with Gemini count_tokens")
        try:
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            context_str = self._vector_memory.format_context(selected_chunks)
            exact_count = client.models.count_tokens(
                model=self.llm.model_name,
                contents=[{"parts": [{"text": context_str}]}]
            ).total_tokens
            
            # Remove chunks if exact count exceeds limit
            while exact_count > MAX_RAG_TOKENS and selected_chunks:
                removed = selected_chunks.pop()
                logger.debug(f"Removed chunk to fit budget: {removed.metadata.get('source')}")
                context_str = self._vector_memory.format_context(selected_chunks)
                exact_count = client.models.count_tokens(
                    model=self.llm.model_name,
                    contents=[{"parts": [{"text": context_str}]}]
                ).total_tokens
            total_tokens = exact_count
        except Exception as e:
            logger.debug(f"Gemini verification failed: {e}, using tiktoken estimate")
    
    # Format selected chunks
    context_str = self._vector_memory.format_context(selected_chunks)
    
    # Build augmented prompt
    augmented = (
        f"RELEVANT CONTEXT ({total_tokens} tokens):\n"
        f"{context_str}\n\n"
        f"USER REQUEST:\n{user_message}"
    )
    
    return augmented
```

### Threshold Tuning Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.3 | Very strict — only highly similar chunks | Precise technical questions |
| 0.55 (default) | Moderately similar | General architecture queries |
| 0.8 | Loose — accepts loosely related chunks | Broad exploratory questions |
| 1.0 | Almost anything | Maximum context retrieval |

**Logging output example**:
```
DEBUG: Chunk score: 0.42, source: auth-api-spec.md
DEBUG: Adding chunk: auth-api-spec.md (487 tokens, score 0.42)
DEBUG: Chunk score: 0.51, source: system-design.md
DEBUG: Adding chunk: system-design.md (512 tokens, score 0.51)
DEBUG: Chunk score: 0.67, source: unrelated-doc.md
DEBUG: Filtered out 1 irrelevant chunks (score > 0.55)
DEBUG: Budget exhausted at 2200 tokens, stopping
```

---

## Change 4: Hard Cutoff for User Inputs

### Proposed Design

**Current Flaw**: `agentic_action()` strips filler words but doesn't limit input length. A 3,000-line config file pasted by user destroys memory budget.

**Proposed Solution**:
- Add token validation step
- If input exceeds 500 tokens, reject with prompt to summarize or upload file

### Current Implementation Analysis

**File**: `agent/dispatcher.py:149-188`

```python
# Current flow:
def agentic_action(user_message: str) -> str:
    if not user_message or not user_message.strip():
        return "Please enter a message..."
    
    # No token limit check here!
    simplified = _simplify_instruction(user_message)
    agent = _get_architect()
    return agent.run(simplified)
```

**Feasibility**: ✅ **Fully feasible**

**Implementation Steps**:

```python
def agentic_action(user_message: str) -> str:
    import logging
    import os
    
    logger = logging.getLogger(__name__)
    
    if not user_message or not user_message.strip():
        return "Please enter a message..."
    
    # Token limit validation
    MAX_USER_TOKENS = 500
    SAFETY_MARGIN = 0.10  # 10% buffer for tokenizer differences
    EFFECTIVE_LIMIT = int(MAX_USER_TOKENS * (1 - SAFETY_MARGIN))  # 450
    
    # Step 1: Try tiktoken for fast token counting
    token_count = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(user_message))
    except ImportError:
        logger.warning("tiktoken not installed, using character-based estimate")
        # Fallback: character-based estimate (1 token ≈ 4 chars)
        # Use stricter char limit (500 tokens * 3 chars = 1500 chars)
        MAX_USER_CHARS = 1500
        if len(user_message) > MAX_USER_CHARS:
            return (
                f"⚠️ Your message is too long ({len(user_message)} characters).\n"
                f"Maximum allowed: ~{MAX_USER_CHARS} characters.\n\n"
                f"Please:\n"
                f"  • Summarize your request in fewer words\n"
                f"  • Or describe what you want and I'll ask clarifying questions\n"
            )
        # Continue with simplified flow (skip Gemini verification)
        simplified = _simplify_instruction(user_message)
        agent = _get_architect()
        return agent.run(simplified)
    except Exception as e:
        # tiktoken.encode() failed on unusual text
        logger.warning(f"tiktoken encoding failed: {e}, using character estimate")
        MAX_USER_CHARS = 1500
        if len(user_message) > MAX_USER_CHARS:
            return (
                f"⚠️ Your message contains unusual content and is too long.\n"
                f"Maximum allowed: ~{MAX_USER_CHARS} characters.\n\n"
                f"Please summarize your request."
            )
        simplified = _simplify_instruction(user_message)
        agent = _get_architect()
        return agent.run(simplified)
    
    # Step 2: If approaching limit, verify with exact Gemini count_tokens
    if token_count > EFFECTIVE_LIMIT:
        try:
            from google import genai
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            exact_count = client.models.count_tokens(
                model="gemini-1.5-flash",
                contents=[{"parts": [{"text": user_message]}]
            ).total_tokens
            token_count = exact_count
        except Exception:
            logger.debug("Gemini verification failed, using tiktoken estimate")
    
    # Step 3: Reject if over limit
    if token_count > MAX_USER_TOKENS:
        return (
            f"⚠️ Your message is too long ({token_count} tokens).\n"
            f"Maximum allowed: {MAX_USER_TOKENS} tokens.\n\n"
            f"Please:\n"
            f"  • Summarize your request in fewer words\n"
            f"  • Or describe what you want and I'll ask clarifying questions\n"
        )
    
    # Step 4: Continue with normal flow
    simplified = _simplify_instruction(user_message)
    agent = _get_architect()
    return agent.run(simplified)
```

---

## Feasibility Summary

| Change | Feasibility | Dependencies Required |
|--------|-------------|----------------------|
| 1. Asymmetric Memory | ✅ Fully feasible | None (existing code) |
| 2. TokenTextSplitter | ✅ Feasible | `tiktoken` package |
| 3. Dynamic Token Budgeting | ✅ Fully feasible | `tiktoken` package |
| 4. User Input Cutoff | ✅ Fully feasible | `tiktoken` package |

---

## Token Budget Model (After Implementation)

```
Total Budget:     ~10,000 tokens (Gemini Flash)

Allocation:
┌─────────────────────────────────────────────┐
│ System Prompt:           ~800 tokens        │
│ User Input (max):        ~500 tokens        │
│ Conversation (5 turns):  ~250 tokens        │ ← Each user msg truncated
│                              (5 × 50)       │
│ RAG Context (dynamic):   ~2,500 tokens      │ ← Capped ceiling
│ LLM Response Buffer:     ~6,000 tokens      │ ← Room for generation
└─────────────────────────────────────────────┘
```

---

## Implementation Order

1. Add `tiktoken` to `requirements.txt`
2. Implement Change 4 (User Input Cutoff) — prevents budget overrun at entry
3. Implement Change 1 (Asymmetric Memory) — reduces conversation footprint
4. Implement Change 2 (TokenTextSplitter) — guarantees chunk token sizes
5. Implement Change 3 (Dynamic RAG Budget) — caps retrieval tokens

---

## Files to Modify

| File | Change | Description |
|------|--------|-------------|
| `requirements.txt` | 2, 3, 4 | Add `tiktoken>=0.5.0` |
| `agent/memory.py` | 1 | Truncate AI messages to placeholder |
| `agent/vector_memory.py` | 2, 3 | Replace splitter + add `similarity_search_with_score()` method |
| `agent/architect.py` | 3 | Two-Pass Filtering in `_build_augmented_message()` |
| `agent/dispatcher.py` | 4 | User input token validation |

---

## Risk: tiktoken vs Gemini Tokenizer

`tiktoken` uses OpenAI tokenizers. Gemini tokenization differs slightly.

**Mitigation Options**:
1. Use `cl100k_base` encoding (GPT-4) — good approximation for English
2. Use Gemini `count_tokens` API for exact validation (adds latency)
3. Add safety margin: budget for 10% more tokens than tiktoken reports

**Recommendation**: Use `tiktoken` for fast validation. Add 10% safety margin. Use Gemini `count_tokens` only when near budget limits.

---

## Implementation Progress Checklist

### Dependencies

- [ ] Add `tiktoken>=0.5.0` to `requirements.txt`

### Change 1: Asymmetric Memory

- [ ] Modify `agent/memory.py` — `add_ai_message()` with failure detection
- [ ] Test: Verify `[AI Action: Failed]` vs `[AI Action: Generated document]` placeholders

### Change 2: TokenTextSplitter

- [ ] Modify `agent/vector_memory.py` — replace splitter with fallback logic
- [ ] Test: Verify chunks are token-accurate (not character-based)

### Change 3: Two-Pass Filtering

- [ ] Add `similarity_search_with_score()` method to `agent/vector_memory.py`
- [ ] Modify `agent/architect.py` — `_build_augmented_message()` with Two-Pass Filtering
- [ ] Test: Verify irrelevant chunks filtered out (score > threshold)
- [ ] Test: Verify oversized chunk warning logged
- [ ] Test: Verify Gemini verification triggers when near budget limit

### Change 4: User Input Cutoff

- [ ] Modify `agent/dispatcher.py` — `agentic_action()` with token validation
- [ ] Test: Verify oversized inputs rejected with helpful message
- [ ] Test: Verify tiktoken fallback works when not installed

### Integration Testing

- [ ] Test: Full conversation flow with token budget constraints
- [ ] Test: RAG retrieval with various relevance thresholds (0.3, 0.55, 0.8)
- [ ] Test: Edge case — oversized chunk in ChromaDB
- [ ] Test: Edge case — unusual Unicode/binary in user input