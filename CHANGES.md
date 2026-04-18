# Change Log

---

## [2026-04-18] Structured JSON Logging

---

### 1. `agent/logging_config.py` — New logging configuration module
- **Why:** You needed visibility into the agent's execution flow to debug slow responses, track tool usage, and monitor memory/token counts.
- **How:** Created new module with:
  - `JSONFormatter` — outputs structured JSON logs to file
  - `ConsoleFormatter` — human-readable colored output to terminal
  - `setup_logging(debug=False)` — INFO level (always visible)
  - `setup_logging(debug=True)` — DEBUG level (verbose)
  - Outputs to both console (stdout) and file (`logs/agent.log`)
  - Log format: `{"timestamp": "...", "level": "INFO", "module": "architect", "method": "run", "message": "...", "duration_ms": 150}`
  - Extra fields: `tokens`, `chunks`, `tool`, `model`, `score`, `filename`

---

### 2. `agent/dispatcher.py` — Added logging to entry point
- **Why:** Track when messages enter the system, token validation, and response timing.
- **How:**
  - Added `time` import and module-level logger
  - `agentic_action()`: logs entry with message preview, token count, agent selection, instruction simplification, and exit with duration
  - `init_architect()`: logs model initialization
  - `switch_model()`: logs model switching
  - `reset_conversation()`: logs memory clearing

---

### 3. `agent/architect.py` — Added logging to core agent
- **Why:** Track RAG retrieval, LLM calls, tool invocations, and ReAct loop iterations.
- **How:**
  - Added `time` import
  - `run()`: logs entry, RAG retrieval timing (with has_rag flag), LLM call timing, each tool invocation with name and duration, ReAct iteration count, and exit with total duration
  - Tool iterations logged with `extra={"tool": tool_name, "iteration": n}`
  - RAG retrieval logged with `extra={"rag": True/False, "duration_ms": ms}`

---

### 4. `agent/tools.py` — Added logging to tool functions
- **Why:** Track when save_document and list_documents are called, and their execution timing.
- **How:**
  - Added `time` import and module-level logger
  - `save_document()`: logs entry with filename, file write timing, vector indexing timing (with chunk count), and exit
  - `list_documents()`: logs entry and number of documents found

---

### 5. `agent/vector_memory.py` — Added logging to vector operations
- **Why:** Track ChromaDB operations which can be slow (embedding + storage).
- **How:**
  - Added `time` import
  - `index_document()`: logs entry with filename, split timing, ChromaDB store timing, and exit with chunk count
  - `similarity_search_with_score()`: logs entry with query preview, search timing, threshold filtering, and exit with result count
  - Each chunk score logged at DEBUG level for threshold tuning

---

### 6. `agent/__init__.py` — Exported logging config
- **Why:** main.py and app.py need to call `setup_logging()` at startup.
- **How:** Added `setup_logging` to import and `__all__` list.

---

### 7. `main.py` — Initialize logging at startup
- **Why:** Logging must be configured before any agent operations.
- **How:** Added `setup_logging(debug=False)` call before model selection.

---

### 8. `app.py` — Initialize logging at startup
- **Why:** Logging must be configured before any agent operations.
- **How:** Added `setup_logging(debug=False)` call after dotenv load.

---

## [2026-04-18] Accurate Token Estimation

---

### 1. `agent/architect.py` — Added `get_exact_token_count()` method
- **Why:** The existing `get_token_estimate()` only counted the sliding window using `chars ÷ 4`. It missed RAG context retrieved from ChromaDB, which can add ~375 tokens. Users saw `~200 tokens` but Gemini received `~600 tokens`.
- **How:** Added new method after `get_token_estimate()`:
  - Uses `google.genai` `count_tokens` API for exact counts
  - Converts LangChain messages to Gemini `Content` objects (following `langchain_google_genai` pattern)
  - Returns breakdown: total, conversation, rag_context, system_prompt, method
  - Detects RAG context in augmented messages by parsing `"RELEVANT CONTEXT"` header
  - Added `_count_rag_tokens()` helper with fallback for parse failures
  - Added `_fallback_token_count()` for when Gemini API unavailable

---

### 2. `agent/dispatcher.py` — Added `get_exact_token_count()` to public API
- **Why:** main.py and app.py needed access to the exact token counts for UI display.
- **How:** Added wrapper function after `get_token_estimate()`:
  - Calls `_architect.get_exact_token_count()` if agent initialized
  - Returns fallback dict with zeros if agent not yet created

---

### 3. `agent/__init__.py` — Exported new function
- **Why:** main.py and app.py import from the `agent` package.
- **How:** Added `get_exact_token_count` to import and `__all__` list.

---

### 4. `app.py` — Updated UI token display
- **Why:** The token display only showed a rough estimate without breakdown.
- **How:** Modified `update_token_display()`:
  - Calls `get_exact_token_count()` instead of `get_token_estimate()`
  - Shows breakdown when exact: Total, Conversation, RAG Context, System Prompt
  - Shows "(approximate)" label when using fallback
  - Updated textbox to 4 lines to accommodate multi-line display
  - Changed info text to "Exact token counts from Gemini API"

---

### 5. `main.py` — Updated CLI `/tokens` command
- **Why:** The `/tokens` command only showed a single rough estimate number.
- **How:** Modified `/tokens` handler in `handle_command()`:
  - Imports and calls `get_exact_token_count()`
  - Shows formatted breakdown when exact method available
  - Shows fallback message when approximate or agent not initialized

---

## [2026-04-18] Token Budget Optimization

---

### 1. `requirements.txt` — Added `tiktoken`
- **Why:** OpenAI's `tiktoken` tokenizer provides accurate token counting for budget validation. Character-based estimates (`chars ÷ 4`) are unreliable for code blocks and non-English text.
- **How:** Added `tiktoken>=0.5.0` under a new `# ── Token counting ──` section.

---

### 2. `agent/memory.py` — Asymmetric Memory (AI message truncation)
- **Why:** `SlidingWindowMemory` stored full AI responses (often 2000+ tokens) in conversation history, consuming the token budget.
- **How:** Modified `add_ai_message()` to truncate AI responses to placeholders:
  - Success: `[AI Action: Generated system-design.md]` (extracts filename if mentioned)
  - Failure: `[AI Action: Failed] ⚠️ your message...` (preserves first 50 chars for debugging)
  - Empty: `[AI Action: No response]`
  - Detects failure indicators: `error:`, `failed to`, `⚠️`, etc.

---

### 3. `agent/vector_memory.py` — TokenTextSplitter with fallback
- **Why:** `RecursiveCharacterTextSplitter(chunk_size=500)` used characters, not tokens. Code blocks have unpredictable token densities, breaking budget calculations.
- **How:** Replaced with `TokenTextSplitter(encoding_name="cl100k_base", chunk_size=500, chunk_overlap=100)` with graceful fallback:
  - If `tiktoken` not installed → falls back to `RecursiveCharacterTextSplitter`
  - Logs warning about approximate token budgets
  - Added `import logging` and `logger` instance

---

### 4. `agent/vector_memory.py` — Added `similarity_search_with_score()` method
- **Why:** RAG retrieval blindly fetched chunks without checking relevance scores. Irrelevant chunks wasted token budget.
- **How:** Added new method after `search()`:
  - Returns `(Document, score)` tuples with L2 distance scores
  - Filters by `distance_threshold=0.55` (lower score = more similar)
  - Comprehensive error handling: ChromaDB init, collection access, search failures
  - Debug logging for threshold tuning: `Chunk score: 0.42, source: auth-spec.md`

---

### 5. `agent/architect.py` — Two-Pass Filtering in `_build_augmented_message()`
- **Why:** Previous implementation blindly added chunks until token limit, ignoring relevance. A chunk about "vacation policy" could be added for a "password recovery" query.
- **How:** Replaced fixed `k=3` search with Two-Pass Filtering:
  - **Pass 1 (Relevance)**: Call `similarity_search_with_score(k=10, threshold=0.55)` → filter irrelevant chunks
  - **Pass 2 (Budget)**: Loop through filtered chunks, count tokens, stop at 2250 token limit (2500 - 10% safety margin)
  - **Gemini verification**: Only when `total_tokens > 2000`, verify with `count_tokens` API and remove chunks if exceeded
  - Added `count_tokens()` helper with fallback for unusual text (binary, malformed unicode)
  - Logs oversized chunks (>600 tokens) as warnings

---

### 6. `agent/dispatcher.py` — User Input Token Validation
- **Why:** `agentic_action()` had no input length limit. A 3000-line config file pasted by user destroyed memory budget.
- **How:** Added token validation at entry:
  - `MAX_USER_TOKENS = 500` with 10% safety margin
  - Try `tiktoken` first → if not installed, fall back to 1500 char limit
  - If approaching limit → verify with Gemini `count_tokens`
  - Reject oversized inputs with helpful message suggesting summarization
  - All edge cases handled: missing tiktoken, unusual text, Gemini API failure

---

## [2026-03-07] Interactive Model Selection + google.genai Migration

---

### 1. `requirements.txt` — Added `questionary`
- **Why:** Arrow-key interactive terminal selection was needed for the `/models` command and startup model picker. No built-in Python library supports this.
- **How:** Added `questionary>=2.0.0` under a new `# ── CLI interaction ──` section.

---

### 2. `agent/architect.py` — Added `list_available_models()` module-level function
- **Why:** The existing `_list_models()` method was broken — it returned only the first matching model and had the wrong return type (`str` instead of `List[str]`). A module-level function was needed so `main.py` could call it before the agent is initialised.
- **How:** Replaced `_list_models()` with a standalone `list_available_models() -> List[str]` that creates a `genai.Client` and returns all models where `'generateContent'` is in `supported_actions`.

---

### 3. `agent/architect.py` — `__init__` accepts `model` param
- **Why:** The model was hardcoded as `"gemini-3.1-flash-lite-preview"`, preventing the user from selecting a different model at startup or switching mid-session.
- **How:** Changed signature to `def __init__(self, model: str = DEFAULT_MODEL)` and passed `model` to `ChatGoogleGenerativeAI`.

---

### 4. `agent/architect.py` — Fixed `genai.configure()` order bug
- **Why:** `genai.configure(api_key=None)` was being called before the API key was validated, meaning an invalid state was set silently before the `ValueError` was raised.
- **How:** Moved the `genai.configure()` call to after the key check. Later removed entirely (see change #8).

---

### 5. `agent/architect.py` — Added `switch_model()` method
- **Why:** Users needed to switch the active Gemini model mid-session (via `/models` command) without losing conversation history.
- **How:** Creates a new `ChatGoogleGenerativeAI` instance with the new model and rebinds tools via `self.llm.bind_tools(self._tools)`. Memory is untouched.

---

### 6. `agent/architect.py` — Removed broken `_list_models()` method
- **Why:** It returned only the first model (early `return` inside a `for` loop), had the wrong return type, and was replaced by `list_available_models()`.
- **How:** Deleted entirely.

---

### 7. `agent/dispatcher.py` — Added `init_architect()` and `switch_model()`
- **Why:** `main.py` needed to pre-initialise the agent with the user-selected model before the chat loop starts, and switch models mid-session without recreating the agent from scratch.
- **How:** `init_architect(model)` creates the singleton with the chosen model. `switch_model(model)` calls `_architect.switch_model()` if the agent exists, or creates it fresh if not.

---

### 8. `agent/architect.py` — Migrated from `google.generativeai` to `google.genai`
- **Why:** `google.generativeai` is deprecated and no longer receiving updates. The new `google.genai` SDK is the officially supported replacement.
- **How:**
  - Changed `import google.generativeai as genai` → `from google import genai`
  - Replaced `genai.configure()` + `genai.list_models()` → `genai.Client(api_key).models.list()`
  - Replaced `m.supported_generation_methods` → `m.supported_actions` (field renamed in new SDK)
  - Removed `genai.configure()` from `__init__` entirely — it was always redundant since `ChatGoogleGenerativeAI` takes `google_api_key` directly

---

### 9. `agent/__init__.py` — Exported new symbols
- **Why:** `main.py` needed access to `list_available_models`, `init_architect`, and `switch_model` from the `agent` package.
- **How:** Added imports from `architect` and `dispatcher`, added all three to `__all__`.

---

### 10. `main.py` — Added `pick_model()` and startup model selection
- **Why:** Users needed to choose a Gemini model at startup before the agent is initialised.
- **How:** `pick_model()` calls `list_available_models()`, renders an arrow-key menu via `questionary.select()` with the default pre-selected, and returns the chosen model name. Called in `main()` before `init_architect()` and `print_banner()`.

---

### 11. `main.py` — Added `/models` command
- **Why:** Users needed a way to switch models mid-session without restarting the program.
- **How:** Added `/models` case in `handle_command()` that calls `pick_model()` then `switch_model()`. Conversation memory is preserved.

---

### 12. `main.py` — `print_banner()` accepts `model` param
- **Why:** The banner was reading `os.getenv('GEMINI_MODEL')` which could be stale or unset. It should always show the model actually in use.
- **How:** Changed signature to `print_banner(model: str)` and replaced the `os.getenv()` call with the passed param.

---

## [2026-03-07] Fix unreadable agent response in terminal

### 13. `agent/architect.py` — Extract plain text from structured response content
- **Why:** Newer Gemini models with thinking enabled (e.g. `gemini-2.5-flash`) return `response.content` as a list of typed content blocks `[{'type': 'text', 'text': '...', 'extras': {...}}]` instead of a plain string. This caused the raw list to be printed directly in the terminal.
- **How:** After the LLM responds, check if `response.content` is a `list`. If so, extract only the blocks where `type == "text"` and join their `text` values. If it's already a plain string, use it directly.