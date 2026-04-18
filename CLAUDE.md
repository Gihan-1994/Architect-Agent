# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

Software Architect Agent — an AI-powered tool that creates professional software architecture documents (system designs, API specs, ERDs, sequence diagrams) through natural conversation. Uses **Google Gemini** + **LangChain** with a two-layer memory system.

---

## Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key (copy .env.example to .env and add your key)
cp .env.example .env
# Edit .env: GOOGLE_API_KEY=your_actual_key_here
# Free key: https://aistudio.google.com/app/apikey
```

### Run
```bash
# Launch Gradio web UI (opens browser at http://localhost:7860)
python app.py
```

---

## Architecture

### Two-Layer Memory System

The agent uses **two independent memory layers** that serve different purposes:

| Layer | File | Purpose | Persistence |
|-------|------|---------|-------------|
| Short-term | `agent/memory.py` | Sliding window (last 5 exchanges) | RAM, lost on restart |
| Long-term | `agent/vector_memory.py` | Semantic search over saved documents | ChromaDB on disk |

**Key insight**: Retrieved context from vector memory is **injected fresh on each call**, NOT stored in the sliding window. This prevents document chunks from pushing out conversational turns.

### Agent Package Structure

```
agent/
├── __init__.py      # Public API exports
├── dispatcher.py    # Entry point — routes to appropriate agent
├── architect.py     # Core agent (ReAct loop + RAG + tool binding)
├── memory.py        # SlidingWindowMemory class
├── vector_memory.py # VectorMemory class (ChromaDB + embeddings)
└── tools.py         # LangChain @tool functions (save_document, list_documents)
```

### Execution Flow

```
User input → dispatcher.agentic_action()
                │
                ├─ _select_agent()        → "architect" (only agent currently)
                ├─ _simplify_instruction() → strip filler phrases
                └
                └─ architect.run()
                     │
                     ├─ memory.add_user_message()     → sliding window
                     ├─ _build_augmented_message()    → RAG retrieval
                     │      └
                     │      └─ vector_memory.search(k=3) → ChromaDB
                     │      └
                     │      └─ prepend context to user message
                     │
                     ├─ llm_with_tools.invoke()       → Gemini call
                     │      │
                     │      ├─ No tool_calls? → return response
                     │      └
                     │      └ Has tool_calls?
                     │           │
                     │           ├─ Execute tool (save_document → indexes in ChromaDB)
                     │           └
                     │           └─ llm_with_tools.invoke() again
                     │
                     └─ memory.add_ai_message()       → sliding window
                     │
                     └─ return final_text
```

### Key Technical Decisions

1. **Singleton pattern**: Both `SoftwareArchitectAgent` and `VectorMemory` use singletons to share state across the session
2. **Manual ReAct loop**: No LangGraph dependency — keeps implementation simple
3. **Tool indexing**: `save_document` tool automatically indexes content to ChromaDB after saving
4. **Embeddings**: all-MiniLM-L6-v2 (384 dimensions, runs on CPU)
5. **Text splitting**: 500-char chunks with 100-char overlap

---

## Code Change Instructions

### 1. Document All Changes
Every code change must be documented in `CHANGES.md` at the project root.
Each entry must include:
- **What changed** — file and line(s) affected
- **Why** — the root cause or requirement that triggered the change
- **How** — a brief description of the approach taken

### 2. Clarify Before Implementing
Never implement until the request is fully understood.
Ask clarifying questions one at a time until:
- The exact behaviour expected is clear
- Edge cases and constraints are understood
- The scope of the change is agreed upon
Only proceed when there is nothing left to clarify.

### 3. Clean and Optimised Code
Always suggest the cleanest, most readable, and most efficient solution.
- Prefer simple over clever
- Remove dead code and redundancy
- Follow existing patterns in the codebase
- Never over-engineer — implement only what the request requires

### 4. Show Pseudocode and Diff Before Implementing
Before making any code change, always present:
1. **Pseudocode** — plain English logic showing the intended flow
2. **Code preview** — the exact change that will be made (diff or full snippet)

Wait for explicit approval before applying the change.