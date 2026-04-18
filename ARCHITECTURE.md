# Software Architect Agent — Full Architecture

> A LangChain-powered agent specialised for creating software architecture documents.
> Uses Google Gemini (free tier) with dual-layer memory, file persistence, and MCP tool integration.

---

## Table of Contents

1. [High-Level System Overview](#1-high-level-system-overview)
2. [Project File Structure](#2-project-file-structure)
3. [Component Responsibilities](#3-component-responsibilities)
4. [Dual-Layer Memory Architecture](#4-dual-layer-memory-architecture)
5. [Complete Request Lifecycle](#5-complete-request-lifecycle)
6. [Component-by-Component Pseudocode](#6-component-by-component-pseudocode)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [Dependency Map](#8-dependency-map)
9. [Technology Stack](#9-technology-stack)
10. [MCP Integration Overview](#10-mcp-integration-overview)

---

## 1. High-Level System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                           │
│                                                                  │
│   CLI (main.py)                  Web UI (app.py / Gradio)        │
│   python main.py                 python app.py → localhost:7860  │
└──────────────────────────┬───────────────────────────────────────┘
                           │ user message (plain string)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DISPATCHER (dispatcher.py)                    │
│                                                                  │
│   agentic_action(user_message)                                   │
│     1. _select_agent()      → decides which agent to use         │
│     2. _simplify_instruction() → strips filler words             │
│     3. routes to chosen agent                                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │ simplified message
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              SOFTWARE ARCHITECT AGENT (architect.py)             │
│                                                                  │
│   run(message)                                                   │
│     1. Save to sliding window (Layer 1 memory)                   │
│     2. Retrieve relevant chunks from ChromaDB (Layer 2 memory)   │
│     3. Augment message with retrieved context                    │
│     4. Send to Gemini LLM                                        │
│     5. Execute tool calls if any (ReAct loop)                    │
│     6. Return final response                                     │
└────────────┬───────────────────────────────┬─────────────────────┘
             │                               │
             ▼                               ▼
┌────────────────────────┐    ┌──────────────────────────────────┐
│  MEMORY LAYER 1        │    │  MEMORY LAYER 2                  │
│  SlidingWindowMemory   │    │  VectorMemory (vector_memory.py) │
│  (memory.py)           │    │                                  │
│                        │    │  all-MiniLM-L6-v2                │
│  Last 5 exchanges      │    │  RecursiveCharacterTextSplitter  │
│  in RAM                │    │  ChromaDB (persisted to disk)    │
│  Lost on session end   │    │  Survives restarts               │
└────────────────────────┘    └──────────────────────────────────┘
             │                               │
             └──────────────┬────────────────┘
                            ▼
             ┌──────────────────────────┐
             │   GOOGLE GEMINI 1.5 FLASH │
             │   (free tier LLM)         │
             └──────────────┬───────────┘
                            │ may call tools
                            ▼
             ┌──────────────────────────┐
             │   TOOLS (tools.py)        │
             │                          │
             │  save_document()         │──► .md file on disk
             │    + auto-indexes in     │──► ChromaDB on disk
             │      ChromaDB            │
             │                          │
             │  list_documents()        │──► reads output folder
             └──────────────────────────┘
```

---

## 2. Project File Structure

```
architect-agent/
│
├── main.py                  ← CLI entry point
├── app.py                   ← Gradio web UI entry point
├── requirements.txt         ← Python dependencies
├── .env.example             ← API key template (copy to .env)
├── .gitignore
├── ARCHITECTURE.md          ← this document
├── MCP-INTEGRATION-PLAN.md  ← MCP integration plan (Phase 1–3)
│
├── agent/                   ← Core agent package
│   ├── __init__.py          ← Public API exports
│   ├── dispatcher.py        ← agentic_action() — main router
│   ├── architect.py         ← SoftwareArchitectAgent class
│   ├── memory.py            ← SlidingWindowMemory (Layer 1)
│   ├── tools.py             ← save_document, list_documents tools (current)
│   └── vector_memory.py     ← VectorMemory / ChromaDB (Layer 2)
│
├── mcp_servers/             ← MCP server package (added in MCP Phase 1)
│   ├── __init__.py          ← empty init
│   └── architect_server.py  ← FastMCP server — exposes tools over stdio
│
├── output/                  ← Generated .md files (created at runtime)
│   ├── system-design.md
│   ├── auth-api-spec.md
│   └── ...
│
└── chroma_db/               ← ChromaDB vector store (created at runtime)
    ├── chroma.sqlite3        ← Metadata, chunk text, collection info
    └── <uuid>/              ← Binary embedding vectors (HNSW index)
        ├── data_level0.bin
        ├── header.bin
        └── ...
```

---

## 3. Component Responsibilities

| File | Class / Function | Single Responsibility |
|---|---|---|
| `dispatcher.py` | `agentic_action()` | Entry point — selects agent, simplifies instruction, routes |
| `dispatcher.py` | `_select_agent()` | Decides which agent handles the message |
| `dispatcher.py` | `_simplify_instruction()` | Strips filler phrases before passing to agent |
| `architect.py` | `list_available_models()` | Queries Gemini API and returns all chat-capable model names |
| `architect.py` | `SoftwareArchitectAgent` | Orchestrates LLM + memory + tools + ReAct loop |
| `architect.py` | `switch_model()` | Swaps the active Gemini model at runtime; preserves memory |
| `architect.py` | `_build_augmented_message()` | RAG read path — retrieves context and wraps message |
| `memory.py` | `SlidingWindowMemory` | Keeps last N exchanges in RAM to minimise tokens |
| `tools.py` | `save_document()` | Writes .md file to disk AND indexes in ChromaDB |
| `tools.py` | `list_documents()` | Reads output folder and returns file listing |
| `vector_memory.py` | `VectorMemory` | Chunks → embeds → stores (write) and searches (read) |
| `main.py` | `main()` | CLI loop — reads user input, calls agent, prints response |
| `app.py` | Gradio `demo` | Web UI — wraps `agentic_action()` in a browser chat interface |
| `mcp_servers/architect_server.py` | `FastMCP("ArchitectServer")` | Exposes save/list tools as an MCP server over stdio (Phase 1) |

---

## 4. Dual-Layer Memory Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         MEMORY COMPARISON                │
                    ├──────────────────┬──────────────────────┤
                    │   Layer 1        │   Layer 2            │
                    │   Sliding Window │   Vector Store       │
                    ├──────────────────┼──────────────────────┤
                    │ File             │ memory.py            │ vector_memory.py
                    │ Storage          │ RAM only             │ Disk (chroma_db/)
                    │ Survives restart │ ✗ No                 │ ✓ Yes
                    │ Content          │ Recent messages      │ All saved docs
                    │ Max size         │ 5 exchanges = ~10 msg│ Unlimited
                    │ Search type      │ Sequential (FIFO)    │ Semantic similarity
                    │ Purpose          │ Conversation flow    │ Document recall
                    │ Token cost       │ Bounded (always ≤N)  │ Only 3 chunks/call
                    └──────────────────┴──────────────────────┘
```

### How they work together in a single LLM call

```
messages sent to Gemini = [
    SystemMessage("You are a Senior Software Architect..."),  ← always first

    HumanMessage("Create an API spec"),     ← from sliding window (turn N-2)
    AIMessage("Here is your system..."),    ← from sliding window (turn N-2)

    HumanMessage("Revise the ERD"),         ← from sliding window (turn N-1)
    AIMessage("Updated ERD below..."),      ← from sliding window (turn N-1)

    HumanMessage(                           ← CURRENT TURN (augmented)
        "RELEVANT CONTEXT FROM SAVED DOCS:
         [From: system-design.md]
         ## Services: OrderService, UserService...
         ---
         [From: auth-api-spec.md]
         ## POST /auth/login  ...
         ══════════════════════
         USER REQUEST:
         Write a sequence diagram for checkout"
    )
]
```

---

## 5. Complete Request Lifecycle

### Pseudocode — full end-to-end flow

```
// ENTRY POINT
// ─────────────────────────────────────────────────────────────
function handle_user_input(raw_input):

    // INTERFACE LAYER (main.py or app.py)
    if raw_input is a slash command (/clear, /tokens, /docs):
        handle locally, do not send to agent
        return

    call agentic_action(raw_input)
    print the response


// DISPATCHER (dispatcher.py)
// ─────────────────────────────────────────────────────────────
function agentic_action(user_message):

    if user_message is empty:
        return "Please enter a message"

    // Step 1: Agent selection
    agent_type = _select_agent(user_message)
    // Currently always returns "architect"
    // Future: keyword matching or LLM classification

    // Step 2: Simplify instruction
    simplified = _simplify_instruction(user_message)
    // Strip leading filler: "could you please ", "can you ", etc.
    // Capitalise first letter

    // Step 3: Route to agent
    if agent_type == "architect":
        return architect_agent.run(simplified)


// ARCHITECT AGENT (architect.py)
// ─────────────────────────────────────────────────────────────
function run(user_message):

    // Step 1: Short-term memory — save original message
    sliding_window.add_user_message(user_message)
    // Internally trims to last 5 exchanges (10 messages)

    // Step 2: Long-term memory — retrieve relevant context
    augmented = _build_augmented_message(user_message)

    // Step 3: Build message list for LLM
    messages = sliding_window.get_messages()
    // = [SystemMessage, ...last 5 exchanges...]
    messages[-1] = HumanMessage(augmented)
    // Replace last message with the context-augmented version

    // Step 4: First LLM call
    response = gemini.invoke(messages)

    // Step 5: ReAct tool loop
    while response has tool_calls:
        messages.append(response)                 // add LLM's tool request
        for each tool_call in response.tool_calls:
            result = execute_tool(tool_call)      // run Python function
            messages.append(ToolMessage(result))  // add tool output
        response = gemini.invoke(messages)        // LLM sees tool result

    // Step 6: Save AI response to sliding window
    sliding_window.add_ai_message(response.content)

    return response.content


// RAG — READ PATH (architect.py)
// ─────────────────────────────────────────────────────────────
function _build_augmented_message(user_message):

    if vector_store has no documents:
        return user_message unchanged               // first-ever call

    docs = vector_store.search(user_message, k=3)
    // Internally:
    //   query_embedding = all-MiniLM-L6-v2.encode(user_message)
    //   → 384 numbers
    //   ChromaDB computes cosine_similarity(query_embedding, all_stored_embeddings)
    //   → returns top 3 Documents sorted by similarity score

    if docs is empty:
        return user_message unchanged

    context_block = format_context(docs)
    // "[ From: file.md ]\n chunk text \n --- \n [ From: other.md ] \n..."

    return:
        "RELEVANT CONTEXT FROM YOUR PREVIOUSLY SAVED DOCUMENTS:\n"
        + context_block
        + "\n══════════════════════════\n"
        + "USER REQUEST:\n"
        + user_message


// TOOL EXECUTION — save_document (tools.py)
// ─────────────────────────────────────────────────────────────
function save_document(filename, content):

    // Step 1: Sanitise filename
    clean_name = filename.lower().replace(" ", "-")
    if not clean_name ends with ".md":
        clean_name += ".md"

    full_path = output_folder + "/" + clean_name

    // Step 2: Write file to disk
    write header + content to full_path
    // header = "<!-- Generated by Agent | 2025-03-06 14:30 -->"

    // Step 3: RAG write path — index in ChromaDB
    chunks = text_splitter.split(content)
    // RecursiveCharacterTextSplitter: chunk_size=500, overlap=100
    // Tries to split on \n\n, then \n, then spaces, then characters

    for each chunk:
        attach metadata = { "source": clean_name }

    vector_store.add_documents(chunks)
    // Internally for each chunk:
    //   embedding = all-MiniLM-L6-v2.encode(chunk.text)
    //   → 384 numbers
    //   ChromaDB stores (text, embedding, metadata) to chroma_db/ on disk

    return "✅ Document saved: /path/to/file.md | Indexed N chunks"


// VECTOR MEMORY — VectorMemory class (vector_memory.py)
// ─────────────────────────────────────────────────────────────
class VectorMemory:

    on init:
        load HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        create RecursiveCharacterTextSplitter(chunk_size=500, overlap=100)
        point to persist_dir = "./chroma_db"
        do NOT load ChromaDB yet (lazy loading)

    function index_document(content, source_filename):
        chunks = splitter.create_documents(
            texts=[content],
            metadatas=[{ "source": source_filename }]
        )
        // Each chunk is a Document object:
        //   .page_content = "## Overview\nThis system..."
        //   .metadata     = { "source": "system-design.md" }

        chroma_store.add_documents(chunks)
        // ChromaDB calls HuggingFaceEmbeddings on each chunk automatically
        // Writes to chroma_db/chroma.sqlite3 + binary vector files

        return len(chunks)

    function search(query, k=3):
        if store is empty:
            return []
        return chroma_store.similarity_search(query, k=k)
        // Internally:
        //   embed query → 384 numbers
        //   HNSW (Hierarchical Navigable Small World) approximate nearest
        //   neighbour search across all stored embeddings
        //   return top k Documents

    function _get_store():
        if self._store is None:
            self._store = Chroma(
                collection_name="architect_docs",
                embedding_function=embeddings,
                persist_directory="./chroma_db"
            )
            // If chroma_db/ already has data → loads existing collection
            // If chroma_db/ is empty → creates fresh collection
        return self._store


// SLIDING WINDOW MEMORY (memory.py)
// ─────────────────────────────────────────────────────────────
class SlidingWindowMemory:

    on init(max_exchanges=5):
        max_messages = max_exchanges * 2    // 5 exchanges = 10 messages
        _messages = []                      // rolling list
        _system = None                      // system message (always kept)

    function add_user_message(content):
        _messages.append(HumanMessage(content))
        _trim()

    function add_ai_message(content):
        _messages.append(AIMessage(content))
        _trim()

    function _trim():
        if len(_messages) > max_messages:
            _messages = _messages[-max_messages:]
            // Drop oldest messages from the front
            // Keeps the window bounded — token cost never grows unbounded

    function get_messages():
        return [_system] + _messages
        // Always prepend system message so the LLM never loses its role
```

---

## 6. Component-by-Component Pseudocode

### `dispatcher.py` — Agent Selector

```
// _select_agent(message) — extensible routing logic
//
// TODAY:                     FUTURE EXTENSION:
// ─────────────────────────  ─────────────────────────────────────
// all messages → "architect" "security threat model" → "security"
//                            "docker compose setup"  → "devops"
//                            "optimize this query"   → "database"

function _select_agent(message):
    // Future: check keywords or call a classifier LLM
    return "architect"


// _simplify_instruction(message) — noise reducer
//
// Input:  "could you please create a system design for my app"
// Output: "Create a system design for my app"

function _simplify_instruction(message):
    message = message.strip()
    for phrase in ["could you please ", "can you please ", "i want you to ", ...]:
        if message starts with phrase (case-insensitive):
            remove the phrase from the front
            break
    capitalise first letter
    return message
```

### `memory.py` — Sliding Window Visualised

```
// State after 6 exchanges with max_exchanges=5:
//
// _messages before trim (12 items):
//   [H1, A1,  H2, A2,  H3, A3,  H4, A4,  H5, A5,  H6, A6]
//    ↑ oldest                                    newest ↑
//
// After _trim() (max_messages = 10):
//   dropped → [H1, A1]
//   kept    → [H2, A2,  H3, A3,  H4, A4,  H5, A5,  H6, A6]
//
// What Gemini receives:
//   [SystemMessage,  H2, A2,  H3, A3,  H4, A4,  H5, A5,  H6, A6_augmented]
//    always first ↑                                           ↑ current turn
```

### `vector_memory.py` — Embedding Visualised

```
// When index_document("# System Design\n## Services...", "system-design.md"):
//
// STEP 1: Split
//   chunk 1: "# System Design\n## Overview\nThe ecommerce platform..."  (500 chars)
//   chunk 2: "...ecommerce platform uses microservices.\n## Services\n"  (500 chars)
//             ↑ first 100 chars of chunk 2 overlap with last 100 of chunk 1
//   chunk 3: "...Services\n### OrderService handles..."                  (500 chars)
//   ...
//
// STEP 2: Embed each chunk (all-MiniLM-L6-v2 → 384 numbers)
//   chunk 1 → [0.12, -0.43, 0.88, 0.03, -0.21, ...]  384 floats
//   chunk 2 → [0.09, -0.47, 0.91, 0.01, -0.19, ...]  384 floats  ← similar!
//   chunk 3 → [0.55,  0.22, 0.11, 0.77,  0.33, ...]  384 floats
//
// STEP 3: Store in ChromaDB
//   Row: { text: "# System Design...", vector: [0.12, ...], source: "system-design.md" }
//   Row: { text: "...microservices...", vector: [0.09, ...], source: "system-design.md" }
//   Row: { text: "...OrderService...", vector: [0.55, ...], source: "system-design.md" }
//
//
// When search("checkout flow sequence diagram"):
//
// STEP 1: Embed the query
//   query → [0.61, 0.19, 0.08, 0.74, 0.28, ...]  384 floats
//
// STEP 2: Cosine similarity against every stored row
//   similarity("checkout flow...", chunk 1) = 0.21  ← low
//   similarity("checkout flow...", chunk 2) = 0.19  ← low
//   similarity("checkout flow...", chunk 3) = 0.87  ← HIGH ← returned
//   similarity("checkout flow...", other docs...) = ...
//
// STEP 3: Return top 3 by score
//   → [ Document(text="...OrderService...", source="system-design.md"), ... ]
```

---

## 7. Data Flow Diagrams

### Write Path (saving a document)

```
User: "Create a system design for ecommerce"
  │
  ▼
agentic_action()
  │
  ▼
architect_agent.run()
  │
  ├─► sliding_window.add_user_message("Create a system design...")
  │
  ├─► _build_augmented_message()  → ChromaDB empty → returns original message
  │
  ├─► gemini.invoke([system, user_msg])
  │         │
  │         └─ Gemini decides to call save_document("ecommerce-system-design", "# System...")
  │
  ├─► tools.save_document() executes:
  │         │
  │         ├─► writes  output/ecommerce-system-design.md  to disk  ✓
  │         │
  │         └─► vector_memory.index_document():
  │                   │
  │                   ├─ splits content → 8 chunks
  │                   ├─ embeds each chunk (all-MiniLM-L6-v2)
  │                   └─ stores 8 rows in chroma_db/  ✓
  │
  ├─► gemini.invoke([..., tool_result]) → "I've created your system design..."
  │
  ├─► sliding_window.add_ai_message("I've created...")
  │
  └─► return "I've created your system design..."
```

### Read Path (retrieving context next session)

```
User: "Now write an API spec consistent with the system design"
  │
  ▼
agentic_action()
  │
  ▼
architect_agent.run()
  │
  ├─► sliding_window.add_user_message("Now write an API spec...")
  │   (new session → window is empty except for this message)
  │
  ├─► _build_augmented_message("Now write an API spec..."):
  │         │
  │         ├─ vector_memory.search("Now write an API spec...", k=3)
  │         │       │
  │         │       ├─ embed query → 384 numbers
  │         │       ├─ cosine_similarity against all 8 stored chunks
  │         │       └─ returns top 3:
  │         │             chunk from "ecommerce-system-design.md" (score 0.82)
  │         │             chunk from "ecommerce-system-design.md" (score 0.74)
  │         │             chunk from "ecommerce-system-design.md" (score 0.69)
  │         │
  │         └─ builds augmented message:
  │               "RELEVANT CONTEXT FROM YOUR PREVIOUSLY SAVED DOCUMENTS:
  │                [From: ecommerce-system-design.md]
  │                ## Tech Stack: Node.js, PostgreSQL, Redis...
  │                ---
  │                [From: ecommerce-system-design.md]
  │                ## Services: OrderService, UserService, PaymentService...
  │                ══════════════════════════════
  │                USER REQUEST:
  │                Now write an API spec consistent with the system design"
  │
  ├─► gemini.invoke([system, augmented_message])
  │   Gemini sees the system design context → produces consistent API spec
  │
  ├─► tools.save_document("auth-api-spec", "# API Specification...")
  │         └─► indexes in ChromaDB too
  │
  └─► return "Here is your API specification..."
```

---

## 8. Dependency Map

```
main.py ──────────────────────────────────────────────────► agent/__init__.py
app.py  ──────────────────────────────────────────────────► agent/__init__.py
                                                                    │
                                              ┌─────────────────────┤
                                              │                     │
                                              ▼                     ▼
                                      dispatcher.py          tools.py (set_save_path)
                                              │
                                              ▼
                                       architect.py
                                         │       │
                              ┌──────────┘       └──────────┐
                              ▼                             ▼
                          memory.py                    tools.py
                    (SlidingWindowMemory)      (save_document, list_documents)
                                                         │
                                              ┌──────────┘
                                              ▼
                                      vector_memory.py ◄─── architect.py
                                    (VectorMemory,           (also imports
                                     get_vector_memory)       get_vector_memory)
                                              │
                              ┌───────────────┼───────────────────┐
                              ▼               ▼                   ▼
                    HuggingFaceEmbeddings  Chroma DB    RecursiveCharacter
                    (all-MiniLM-L6-v2)   (chroma_db/)  TextSplitter
```

**Key design rule:** `tools.py` and `architect.py` both import the **same singleton** (`get_vector_memory()`) from `vector_memory.py`. This guarantees they both read from and write to the same ChromaDB collection. There is no duplication.

---

## 9. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Google Gemini (free tier) | Free tier, fast, supports tool calling; model switchable at runtime |
| LLM Framework | LangChain | Tool binding, message types, LLM wrappers |
| Gemini SDK | google-genai | Official Google Gemini SDK — used for model listing |
| Embedding Model | all-MiniLM-L6-v2 | Small (90 MB), fast on CPU, 384 dimensions, excellent for semantic search |
| Text Splitting | RecursiveCharacterTextSplitter | Respects document structure — splits on paragraphs before characters |
| Vector Database | ChromaDB | Zero-config, persists to local disk, no server needed |
| Short-term Memory | Custom SlidingWindowMemory | Bounded token cost, preserves conversational flow |
| Web UI | Gradio | Zero HTML/CSS/JS — pure Python chat interface |
| Environment | python-dotenv | Keeps API key out of source code |
| MCP Server | FastMCP (`mcp` package) | Exposes tools as an MCP-compatible server over stdio |
| MCP Client | MultiServerMCPClient | Connects agent to one or more MCP servers; merges all tools |

---

## 10. MCP Integration Overview

MCP (Model Context Protocol) is a standard protocol for connecting AI agents to external
tool servers. It acts like a USB port — any agent that speaks MCP can plug into any
MCP server.

### Current tools vs MCP tools

```
CURRENT (hardwired)                    TARGET (MCP-based)
──────────────────────────────────     ─────────────────────────────────────
architect.py                           architect.py
  │                                      │
  ├── import save_document               └── MultiServerMCPClient({
  ├── import list_documents                      "architect": architect_server.py,
  └── llm.bind_tools([...])                      "filesystem": filesystem_server,
      ↑ direct Python import                 })
                                             tools = await client.get_tools()
                                             llm.bind_tools(tools)
                                             ↑ dynamically loaded from server
```

### Three-phase integration plan

```
Phase 1 ──────────────────────────────────────────────────────────────────
  Create mcp_servers/architect_server.py
  FastMCP server that exposes save_document and list_documents over stdio
  Lab 7 reference: Task 1 (Calculator server pattern)

Phase 2 ──────────────────────────────────────────────────────────────────
  Update agent/architect.py to use MultiServerMCPClient
  Load tools from the MCP server instead of direct Python imports
  Requires: async migration (asyncio.run in main.py, ainvoke in architect.py)
  Lab 7 reference: Task 2 (LangGraph + MCP integration pattern)

Phase 3 ──────────────────────────────────────────────────────────────────
  Add external MCP servers (filesystem, GitHub)
  Agent gains ability to read actual codebase files from disk
  Lab 7 reference: Task 3 (Multiple MCP servers pattern)
```

### Updated system overview (after Phase 2+3)

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                           │
│   CLI (main.py)                  Web UI (app.py / Gradio)        │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DISPATCHER (dispatcher.py)                    │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              SOFTWARE ARCHITECT AGENT (architect.py)             │
│   run() → memory → RAG → LLM → ReAct loop                       │
└──────────────┬─────────────────────────────┬─────────────────────┘
               │                             │
               ▼                             ▼
┌──────────────────────┐    ┌────────────────────────────────────┐
│  DUAL-LAYER MEMORY   │    │   MultiServerMCPClient             │
│  Layer 1: Sliding    │    │                                    │
│  Layer 2: ChromaDB   │    │  ┌─────────────────────────────┐  │
└──────────────────────┘    │  │  architect_server.py (stdio) │  │
                            │  │  save_document               │  │
                            │  │  list_documents              │  │
                            │  └─────────────────────────────┘  │
                            │  ┌─────────────────────────────┐  │
                            │  │  filesystem server (Phase 3) │  │
                            │  │  read_file, list_directory   │  │
                            │  └─────────────────────────────┘  │
                            └────────────────────────────────────┘
```

> Full implementation details are in `MCP-INTEGRATION-PLAN.md`

---

*Generated for: Software Architect Agent project*
*Location: `/home/gihan/cowork/langchain/architect-agent/`*
*Last updated: Added MCP integration overview and new agent methods*
