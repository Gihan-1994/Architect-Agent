# MCP Integration Plan — Software Architect Agent

> **Goal:** Evolve the agent's tools from hardwired LangChain functions into a proper
> MCP (Model Context Protocol) server, and optionally connect to external MCP servers
> to give the agent new abilities (e.g. reading real code files from disk).
>
> This plan maps directly to **Lab 7** of your LangChain course:
> - Task 1 → Phase 1 (FastMCP server)
> - Task 2 → Phase 2 (MultiServerMCPClient integration)
> - Task 3 → Phase 3 (Multiple servers)

---

## Table of Contents

1. [What is MCP and Why Integrate It?](#1-what-is-mcp-and-why-integrate-it)
2. [Current vs Target Architecture](#2-current-vs-target-architecture)
3. [New File Structure](#3-new-file-structure)
4. [Phase 1 — Create the Architect MCP Server](#4-phase-1--create-the-architect-mcp-server)
5. [Phase 2 — Connect Agent to MCP Server](#5-phase-2--connect-agent-to-mcp-server)
6. [Phase 3 — Add External MCP Servers](#6-phase-3--add-external-mcp-servers)
7. [Async Migration Guide](#7-async-migration-guide)
8. [New Packages Required](#8-new-packages-required)
9. [Testing Checklist](#9-testing-checklist)

---

## 1. What is MCP and Why Integrate It?

### What is MCP?

MCP (Model Context Protocol) is a standard protocol that lets AI agents communicate
with external tool servers over a well-defined interface. Think of it as a **USB port
for AI** — any agent that speaks MCP can plug into any MCP server.

```
┌─────────────────┐     MCP Protocol     ┌─────────────────┐
│   AI Agent      │◄────────────────────►│   MCP Server    │
│  (LangChain)    │   stdio/SSE/HTTP     │   (Your Tools)  │
└─────────────────┘                      └─────────────────┘
```

An MCP server has three components:

```
┌──────────────────────────────────────┐
│            MCP Server                │
├──────────────────────────────────────┤
│ 1. Tools     → functions the AI calls│
│ 2. Resources → files, data, configs  │
│ 3. Prompts   → pre-defined templates │
└──────────────────────────────────────┘
```

### Why integrate it into this project?

| Without MCP (current) | With MCP (target) |
|---|---|
| Tools are hardwired into `architect.py` | Tools live in a standalone server process |
| Only this agent can use the tools | Any agent, app, or Claude Desktop can connect |
| Adding a new tool means editing agent code | Add a new `@mcp.tool()` — agent auto-discovers it |
| No access to external tools | Connect to filesystem, GitHub, or any public MCP server |
| Synchronous only | Async-ready — supports concurrent tool calls |

---

## 2. Current vs Target Architecture

### Current (hardwired tools)

```
architect.py
  │
  ├── imports save_document from tools.py    ← direct Python import
  ├── imports list_documents from tools.py   ← direct Python import
  └── binds them with llm.bind_tools([...])
```

### Target (MCP-based tools)

```
architect.py
  │
  └── MultiServerMCPClient({
          "architect": architect_server.py,   ← subprocess via stdio
          "filesystem": filesystem_server,    ← (Phase 3) reads real files
      })
        │
        └── tools = await client.get_tools()  ← dynamically loaded
            llm.bind_tools(tools)             ← same binding, different source
```

The agent's ReAct loop does not change. Only where the tools come from changes.

---

## 3. New File Structure

```
architect-agent/
│
├── agent/
│   ├── __init__.py          ← updated exports
│   ├── dispatcher.py        ← updated: async wrapper
│   ├── architect.py         ← updated: MultiServerMCPClient
│   ├── memory.py            ← unchanged
│   ├── tools.py             ← kept as fallback reference
│   └── vector_memory.py     ← unchanged
│
├── mcp_servers/             ← NEW DIRECTORY
│   ├── __init__.py          ← new (empty)
│   └── architect_server.py  ← new: FastMCP server with save/list tools
│
├── main.py                  ← updated: asyncio.run() wrapper
├── app.py                   ← updated: async Gradio handler
├── requirements.txt         ← add: mcp, langchain-mcp-adapters
└── MCP-INTEGRATION-PLAN.md  ← this file
```

---

## 4. Phase 1 — Create the Architect MCP Server

### Reason

Right now `save_document` and `list_documents` are plain Python functions decorated
with LangChain's `@tool`. To make them MCP-compatible, we wrap them in a `FastMCP`
server — exactly the same pattern as Lab 7 Task 1 (the Calculator server).

The server runs as a **subprocess** communicating over `stdio`. The agent launches it,
sends tool call requests over stdin, reads results from stdout.

### Pseudocode

```
// mcp_servers/architect_server.py

// 1. Create FastMCP server instance
mcp = FastMCP("ArchitectServer")

// 2. Expose save_document as an MCP tool
@mcp.tool()
function save_document(filename, content, save_path):
    // Same logic as current tools.py save_document
    // sanitise filename → write .md file → index in ChromaDB
    return "✅ Document saved: /path/to/file.md | Indexed N chunks"

// 3. Expose list_documents as an MCP tool
@mcp.tool()
function list_documents(save_path):
    // Same logic as current tools.py list_documents
    // list all .md files in save_path
    return "Files: ..."

// 4. Start the server over stdio (blocks until killed)
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Key points

- `FastMCP` is from the `mcp` package (same as your lab)
- `transport="stdio"` means the server communicates via standard input/output
- The server is a standalone script — you can test it independently before connecting the agent
- ChromaDB indexing logic stays the same — just moved into the server

---

## 5. Phase 2 — Connect Agent to MCP Server

### Reason

The agent needs to load its tools from the MCP server instead of importing them
directly. This is done with `MultiServerMCPClient` from `langchain-mcp-adapters`
— exactly the pattern from Lab 7 Task 2.

The client launches `architect_server.py` as a subprocess, fetches its tool list,
and returns standard LangChain tool objects. The rest of the agent (ReAct loop,
memory, RAG) stays exactly the same.

### Pseudocode

```
// Updated agent/architect.py

// On init:
client = MultiServerMCPClient({
    "architect": {
        "command": "python",
        "args": ["/path/to/mcp_servers/architect_server.py"],
        "transport": "stdio"
    }
})

// Async setup (called once at startup):
async function _setup_tools():
    tools = await client.get_tools()
    // tools is now a list of LangChain-compatible Tool objects
    // same shape as before — bind_tools() works unchanged
    llm_with_tools = llm.bind_tools(tools)
    tool_map = { t.name: t for t in tools }

// run() becomes async:
async function run(user_message):
    // Steps 1-6 remain identical
    // Only difference: tool execution may be async
    response = await llm_with_tools.ainvoke(messages)
    while response.tool_calls:
        for tool_call in response.tool_calls:
            result = await tool_map[tool_call.name].ainvoke(tool_call.args)
            messages.append(ToolMessage(result, tool_call_id=tool_call.id))
        response = await llm_with_tools.ainvoke(messages)
    return response.content
```

### Important: async ripple effect

Making `run()` async means everything that calls it must also be async:

```
main.py    → asyncio.run(main())
app.py     → async def respond(message, history)
dispatcher → async def agentic_action(message)
```

This is a small but important change — covered in the Async Migration Guide (Section 7).

---

## 6. Phase 3 — Add External MCP Servers

### Reason

With `MultiServerMCPClient`, you can connect to multiple MCP servers simultaneously.
All their tools are merged into one flat list and given to the LLM. This is the
pattern from Lab 7 Task 3 (Calculator + Weather servers).

For a software architect agent, two external servers are especially useful:

### Option A — Filesystem Server (read your actual codebase)

The official MCP filesystem server lets the agent read files from your computer.
This means you can say "analyse my NestJS project" and the agent will actually
read the code.

```
// Install the official MCP filesystem server (Node.js-based)
npm install -g @modelcontextprotocol/server-filesystem

// Connect in architect.py:
client = MultiServerMCPClient({
    "architect": {
        "command": "python",
        "args": ["mcp_servers/architect_server.py"],
        "transport": "stdio"
    },
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/home/gihan"          ← root path the agent can read
        ],
        "transport": "stdio"
    }
})
```

**New capabilities unlocked:**
- "Read my ecommerce-api project and generate an architecture document"
- "Look at my auth module and write an API spec for it"
- "Check what routes exist in my NestJS controller"

### Option B — GitHub Server (read repos directly)

```
// Connect to GitHub MCP server:
"github": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "transport": "stdio",
    "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token" }
}
```

**New capabilities unlocked:**
- "Analyse my GitHub repo and create a system design"
- "Read the open pull requests and suggest architectural improvements"

### How tool merging works

```
// tools from architect server:
//   → save_document, list_documents

// tools from filesystem server:
//   → read_file, list_directory, search_files, ...

// tools = await client.get_tools()
// Result: [ save_document, list_documents, read_file, list_directory, ... ]
//
// LLM now has ALL tools — it automatically picks the right one per request
```

---

## 7. Async Migration Guide

MCP clients are async. Here is every file that needs updating and exactly what changes:

### `main.py`

```python
# BEFORE (sync)
def main():
    response = agentic_action(user_input)

# AFTER (async)
import asyncio

async def main():
    response = await agentic_action(user_input)

if __name__ == "__main__":
    asyncio.run(main())
```

### `app.py` (Gradio)

```python
# BEFORE
def respond(message, history):
    return agentic_action(message)

# AFTER
async def respond(message, history):
    return await agentic_action(message)
```

### `agent/dispatcher.py`

```python
# BEFORE
def agentic_action(user_message: str) -> str:
    return architect_agent.run(simplified)

# AFTER
async def agentic_action(user_message: str) -> str:
    return await architect_agent.run(simplified)
```

### `agent/architect.py`

```python
# BEFORE
def run(self, user_message: str) -> str:
    response = self._llm_with_tools.invoke(messages)

# AFTER
async def run(self, user_message: str) -> str:
    response = await self._llm_with_tools.ainvoke(messages)
```

---

## 8. New Packages Required

```bash
pip install mcp langchain-mcp-adapters
```

Add to `requirements.txt`:

```
# ── MCP (Model Context Protocol) ─────────────────────────────
mcp>=1.0.0                     # FastMCP server framework — @mcp.tool() decorator
langchain-mcp-adapters>=0.1.0  # MultiServerMCPClient — connects agent to MCP servers
```

For Phase 3 filesystem server (Node.js — install separately):

```bash
npm install -g @modelcontextprotocol/server-filesystem
```

---

## 9. Testing Checklist

### Phase 1 — Test the MCP server standalone

```bash
# Run the server directly and send a raw tool call via stdin
python mcp_servers/architect_server.py
# (It will block waiting for MCP protocol messages)
# Press Ctrl+C to stop — if it starts without error, Phase 1 is working
```

### Phase 2 — Test the connected agent

```bash
python main.py
# Try: "Create a system design for a blog platform"
# Expected: agent calls save_document via MCP, file appears in output/
```

### Phase 3 — Test filesystem access

```bash
python main.py
# Try: "Read my NestJS project at /home/gihan/Nest backend/ecommerce-api and
#       generate an architecture document"
# Expected: agent reads files, then calls save_document with a generated doc
```

---

## Implementation Order

Implement one phase at a time. Each phase is self-contained and independently testable:

```
Phase 1  ──►  Phase 2  ──►  Phase 3
(server)      (connect)      (extend)
  │               │               │
test alone    test CLI        test with
  ✓           + Gradio          real files
```

---

*Plan for: Software Architect Agent — MCP Integration*
*Location: `/home/gihan/cowork/langchain/architect-agent/`*
*Lab reference: Lab 7 — MCP (Tasks 1, 2, 3)*
