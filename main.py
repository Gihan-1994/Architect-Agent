"""
main.py - CLI Interface for the Software Architect Agent

HOW TO RUN:
    python main.py                        # will prompt you for save path
    python main.py --save-path ./my-docs  # specify save path upfront

COMMANDS DURING CHAT:
    /clear   → Reset conversation history (but agent stays alive)
    /tokens  → Show estimated token usage of current memory
    /docs    → List all saved documents
    /quit    → Exit the program

WHAT HAPPENS UNDER THE HOOD:
  Your input → agentic_action() → SoftwareArchitectAgent → Gemini LLM
  If Gemini decides a tool is needed → tool runs → result fed back to Gemini
  Gemini writes final response → displayed here
"""

import os
import sys
import argparse
import questionary
from dotenv import load_dotenv
from agent import (
    agentic_action, reset_conversation, get_token_estimate, get_exact_token_count, set_save_path,
    list_available_models, init_architect, switch_model, setup_logging,
    agentic_action_langgraph,
)
# Load .env file (must happen before importing agent, which reads env vars)
load_dotenv()



# ── Initialize logging ─────────────────────────────────────────────────────
# Set debug=False for INFO level (normal operation)
# Set debug=True for DEBUG level (verbose, for troubleshooting)
setup_logging(debug=True)

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
_current_thread_id: str | None = None
_use_langgraph = False  # Toggle between original and LangGraph mode


# ─────────────────────────────────────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pick_model() -> str:
    """
    Fetch available Gemini models and let the user pick one interactively.

    Uses questionary for arrow-key navigation:
      - Press Enter on the pre-selected default to use it
      - Press ↓/↑ to navigate, then Enter to confirm another choice

    Returns:
        The selected model name string.
    """
    print("Fetching available Gemini models...\n")
    try:
        models = list_available_models()
    except Exception:
        return DEFAULT_MODEL

    if not models:
        return DEFAULT_MODEL

    # Pre-select the default model in the list, fall back to first entry
    default = next((m for m in models if DEFAULT_MODEL in m), models[0])

    chosen = questionary.select(
        "Select a Gemini model (↑↓ to navigate, Enter to confirm):",
        choices=models,
        default=default,
    ).ask()

    return chosen or default


def print_banner(model: str) -> None:
    """Print the welcome banner showing the active model."""
    print(f"   🤖 Model: {model}")
    print("""
╔════════════════════════════════════════════════════════════╗
║      🏗️   Software Architect Agent  (CLI Mode)            ║
║      Powered by Google Gemini 1.5 Flash + LangChain        ║
╠════════════════════════════════════════════════════════════╣
║  I can create:                                             ║
║    • System Design Documents                               ║
║    • API Specifications                                    ║
║    • Database Schemas / ERDs (Mermaid)                     ║
║    • Sequence / Flow Diagrams (Mermaid)                    ║
╠════════════════════════════════════════════════════════════╣
║  Commands: /clear  /tokens  /docs  /quit  /models  /langgraph ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")


def ask_save_path() -> str:
    """
    Prompt the user for a save directory.
    Returns the path string (creates the directory if needed).
    """
    print("📁 Where should generated documents be saved?")
    print("   Press Enter to use './output', or type a custom path:")
    raw = input("   Save path: ").strip()
    path = raw if raw else "./output"
    os.makedirs(path, exist_ok=True)
    print(f"   ✅ Documents will be saved to: {os.path.abspath(path)}\n")
    return path


def handle_command(cmd: str) -> bool:
    """
    Handle a slash command entered by the user.

    Returns:
        True  if the command was handled (caller should NOT send to agent)
        False if the command is not recognised (treat as normal message)
    """
    cmd_lower = cmd.lower().strip()

    if cmd_lower == "/models":
        chosen = pick_model()
        switch_model(chosen)
        print(f"✅ Model switched to: {chosen}\n")
        return True

    if cmd_lower == "/langgraph":
        global _use_langgraph
        _use_langgraph = not _use_langgraph
        status = "ENABLED" if _use_langgraph else "DISABLED"
        print(f"✅ LangGraph mode: {status}\n")
        if _use_langgraph:
            print("   Features: HITL approval, state persistence, Self-RAG\n")
        return True

    if cmd_lower == "/quit":
        print("👋 Goodbye!")
        sys.exit(0)

    elif cmd_lower == "/clear":
        reset_conversation()
        print("🔄 Conversation history cleared. Starting fresh.\n")
        return True

    elif cmd_lower == "/tokens":
        counts = get_exact_token_count()
        if counts["method"] == "exact":
            print("📊 Token Usage (exact counts from Gemini API):")
            print(f"   Total:        {counts['total']:,} tokens")
            print(f"   Conversation: {counts['conversation']:,} tokens")
            print(f"   RAG Context:  {counts['rag_context']:,} tokens")
            print(f"   System Prompt: {counts['system_prompt']:,} tokens")
        elif counts["method"] == "approximate":
            print(f"📊 Token Usage (approximate): ~{counts['total']:,} tokens")
            print("   Note: Install tiktoken for exact counts")
        else:
            print("📊 Token Usage: ~0 tokens (agent not initialized)")
        print()
        return True

    elif cmd_lower == "/docs":
        # Let the agent handle listing (uses the list_documents tool)
        return False  # Pass "/docs" → agent will call list_documents tool

    return False  # Not a known command


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Software Architect Agent — create architecture documents via chat"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory to save generated markdown documents (default: ./output)",
    )
    args = parser.parse_args()

    # ── Model selection ───────────────────────────────────────────────────
    chosen_model = pick_model()
    init_architect(chosen_model)

    # ── Welcome banner ────────────────────────────────────────────────────
    print_banner(chosen_model)

    # ── Set save path ─────────────────────────────────────────────────────
    save_path = args.save_path if args.save_path else ask_save_path()
    set_save_path(save_path)

    # ── Conversation loop ─────────────────────────────────────────────────
    print('💬 Start chatting! Type "/quit" to exit.\n')
    print("Example: Create a system design for a food delivery platform\n")

    while True:
        try:
            # Read user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Check for slash commands first
            if user_input.startswith("/"):
                handled = handle_command(user_input)
                if handled:
                    continue
                # If /docs or unknown → fall through to agent

            # Send to the agent
            print("\n🤔 Thinking...\n")

            if _use_langgraph:
                # LangGraph mode with HITL support
                global _current_thread_id
                result = agentic_action_langgraph(user_input, thread_id=_current_thread_id)
                _current_thread_id = result["thread_id"]

                if result["needs_approval"]:
                    # Human-in-the-Loop approval prompt
                    tool = result["tool_name"]
                    print(f"\n⚠️  Approval needed: {tool}")
                    print("   This will save/modify a document.")
                    approval = input("   Approve? [y/n]: ").strip().lower()

                    if approval in ["y", "yes"]:
                        print("\n✅ Approved. Executing...\n")
                        result = agentic_action_langgraph(
                            "",
                            thread_id=_current_thread_id,
                            resume_approval=True,
                            approval_decision=True,
                        )
                    else:
                        print("\n❌ Rejected. Cancelling operation.\n")
                        result = agentic_action_langgraph(
                            "",
                            thread_id=_current_thread_id,
                            resume_approval=True,
                            approval_decision=False,
                        )

                response = result["response"]
            else:
                # Original mode (backward compatible)
                response = agentic_action(user_input)

            print(f"🏗️  Architect: {response}\n")
            print("─" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("   Try again, or type /clear to reset.\n")


if __name__ == "__main__":
    main()
