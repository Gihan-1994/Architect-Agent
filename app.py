"""
app.py - Gradio Web Interface for the Software Architect Agent

HOW TO RUN:
    python app.py

Then open your browser at: http://localhost:7860

WHY GRADIO?
  Gradio is the simplest way to wrap a Python function in a web UI.
  It gives you a chat interface with zero HTML/CSS/JS — just Python.
  It's widely used in AI/ML demos and is free to use locally.

INTERFACE LAYOUT:
  ┌─────────────────────────────────┬──────────────────────┐
  │  Chat window                    │  Settings panel       │
  │  (shows conversation history)   │  - Save path input    │
  │                                 │  - Memory usage       │
  │  [Text input box]  [Send]       │  - Clear button       │
  │                                 │  - Example prompts    │
  └─────────────────────────────────┴──────────────────────┘
"""

import os
import gradio as gr
from dotenv import load_dotenv
from agent import agentic_action, reset_conversation, get_token_estimate, get_exact_token_count, set_save_path, get_save_path, setup_logging
from agent import agentic_action_langgraph
# Load environment variables before importing the agent
load_dotenv()



# ── Initialize logging ─────────────────────────────────────────────────────
# Set debug=False for INFO level (normal operation)
# Set debug=True for DEBUG level (verbose, for troubleshooting)
setup_logging(debug=False)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SAVE_PATH = "./output"
set_save_path(DEFAULT_SAVE_PATH)

# LangGraph session state
_session_thread_id: str | None = None
_pending_approval: dict | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Event Handlers
# ─────────────────────────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list,
    save_path: str,
    use_langgraph: bool = False,
) -> tuple:
    """
    Handle a single chat turn from the Gradio UI.

    Args:
        user_message: What the user typed.
        history:      Gradio chat history — list of [user_msg, bot_msg] pairs.
        save_path:    Directory to save documents (from the settings panel).
        use_langgraph: Enable LangGraph mode with HITL approval.

    Returns:
        ("", updated_history, approval_visible) — clears input, updates chat, shows/hides approval.
    """
    global _session_thread_id, _pending_approval

    if not user_message.strip():
        return "", history, gr.update(visible=False)

    # Update save path if the user changed it in the settings panel
    current_path = get_save_path()
    if save_path and save_path.strip() and save_path.strip() != current_path:
        set_save_path(save_path.strip())

    # Call the agent
    if use_langgraph:
        # LangGraph mode with HITL support
        result = agentic_action_langgraph(user_message, thread_id=_session_thread_id)
        _session_thread_id = result["thread_id"]

        if result["needs_approval"]:
            # Store pending approval for later resume
            _pending_approval = result

            # Show approval prompt in chat
            tool = result.get("tool_name", "unknown")
            approval_msg = (
                f"⚠️ **Approval Needed**: `{tool}`\n\n"
                f"This will save or modify a document.\n\n"
                f"Click **Approve** or **Reject** below to continue."
            )
            history.append([user_message, approval_msg])
            return "", history, gr.update(visible=True)  # Show approval buttons

        response = result["response"]
    else:
        # Original mode (backward compatible)
        response = agentic_action(user_message)

    # Append to Gradio history format: [[user, bot], ...]
    history.append([user_message, response])

    return "", history, gr.update(visible=False)  # Hide approval buttons


def clear_chat() -> list:
    """Reset conversation history and return an empty chat history."""
    reset_conversation()
    global _session_thread_id, _pending_approval
    _session_thread_id = None
    _pending_approval = None
    return []


def handle_approval(approved: bool, history: list) -> tuple:
    """
    Handle approval decision from UI buttons.

    Args:
        approved: True = approve, False = reject
        history: Current chat history

    Returns:
        (updated_history, approval_visible) — updates chat and hides buttons.
    """
    global _pending_approval, _session_thread_id

    if _pending_approval:
        result = agentic_action_langgraph(
            "",
            thread_id=_session_thread_id,
            resume_approval=True,
            approval_decision=approved,
        )

        if approved:
            status_msg = "✅ **Approved**. Executing operation..."
        else:
            status_msg = "❌ **Rejected**. Cancelling operation."

        # Update chat with result
        history.append([None, status_msg + "\n\n" + result["response"]])
        _pending_approval = None

    return history, gr.update(visible=False)  # Hide approval buttons


def update_token_display() -> str:
    """
    Return a formatted string showing current memory token count.

    Uses exact Gemini count_tokens when available, shows breakdown:
    - Total tokens
    - Conversation tokens (sliding window)
    - RAG context tokens (retrieved chunks)
    - System prompt tokens

    Falls back to rough estimate if Gemini API unavailable.
    """
    counts = get_exact_token_count()

    if counts["method"] == "exact":
        # Show breakdown with exact counts
        return (
            f"Total: {counts['total']:,} tokens (exact)\n"
            f"Conversation: {counts['conversation']:,}\n"
            f"RAG Context: {counts['rag_context']:,}\n"
            f"System Prompt: {counts['system_prompt']:,}"
        )
    elif counts["method"] == "approximate":
        # Rough estimate only
        return f"~{counts['total']:,} tokens (approximate)\nInstall tiktoken for exact counts"
    else:
        # Fallback (agent not initialized)
        return "~0 tokens in memory"


def update_save_path(path: str) -> str:
    """Apply a new save path from the settings panel."""
    if path and path.strip():
        set_save_path(path.strip())
        return f"✅ Saving to: {os.path.abspath(path.strip())}"
    return "⚠️ Please enter a valid path."


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI Definition
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_PROMPTS = [
    "Create a system design for a food delivery app with microservices",
    "Write an API specification for user authentication (JWT)",
    "Generate a database schema for an e-commerce platform",
    "Create a sequence diagram for the order checkout flow",
    "Design a high-level architecture for a real-time chat application",
    "List all saved documents",
]

with gr.Blocks(
    title="🏗️ Software Architect Agent",
    theme=gr.themes.Soft(),
    css="""
        .message-bubble { font-size: 14px; }
        footer { display: none !important; }
    """,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 🏗️ Software Architect Agent
        *Powered by **Google Gemini 1.5 Flash** + **LangChain***

        Create professional software architecture documents through natural conversation.
        """
    )

    with gr.Row():
        # ── Left: Chat Panel ──────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[],
                height=520,
                label="Conversation",
                bubble_full_width=False,
                show_copy_button=True,
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask me to create a system design, API spec, ERD or diagram…",
                    scale=5,
                    show_label=False,
                    container=False,
                    autofocus=True,
                )
                send_btn = gr.Button("Send ▶", variant="primary", scale=1, min_width=80)

            # LangGraph mode toggle
            langgraph_toggle = gr.Checkbox(
                value=False,
                label="🧪 LangGraph Mode (HITL approval)",
                info="Enable for state persistence + approval prompts",
            )

            # Approval buttons (hidden by default, shown when approval needed)
            with gr.Row(visible=False) as approval_row:
                approve_btn = gr.Button("✅ Approve", variant="primary", scale=1)
                reject_btn = gr.Button("❌ Reject", variant="secondary", scale=1)

            # Example prompts (clickable chips)
            gr.Markdown("**💡 Try these:**")
            example_btns = []
            for prompt in EXAMPLE_PROMPTS:
                btn = gr.Button(prompt, size="sm", variant="secondary")
                example_btns.append(btn)

        # ── Right: Settings Panel ─────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")

            save_path_box = gr.Textbox(
                value=DEFAULT_SAVE_PATH,
                label="📁 Save documents to:",
                placeholder="./output",
                info="Documents are saved here as .md files",
            )
            apply_path_btn = gr.Button("Apply Path", size="sm")
            path_status = gr.Textbox(
                value=f"✅ Saving to: {os.path.abspath(DEFAULT_SAVE_PATH)}",
                label="Status",
                interactive=False,
                lines=1,
            )

            gr.Markdown("---")

            token_display = gr.Textbox(
                value="~0 tokens in memory",
                label="📊 Memory Usage",
                interactive=False,
                lines=4,
                info="Exact token counts from Gemini API",
            )
            token_btn = gr.Button("Refresh Token Count", size="sm")

            gr.Markdown("---")

            clear_btn = gr.Button(
                "🔄 Clear Conversation",
                variant="secondary",
            )

            gr.Markdown(
                """
                ---
                ### 📄 Document Types
                - System Design
                - API Specification
                - Database Schema (ERD)
                - Sequence Diagram

                All documents saved as **Markdown** with
                embedded **Mermaid diagrams**.

                ---
                ### 🔑 Gemini Free Tier
                Get your free API key at
                [aistudio.google.com](https://aistudio.google.com/app/apikey)
                """
            )

    # ─────────────────────────────────────────────────────────────────────
    # Event Wiring
    # ─────────────────────────────────────────────────────────────────────

    # Send button - LangGraph mode support
    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, save_path_box, langgraph_toggle],
        outputs=[msg_box, chatbot, approval_row],
    )

    # Enter key in text box - LangGraph mode support
    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, save_path_box, langgraph_toggle],
        outputs=[msg_box, chatbot, approval_row],
    )

    # Approval buttons (HITL)
    approve_btn.click(
        fn=lambda h: handle_approval(True, h),
        inputs=[chatbot],
        outputs=[chatbot, approval_row],
    )
    reject_btn.click(
        fn=lambda h: handle_approval(False, h),
        inputs=[chatbot],
        outputs=[chatbot, approval_row],
    )

    # Clear conversation
    clear_btn.click(fn=clear_chat, outputs=[chatbot])

    # Token refresh
    token_btn.click(fn=update_token_display, outputs=[token_display])

    # Apply save path
    apply_path_btn.click(
        fn=update_save_path,
        inputs=[save_path_box],
        outputs=[path_status],
    )

    # Example prompt buttons — clicking fills the text box
    for btn in example_btns:
        btn.click(
            fn=lambda p=btn.value: p,
            inputs=None,
            outputs=[msg_box],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible from any network interface
        server_port=7860,
        share=False,              # set True to get a public ngrok URL
        inbrowser=True,           # auto-open browser tab
    )
