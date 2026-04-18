# agent/__init__.py
# Exposes the public API of the agent package.
# Import agentic_action from here for clean usage in main.py and app.py.

from .dispatcher import agentic_action, reset_conversation, get_token_estimate, get_exact_token_count, init_architect, switch_model
from .tools import set_save_path, get_save_path
from .architect import list_available_models
from .logging_config import setup_logging

__all__ = [
    "agentic_action",
    "reset_conversation",
    "get_token_estimate",
    "get_exact_token_count",
    "set_save_path",
    "get_save_path",
    "list_available_models",
    "init_architect",
    "switch_model",
    "setup_logging",
]
