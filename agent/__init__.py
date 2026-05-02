# agent/__init__.py
# Exposes the public API of the agent package.
# Import agentic_action from here for clean usage in main.py and app.py.

from .dispatcher import agentic_action, reset_conversation, get_token_estimate, get_exact_token_count, init_architect, switch_model
from .dispatcher import agentic_action_langgraph, get_interrupted_state
from .tools import set_save_path, get_save_path
from .architect import list_available_models
from .logging_config import setup_logging

# NEW (Phase 1): Vector memory exports
from .vector_memory import get_vector_memory, VectorMemory, SemanticChunkingStrategy

# NEW (Phase 2): Hierarchical memory exports
from .hierarchical_memory import get_hierarchical_memory, HierarchicalVectorMemory, get_retriever

# NEW (Phase 3): LangGraph orchestration exports
from .langgraph_architect import get_langgraph_architect, LangGraphArchitect

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
    # Phase 1
    "get_vector_memory",
    "VectorMemory",
    "SemanticChunkingStrategy",
    # Phase 2
    "get_hierarchical_memory",
    "HierarchicalVectorMemory",
    "get_retriever",
    # Phase 3
    "get_langgraph_architect",
    "LangGraphArchitect",
    # Phase 4
    "agentic_action_langgraph",
    "get_interrupted_state",
]
