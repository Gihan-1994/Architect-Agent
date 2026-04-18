"""
agent/logging_config.py — Structured JSON Logging Configuration

WHY THIS EXISTS:
  Provides a centralized logging setup that outputs structured JSON logs
  to both console and file. This enables:
    - Debugging slow responses (see which step takes time)
    - Tracking tool usage (when save_document/list_documents called)
    - Monitoring memory (token counts, chunk retrieval)
    - Machine-readable logs for analysis

USAGE:
  # At application startup (main.py or app.py):
  from agent.logging_config import setup_logging
  setup_logging(debug=False)  # INFO level
  setup_logging(debug=True)   # DEBUG level (verbose)

LOG FORMAT (JSON):
  {"timestamp": "2026-04-18T10:30:45", "level": "INFO", "module": "architect",
   "method": "run", "message": "Processing user message", "duration_ms": 150}

OUTPUT DESTINATIONS:
  - Console: stdout (visible in terminal)
  - File: logs/agent.log (persistent, can be analyzed later)
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as structured JSON.

    Each log record becomes a JSON object with:
      - timestamp: ISO 8601 format
      - level: INFO, DEBUG, WARNING, ERROR
      - module: which Python module (dispatcher, architect, tools, vector_memory)
      - method: which function/method was called
      - message: human-readable description
      - extra fields: duration_ms, tokens, chunks, etc. (when provided)
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "level": record.levelname,
            "module": record.name.split(".")[-1],  # e.g., "architect" from "agent.architect"
            "method": record.funcName if hasattr(record, "funcName") else "unknown",
            "message": record.getMessage(),
        }

        # Add extra fields if present (duration, tokens, chunks, etc.)
        if hasattr(record, "duration_ms"):
            log_obj["duration_ms"] = record.duration_ms
        if hasattr(record, "tokens"):
            log_obj["tokens"] = record.tokens
        if hasattr(record, "chunks"):
            log_obj["chunks"] = record.chunks
        if hasattr(record, "tool"):
            log_obj["tool"] = record.tool
        if hasattr(record, "model"):
            log_obj["model"] = record.model
        if hasattr(record, "score"):
            log_obj["score"] = record.score
        if hasattr(record, "saved_file"):
            log_obj["saved_file"] = record.saved_file

        return json.dumps(log_obj)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console output with colors.

    Shows the same JSON structure but formatted for easy reading:
      [INFO] architect.run → Processing user message (150ms)
    """

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        module = record.name.split(".")[-1]
        method = record.funcName

        # Build base message
        base = f"{color}[{record.levelname}]{reset} {module}.{method} → {record.getMessage()}"

        # Add extra fields as inline annotations
        extras = []
        if hasattr(record, "duration_ms"):
            extras.append(f"{record.duration_ms}ms")
        if hasattr(record, "tokens"):
            extras.append(f"{record.tokens} tok")
        if hasattr(record, "chunks"):
            extras.append(f"{record.chunks} chunks")
        if hasattr(record, "tool"):
            extras.append(f"tool={record.tool}")
        if hasattr(record, "results"):
            extras.append(f"{record.results} results")
        if hasattr(record, "threshold"):
            extras.append(f"thresh={record.threshold}")
        if hasattr(record, "rag"):
            extras.append(f"rag={record.rag}")
        if hasattr(record, "model"):
            extras.append(f"model={record.model}")
        if hasattr(record, "saved_file"):
            extras.append(f"file={record.saved_file}")
        if hasattr(record, "score"):
            extras.append(f"score={record.score}")

        if extras:
            base += f" ({', '.join(extras)})"

        return base


def setup_logging(debug: bool = False, log_dir: str = "logs") -> None:
    """
    Configure structured logging for the entire agent package.

    Args:
        debug: If True, enable DEBUG level (verbose). If False, INFO level only.
        log_dir: Directory for log files (default: logs/). Created if needed.

    Call this once at application startup before importing other agent modules.
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Determine log level
    level = logging.DEBUG if debug else logging.INFO

    # Configure root logger for agent package
    agent_logger = logging.getLogger("agent")
    agent_logger.setLevel(level)

    # Clear existing handlers (avoid duplicates on restart)
    agent_logger.handlers.clear()

    # ── Console Handler (human-readable with colors) ───────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ConsoleFormatter())
    agent_logger.addHandler(console_handler)

    # ── File Handler (structured JSON) ─────────────────────────────────────
    log_file = os.path.join(log_dir, "agent.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    agent_logger.addHandler(file_handler)

    # Log initialization
    agent_logger.info(
        "Logging initialized",
        extra={"model": "logging_config", "debug": debug, "log_file": log_file}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper function for timing operations
# ─────────────────────────────────────────────────────────────────────────────

import time
from functools import wraps
from typing import Callable

def log_duration(logger: logging.Logger) -> Callable:
    """
    Decorator to log method execution duration.

    Usage:
        @log_duration(logger)
        def my_method(self, ...):
            ...

    Logs both entry and exit with duration in milliseconds.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.debug(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                duration_ms = int((time.time() - start) * 1000)
                logger.info(
                    f"Completed {func.__name__}",
                    extra={"duration_ms": duration_ms}
                )
                return result
            except Exception as e:
                duration_ms = int((time.time() - start) * 1000)
                logger.error(
                    f"Failed {func.__name__}: {e}",
                    extra={"duration_ms": duration_ms}
                )
                raise
        return wrapper
    return decorator