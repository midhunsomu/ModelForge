"""
ModelMesh — Structured Logging Configuration
---------------------------------------------
Uses structlog for consistent JSON log output in production.
In development, pretty-prints with colors.
All logs include: timestamp, level, service, trace_id, and message.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict

import structlog
from structlog.types import EventDict, WrappedLogger

# ── Context variable for request tracing ─────────────────────────────────────
# This is set per-request and automatically injected into every log line
# produced during that request's lifecycle — even in nested function calls.
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
_model_version_var: ContextVar[str] = ContextVar("model_version", default="")


def get_trace_id() -> str:
    return _trace_id_var.get() or str(uuid.uuid4())


def set_trace_id(trace_id: str) -> None:
    _trace_id_var.set(trace_id)


def set_model_version(version: str) -> None:
    _model_version_var.set(version)


# ── Custom processors ─────────────────────────────────────────────────────────

def add_trace_id(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Inject the current request's trace_id into every log line."""
    trace_id = _trace_id_var.get()
    if trace_id:
        event_dict["trace_id"] = trace_id
    model_ver = _model_version_var.get()
    if model_ver:
        event_dict["model_version"] = model_ver
    return event_dict


def add_service_context(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Add service-level metadata to every log entry."""
    event_dict["service"] = "modelmesh-api"
    return event_dict


def drop_color_message_key(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """
    Uvicorn logs a 'color_message' key with ANSI codes.
    Drop it to keep JSON output clean.
    """
    event_dict.pop("color_message", None)
    return event_dict


# ── Setup function ────────────────────────────────────────────────────────────

def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Configure structlog for the application.
    Call once at startup in main.py.

    Args:
        log_level: e.g. "INFO", "DEBUG", "WARNING"
        log_format: "json" for production, "text" for local dev
    """
    log_level_num = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_trace_id,
        add_service_context,
        drop_color_message_key,
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level_num)

    # Silence noisy third-party loggers
    for noisy in ["uvicorn.access", "kafka", "mlflow.tracking"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a structlog logger bound to the given module name."""
    return structlog.get_logger(name)
