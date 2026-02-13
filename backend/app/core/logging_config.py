"""
Structured JSON logging configuration with request_id and user_id context.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from app.core.config import settings

# Context variables for request tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[int | None] = ContextVar("user_id", default=None)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        user_id = user_id_var.get()
        if user_id:
            log_data["user_id"] = user_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """Plain text formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as plain text."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        message = record.getMessage()

        # Add context if available
        request_id = request_id_var.get()
        user_id = user_id_var.get()

        context_parts = []
        if request_id:
            context_parts.append(f"req={request_id[:8]}")
        if user_id:
            context_parts.append(f"user={user_id}")

        context = f" [{', '.join(context_parts)}]" if context_parts else ""

        base = f"{timestamp} [{record.levelname}] {record.name}{context}: {message}"

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


def setup_logging() -> None:
    """Configure application logging."""
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Choose formatter based on settings
    formatter = JSONFormatter() if settings.log_json else PlainFormatter()

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def set_request_context(request_id: str, user_id: int | None = None) -> None:
    """Set request context for logging."""
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context() -> None:
    """Clear request context."""
    request_id_var.set(None)
    user_id_var.set(None)
