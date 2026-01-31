"""
Logging Setup Module
====================

JSON-based logging configuration for production use.
All logs are output to stdout in JSON format.

Log format:
    {
        "timestamp": "2024-01-27T12:00:00.000Z",
        "level": "INFO",
        "module": "main",
        "message": "Collector starting"
    }
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Any

import orjson


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Outputs log records as JSON objects with consistent fields:
    - timestamp: ISO8601 format with timezone
    - level: Log level name
    - module: Module name where log was created
    - message: Log message
    
    Additional fields from extra dict are included if present.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.
        
        Args:
            record: Log record to format.
            
        Returns:
            JSON-encoded log entry.
        """
        # Build base log entry
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields (excluding standard LogRecord attributes)
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message",
        }
        
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                # Ensure value is JSON-serializable
                log_entry[key] = self._make_serializable(value)
        
        # Serialize to JSON using orjson for performance
        return orjson.dumps(log_entry).decode("utf-8")
    
    def _make_serializable(self, value: Any) -> Any:
        """
        Convert value to JSON-serializable format.
        
        Handles:
        - dicts with non-string keys -> convert keys to strings
        - objects that aren't serializable -> convert to string
        """
        if isinstance(value, dict):
            return {str(k): self._make_serializable(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._make_serializable(v) for v in value]
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        else:
            # Fallback: convert to string
            return str(value)


class JsonLogHandler(logging.StreamHandler):
    """
    Stream handler that outputs JSON-formatted logs to stdout.
    """
    
    def __init__(self):
        super().__init__(stream=sys.stdout)
        self.setFormatter(JsonFormatter())


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application logging.
    
    Sets up JSON-formatted logging to stdout with the specified level.
    Removes any existing handlers and configures the root logger.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Example:
        >>> setup_logging("DEBUG")
        >>> logging.info("Application started")
        # {"timestamp":"2024-01-27T12:00:00.000Z","level":"INFO","module":"main","message":"Application started"}
    """
    # Get numeric level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add JSON handler
    json_handler = JsonLogHandler()
    json_handler.setLevel(numeric_level)
    root_logger.addHandler(json_handler)
    
    # Set root logger level
    root_logger.setLevel(numeric_level)
    
    # Reduce noise from third-party libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name, typically __name__.
        
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
