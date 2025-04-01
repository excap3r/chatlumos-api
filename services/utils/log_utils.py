#!/usr/bin/env python3
"""
Logging Utilities

This module provides standardized logging setup and helper functions.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Any, Optional
from flask import Request, Response, g
import structlog

# --- structlog configuration ---

# Processors define how log records are enriched and formatted
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level, # Adds log level (e.g., 'info')
        structlog.stdlib.add_logger_name, # Adds logger name
        structlog.processors.TimeStamper(fmt="iso"), # Adds ISO timestamp
        structlog.processors.StackInfoRenderer(), # Adds stack info on exception
        structlog.dev.set_exc_info, # Adds exception info automatically
        structlog.processors.dict_tracebacks, # Formats tracebacks nicely
        structlog.processors.UnicodeDecoder(), # Decodes unicode strings
        # Add context variables (like request_id) if set via structlog.contextvars
        structlog.contextvars.merge_contextvars, 
        # Final step: Render the event dictionary to JSON
        structlog.processors.JSONRenderer() 
    ],
    # Use standard logging infrastructure for output
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# --- Standard logging setup (for handlers) ---

def setup_standard_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Sets up standard logging handlers (Console, File).
    structlog will use these handlers for output.
    """
    # Configure root logger handler
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers (important for reconfiguration)
    root_logger.handlers.clear() 

    # Create console handler (writes JSON formatted by structlog)
    console_handler = logging.StreamHandler(sys.stdout)
    # No formatter needed here, structlog handles the formatting
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except OSError as e:
                # Handle potential race condition if dir created between check and makedirs
                if not os.path.isdir(log_dir):
                    logging.getLogger(__name__).error(f"Failed to create log directory: {log_dir}", exc_info=True)
                    raise e
                    
        try:
            file_handler = logging.FileHandler(log_file)
            # No formatter needed
            root_logger.addHandler(file_handler)
        except Exception as e:
             logging.getLogger(__name__).error(f"Failed to create file handler for: {log_file}", exc_info=True)
             # Decide if this should be fatal or just log the error


# --- Deprecated function (use structlog.get_logger directly) ---
# Keep for now if other modules rely on it, but plan removal

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> structlog.stdlib.BoundLogger:
    """
    DEPRECATED: Use setup_standard_logging() once at startup, 
                 then structlog.get_logger() everywhere else.
                 
    Sets up standard logging handlers and returns a structlog logger.
    
    Args:
        name: Logger name (will be included in logs)
        log_file: Optional log file path for standard logging handler
        level: Logging level for standard logging handler
        
    Returns:
        Configured structlog logger instance
    """
    # Ensure standard logging is configured (can be called multiple times, but ideally once)
    # Note: This approach might reconfigure handlers repeatedly if called often.
    # Better: Call setup_standard_logging() once at app startup.
    setup_standard_logging(log_file=log_file, level=level) 
    
    # Return a structlog logger bound to the name
    return structlog.get_logger(name)


# --- Updated Request/Response Logging ---

def log_request(request: Request) -> None:
    """
    Log details about an HTTP request using structlog.
    Assumes structlog contextvars are used for request_id.
    """
    logger = structlog.get_logger('request_logger') # Get structlog logger
    
    # Store request start time for calculating duration
    g.request_start_time = time.time()
    
    # Extract request data safely
    request_data = None
    if request.is_json:
        try:
            request_data = request.get_json()
        except Exception:
            request_data = "<Invalid JSON>"
    elif request.form:
         request_data = dict(request.form)
    # Add handling for other content types if needed

    # Log request details as key-value pairs
    log_details = {
        # request_id should be added via contextvars if used
        "event": "request_received", # Clear event name
        "method": request.method,
        "url": request.url,
        "path": request.path,
        "remote_addr": request.remote_addr,
        "headers": dict(request.headers),
        "args": dict(request.args),
        "body": request_data 
    }
    
    # Remove sensitive information
    if "Authorization" in log_details["headers"]:
        log_details["headers"]["Authorization"] = "<redacted>"
    # Redact other sensitive headers if needed (e.g., Cookies)

    logger.info("Incoming request", **log_details)


def log_response(response: Response) -> None:
    """
    Log details about an HTTP response using structlog.
    Assumes structlog contextvars are used for request_id.
    """
    logger = structlog.get_logger('request_logger') # Get structlog logger
    
    # Calculate request duration
    duration_ms = None
    if hasattr(g, 'request_start_time'):
        duration_ms = (time.time() - g.request_start_time) * 1000 # Log duration in ms
    
    # Extract response data safely - Limit size?
    response_data = None
    # Avoid loading large response bodies into memory/logs
    if response.content_length is not None and response.content_length < 1024 * 10: # e.g., < 10KB
        try:
            if response.is_json:
                 response_data = json.loads(response.get_data(as_text=True))
            # Add handling for other response types if needed (e.g., text)
            # else:
            #    response_data = response.get_data(as_text=True) 
        except Exception:
            response_data = "<Non-JSON or Unparseable Response Body>"
    elif response.content_length is not None:
         response_data = f"<Response Body Too Large: {response.content_length} bytes>"
    else:
         response_data = "<Response Body Skipped (Unknown Length)>"

    # Log response details
    log_details = {
        "event": "response_sent",
        # request_id from contextvars
        "status_code": response.status_code,
        "duration_ms": duration_ms,
        "headers": dict(response.headers),
        "body": response_data
    }
    
    logger.info("Outgoing response", **log_details)


def initialize_logging(log_level_name: str = 'INFO', log_file: Optional[str] = None):
    """
    Call this once at application startup to configure logging handlers.
    """
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    setup_standard_logging(log_file=log_file, level=log_level)
    structlog.get_logger(__name__).info("Logging initialized", log_level=log_level_name, log_file=log_file or 'console')


# --- Flask Request Hook Setup ---

def setup_request_logging(app):
    """
    Set up Flask before/after request hooks for logging.
    Requires contextvars setup for request_id.
    """
    import uuid
    from structlog.contextvars import bind_contextvars, clear_contextvars

    @app.before_request
    def before_request_log():
        # Start context for each request
        clear_contextvars()
        request_id = str(uuid.uuid4())
        g.request_id = request_id # Keep in g for potential use elsewhere
        bind_contextvars(request_id=request_id) 
        
        log_request(request)

    @app.after_request
    def after_request_log(response):
        # Response logging happens before context is cleared
        log_response(response) 
        return response

    @app.teardown_request
    def teardown_request_log(exc=None):
        # Clear context after request is finished
        clear_contextvars() 