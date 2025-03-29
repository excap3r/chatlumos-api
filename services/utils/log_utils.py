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

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with standardized formatting.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file provided
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_request(request: Request, logger: Optional[logging.Logger] = None) -> None:
    """
    Log details about an HTTP request.
    
    Args:
        request: Flask request object
        logger: Optional logger to use (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Store request start time for calculating duration
    g.request_start_time = time.time()
    
    # Extract request data safely
    try:
        request_data = request.get_json() if request.is_json else None
    except Exception:
        request_data = "<Invalid JSON>"
    
    # Log request details
    log_data = {
        "request_id": getattr(g, 'request_id', None),
        "method": request.method,
        "url": request.url,
        "path": request.path,
        "remote_addr": request.remote_addr,
        "headers": dict(request.headers),
        "args": dict(request.args),
        "data": request_data
    }
    
    # Remove sensitive information
    if "Authorization" in log_data["headers"]:
        log_data["headers"]["Authorization"] = "<redacted>"
    
    logger.info(f"Request: {json.dumps(log_data, default=str)}")

def log_response(response: Response, logger: Optional[logging.Logger] = None) -> None:
    """
    Log details about an HTTP response.
    
    Args:
        response: Flask response object
        logger: Optional logger to use (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Calculate request duration
    duration = None
    if hasattr(g, 'request_start_time'):
        duration = time.time() - g.request_start_time
    
    # Extract response data safely
    try:
        response_data = json.loads(response.get_data(as_text=True))
    except Exception:
        response_data = "<Non-JSON response>"
    
    # Log response details
    log_data = {
        "request_id": getattr(g, 'request_id', None),
        "status_code": response.status_code,
        "duration": duration,
        "headers": dict(response.headers),
        "data": response_data
    }
    
    logger.info(f"Response: {json.dumps(log_data, default=str)}")

def create_request_logger(app):
    """
    Create a Flask before/after request logger.
    
    Args:
        app: Flask application
    """
    logger = logging.getLogger('request_logger')
    
    @app.before_request
    def before_request():
        log_request(request, logger)
    
    @app.after_request
    def after_request(response):
        log_response(response, logger)
        return response 