"""
Utilities Package

This package provides common utilities shared across services:
- Environment variable management
- API key handling
- Configuration management
- Error handling
- Logging utilities
"""

# from .env_utils import load_env_var, get_api_key, get_config # Removed missing import
from .error_utils import handle_error, APIError, ValidationError
from .log_utils import setup_logger, log_request, log_response

__all__ = [
    # 'load_env_var', # Removed
    # 'get_api_key', # Removed
    # 'get_config', # Removed
    'handle_error',
    'APIError',
    'ValidationError',
    'setup_logger',
    'log_request',
    'log_response'
] 