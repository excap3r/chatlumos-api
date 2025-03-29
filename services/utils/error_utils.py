#!/usr/bin/env python3
"""
Error Handling Utilities

This module provides standardized error handling and custom exception classes.
"""

import sys
import traceback
import logging
from typing import Dict, Any, Optional, Tuple, Callable

# Initialize logger
logger = logging.getLogger('error_utils')

class APIError(Exception):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, status_code: int = 500, details: Any = None):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        error_dict = {
            "error": self.message,
            "status_code": self.status_code
        }
        
        if self.details:
            error_dict["details"] = self.details
            
        return error_dict

class ValidationError(APIError):
    """Error for validation failures."""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Any = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Name of the field that failed validation
            details: Additional error details
        """
        super().__init__(message, status_code=400, details=details)
        self.field = field
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation error to dictionary representation."""
        error_dict = super().to_dict()
        
        if self.field:
            error_dict["field"] = self.field
            
        return error_dict

def handle_error(func: Callable) -> Callable:
    """
    Decorator for API endpoint error handling.
    
    This decorator catches exceptions, logs them, and returns 
    standardized error responses.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            # Log API errors
            logger.error(f"API Error: {str(e)}")
            return e.to_dict(), e.status_code
        except Exception as e:
            # Log unexpected errors
            error_traceback = traceback.format_exc()
            logger.error(f"Unexpected error: {str(e)}\n{error_traceback}")
            
            # Return generic error response
            return {
                "error": "Internal server error",
                "status_code": 500,
                "details": str(e)
            }, 500
    
    return wrapper

def format_error_response(error: Exception, include_traceback: bool = False) -> Tuple[Dict[str, Any], int]:
    """
    Format a standardized error response.
    
    Args:
        error: The exception to format
        include_traceback: Whether to include the traceback in the response
        
    Returns:
        Tuple of (error_dict, status_code)
    """
    if isinstance(error, APIError):
        return error.to_dict(), error.status_code
    
    # Generic error
    error_dict = {
        "error": str(error),
        "status_code": 500
    }
    
    if include_traceback:
        error_dict["traceback"] = traceback.format_exc()
        
    return error_dict, 500 