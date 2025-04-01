#!/usr/bin/env python3
"""
Error Handling Utilities

This module provides standardized error handling and custom exception classes.
"""

import sys
import traceback
import structlog
from typing import Dict, Any, Optional, Tuple, Callable

# Initialize logger
logger = structlog.get_logger(__name__)

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

class DatabaseError(APIError):
    """Error for database operation failures."""
    def __init__(self, message: str = "Database operation failed", details: Any = None):
        super().__init__(message, status_code=500, details=details)

class NotFoundError(APIError):
    """Error when a requested resource is not found."""
    def __init__(self, message: str = "Resource not found", details: Any = None):
        super().__init__(message, status_code=404, details=details)

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
    error_logger = logger
    
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            # Log API errors with details
            error_logger.warning("API Error occurred", 
                                 error_message=e.message, 
                                 status_code=e.status_code, 
                                 details=e.details, 
                                 exception_type=type(e).__name__)
            return e.to_dict(), e.status_code
        except Exception as e:
            # Log unexpected errors with traceback
            error_logger.error("Unexpected server error occurred", 
                               error=str(e), 
                               exception_type=type(e).__name__,
                               exc_info=True)
            
            # Return generic error response
            generic_error_response = {
                "error": "Internal server error",
            }
            return generic_error_response, 500
    
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