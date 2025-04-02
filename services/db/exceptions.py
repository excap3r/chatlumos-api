#!/usr/bin/env python3
"""
Database Exceptions

Defines custom exceptions for database operations, integrated with APIError for proper HTTP responses.
"""

from ..utils.error_utils import APIError, ValidationError, NotFoundError # Import base API errors

class DatabaseError(APIError):
    """Base class for generic database errors (500 Internal Server Error)."""
    def __init__(self, message: str = "A database error occurred", details: any = None):
        super().__init__(message, status_code=500, details=details)

class ConnectionError(DatabaseError):
    """Error occurred while connecting to the database (500 Internal Server Error)."""
    def __init__(self, message: str = "Database connection error", details: any = None):
        super().__init__(message, details=details)

class QueryError(DatabaseError):
    """Error occurred while executing a database query (500 Internal Server Error)."""
    def __init__(self, message: str = "Database query error", details: any = None):
        super().__init__(message, details=details)

# --- User-related exceptions --- 

class UserError(APIError):
    """Base class for user-related API errors."""
    # Often uses specific subclasses below with defined status codes.
    # Defaulting to 500 if used directly, but subclassing is preferred.
    def __init__(self, message: str = "User-related error", status_code: int = 500, details: any = None):
        super().__init__(message, status_code=status_code, details=details)

class UserNotFoundError(NotFoundError):
    """User not found in the database (404 Not Found). Inherits from general NotFoundError."""
    def __init__(self, message: str = "User not found", details: any = None):
        super().__init__(message, details=details)

class UserAlreadyExistsError(APIError):
    """User already exists in the database (409 Conflict)."""
    # Could also inherit ValidationError (400), but 409 is often more specific for existing resources.
    def __init__(self, message: str = "User already exists", details: any = None):
        super().__init__(message, status_code=409, details=details)

class InvalidCredentialsError(APIError):
    """Invalid user credentials (401 Unauthorized)."""
    def __init__(self, message: str = "Invalid username or password", details: any = None):
        super().__init__(message, status_code=401, details=details)

# --- General Data Integrity Exceptions ---

class DuplicateEntryError(APIError):
    """A unique constraint violation occurred (409 Conflict)."""
    def __init__(self, message: str = "Duplicate entry detected", details: any = None):
        super().__init__(message, status_code=409, details=details)

# --- API key related exceptions --- 

class ApiKeyError(APIError):
    """Base class for API key related errors (default 500)."""
    def __init__(self, message: str = "API Key error", status_code: int = 500, details: any = None):
        super().__init__(message, status_code=status_code, details=details)

class ApiKeyNotFoundError(NotFoundError):
    """API key not found (404 Not Found). Inherits from general NotFoundError."""
    def __init__(self, message: str = "API key not found", details: any = None):
        super().__init__(message, details=details)

class ApiKeyInvalidError(APIError):
    """API key is invalid (401 Unauthorized)."""
    def __init__(self, message: str = "Invalid or expired API key", details: any = None):
        super().__init__(message, status_code=401, details=details)

class ApiKeyRevokedError(APIError):
    """API key has been revoked (403 Forbidden)."""
    def __init__(self, message: str = "API key has been revoked", details: any = None):
        super().__init__(message, status_code=403, details=details)

# --- Permission related exceptions --- 

class PermissionError(APIError):
    """Base class for permission related errors (default 500)."""
    def __init__(self, message: str = "Permission error", status_code: int = 500, details: any = None):
        super().__init__(message, status_code=status_code, details=details)

class InsufficientPermissionsError(APIError):
    """User lacks required permissions (403 Forbidden)."""
    def __init__(self, message: str = "Insufficient permissions", details: any = None):
        super().__init__(message, status_code=403, details=details) 