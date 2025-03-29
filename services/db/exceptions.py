#!/usr/bin/env python3
"""
Database Exceptions

Defines custom exceptions for database operations.
"""

class DatabaseError(Exception):
    """Base class for all database errors."""
    pass

class ConnectionError(DatabaseError):
    """Error occurred while connecting to the database."""
    pass

class QueryError(DatabaseError):
    """Error occurred while executing a database query."""
    pass

# User-related exceptions
class UserError(DatabaseError):
    """Base class for user-related errors."""
    pass

class UserNotFoundError(UserError):
    """User not found in the database."""
    pass

class UserAlreadyExistsError(UserError):
    """User already exists in the database."""
    pass

class InvalidCredentialsError(UserError):
    """Invalid user credentials."""
    pass

# API key related exceptions
class ApiKeyError(DatabaseError):
    """Base class for API key related errors."""
    pass

class ApiKeyNotFoundError(ApiKeyError):
    """API key not found in the database."""
    pass

class ApiKeyInvalidError(ApiKeyError):
    """API key is invalid or expired."""
    pass

class ApiKeyRevokedError(ApiKeyError):
    """API key has been revoked."""
    pass

# Permission related exceptions
class PermissionError(DatabaseError):
    """Base class for permission related errors."""
    pass

class InsufficientPermissionsError(PermissionError):
    """User lacks required permissions."""
    pass 