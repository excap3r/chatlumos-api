#!/usr/bin/env python3
"""
Authentication Utilities

Provides utility functions for JWT token handling, password hashing,
and permission/role checking.
"""

import os
import jwt
import time
import logging
import secrets
import hashlib
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# Set up logging
logger = logging.getLogger("auth_utils")

# Custom exceptions
class AuthenticationError(Exception):
    """Base class for authentication errors"""
    pass

class MissingSecretError(AuthenticationError):
    """JWT secret key is missing or invalid"""
    pass

class InvalidTokenError(AuthenticationError):
    """JWT token is invalid or corrupted"""
    pass

class ExpiredTokenError(AuthenticationError):
    """JWT token has expired"""
    pass

class InsufficientPermissionsError(AuthenticationError):
    """User lacks required permissions"""
    pass


def get_jwt_secret() -> str:
    """
    Get JWT secret key from environment or raise exception.
    
    Returns:
        str: JWT secret key
        
    Raises:
        MissingSecretError: If JWT secret is not configured
    """
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        logger.error("JWT_SECRET environment variable not set")
        raise MissingSecretError("JWT secret key is not configured")
    return secret


def create_token(user_id: str, 
                 username: str,
                 type: str = "access", 
                 roles: List[str] = None, 
                 permissions: List[str] = None,
                 expiry: int = None) -> str:
    """
    Create a JWT token for a user.
    
    Args:
        user_id: User identifier
        username: User's username
        type: Token type (access or refresh)
        roles: List of user roles
        permissions: List of user permissions
        expiry: Expiration time in seconds
        
    Returns:
        JWT token string
        
    Raises:
        MissingSecretError: If JWT secret is not configured
    """
    if roles is None:
        roles = []
    if permissions is None:
        permissions = []
    
    # Default expiry times
    if expiry is None:
        if type == "access":
            expiry = 3600  # 1 hour
        else:  # refresh token
            expiry = 2592000  # 30 days
    
    payload = {
        "sub": user_id,
        "username": username,
        "type": type,
        "roles": roles,
        "permissions": permissions,
        "iat": int(time.time()),
        "exp": int(time.time() + expiry)
    }
    
    return jwt.encode(payload, get_jwt_secret(), algorithm="HS256")


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Dict containing token payload
        
    Raises:
        MissingSecretError: If JWT secret is not configured
        InvalidTokenError: If token is invalid
        ExpiredTokenError: If token has expired
    """
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise ExpiredTokenError("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {str(e)}")
        raise InvalidTokenError(f"Invalid token: {str(e)}")


def hash_password(password: str) -> Tuple[str, str]:
    """
    Generate a secure hash of a password with a random salt.
    
    Args:
        password: Plain text password
        
    Returns:
        Tuple of (hashed_password, salt)
    """
    # Generate a random salt
    salt = secrets.token_hex(16)
    
    # Hash the password with the salt
    pwdhash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()
    
    return pwdhash, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """
    Verify a password against a stored hash and salt.
    
    Args:
        password: Plain text password to verify
        stored_hash: Previously stored password hash
        salt: Salt used for hashing
        
    Returns:
        True if password matches, False otherwise
    """
    calculated_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()
    
    return calculated_hash == stored_hash


def generate_api_key() -> str:
    """
    Generate a secure random API key.
    
    Returns:
        API key string
    """
    # Format: prefix.random_part
    prefix = "pdf_wisdom"
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}.{random_part}"


def hash_api_key(api_key: str) -> str:
    """
    Generate a secure hash of an API key for storage.
    
    Args:
        api_key: API key to hash
        
    Returns:
        Hashed API key
    """
    # Use a strong hash algorithm for API keys
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()


def has_role(payload: Dict[str, Any], role: str) -> bool:
    """
    Check if user has a specific role.
    
    Args:
        payload: JWT payload or user data
        role: Role to check
        
    Returns:
        True if user has the role, False otherwise
    """
    roles = payload.get("roles", [])
    
    # Special case: "admin" role has all roles
    if "admin" in roles:
        return True
    
    return role in roles


def has_permission(payload: Dict[str, Any], permission: str) -> bool:
    """
    Check if user has a specific permission.
    
    Args:
        payload: JWT payload or user data
        permission: Permission to check
        
    Returns:
        True if user has the permission, False otherwise
    """
    # Get user roles and permissions
    roles = payload.get("roles", [])
    permissions = payload.get("permissions", [])
    
    # Special case: "admin" role has all permissions
    if "admin" in roles:
        return True
    
    return permission in permissions 