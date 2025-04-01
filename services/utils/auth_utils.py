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
import structlog
import bcrypt
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

# Added import for Flask context
from flask import current_app

# Configure logger
logger = structlog.get_logger(__name__)

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

# Removed: Global JWT constants - Use current_app.config
# JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default-super-secret-key")
# JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Removed: get_jwt_secret() - Use current_app.config['JWT_SECRET_KEY'] directly


def create_token(user_id: str,
                 username: str,
                 type: str = "access",
                 roles: Optional[List[str]] = None,
                 permissions: Optional[List[str]] = None,
                 expiry: Optional[int] = None) -> str:
    """
    Create a JWT token for a user, using settings from current_app.config.

    Args:
        user_id: User identifier.
        username: User's username.
        type: Token type ('access' or 'refresh').
        roles: List of user roles.
        permissions: List of user permissions.
        expiry: Expiration time in seconds (overrides config).

    Returns:
        JWT token string.

    Raises:
        MissingSecretError: If JWT secret is not configured in the app.
        RuntimeError: If token generation fails.
        RuntimeError: If called outside of an active Flask application context.
    """
    if not current_app:
        raise RuntimeError("Cannot create token outside of Flask application context.")

    # Ensure roles/permissions are lists
    roles = roles if roles is not None else []
    permissions = permissions if permissions is not None else []

    # Get config values from current_app
    secret_key = current_app.config.get('JWT_SECRET_KEY')
    algorithm = current_app.config.get('JWT_ALGORITHM', 'HS256')

    if not secret_key:
        logger.error("JWT_SECRET_KEY not found in Flask app config.")
        raise MissingSecretError("JWT secret key is not configured in the application.")

    # Determine expiry from args or config
    if expiry is None:
        if type == "access":
            # Use ACCESS_TOKEN_EXPIRES from config, default 3600
            expiry = current_app.config.get('ACCESS_TOKEN_EXPIRES', 3600)
        else:  # refresh token
            # Use REFRESH_TOKEN_EXPIRES from config, default 2592000 (30 days)
            expiry = current_app.config.get('REFRESH_TOKEN_EXPIRES', 2592000)

    now = int(time.time())
    exp_time = now + expiry

    payload = {
        "sub": user_id,
        "username": username,
        "type": type,
        "roles": roles,
        "permissions": permissions,
        "iat": now,
        "exp": exp_time,
        "jti": uuid.uuid4().hex # Add unique token identifier
    }

    try:
        token = jwt.encode(payload, secret_key, algorithm=algorithm)
        logger.info("JWT token generated successfully", user_id=user_id, username=username, token_type=type)
        return token
    except Exception as e:
        logger.error("Error generating JWT token", user_id=user_id, error=str(e), exc_info=True)
        raise RuntimeError(f"Token generation failed: {e}") from e


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token using settings from current_app.config.

    Args:
        token: JWT token string.

    Returns:
        Dictionary containing token payload if valid, otherwise None.

    Raises:
        RuntimeError: If called outside of an active Flask application context.
    """
    if not current_app:
        raise RuntimeError("Cannot decode token outside of Flask application context.")

    secret_key = current_app.config.get('JWT_SECRET_KEY')
    algorithm = current_app.config.get('JWT_ALGORITHM', 'HS256')

    if not secret_key:
        logger.error("JWT_SECRET_KEY not found in Flask app config during decode.")
        # Treat missing secret as an invalid token scenario for security
        # Raise an error instead of returning None, as this is a config issue
        raise MissingSecretError("JWT secret key is not configured.")

    # Default options for jwt.decode
    decode_options = {
        "verify_signature": True,
        "verify_exp": True,
        "verify_nbf": True,
        "verify_iat": True,
        "verify_aud": False, # Audience verification not typically used here
        "require": ["exp", "iat", "sub", "jti"] # Ensure required claims are present
    }

    try:
        # The leeway option accounts for clock skew between servers
        # Decode *without* expiry verification first to check invalidation list
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
            leeway=10, # Allow 10 seconds clock skew
            options={"verify_exp": False} # Decode even if expired initially
        )

        # Check if the token jti is in the invalidation list (implementation in the caller)
        # Now verify expiration manually if required
        if decode_options["verify_exp"] and payload['exp'] < time.time() - 10: # Apply leeway
             raise jwt.ExpiredSignatureError("Token has expired")

        logger.debug("JWT token decoded successfully", user_id=payload.get('sub'), jti=payload.get('jti'))
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("Attempted to use expired JWT token", jti=jwt.get_unverified_header(token).get('jti', 'unknown'))
        raise ExpiredTokenError("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT token provided", error=str(e))
        raise InvalidTokenError(f"Token is invalid: {e}")
    except Exception as e:
        # Log unexpected errors but raise a generic invalid token error
        logger.error("Unexpected error decoding JWT token", error=str(e), exc_info=True)
        raise InvalidTokenError("Failed to decode token due to an unexpected error.")


def hash_password(password: str) -> Tuple[str, str]:
    """
    Generate a secure hash of a password with a random salt using bcrypt.

    Args:
        password: Plain text password.

    Returns:
        Tuple of (hashed_password_str, salt_str).
    """
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt)

    logger.debug("Password hashed successfully using bcrypt")
    # Return as strings for database storage
    return hashed_password.decode('utf-8'), salt.decode('utf-8')


def verify_password(password: str, stored_hash: str, salt: Optional[str] = None) -> bool:
    """
    Verify a password against a stored bcrypt hash.
    The salt is embedded within the bcrypt hash itself.

    Args:
        password: Plain text password to verify.
        stored_hash: Previously stored bcrypt password hash.
        salt: Optional, not used for bcrypt but kept for compatibility (will be ignored).

    Returns:
        True if password matches, False otherwise.
    """
    # Convert inputs to bytes
    password_bytes = password.encode('utf-8')
    stored_hash_bytes = stored_hash.encode('utf-8')

    try:
        # bcrypt.checkpw handles salt extraction and comparison
        is_valid = bcrypt.checkpw(password_bytes, stored_hash_bytes)
        logger.debug("Password verification result (bcrypt)", is_valid=is_valid)
        return is_valid
    except ValueError as e:
        # Handle cases where the stored_hash is not a valid bcrypt hash
        logger.warning("Attempted to verify password against an invalid hash format", error=str(e))
        return False
    except Exception as e:
        logger.error("Unexpected error during password verification", error=str(e), exc_info=True)
        return False


def generate_api_key(prefix: str = "sk") -> str:
    """
    Generate a secure random API key with a prefix.

    Args:
        prefix: A short prefix for the key (e.g., 'sk' for secret key).

    Returns:
        API key string.
    """
    random_part = secrets.token_urlsafe(32) # Generate 32 random bytes, URL-safe base64 encoded
    key = f"{prefix}_{random_part}"
    logger.debug("Generated new API key", key_prefix=prefix)
    return key


def hash_api_key(api_key: str) -> str:
    """
    Generate a secure hash of an API key for storage using bcrypt.
    Note: Bcrypt is generally for passwords, but can be used here. SHA-256 might also be suitable.
          Using bcrypt makes verification consistent with passwords.

    Args:
        api_key: API key to hash.

    Returns:
        Hashed API key string.
    """
    hashed_key = bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt())
    logger.debug("API key hashed using bcrypt")
    return hashed_key.decode('utf-8')


def verify_api_key_hash(api_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against its stored bcrypt hash.

    Args:
        api_key: The plain text API key.
        stored_hash: The stored bcrypt hash of the key.

    Returns:
        True if the key matches the hash, False otherwise.
    """
    try:
        return bcrypt.checkpw(api_key.encode('utf-8'), stored_hash.encode('utf-8'))
    except ValueError:
        logger.warning("Attempted to verify API key against an invalid hash format")
        return False
    except Exception as e:
        logger.error("Unexpected error during API key hash verification", error=str(e), exc_info=True)
        return False


def has_role(payload: Dict[str, Any], role: str) -> bool:
    """
    Check if user payload contains a specific role.

    Args:
        payload: User data dictionary (e.g., from JWT or DB).
        role: Role name to check.

    Returns:
        True if user has the role, False otherwise.
    """
    roles = payload.get("roles", [])

    # Special case: "admin" role implicitly has all roles
    if "admin" in roles:
        return True

    return role in roles


def has_permission(payload: Dict[str, Any], permission: str) -> bool:
    """
    Check if user payload contains a specific permission.
    (Currently assumes permissions might be directly listed or inferred from roles).

    Args:
        payload: User data dictionary (e.g., from JWT or DB).
        permission: Permission name to check.

    Returns:
        True if user has the permission, False otherwise.
    """
    roles = payload.get("roles", [])
    permissions_list = payload.get("permissions", [])

    # Special case: "admin" role implicitly has all permissions
    if "admin" in roles:
        return True

    # Check direct permissions
    if permission in permissions_list:
        return True

    # Add logic here if permissions are derived from roles
    # e.g., if roles == ['editor'] and permission == 'edit_content': return True

    return False 