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
        Dictionary containing token payload if valid.

    Raises:
        MissingSecretError: If JWT secret is not configured.
        ExpiredTokenError: If the token has expired.
        InvalidTokenError: If the token is invalid for any other reason.
        RuntimeError: If called outside of an active Flask application context.
    """
    if not current_app:
        raise RuntimeError("Cannot decode token outside of Flask application context.")

    secret_key = current_app.config.get('JWT_SECRET_KEY')
    algorithm = current_app.config.get('JWT_ALGORITHM', 'HS256')

    if not secret_key:
        logger.error("JWT_SECRET_KEY not found in Flask app config during decode.")
        raise MissingSecretError("JWT secret key is not configured.")

    # Options for jwt.decode, including default leeway for clock skew
    decode_options = {
        "verify_signature": True,
        "verify_exp": True,
        "verify_nbf": True,
        "verify_iat": True,
        "verify_aud": False, # Audience verification not typically used here
        "require": ["exp", "iat", "sub", "jti"] # Ensure required claims are present
    }

    try:
        # Let jwt.decode handle expiry verification with leeway
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
            leeway=current_app.config.get('JWT_LEEWAY', 10), # Use config or default leeway
            options=decode_options
        )

        # Optional: Add check here for invalidated tokens (e.g., check JTI against a denylist)
        # if is_token_invalidated(payload.get('jti')):
        #     raise InvalidTokenError("Token has been invalidated.")

        logger.debug("JWT token decoded successfully", user_id=payload.get('sub'), jti=payload.get('jti'))
        return payload

    except jwt.ExpiredSignatureError:
        # Log JTI if possible, even from expired token header
        try:
            unverified_header = jwt.get_unverified_header(token)
            jti = unverified_header.get('jti', 'unknown')
        except Exception:
            jti = 'unknown'
        logger.warning("Attempted to use expired JWT token", jti=jti)
        raise ExpiredTokenError("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT token provided", error=str(e))
        raise InvalidTokenError(f"Token is invalid: {e}")
    except Exception as e:
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
    Ensures the random part does not contain underscores to allow reliable splitting.

    Args:
        prefix: A short prefix for the key (e.g., 'sk' for secret key).

    Returns:
        API key string in the format "prefix_randompart".
    """
    # Generate 32 random bytes, URL-safe base64 encoded
    random_part_raw = secrets.token_urlsafe(32)
    # Replace underscores to avoid conflicts with the prefix separator
    random_part = random_part_raw.replace('_', '-')
    key = f"{prefix}_{random_part}"
    logger.debug("Generated new API key", key_prefix=prefix)
    return key


def hash_api_key(api_key: str) -> str:
    """
    Generate a secure bcrypt hash of the SECRET part of an API key for storage.
    Assumes the key format is "prefix_secretPart".
    Includes automatically generated salt.

    Args:
        api_key: The plain text API key (e.g., 'sk_randompart').

    Returns:
        bcrypt hash string of the secret part of the API key.

    Raises:
        ValueError: If the API key format is invalid.
    """
    try:
        parts = api_key.split('_')
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Invalid API key format for hashing")
        secret_key_part = parts[1]
        api_key_bytes = secret_key_part.encode('utf-8')
    except Exception as e:
        logger.error("Error extracting secret part from API key for hashing", error=str(e))
        # Re-raise as ValueError to indicate bad input format
        raise ValueError("Invalid API key format for hashing") from e

    # Generate salt and hash the secret part
    hashed_key = bcrypt.hashpw(api_key_bytes, bcrypt.gensalt())
    logger.debug("API Key secret part hashed successfully using bcrypt")
    # Return as string for database storage
    return hashed_key.decode('utf-8')


def verify_api_key_hash(api_key: str, stored_hash: str) -> bool:
    """
    Verify an API key against a stored bcrypt hash.
    This is specifically for keys hashed using bcrypt during creation.

    Args:
        api_key: The plain text API key provided by the user.
        stored_hash: The bcrypt hash stored in the database.

    Returns:
        True if the key matches the hash, False otherwise.
    """
    logger.debug("Verifying API key against stored bcrypt hash")
    # Extract the actual secret part of the key to compare against the bcrypt hash.
    # Assumption: create_api_key hashed *only* the secret part generated by create_raw_api_key.
    # If the full key (prefix + secret) was hashed, this needs adjustment.
    # Let's re-evaluate create_api_key's hashing logic... it hashes `api_key_val` which is the secret part.
    try:
        parts = api_key.split('_')
        if len(parts) != 2:
            logger.warning("Invalid API key format for hash verification", key_prefix=api_key[:8] + "...")
            return False
        secret_key_part = parts[1]
        api_key_bytes = secret_key_part.encode('utf-8')
        stored_hash_bytes = stored_hash.encode('utf-8')
    except Exception as e:
        logger.error("Error preparing API key/hash for verification", error=str(e))
        return False

    try:
        # Use bcrypt.checkpw for verification
        is_valid = bcrypt.checkpw(api_key_bytes, stored_hash_bytes)
        logger.debug("API key bcrypt verification result", is_valid=is_valid)
        return is_valid
    except ValueError:
        # Handle cases where the stored_hash is not a valid bcrypt hash
        logger.warning("Attempted to verify API key against an invalid bcrypt hash format")
        return False
    except Exception as e:
        logger.error("Unexpected error during API key bcrypt verification", error=str(e), exc_info=True)
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