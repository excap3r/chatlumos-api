#!/usr/bin/env python3
"""
Authentication Middleware

Provides middleware functions for authentication and authorization
in the API, handling both JWT tokens and API keys.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from flask import request, jsonify, g, current_app
import structlog

# Import utilities
from .utils.auth_utils import (
    decode_token, 
    has_permission, 
    has_role, 
    MissingSecretError, 
    InvalidTokenError, 
    ExpiredTokenError,
    InsufficientPermissionsError
)

# Import API key verification
from .db.user_db import verify_api_key

# Configure logger
logger = structlog.get_logger(__name__)

def get_token_from_header() -> Optional[str]:
    """
    Extract JWT token from Authorization header.
    
    Returns:
        Token string or None if not found
    """
    auth_header = request.headers.get("Authorization", "")
    
    if not auth_header:
        return None
    
    # Check for Bearer token
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    
    return None

def get_api_key_from_request() -> Optional[str]:
    """
    Extract API key from request.
    
    First checks the Authorization header for 'ApiKey' prefix,
    then checks 'X-API-Key' header, then checks query parameter 'api_key'.
    
    Returns:
        API key string or None if not found
    """
    # Check Authorization header first
    auth_header = request.headers.get("Authorization", "")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "apikey":
            return parts[1]
    
    # Then check X-API-Key header
    api_key_header = request.headers.get("X-API-Key")
    if api_key_header:
        return api_key_header
    
    # Finally check query parameter
    api_key_param = request.args.get("api_key")
    if api_key_param:
        return api_key_param
    
    return None

def require_auth(roles: Optional[List[str]] = None, allow_api_key: bool = True):
    """
    Decorator to enforce authentication and optional role/permission checks.
    Supports both JWT Bearer tokens and API keys.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = None
            auth_method = None
            
            # 1. Check for JWT Bearer token
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
                payload = decode_token(token)
                if payload:
                    # Standardize user structure for JWT
                    user = {
                        'id': payload.get('sub'),
                        'username': payload.get('username'),
                        'roles': payload.get('roles', []),
                        'permissions': payload.get('permissions', []),
                        'auth_type': 'jwt',
                        'jti': payload.get('jti')
                    }
                    auth_method = 'jwt' # Keep tracking auth method separately if needed
                    logger.debug("Authenticated via JWT", user_id=user['id'])
                else:
                     logger.warning("Invalid or expired JWT token received")
                     return jsonify({"error": "Invalid or expired token"}), 401
            
            # 2. Check for API Key (if JWT not found/valid and allow_api_key is True)
            elif allow_api_key:
                api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
                if api_key:
                    try:
                        # Validate key against DB (verify_api_key checks hash)
                        user_info = verify_api_key(api_key)
                        if user_info:
                            # Standardize user structure for API Key
                            # Assuming verify_api_key returns user_id, username, roles, permissions, id (key id)
                            user = {
                                'id': user_info.get('user_id'), # Map user_id from DB to id
                                'username': user_info.get('username'),
                                'roles': user_info.get('roles', []),
                                'permissions': user_info.get('permissions', []),
                                'auth_type': 'api_key',
                                'api_key_id': user_info.get('id') # Map key id from DB
                            }
                            auth_method = 'api_key'
                            logger.debug("Authenticated via API Key", user_id=user['id'], key_id=user['api_key_id'])
                        else:
                             logger.warning("Invalid API key provided", key_preview=api_key[:4]+"...")
                             return jsonify({"error": "Invalid API key"}), 401
                    except DatabaseError as db_err:
                        logger.error("Database error during API key validation", error=str(db_err), exc_info=True)
                        return jsonify({"error": "API key validation failed due to database issue"}), 500
                    except Exception as e:
                        logger.error("Unexpected error during API key validation", error=str(e), exc_info=True)
                        return jsonify({"error": "An unexpected error occurred during API key validation"}), 500

            # 3. No valid authentication found
            if not user:
                logger.warning("Authentication required but none provided or invalid")
                return jsonify({"error": "Authentication required"}), 401

            # 4. Role check (if roles are specified)
            if roles:
                user_roles = user.get('roles', [])
                if not any(has_role(user, role) for role in roles):
                    logger.warning("Authorization failed: Insufficient roles", 
                                 user_id=user['id'], 
                                 required_roles=roles, 
                                 user_roles=user_roles)
                    return jsonify({"error": "Forbidden: Insufficient roles"}), 403

            # Authentication successful, store user info in g for access in the route
            g.user = user
            g.auth_method = auth_method
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Convenience decorators
def admin_required():
    """Decorator requiring 'admin' role."""
    return require_auth(roles=['admin'])

def permission_required(permission: str):
    """
    Decorator factory requiring a specific permission.
    Ensures the user is authenticated and has the required permission.

    Args:
        permission: The permission string required to access the route.
    """
    def decorator(f):
        # Apply require_auth first to ensure g.user exists and is valid
        @require_auth() # No specific roles needed here, just authentication
        @wraps(f) # Apply wraps to the inner function
        def decorated_function(*args, **kwargs):
            # g.user should be set by @require_auth
            if not hasattr(g, 'user') or not g.user:
                # This shouldn't happen if @require_auth runs first, but safeguard
                logger.error("Permission check failed: User not found in g after @require_auth")
                return jsonify({"error": "Authentication context error"}), 500

            # Check permission using the utility function
            if not has_permission(g.user, permission):
                logger.warning("Authorization failed: Missing required permission",
                             user_id=g.user.get('id'),
                             required_permission=permission,
                             user_permissions=g.user.get('permissions', []),
                             user_roles=g.user.get('roles', []))
                return jsonify({"error": f"Forbidden: Requires permission '{permission}'"}), 403

            # Permission check passed
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def optional_auth(f):
    """
    Decorator for routes where authentication is optional.
    
    If authentication is provided, it will be verified and user info will be available,
    but the route will still be accessible without authentication.
    
    Returns:
        Decorator function
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        # Try JWT token first
        g.user = None # Initialize g.user
        token = get_token_from_header()
        if token:
            try:
                payload = decode_token(token)
                # Standardize g.user for optional JWT auth
                g.user = {
                    'id': payload.get('sub'),
                    'username': payload.get('username'),
                    'roles': payload.get('roles', []),
                    'permissions': payload.get('permissions', []),
                    'auth_type': 'jwt',
                    'jti': payload.get('jti')
                }
            except Exception as e:
                # Just log the error but continue
                logger.warning(f"Optional JWT auth failed: {str(e)}")
                # g.user remains None if auth fails
        else:
            # Try API key
            api_key = get_api_key_from_request()
            if api_key:
                try:
                    # Verify API key
                    user_info = verify_api_key(api_key)
                    if user_info:
                        # Standardize g.user for optional API key auth
                        g.user = {
                            'id': user_info.get('user_id'),
                            'username': user_info.get('username'),
                            'roles': user_info.get('roles', []),
                            'permissions': user_info.get('permissions', []),
                            'auth_type': 'api_key',
                            'api_key_id': user_info.get('id')
                        }
                    else:
                        # Log invalid key but continue
                        logger.warning("Optional API key auth failed: Invalid key")
                        # g.user remains None
                except Exception as e:
                    # Just log the error but continue
                    logger.warning(f"Optional API key auth failed: {str(e)}")
                    # g.user remains None
        # If no auth provided or auth failed, g.user is None

        return f(*args, **kwargs)
    
    return wrapped 