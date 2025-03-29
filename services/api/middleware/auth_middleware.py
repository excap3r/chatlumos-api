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

# Set up logging
logger = logging.getLogger("auth_middleware")

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

def auth_required(roles: List[str] = None, permissions: List[str] = None):
    """
    Decorator for routes that require authentication.
    
    If roles or permissions are specified, the user must have at least one of them.
    
    Args:
        roles: List of required roles
        permissions: List of required permissions
        
    Returns:
        Decorator function
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Try JWT token first
            token = get_token_from_header()
            if token:
                try:
                    payload = decode_token(token)
                    
                    # Store user info in flask global 'g'
                    g.user = {
                        "id": payload.get("sub"),
                        "roles": payload.get("roles", []),
                        "permissions": payload.get("permissions", []),
                        "token_type": payload.get("type")
                    }
                    
                    # Check token type
                    if payload.get("type") != "access":
                        return jsonify({
                            "error": "Invalid token type",
                            "message": "Please use an access token for authentication"
                        }), 401
                    
                    # Check permissions or roles if specified
                    if permissions:
                        has_required_perm = any(has_permission(payload, perm) for perm in permissions)
                        if not has_required_perm:
                            return jsonify({
                                "error": "Insufficient permissions",
                                "message": f"Required permissions: {', '.join(permissions)}"
                            }), 403
                    
                    if roles:
                        has_required_role = any(has_role(payload, role) for role in roles)
                        if not has_required_role:
                            return jsonify({
                                "error": "Insufficient roles",
                                "message": f"Required roles: {', '.join(roles)}"
                            }), 403
                    
                    return f(*args, **kwargs)
                    
                except ExpiredTokenError:
                    return jsonify({
                        "error": "Token expired",
                        "message": "Your token has expired, please login again"
                    }), 401
                except (InvalidTokenError, MissingSecretError):
                    pass  # Try API key instead
            
            # Try API key
            api_key = get_api_key_from_request()
            if api_key:
                try:
                    # Verify API key
                    api_key_data = verify_api_key(api_key)
                    
                    # Store user info in flask global 'g'
                    g.user = {
                        "id": api_key_data.get("user_id"),
                        "username": api_key_data.get("username"),
                        "roles": api_key_data.get("roles", []),
                        "permissions": api_key_data.get("permissions", []),
                        "auth_type": "api_key",
                        "api_key_id": api_key_data.get("id")
                    }
                    
                    # Check permissions
                    if permissions:
                        has_required_perm = any(perm in api_key_data.get("permissions", []) for perm in permissions)
                        if not has_required_perm:
                            return jsonify({
                                "error": "Insufficient permissions",
                                "message": f"Required permissions: {', '.join(permissions)}"
                            }), 403
                    
                    # Check roles
                    if roles:
                        has_required_role = any(role in api_key_data.get("roles", []) for role in roles)
                        if not has_required_role:
                            return jsonify({
                                "error": "Insufficient roles",
                                "message": f"Required roles: {', '.join(roles)}"
                            }), 403
                    
                    return f(*args, **kwargs)
                    
                except Exception as e:
                    logger.error(f"API key authentication error: {str(e)}")
            
            # No valid authentication found
            return jsonify({
                "error": "Authentication required",
                "message": "Please provide a valid access token or API key"
            }), 401
            
        return wrapped
    return decorator

def admin_required(f):
    """
    Decorator for routes that require admin role.
    
    Returns:
        Decorator function
    """
    return auth_required(roles=["admin"])(f)

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
        token = get_token_from_header()
        if token:
            try:
                payload = decode_token(token)
                
                # Store user info in flask global 'g'
                g.user = {
                    "id": payload.get("sub"),
                    "roles": payload.get("roles", []),
                    "permissions": payload.get("permissions", []),
                    "token_type": payload.get("type"),
                    "auth_type": "jwt"
                }
            except Exception as e:
                # Just log the error but continue
                logger.warning(f"Optional JWT auth failed: {str(e)}")
                g.user = None
        else:
            # Try API key
            api_key = get_api_key_from_request()
            if api_key:
                try:
                    # Verify API key
                    api_key_data = verify_api_key(api_key)
                    
                    # Store user info in flask global 'g'
                    g.user = {
                        "id": api_key_data.get("user_id"),
                        "username": api_key_data.get("username"),
                        "roles": api_key_data.get("roles", []),
                        "permissions": api_key_data.get("permissions", []),
                        "auth_type": "api_key",
                        "api_key_id": api_key_data.get("id")
                    }
                except Exception as e:
                    # Just log the error but continue
                    logger.warning(f"Optional API key auth failed: {str(e)}")
                    g.user = None
            else:
                g.user = None
        
        return f(*args, **kwargs)
    
    return wrapped 