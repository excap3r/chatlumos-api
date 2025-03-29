#!/usr/bin/env python3
"""
Authentication API Routes

Provides API endpoints for authentication, including user registration,
login, logout, token refresh, and API key management.
"""

import logging
from flask import Blueprint, request, jsonify, g
from typing import Dict, Any, List, Optional, Tuple

# Import authentication utilities
from ..utils.auth_utils import (
    create_token,
    hash_password,
    verify_password,
    generate_api_key,
    hash_api_key,
    MissingSecretError,
    InvalidTokenError,
    ExpiredTokenError
)

# Import database services
from ..db.user_db import (
    create_user,
    authenticate_user,
    create_api_key as db_create_api_key,
    get_user_api_keys,
    revoke_api_key,
    UserAlreadyExistsError,
    InvalidCredentialsError,
    UserNotFoundError
)

# Import middleware
from ..auth_middleware import auth_required, admin_required

# Set up logging
logger = logging.getLogger("auth_routes")

# Create Blueprint
auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register a new user.
    
    Request Body:
        username: User's username
        password: User's password
        email: User's email
        
    Returns:
        201: User created successfully
        400: Invalid request
        409: Username or email already exists
    """
    data = request.get_json()
    
    # Validate request data
    if not data:
        return jsonify({"error": "Invalid request", "message": "Request body is required"}), 400
    
    username = data.get("username")
    password = data.get("password")
    email = data.get("email")
    
    if not all([username, password, email]):
        return jsonify({
            "error": "Invalid request", 
            "message": "Username, password, and email are required"
        }), 400
    
    # Hash password
    password_hash, salt = hash_password(password)
    
    try:
        # Create user in database
        user_id = create_user(
            username=username,
            email=email,
            password_hash=password_hash,
            password_salt=salt,
            # Default role is 'user'
            roles=["user"]
        )
        
        return jsonify({
            "message": "User registered successfully",
            "user_id": user_id
        }), 201
        
    except UserAlreadyExistsError as e:
        return jsonify({
            "error": "User already exists",
            "message": str(e)
        }), 409
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": "Failed to create user"
        }), 500

@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Authenticate user and return access and refresh tokens.
    
    Request Body:
        username: User's username
        password: User's password
        
    Returns:
        200: Authentication successful
        400: Invalid request
        401: Invalid credentials
    """
    data = request.get_json()
    
    # Validate request data
    if not data:
        return jsonify({"error": "Invalid request", "message": "Request body is required"}), 400
    
    username = data.get("username")
    password = data.get("password")
    
    if not all([username, password]):
        return jsonify({
            "error": "Invalid request", 
            "message": "Username and password are required"
        }), 400
    
    try:
        # Authenticate user
        user = authenticate_user(username, password)
        
        # Generate tokens
        access_token = create_token(
            user_id=user["id"],
            username=user["username"],
            type="access",
            roles=user["roles"],
            permissions=user["permissions"]
        )
        
        refresh_token = create_token(
            user_id=user["id"],
            username=user["username"],
            type="refresh"
        )
        
        return jsonify({
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 3600,  # 1 hour
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "roles": user["roles"],
                "permissions": user["permissions"]
            }
        }), 200
        
    except InvalidCredentialsError:
        return jsonify({
            "error": "Invalid credentials",
            "message": "Invalid username or password"
        }), 401
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": "Authentication failed"
        }), 500

@auth_bp.route("/refresh", methods=["POST"])
def refresh_token():
    """
    Refresh access token using refresh token.
    
    Request Body:
        refresh_token: Refresh token
        
    Returns:
        200: Token refreshed successfully
        400: Invalid request
        401: Invalid or expired token
    """
    data = request.get_json()
    
    # Validate request data
    if not data:
        return jsonify({"error": "Invalid request", "message": "Request body is required"}), 400
    
    refresh_token = data.get("refresh_token")
    
    if not refresh_token:
        return jsonify({
            "error": "Invalid request", 
            "message": "Refresh token is required"
        }), 400
    
    try:
        # Decode refresh token
        from ..utils.auth_utils import decode_token
        payload = decode_token(refresh_token)
        
        # Verify token type
        if payload.get("type") != "refresh":
            return jsonify({
                "error": "Invalid token",
                "message": "Token is not a refresh token"
            }), 401
        
        # Generate new access token
        access_token = create_token(
            user_id=payload["sub"],
            username=payload["username"],
            type="access",
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", [])
        )
        
        return jsonify({
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600  # 1 hour
        }), 200
        
    except (InvalidTokenError, ExpiredTokenError, MissingSecretError) as e:
        return jsonify({
            "error": "Invalid token",
            "message": str(e)
        }), 401
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": "Failed to refresh token"
        }), 500

@auth_bp.route("/api-keys", methods=["POST"])
@auth_required()
def create_api_key():
    """
    Create a new API key for the authenticated user.
    
    Request Body:
        name: Name/description for the API key
        
    Returns:
        201: API key created successfully
        400: Invalid request
        401: Unauthorized
    """
    data = request.get_json()
    
    # Validate request data
    if not data:
        return jsonify({"error": "Invalid request", "message": "Request body is required"}), 400
    
    name = data.get("name", "API Key")
    
    # Generate API key
    api_key = generate_api_key()
    api_key_hash = hash_api_key(api_key)
    
    try:
        # Save API key to database
        key_id = db_create_api_key(
            user_id=g.user["id"],
            key_hash=api_key_hash,
            name=name
        )
        
        # Return the API key (only time it will be visible)
        return jsonify({
            "message": "API key created successfully",
            "key_id": key_id,
            "api_key": api_key,
            "name": name,
            "warning": "This API key will only be shown once. Please save it securely."
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating API key: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": "Failed to create API key"
        }), 500

@auth_bp.route("/api-keys", methods=["GET"])
@auth_required()
def list_api_keys():
    """
    List all API keys for the authenticated user.
    
    Returns:
        200: List of API keys
        401: Unauthorized
    """
    try:
        # Get API keys from database
        api_keys = get_user_api_keys(g.user["id"])
        
        return jsonify({
            "api_keys": api_keys
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing API keys: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": "Failed to list API keys"
        }), 500

@auth_bp.route("/api-keys/<key_id>", methods=["DELETE"])
@auth_required()
def delete_api_key(key_id):
    """
    Revoke an API key.
    
    Path Parameters:
        key_id: ID of the API key to revoke
        
    Returns:
        200: API key revoked successfully
        401: Unauthorized
        404: API key not found
    """
    try:
        # Revoke API key
        revoked = revoke_api_key(key_id, g.user["id"])
        
        if not revoked:
            return jsonify({
                "error": "API key not found",
                "message": "API key not found or does not belong to user"
            }), 404
        
        return jsonify({
            "message": "API key revoked successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error revoking API key: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": "Failed to revoke API key"
        }), 500

@auth_bp.route("/users", methods=["GET"])
@admin_required
def list_users():
    """
    List all users (admin only).
    
    Returns:
        200: List of users
        401: Unauthorized
        403: Forbidden
    """
    try:
        # Get users from database
        from ..db.user_db import get_all_users
        users = get_all_users()
        
        return jsonify({
            "users": users
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": "Failed to list users"
        }), 500

@auth_bp.route("/users/<user_id>/roles", methods=["PUT"])
@admin_required
def update_user_roles(user_id):
    """
    Update user roles (admin only).
    
    Path Parameters:
        user_id: ID of the user to update
        
    Request Body:
        roles: List of roles to assign to the user
        
    Returns:
        200: Roles updated successfully
        400: Invalid request
        401: Unauthorized
        403: Forbidden
        404: User not found
    """
    data = request.get_json()
    
    # Validate request data
    if not data:
        return jsonify({"error": "Invalid request", "message": "Request body is required"}), 400
    
    roles = data.get("roles")
    
    if not roles or not isinstance(roles, list):
        return jsonify({
            "error": "Invalid request", 
            "message": "Roles must be a non-empty list"
        }), 400
    
    try:
        # Update user roles
        from ..db.user_db import update_user_roles as db_update_user_roles
        updated = db_update_user_roles(user_id, roles)
        
        if not updated:
            return jsonify({
                "error": "User not found",
                "message": "User not found"
            }), 404
        
        return jsonify({
            "message": "User roles updated successfully",
            "roles": roles
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating user roles: {str(e)}")
        return jsonify({
            "error": "Server error",
            "message": "Failed to update user roles"
        }), 500 