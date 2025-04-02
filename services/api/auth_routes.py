#!/usr/bin/env python3
"""
Authentication API Routes

Provides API endpoints for authentication, including user registration,
login, logout, token refresh, and API key management.
"""

import logging
import structlog
from flask import Blueprint, request, jsonify, g, current_app
from typing import Dict, Any, List, Optional, Tuple
from functools import wraps
import re # Added for password regex
import time # Added time for Redis expiry calculation

# Import Pydantic for validation
from pydantic import BaseModel, EmailStr, Field, validator, ValidationError as PydanticValidationError

# Import authentication utilities
from ..utils.auth_utils import create_token, decode_token, MissingSecretError, InvalidTokenError, ExpiredTokenError

# Import User Service
from ..user_service import (
    register_new_user, 
    login_user,
    get_keys_for_user,
    add_api_key,
    remove_api_key,
    get_all_users_list as service_get_all_users,
    update_user_roles_by_id as service_update_user_roles
)

# Import specific exceptions used for flow control or specific responses
from ..db.exceptions import UserAlreadyExistsError, InvalidCredentialsError, UserNotFoundError, DatabaseError

# Import middleware
from .middleware.auth_middleware import require_auth

# Import base error class
from ..utils.error_utils import APIError, ValidationError, format_error_response

# Configure logger
logger = structlog.get_logger(__name__)

# Create Blueprint
auth_bp = Blueprint("auth", __name__)

# --- Pydantic Schemas ---

def validate_password_complexity(password: str) -> str:
    """Custom validator for password complexity."""
    if len(password) < 8:
        raise ValueError('Password must be at least 8 characters long')
    if not re.search(r"[a-z]", password):
        raise ValueError('Password must contain a lowercase letter')
    if not re.search(r"[A-Z]", password):
        raise ValueError('Password must contain an uppercase letter')
    if not re.search(r"\d", password):
        raise ValueError('Password must contain a digit')
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise ValueError('Password must contain a special character')
    return password

class UserRegistrationSchema(BaseModel):
    username: str = Field(..., min_length=3)
    email: EmailStr
    password: str

    _validate_password = validator('password', allow_reuse=True)(validate_password_complexity)

class UserLoginSchema(BaseModel):
    username: str
    password: str

# --- API Routes ---

@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register a new user. Validates input using Pydantic.

    Request Body (JSON):
        username: User's username (min 3 chars)
        email: User's valid email address
        password: User's password (min 8 chars, incl. upper, lower, digit, symbol)

    Returns:
        201: User created successfully
        400: Invalid request data (validation error)
        409: Username or email already exists
        500: Server error
    """
    try:
        # Validate request data using Pydantic
        payload = request.get_json()
        if not payload:
            raise ValidationError("Request body cannot be empty.")
        
        # Explicitly handle Pydantic validation errors here
        try:
            user_data = UserRegistrationSchema.model_validate(payload)
        except PydanticValidationError as e:
            logger.warning("Registration validation failed", errors=e.errors())
            raise ValidationError(f"Input validation failed: {e.errors()}")

        username = user_data.username
        email = user_data.email
        password = user_data.password

        logger.info("Registration request validation successful", username=username, email=email)

        # Call user service to handle registration (incl. hashing)
        user_id = register_new_user(username, email, password)

        # Optionally create a token upon successful registration
        token = create_token(user_id=user_id, username=username)
        logger.info("User registered successfully", user_id=user_id, username=username)
        return jsonify({"message": "User registered successfully", "user_id": user_id, "token": token}), 201

    except UserAlreadyExistsError as e:
        logger.warning("Registration failed: User already exists", username=username, email=email)
        # Use the existing APIError handling mechanism if format_error_response isn't defined/imported
        raise APIError(str(e), status_code=409)
        # response, status_code = format_error_response(e) # Assumes format_error_response exists and works
        # return jsonify(response), status_code

    except DatabaseError as e: # Catch potential DB errors during user creation
        logger.error("Database error during registration", username=username, error=str(e), exc_info=True)
        raise APIError("Failed to register user due to a database issue.", status_code=500)
    
    # Catch our custom ValidationError raised above or from empty payload check
    except ValidationError as e:
        logger.warning("Registration validation error", message=e.message)
        # Re-raise to be caught by the global handler (which should return 400)
        raise e

    except Exception as e: # Generic catch-all for unexpected errors
        logger.error("Unexpected error during registration", username=username if 'username' in locals() else 'unknown', error=str(e), exc_info=True)
        raise APIError("An unexpected error occurred during registration.", status_code=500)

@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Authenticate user and return access and refresh tokens. Validates input using Pydantic.

    Request Body (JSON):
        username: User's username
        password: User's password

    Returns:
        200: Authentication successful
        400: Invalid request data (validation error)
        401: Invalid credentials
        500: Server error
    """
    try:
        # Validate request data using Pydantic
        payload = request.get_json()
        if not payload:
            raise ValidationError("Request body cannot be empty.")

        login_data = UserLoginSchema.model_validate(payload)
        username = login_data.username
        password = login_data.password

        logger.info("Login request validation successful", username=username)

        # Call user service to handle authentication
        user = login_user(username, password)

        # Generate tokens
        # Ensure create_token handles potential errors (e.g., missing secret)
        try:
            access_token = create_token(
                user_id=user["id"],
                username=user["username"],
                type="access",
                roles=user.get("roles", []), # Use .get for safety
                permissions=user.get("permissions", []) # Use .get for safety
            )

            refresh_token = create_token(
                user_id=user["id"],
                username=user["username"],
                type="refresh"
                # No roles/permissions needed in refresh usually
            )
        except MissingSecretError as e:
             logger.critical("JWT Secret Key is missing or invalid!", error=str(e), exc_info=True)
             raise APIError("Server configuration error: Cannot generate tokens.", status_code=500)
        except Exception as e: # Catch other potential token errors
             logger.error("Token generation failed", user_id=user.get("id"), username=username, error=str(e), exc_info=True)
             raise APIError("Failed to generate authentication tokens.", status_code=500)


        logger.info("User logged in successfully", user_id=user["id"], username=user["username"])

        # Prepare user data for response (avoid sending sensitive info like hash/salt)
        # Get expiry for the access token to return in response
        access_token_expires_in = current_app.config.get('ACCESS_TOKEN_EXPIRES', 3600)

        user_response = {
            "id": user["id"],
            "username": user["username"],
            "roles": user.get("roles", []),
            "permissions": user.get("permissions", [])
        }

        return jsonify({
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": access_token_expires_in,
            "user": user_response
        }), 200

    except PydanticValidationError as e:
        logger.warning("Login validation failed", errors=e.errors())
        raise ValidationError(f"Input validation failed: {e.errors()}")

    except InvalidCredentialsError as e:
        logger.warning(f"Login failed for user {username}: Invalid credentials") # Avoid logging password
        raise APIError("Invalid username or password", status_code=401)

    except UserNotFoundError as e: # Assuming authenticate_user might raise this
        logger.warning(f"Login failed: User not found - {username}")
        raise APIError("Invalid username or password", status_code=401) # Same message as invalid creds for security

    except DatabaseError as e: # Catch potential DB errors during authentication
        logger.error("Database error during login", username=username, error=str(e), exc_info=True)
        raise APIError("Failed to login due to a database issue.", status_code=500)

    except Exception as e: # Generic catch-all
        logger.error("Unexpected error during login", username=username if 'username' in locals() else 'unknown', error=str(e), exc_info=True)
        raise APIError("An unexpected error occurred during login.", status_code=500)

@auth_bp.route("/refresh", methods=["POST"])
def refresh_token():
    """
    Refresh access token using refresh token.
    
    Request Body:
        refresh_token: Refresh token
        
    Returns:
        200: Tokens refreshed successfully (new access and refresh token)
        400: Invalid request
        401: Invalid or expired token, or token reuse detected
        500: Server error (e.g., Redis connection)
    """
    data = request.get_json()
    redis_client = current_app.redis_client # Get Redis client from app context

    if not redis_client:
        logger.error("Redis client not configured or available for token refresh.")
        raise APIError("Server configuration error: Cannot process token refresh.", status_code=500)

    # Validate request data presence
    if not data:
        raise ValidationError("Request body is required")

    refresh_token = data.get("refresh_token")

    if not refresh_token:
        raise ValidationError("Refresh token is required")

    try:
        # Decode refresh token (raises exceptions on failure)
        payload = decode_token(refresh_token)
        user_id = payload["sub"]
        username = payload["username"]
        token_jti = payload["jti"]
        token_exp = payload["exp"]

        # 1. Check if this token JTI is already invalidated in Redis
        invalidation_key = f"invalidated_jti:{token_jti}"
        if redis_client.exists(invalidation_key):
            logger.warning("Attempted reuse of invalidated refresh token", jti=token_jti, user_id=user_id)
            raise APIError("Invalid token: Potential reuse detected.", status_code=401)

        # 2. Verify token type
        if payload.get("type") != "refresh":
            logger.warning("Attempted refresh with non-refresh token type", jti=token_jti, user_id=user_id, type=payload.get("type"))
            raise APIError("Invalid token type for refresh operation.", status_code=401)

        # 3. Invalidate the used refresh token in Redis
        # Calculate remaining validity time for the Redis key expiry
        # Add a small buffer (e.g., 60s) to ensure it outlives the token slightly
        now = int(time.time())
        redis_expiry_seconds = max(60, token_exp - now + 60) # Ensure minimum 60s expiry
        try:
            redis_client.setex(invalidation_key, redis_expiry_seconds, "invalidated")
            logger.info("Invalidated used refresh token", jti=token_jti, user_id=user_id, expiry_seconds=redis_expiry_seconds)
        except Exception as redis_error: # Catch potential Redis errors
            logger.error("Failed to invalidate refresh token in Redis", jti=token_jti, user_id=user_id, error=str(redis_error), exc_info=True)
            # Decide if this should be fatal. For now, proceed but log critically.
            # If Redis is critical for security, raise APIError(..., 500) here.

        # 4. Generate *new* access token and *new* refresh token
        try:
            new_access_token = create_token(
                user_id=user_id,
                username=username,
                type="access",
                # Get roles/permissions from the *original* refresh token payload
                # Or potentially re-fetch from DB if roles can change rapidly?
                # For now, assume roles/permissions in refresh token are sufficient
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", [])
            )
            new_refresh_token = create_token(
                user_id=user_id,
                username=username,
                type="refresh"
                # No roles/permissions needed in refresh token itself usually
            )
        except (MissingSecretError, RuntimeError) as token_error:
            logger.critical("Failed to generate new tokens during refresh", user_id=user_id, error=str(token_error), exc_info=True)
            # This is a server error if token generation fails
            raise APIError("Server error: Could not generate new tokens.", status_code=500)

        # 5. Return both tokens
        # Get expiry for the access token to return in response
        access_token_expires_in = current_app.config.get('ACCESS_TOKEN_EXPIRES', 3600)

        logger.info("Token refresh successful, new tokens issued", user_id=user_id)
        return jsonify({
            "access_token": new_access_token,
            "refresh_token": new_refresh_token, # Return the NEW refresh token
            "token_type": "bearer",
            "expires_in": access_token_expires_in
        }), 200

    # Catch specific exceptions from decode_token
    except (InvalidTokenError, ExpiredTokenError, MissingSecretError) as e:
        logger.warning(f"Token refresh failed: {type(e).__name__}", error=str(e))
        # Use a generic message for security
        raise APIError(f"Invalid or expired refresh token.", status_code=401)

    except APIError: # Re-raise APIErrors directly (like reuse detected)
        raise
    except ValidationError: # Re-raise ValidationErrors
        raise
    except Exception as e: # Catch unexpected errors
        logger.error("Unexpected error during token refresh", error=str(e), exc_info=True)
        raise APIError("An unexpected error occurred during token refresh.", status_code=500)

@auth_bp.route("/users", methods=["GET"])
@require_auth(roles=['admin'])
def list_users():
    """
    List all users (admin only).
    
    Returns:
        200: List of users
        401: Unauthorized
        403: Forbidden
    """
    try:
        # Call user service
        users = service_get_all_users()
        
        return jsonify({
            "users": users
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        raise APIError("Server error", status_code=500)

@auth_bp.route("/users/<user_id>/roles", methods=["PUT"])
@require_auth(roles=['admin'])
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
    
    # Basic validation
    if not data or "roles" not in data:
        raise ValidationError('Missing required field: "roles"')

    roles = data.get("roles")
    
    if not isinstance(roles, list):
        # Service layer might add more validation if needed
        raise ValidationError("Roles must be a non-empty list")
    
    try:
        # Call user service
        updated = service_update_user_roles(user_id, roles)
        
        if updated:
            logger.info(f"Roles updated for user {user_id}")
            return jsonify({
                "message": "User roles updated successfully",
                "roles": roles
            }), 200
        
    except DatabaseError as db_err:
        # Handle potential DB errors specifically if needed, else rely on generic handler
        logger.error(f"Database error updating user roles for {user_id}", error=str(db_err), exc_info=True)
        raise APIError(f"Database error updating roles: {db_err}", status_code=500)
    except Exception as e:
        logger.error(f"Unexpected error updating user roles for {user_id}", error=str(e), exc_info=True)
        raise APIError(f"Server error updating roles: {e}", status_code=500)

@auth_bp.route("/validate", methods=["GET"])
@require_auth()
def validate():
    # If decorator fails, it returns 401/403 directly.
    # If it succeeds, g.user is populated.
    user_info = g.user
    logger.debug("Token validated successfully", user_id=user_info.get('id')) 
    return jsonify({
        "id": user_info.get('id'),
        "username": user_info.get('username'),
        "roles": user_info.get('roles', [])
    })
    # No try/except needed here unless the decorator logic itself can fail unexpectedly
    # (which would be caught by the global 500 handler)

@auth_bp.route("/api-keys", methods=["GET"])
@require_auth()
def get_api_keys():
    """
    Get all active API keys for the authenticated user.
    
    Returns:
        200: List of API keys (key_id, name, created_at, last_used_at, is_active, key_prefix)
        401: Unauthorized
    """
    user_id = g.user['id']
    try:
        keys = get_keys_for_user(user_id)
        # Convert datetime objects to ISO strings for JSON serialization
        keys_serializable = [
            {**key, 
             'created_at': key['created_at'].isoformat() if key.get('created_at') else None, 
             'last_used_at': key['last_used_at'].isoformat() if key.get('last_used_at') else None
            } 
            for key in keys
        ]
        return jsonify(keys_serializable), 200
    except Exception as e:
        logger.error("Failed to get API keys for user", user_id=user_id, error=str(e), exc_info=True)
        raise APIError("Failed to retrieve API keys.", status_code=500)

@auth_bp.route("/api-keys", methods=["POST"])
@require_auth()
def create_api_key_route():
    """
    Create a new API key for the authenticated user.
    
    Request Body:
        name: Descriptive name for the key (required)
        
    Returns:
        201: Key created successfully (includes the plain text key)
        400: Bad request (missing name)
        401: Unauthorized
        500: Server error
    """
    user_id = g.user['id']
    data = request.get_json()
    
    if not data or not data.get('name'):
        raise ValidationError("Missing required field: 'name'")
        
    key_name = data['name']

    try:
        key_id, plain_key, created_ts = add_api_key(user_id, key_name)
        
        return jsonify({
            "message": "API key created successfully. Store the key securely, it will not be shown again.",
            "key_id": key_id,
            "key_name": key_name,
            "api_key": plain_key, # Return the plain text key ONCE upon creation
            "created_at_ts": created_ts # Return the timestamp
        }), 201
    except DatabaseError as e:
        logger.error("Database error creating API key", user_id=user_id, key_name=key_name, error=str(e), exc_info=True)
        raise APIError("Failed to store API key due to database error.", status_code=500)
    except Exception as e:
        logger.error("Unexpected error creating API key", user_id=user_id, key_name=key_name, error=str(e), exc_info=True)
        raise APIError("Failed to create API key.", status_code=500)
        

@auth_bp.route("/api-keys/<key_id>", methods=["DELETE"])
@require_auth()
def delete_api_key_route(key_id):
    """
    Delete (revoke) an API key.
    
    Path Parameters:
        key_id: The ID of the key to delete.
        
    Returns:
        204: Key deleted successfully
        401: Unauthorized
        404: Key not found or not owned by user
        500: Server error
    """
    user_id = g.user['id']
    try:
        success = remove_api_key(user_id, key_id)
        if success:
            return "", 204 # No Content
        else:
            # Explicitly raise the 404 error for not found / permission denied
            raise APIError("API key not found or you do not have permission to delete it.", status_code=404)
            
    except APIError as e: # Catch the APIError we just raised (or others)
        logger.warning("API Error during key deletion", user_id=user_id, key_id=key_id, status_code=e.status_code, error=e.message)
        # Re-raise the caught APIError to be handled by the global handler
        raise e 
    except DatabaseError as e:
        logger.error("Database error deleting API key", user_id=user_id, key_id=key_id, error=str(e), exc_info=True)
        raise APIError("Failed to delete API key due to database error.", status_code=500)
    except Exception as e:
        logger.error("Unexpected error deleting API key", user_id=user_id, key_id=key_id, error=str(e), exc_info=True)
        # Ensure this generic handler doesn't accidentally catch the APIError raised above
        raise APIError("An unexpected server error occurred while deleting the API key.", status_code=500) 