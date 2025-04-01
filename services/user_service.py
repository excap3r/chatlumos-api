#!/usr/bin/env python3
"""
User Service Layer

This module provides a service layer for user-related operations,
acting as an intermediary between API routes and the database access layer.
It encapsulates business logic related to user management, authentication,
and API key handling.
"""

import structlog
from typing import Dict, Any, List, Optional, Tuple

# Import database functions
from .db.user_db import (
    create_user as db_create_user,
    authenticate_user as db_authenticate_user,
    create_api_key as db_create_api_key,
    get_user_api_keys as db_get_user_api_keys,
    revoke_api_key as db_revoke_api_key,
    get_all_users as db_get_all_users,
    update_user_roles as db_update_user_roles,
)

# Import database exceptions
from .db.exceptions import (
    UserAlreadyExistsError, # Should be DuplicateUserError based on auth_routes?
    InvalidCredentialsError,
    UserNotFoundError,
    DatabaseError,
    DuplicateUserError # Confirm this is the correct one raised by create_user
)

# Import utility functions
from .utils.auth_utils import hash_password, generate_api_key, hash_api_key

# Configure logger
logger = structlog.get_logger(__name__)

def register_new_user(username: str, email: str, password: str, roles: Optional[List[str]] = None) -> int:
    """
    Registers a new user, handling password hashing.

    Args:
        username: User's username.
        email: User's email.
        password: User's plain text password.
        roles: Optional list of roles (defaults to ['user'] if None).

    Returns:
        The ID of the newly created user.

    Raises:
        DuplicateUserError: If the username or email already exists.
        DatabaseError: For other database-related issues.
        Exception: For unexpected errors during hashing or creation.
    """
    if roles is None:
        roles = ["user"] # Default role

    try:
        password_hash, salt = hash_password(password)
        logger.info("Hashing password for new user registration", username=username)
    except Exception as e:
        logger.error("Password hashing failed during registration", username=username, error=str(e), exc_info=True)
        # Re-raise a generic exception or a custom HashingError if defined
        raise Exception("Failed to process password for registration.") from e

    try:
        user_id = db_create_user(
            username=username,
            email=email,
            password_hash=password_hash,
            password_salt=salt,
            roles=roles
        )
        logger.info("User created successfully via service layer", user_id=user_id, username=username)
        return user_id
    except DuplicateUserError as e: # Assuming db_create_user raises this
        logger.warning("User registration failed: Duplicate user (service layer)", username=username, email=email)
        raise e # Re-raise the specific error
    except DatabaseError as e:
        logger.error("Database error during user creation (service layer)", username=username, error=str(e), exc_info=True)
        raise e # Re-raise the specific error
    except Exception as e:
        logger.error("Unexpected error during user creation (service layer)", username=username, error=str(e), exc_info=True)
        raise Exception("An unexpected error occurred while creating the user.") from e

def login_user(username: str, password: str) -> Dict[str, Any]:
    """
    Authenticates a user against the database.

    Args:
        username: User's username.
        password: User's plain text password.

    Returns:
        Dictionary containing user details upon successful authentication.

    Raises:
        InvalidCredentialsError: If the username/password combination is incorrect.
        UserNotFoundError: If the user does not exist (might be raised by db_authenticate_user).
        DatabaseError: For database-related issues during authentication.
    """
    try:
        user = db_authenticate_user(username, password)
        logger.info("User authenticated successfully via service layer", username=username, user_id=user.get('id'))
        return user
    except (InvalidCredentialsError, UserNotFoundError) as e:
        logger.warning("User authentication failed (service layer)", username=username, reason=type(e).__name__)
        raise e # Re-raise the specific error
    except DatabaseError as e:
        logger.error("Database error during authentication (service layer)", username=username, error=str(e), exc_info=True)
        raise e # Re-raise the specific error
    except Exception as e:
        logger.error("Unexpected error during authentication (service layer)", username=username, error=str(e), exc_info=True)
        raise Exception("An unexpected error occurred during login.") from e


def add_api_key(user_id: str, key_name: str, prefix: str = "sk") -> Tuple[str, str, int]:
    """
    Generates, hashes, and stores a new API key for a user.

    Args:
        user_id: The ID of the user to create the key for.
        key_name: A descriptive name for the API key.
        prefix: Prefix for the generated key (default: 'sk').

    Returns:
        Tuple containing (api_key_id, generated_api_key, creation_timestamp).
        The generated_api_key is the *plain text key* and should only be shown once.

    Raises:
        DatabaseError: If storing the key fails.
        Exception: For unexpected errors during key generation or hashing.
    """
    try:
        generated_key = generate_api_key(prefix=prefix)
        hashed_key = hash_api_key(generated_key)
        logger.info("Generated and hashed new API key", user_id=user_id, key_name=key_name)
    except Exception as e:
        logger.error("API key generation or hashing failed", user_id=user_id, key_name=key_name, error=str(e), exc_info=True)
        raise Exception("Failed to generate or process the API key.") from e

    try:
        # db_create_api_key should return the necessary details
        key_details = db_create_api_key(user_id, key_name, hashed_key)
        logger.info("API key stored successfully via service layer", user_id=user_id, key_name=key_name, key_id=key_details.get('id'))

        # Return the *plain text* key along with its DB ID and timestamp
        # Ensure db_create_api_key returns a dict with 'id' and 'created_at_ts'
        return key_details['id'], generated_key, key_details['created_at_ts']

    except DatabaseError as e:
        logger.error("Database error storing API key (service layer)", user_id=user_id, key_name=key_name, error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.error("Unexpected error storing API key (service layer)", user_id=user_id, key_name=key_name, error=str(e), exc_info=True)
        raise Exception("An unexpected error occurred while storing the API key.") from e

def get_keys_for_user(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all non-revoked API keys for a specific user.

    Args:
        user_id: The ID of the user.

    Returns:
        List of dictionaries, each representing an API key (excluding the hash).

    Raises:
        DatabaseError: If fetching keys fails.
    """
    try:
        keys = db_get_user_api_keys(user_id)
        logger.info("Retrieved API keys for user via service layer", user_id=user_id, count=len(keys))
        return keys
    except DatabaseError as e:
        logger.error("Database error retrieving API keys (service layer)", user_id=user_id, error=str(e), exc_info=True)
        raise e

def remove_api_key(user_id: str, key_id: str) -> bool:
    """
    Revokes an API key for a user.

    Args:
        user_id: The ID of the user owning the key.
        key_id: The ID of the API key to revoke.

    Returns:
        True if the key was successfully revoked, False otherwise.

    Raises:
        DatabaseError: If the revocation operation fails in the DB.
    """
    try:
        success = db_revoke_api_key(user_id, key_id)
        if success:
            logger.info("API key revoked successfully via service layer", user_id=user_id, key_id=key_id)
        else:
            logger.warning("API key revocation failed or key not found (service layer)", user_id=user_id, key_id=key_id)
        return success
    except DatabaseError as e:
        logger.error("Database error revoking API key (service layer)", user_id=user_id, key_id=key_id, error=str(e), exc_info=True)
        raise e

def get_all_users_list() -> List[Dict[str, Any]]:
    """
    Retrieves a list of all users (typically for admin purposes).

    Returns:
        List of user dictionaries.

    Raises:
        DatabaseError: If fetching users fails.
    """
    try:
        users = db_get_all_users()
        logger.info("Retrieved all users via service layer", count=len(users))
        return users
    except DatabaseError as e:
        logger.error("Database error retrieving all users (service layer)", error=str(e), exc_info=True)
        raise e

def update_user_roles_by_id(user_id: str, roles: List[str]) -> bool:
    """
    Updates the roles for a specific user.

    Args:
        user_id: The ID of the user to update.
        roles: The new list of roles for the user.

    Returns:
        True if roles were updated successfully, False otherwise (e.g., user not found).

    Raises:
        DatabaseError: If the update operation fails in the DB.
    """
    try:
        updated = db_update_user_roles(user_id, roles)
        if updated:
            logger.info("User roles updated successfully via service layer", user_id=user_id, new_roles=roles)
        else:
            logger.warning("User roles update failed or user not found (service layer)", user_id=user_id)
        return updated
    except DatabaseError as e:
        logger.error("Database error updating user roles (service layer)", user_id=user_id, error=str(e), exc_info=True)
        raise e
