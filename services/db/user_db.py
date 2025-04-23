#!/usr/bin/env python3
"""
User Database Service

Provides database operations for user management, authentication,
and API key management using a connection pool managed by the main application.
"""

# import logging # Removed
# import time # Removed
from typing import Dict, Any, List, Optional # Removed Tuple, Union
from datetime import datetime, timedelta, timezone
# import mysql.connector # Removed
# from mysql.connector import pooling, Error # Removed
import uuid
import bcrypt
import structlog # Added
from flask import current_app # Added
import redis
import hashlib
import json

# Import SQLAlchemy components
from sqlalchemy.orm import Session # Add Session import
from sqlalchemy.exc import IntegrityError, SQLAlchemyError # Add SQLAlchemy exceptions
from sqlalchemy.orm import joinedload, selectinload # To fetch User efficiently
from sqlalchemy import delete, select, insert, update # Import delete statement and other SQLAlchemy statements

# Import Models
from .models.user_models import User, UserRoleAssociation, APIKey # Import User, association, and APIKey

# Import utilities
# from ..utils.auth_utils import hash_password, hash_api_key, verify_password # Assuming these might be deprecated if bcrypt is used directly
from ..utils.auth_utils import verify_api_key_hash
from .exceptions import (
    DatabaseError, ConnectionError, QueryError, 
    UserNotFoundError, UserAlreadyExistsError, InvalidCredentialsError, ApiKeyNotFoundError
)
# Import DB session handler decorator
from .db_utils import handle_db_session, _get_session

# Configure logger
logger = structlog.get_logger(__name__) # Changed

# --- Redis Client Helper --- #
# (Assumes redis client is initialized and stored on current_app, e.g., current_app.redis_client)
def _get_redis_client() -> Optional[redis.Redis]:
    """Gets the Redis client from Flask's current_app."""
    try:
        client = getattr(current_app, 'redis_client', None)
        if client and client.ping(): # Check connection
            return client
        elif client:
            logger.warning("Redis client found but ping failed in user_db.")
            return None
        else:
            logger.warning("Redis client not found on current_app in user_db.")
            return None
    except RuntimeError:
        logger.error("Cannot access current_app outside of application context in user_db.")
        return None
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error in user_db: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting Redis client in user_db: {e}", exc_info=True)
        return None

# --- Cache Constants --- #
API_KEY_CACHE_PREFIX = "apikey_verify_cache:"
API_KEY_CACHE_TTL = 60 # Cache results for 60 seconds
API_KEY_CACHE_INVALID_MARKER = "__INVALID__"
API_KEY_CACHE_TTL_INVALID = 10 # Cache results for 10 seconds

# Removed global pool variable and setter
# user_db_pool: Optional[pooling.MySQLConnectionPool] = None
# def set_db_pool(pool: pooling.MySQLConnectionPool):
#     ...

# Removed UserDB class and create_pool method as pool management is deprecated
# class UserDB:
#     ...

# Removed get_connection function as session management is handled by app context
# def get_connection() -> mysql.connector.connection.MySQLConnection:
#    ...

# --- Helper function to get session --- (Optional, but can centralize access)
def _get_session() -> Session:
    """Get the SQLAlchemy session from Flask's current_app."""
    if not hasattr(current_app, 'db_session') or current_app.db_session is None:
        logger.error("SQLAlchemy session not initialized in current_app.")
        raise ConnectionError("Database session not available.")
    # db_session is a scoped_session factory, call it to get the actual session
    return current_app.db_session()

# --- Helper function to convert User model to dict ---
def _user_to_dict(user: User, roles: Optional[List[str]] = None) -> Dict[str, Any]:
    """Converts a User SQLAlchemy model instance to a dictionary.
    Optionally includes roles if provided separately.
    """
    if not user:
        return {}
        
    # If roles are not provided, try to get them from the relationship
    # This requires the relationship to be loaded (e.g., via selectinload)
    if roles is None:
        try:
            # Check if roles relationship is loaded or accessible
            if hasattr(user, 'roles') and user.roles:
                 roles = [assoc.role for assoc in user.roles]
            else:
                 roles = [] # Default to empty if not loaded/available
        except Exception as e:
            # Log error if accessing roles fails unexpectedly
            logger.warning("Failed to access user roles relationship in _user_to_dict", user_id=str(user.id), error=str(e))
            roles = []

    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
        "last_login": user.last_login,
        "roles": roles,
        "permissions": [] # Permissions not directly handled here
    }

# User Operations

@handle_db_session
def create_user(
    username: str,
    email: str,
    password: str,
    roles: List[str] = None
) -> Dict[str, Any]:
    """
    Create a new user using SQLAlchemy ORM.

    Raises:
        UserAlreadyExistsError: If a user with the same username or email exists.
        QueryError: If there's a database error during creation.
    """
    session = _get_session()
    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        if roles is None:
            roles = ["user"]

        new_user = User(
            username=username,
            email=email,
            password_hash=hashed_password
        )

        session.add(new_user)
        session.flush() # Flush to get the new_user.id

        # Assign roles using the association class
        if roles:
            for role_name in roles:
                user_role = UserRoleAssociation(user_id=new_user.id, role=role_name)
                session.add(user_role)

        session.commit()
        logger.info("User created and committed successfully", user_id=str(new_user.id), username=username)

        # Refresh to load defaults and relationships (like created_at)
        session.refresh(new_user)

        # Explicitly query roles AFTER refresh, instead of relying on relationship
        # This is more robust in mocked environments
        committed_roles = session.query(UserRoleAssociation.role).filter_by(user_id=new_user.id).all()
        role_list = [role[0] for role in committed_roles] # Extract role strings

        return {
            "id": str(new_user.id),
            "username": new_user.username,
            "email": new_user.email,
            "is_active": new_user.is_active,
            "created_at": new_user.created_at,
            "updated_at": new_user.updated_at,
            "last_login": new_user.last_login,
            "roles": role_list, # Use the queried list
            "permissions": []
        }
    except IntegrityError as e:
        session.rollback() # Add rollback here
        # Specific handling for duplicates, caught by the decorator's IntegrityError handler
        logger.warning("Attempted to create user with duplicate username/email", username=username, email=email, error=str(e))
        raise UserAlreadyExistsError(f"User with username '{username}' or email '{email}' already exists.") from e
    # Other SQLAlchemyError or general Exception will be caught by handle_db_session decorator

@handle_db_session
def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a user with username and password using SQLAlchemy ORM and bcrypt.
    """
    session = _get_session()
    user_id_for_auth = None # Keep for logging consistency

    # Query user by username using ORM
    user = session.query(User).filter(User.username == username).first()

    if not user:
        logger.warning("Authentication failed: User not found", username=username)
        raise InvalidCredentialsError()

    user_id_for_auth = user.id # Store UUID for logging

    # Verify password using bcrypt against the stored hash
    stored_hash_bytes = user.password_hash.encode('utf-8')
    if not bcrypt.checkpw(password.encode('utf-8'), stored_hash_bytes):
        logger.warning("Authentication failed: Incorrect password", username=username, user_id=str(user_id_for_auth))
        raise InvalidCredentialsError()

    # Check if user is active
    if not user.is_active:
        logger.warning("Authentication failed: User account inactive", username=username, user_id=str(user_id_for_auth))
        raise InvalidCredentialsError("User account is not active")

    # --- Update last login time (Keep inner try/except for this specific commit) ---
    try:
        user.last_login = datetime.utcnow() # Use UTC time now
        session.commit() # Commit this specific change
        logger.debug("Updated last_login for user", user_id=str(user_id_for_auth))
    except SQLAlchemyError as update_err:
        session.rollback() # Rollback only the failed last_login update attempt
        # Log failure but don't fail the overall authentication
        logger.warning("Failed to update last_login time after successful authentication",
                         user_id=str(user_id_for_auth), error=str(update_err))
        # User object might be in detached state, refresh if needed for subsequent reads
        # session.refresh(user) # Or re-query

    # --- Get full user info (roles/permissions) ---
    # Construct the return dictionary directly from the user object
    # Query roles explicitly as the relationship might be viewonly or lazy
    # If session.refresh was called above, this query is fine.
    # If not, and last_login failed, user object might be stale.
    # Let's assume for now the user object state is acceptable or refresh was done.
    committed_roles = session.query(UserRoleAssociation.role).filter_by(user_id=user.id).all()
    role_list = [role[0] for role in committed_roles]

    # Permissions are not managed via this table directly in the model, return empty
    permission_list = []

    user_info = {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
        "last_login": user.last_login, # Reflects the updated time (or original if update failed)
        "roles": role_list,
        "permissions": permission_list
    }

    logger.info("User authenticated successfully", username=username, user_id=str(user_id_for_auth))
    return user_info

@handle_db_session
def get_user_by_id(user_id: str) -> Dict[str, Any]:
    """
    Get user information by ID using SQLAlchemy ORM, including roles.
    """
    session = _get_session()
    # Convert user_id string to UUID if necessary (model uses UUID type)
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        logger.warning("Invalid UUID format provided for get_user_by_id", user_id=user_id)
        raise UserNotFoundError(f"Invalid user ID format: {user_id}")

    # Query user by ID
    # Eager load roles using joinedload or selectinload
    user = session.query(User).options(selectinload(User.roles)).filter(User.id == user_uuid).first()

    if not user:
        logger.warning("User not found by ID", user_id=user_id)
        raise UserNotFoundError(f"User with ID {user_id} not found")

    # Extract roles from the loaded relationship
    role_list = [assoc.role for assoc in user.roles]

    # Permissions not handled here yet
    permission_list = []

    # Construct dictionary
    user_info = {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
        "last_login": user.last_login,
        "roles": role_list,
        "permissions": permission_list
    }
    return user_info

# Define fields allowed for update via this generic function
ALLOWED_UPDATE_FIELDS = {'username', 'email', 'is_active'}

@handle_db_session
def update_user(
    user_id: str,
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update user information using SQLAlchemy ORM.
    Only allows updating specific fields (username, email, is_active).
    """
    session = _get_session()
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        logger.warning("Invalid UUID format provided for update_user", user_id=user_id)
        raise UserNotFoundError(f"Invalid user ID format: {user_id}")

    try:
        user = session.query(User).filter(User.id == user_uuid).first()

        if not user:
            logger.warning("Update failed: User not found by ID", user_id=user_id)
            raise UserNotFoundError(f"User with ID {user_id} not found")

        update_performed = False
        for key, value in data.items():
            if key in ALLOWED_UPDATE_FIELDS:
                if hasattr(user, key):
                    setattr(user, key, value)
                    update_performed = True
                    logger.debug("Updating user field", user_id=user_id, field=key)
                else:
                     # This shouldn't happen if ALLOWED_UPDATE_FIELDS is correct
                     logger.warning("Attempted to update non-existent attribute", field=key, user_id=user_id)
            elif key in {'id', 'password', 'password_hash', 'roles', 'permissions', 'created_at', 'updated_at', 'last_login'}:
                logger.warning("Attempted prohibited update via update_user", field=key, user_id=user_id)
            else:
                logger.warning("Attempted to update unknown field", field=key, user_id=user_id)


        if not update_performed:
             logger.info("No valid fields provided for user update", user_id=user_id, provided_data=data.keys())
             # Return current user data if no valid updates were provided
             return get_user_by_id(user_id)

        # updated_at is handled by onupdate=func.now() in the model definition

        session.commit()
        logger.info("User updated successfully", user_id=user_id)

        # Return the updated user data
        return get_user_by_id(user_id) # Fetches the refreshed data including updated_at

    except UserNotFoundError:
        session.rollback() # Should not be strictly needed if user wasn't found, but good practice
        raise
    except IntegrityError as e:
        session.rollback()
        logger.warning("Update failed due to integrity error (likely duplicate username/email)", user_id=user_id, error=str(e))
        raise UserAlreadyExistsError(f"Update failed: New username or email might already exist.")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error updating user", user_id=user_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to update user: {e}")
    except Exception as e:
        session.rollback()
        logger.error("Unexpected error updating user", user_id=user_id, error=str(e), exc_info=True)
        raise DatabaseError(f"An unexpected error occurred updating user: {e}")

@handle_db_session
def update_user_roles(user_id_str: str, roles: list[str]) -> dict:
    """
    Update user roles in the database using SQLAlchemy.
    Removes roles not in the input list and adds new ones.
    """
    session = _get_session()
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        logger.error("Invalid user ID format for role update", user_id=user_id_str)
        raise UserNotFoundError("Invalid user ID format")

    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning("User not found for role update", user_id=user_id_str)
        raise UserNotFoundError(f"User with ID {user_id_str} not found")

    # Get current roles for the user
    current_role_assocs = session.query(UserRoleAssociation).filter(
        UserRoleAssociation.user_id == user_id
    ).all()
    current_roles = {assoc.role for assoc in current_role_assocs}
    target_roles = set(roles)

    roles_to_add = target_roles - current_roles
    roles_to_remove = current_roles - target_roles

    if not roles_to_add and not roles_to_remove:
        logger.info("No role changes needed for user", user_id=user_id_str)
        # Re-query roles to return current state accurately
        updated_roles = [role[0] for role in session.query(UserRoleAssociation.role).filter(UserRoleAssociation.user_id == user_id).all()]
        return _user_to_dict(user, roles=updated_roles)

    # Remove roles
    if roles_to_remove:
        logger.info("Removing roles from user", user_id=user_id_str, roles_to_remove=list(roles_to_remove))
        session.query(UserRoleAssociation).filter(
            UserRoleAssociation.user_id == user_id,
            UserRoleAssociation.role.in_(roles_to_remove)
        ).delete(synchronize_session=False)

    # Add new roles
    if roles_to_add:
        logger.info("Adding roles to user", user_id=user_id_str, roles_to_add=list(roles_to_add))
        for role_name in roles_to_add:
            new_assoc = UserRoleAssociation(user_id=user_id, role=role_name)
            session.add(new_assoc)
    
    # Update timestamp and commit
    user.updated_at = datetime.now(timezone.utc)
    session.commit()
    logger.info("User roles updated successfully", user_id=user_id_str, new_roles=roles)
    
    # Re-query roles to return the final state
    final_roles = [role[0] for role in session.query(UserRoleAssociation.role).filter(UserRoleAssociation.user_id == user_id).all()]
    return _user_to_dict(user, roles=final_roles)

@handle_db_session
def update_user_password(user_id_str: str, new_password: str) -> bool:
    """
    Update user password using bcrypt and SQLAlchemy.
    """
    session = _get_session()
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        logger.error("Invalid user ID format for password update", user_id=user_id_str)
        raise UserNotFoundError("Invalid user ID format")

    user = session.query(User).filter(User.id == user_id).first()

    if not user:
        logger.warning("User not found for password update", user_id=user_id_str)
        raise UserNotFoundError(f"User with ID {user_id_str} not found for password update.")

    # Hash the new password using bcrypt
    new_hashed_password_bytes = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    new_hashed_password_str = new_hashed_password_bytes.decode('utf-8')

    user.password_hash = new_hashed_password_str
    user.updated_at = datetime.now(timezone.utc)

    session.commit()
    logger.info("User password updated successfully", user_id=user_id_str)
    return True

# --- API Keys ---

@handle_db_session
def create_api_key(user_id_str: str, name: str, expires_at: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Create a new API key for a user using SQLAlchemy ORM.
    Generates a new key, hashes it, and stores it.
    Returns the key details including the *unhashed* key.
    """
    session = _get_session()
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        logger.error("Invalid user ID format for API key creation", user_id=user_id_str)
        raise UserNotFoundError("Invalid user ID format")

    # Check if user exists
    user = session.query(User.id).filter(User.id == user_id).first()
    if not user:
        logger.warning("Failed to create API key: User not found", user_id=user_id_str)
        raise UserNotFoundError(f"Cannot create API key: User with ID {user_id_str} not found")

    # Generate a secure API key (e.g., using UUID or secrets module)
    # Using a simple prefix + UUID for illustration, enhance as needed.
    key_prefix = "sk_" # Example prefix
    key_suffix = str(uuid.uuid4()).replace('-','') # Remove hyphens for cleaner key
    api_key = f"{key_prefix}{key_suffix}"
    
    # Hash the key using bcrypt
    hashed_key = bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Create the new APIKey object
    new_api_key = APIKey(
        user_id=user_id,
        name=name,
        key_hash=hashed_key,
        expires_at=expires_at
        # id, created_at, updated_at, last_used, is_active have defaults
    )
    new_api_key.prefix = key_prefix # Store the prefix for potential future filtering

    session.add(new_api_key)
    session.commit()
    logger.info("API key created successfully", user_id=user_id_str, key_id=str(new_api_key.id), key_name=name)
    
    # Refresh to get default values like created_at and id
    session.refresh(new_api_key)

    # Return the *unhashed* key and its details
    return {
        'key_id': str(new_api_key.id),
        'user_id': user_id_str,
        'name': new_api_key.name,
        'api_key': api_key,  # Important: Return the actual key here
        'prefix': new_api_key.prefix,
        'created_at': new_api_key.created_at,
        'expires_at': new_api_key.expires_at,
        'last_used': new_api_key.last_used,
        'is_active': new_api_key.is_active
    }

@handle_db_session
def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Verify an API key by comparing against stored bcrypt hashes, with Redis caching.
    Fetches the associated user information if the key is valid, active, and not expired.

    Args:
        api_key: The plain text API key provided by the user (e.g., 'sk_abcdef123').

    Returns:
        Dictionary containing user information if the key is valid, otherwise None.
        Raises InvalidCredentialsError if the key format is invalid, not found, or inactive.
    """
    logger.debug("Attempting to verify API key...")

    # --- Extract Key ID and Prefix --- #
    try:
        # Expecting format like "prefix_keyIdPart"
        parts = api_key.split('_')
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Invalid API key format")
        key_prefix = parts[0]
        key_id_str = parts[1] # The part used for DB lookup and potentially cache key
        # Validate key_id_str looks like a UUID hex or similar identifier if needed
        # uuid.UUID(key_id_str) # Optionally validate format, but DB query will fail anyway
        logger.debug("Extracted key components", prefix=key_prefix, key_id=key_id_str)
    except (ValueError, IndexError) as e:
        logger.warning("Invalid API key format provided", api_key_prefix=api_key[:8] + "..." if len(api_key) > 8 else api_key, error=str(e))
        raise InvalidCredentialsError("Invalid API Key format")

    # --- Caching Logic --- #
    redis_client = _get_redis_client()
    cache_key = None
    if redis_client:
        try:
            # Use SHA256 hash of the KEY_ID part for the cache key
            key_id_hash_for_cache = hashlib.sha256(key_id_str.encode('utf-8')).hexdigest()
            cache_key = f"{API_KEY_CACHE_PREFIX}{key_id_hash_for_cache}"
            logger.debug("Generated cache key", cache_key=cache_key)

            cached_result_bytes = redis_client.get(cache_key)
            if cached_result_bytes:
                logger.debug("API key verification cache hit", cache_key=cache_key)

                # *** Check for invalid marker FIRST ***
                if cached_result_bytes == API_KEY_CACHE_INVALID_MARKER.encode('utf-8'): # Compare bytes
                    logger.warning("API key known to be invalid (cache marker hit)", key_id=key_id_str)
                    raise InvalidCredentialsError("Invalid API Key (cached)") # Fail fast

                # *** If not the marker, proceed to decode and parse ***
                cached_result_str = cached_result_bytes.decode('utf-8') # Decode bytes

                # Potential valid data found in cache
                try:
                    cached_data = json.loads(cached_result_str)
                    # Verify the hash from the cache against the provided full key
                    cached_hash = cached_data.get('key_hash')
                    if not cached_hash or not verify_api_key_hash(api_key, cached_hash):
                        logger.warning("Cache hash mismatch for API key", key_id=key_id_str)
                        # Invalidate cache and raise error (treat as invalid)
                        try: redis_client.delete(cache_key) # Remove bad cache entry
                        except redis.exceptions.RedisError: pass # Ignore delete error
                        raise InvalidCredentialsError("Invalid API Key (cache mismatch)")
                    
                    # Check expirations/activity from cache
                    if not cached_data.get('is_active') or not cached_data.get('user_is_active'):
                         logger.warning("Cached key/user marked inactive", key_id=key_id_str)
                         raise InvalidCredentialsError("Invalid API Key (cached inactive)")
                    expires_at_str = cached_data.get('expires_at')
                    if expires_at_str:
                         expires_at_dt = datetime.fromisoformat(expires_at_str)
                         if expires_at_dt < datetime.now(timezone.utc):
                             logger.warning("Cached key expired", key_id=key_id_str)
                             raise InvalidCredentialsError("Invalid API Key (cached expired)")
                    
                    # If all checks pass, return the necessary user info from cache
                    # Construct user info from cached data (or cache the exact dict needed)
                    user_info = {
                        'id': cached_data.get('user_id'),
                        'key_id': key_id_str, # Add key ID to returned info
                        # Include other relevant fields directly from cache if needed
                        # 'username': cached_data.get('username'), 
                        # 'roles': cached_data.get('roles'), 
                    }
                    logger.info("API key verified successfully via cache", key_id=key_id_str, user_id=user_info['id'])
                    # Skipping DB last_used update on cache hit for performance
                    return user_info
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.warning("Failed to process cached data for API key", cache_key=cache_key, error=str(e))
                    # Proceed to DB check if cache data is corrupt or missing fields
            else:
                logger.debug("API key verification cache miss", cache_key=cache_key)
        except redis.exceptions.RedisError as e:
            logger.warning(f"Redis error during API key cache check: {e}. Proceeding without cache.")
            redis_client = None # Disable further cache operations on error
        except InvalidCredentialsError: # Explicitly catch and re-raise auth errors from cache logic
            raise
        except Exception as e:
            logger.error(f"Unexpected error during cache check: {e}", exc_info=True)
            redis_client = None # Disable cache on unexpected error
    # --- End Caching Logic --- #

    session = _get_session()
    user_info: Optional[Dict[str, Any]] = None

    try:
        # --- Database Lookup & Verification (executed if cache miss) --- #
        logger.debug("Performing DB lookup for API key verification", key_id=key_id_str)
        current_time = datetime.now(timezone.utc)

        # Query specifically by key_id (should be unique or identifiable)
        # Assuming key_id_str is the primary key (UUID) of the APIKey table
        try:
            key_uuid = uuid.UUID(key_id_str)
        except ValueError:
             logger.warning("Invalid UUID format for API key ID part", key_id=key_id_str)
             raise InvalidCredentialsError("Invalid API Key format")

        matched_key_record = session.query(APIKey).options(
            joinedload(APIKey.user) # Eager load user
        ).filter(
            APIKey.id == key_uuid
        ).first()

        # --- Verification Checks --- #
        if not matched_key_record:
            logger.warning("API key not found in DB", key_id=key_id_str)
            # Cache invalid result
            if redis_client and cache_key: _cache_invalid_result(redis_client, cache_key)
            raise InvalidCredentialsError("Invalid API Key")

        # *** Check activity and expiration BEFORE hash verification ***
        if not matched_key_record.is_active:
            logger.warning("API key is inactive", key_id=key_id_str)
            if redis_client and cache_key: _cache_invalid_result(redis_client, cache_key)
            raise InvalidCredentialsError("Invalid API Key")

        if matched_key_record.expires_at and matched_key_record.expires_at < current_time:
            logger.warning("API key has expired", key_id=key_id_str, expires_at=matched_key_record.expires_at)
            if redis_client and cache_key: _cache_invalid_result(redis_client, cache_key)
            raise InvalidCredentialsError("Invalid API Key")

        # *** Verify hash only if key is active and not expired ***
        if not verify_api_key_hash(api_key, matched_key_record.key_hash):
            logger.warning("API key hash mismatch", key_id=key_id_str)
            # Cache invalid result
            if redis_client and cache_key: _cache_invalid_result(redis_client, cache_key)
            raise InvalidCredentialsError("Invalid API Key")

        # Verify associated user
        user = matched_key_record.user
        if not user or not user.is_active:
            logger.warning("Associated user is inactive or not found", key_id=key_id_str, user_id=str(user.id) if user else None)
            # Cache invalid result
            if redis_client and cache_key: _cache_invalid_result(redis_client, cache_key)
            raise InvalidCredentialsError("Invalid API Key")

        # --- Key is valid, construct user info --- #
        # Fetch roles if needed (consider if they should be cached too)
        # role_list = [assoc.role for assoc in user.roles] # Requires eager loading or separate query
        user_info = {
            "id": str(user.id),
            "key_id": key_id_str, # Include key ID
            # "username": user.username,
            # "roles": role_list,
        }

        # --- Update last used time --- #
        try:
            matched_key_record.last_used = datetime.now(timezone.utc)
            session.commit()
            logger.debug("Updated last_used for API key", key_id=key_id_str)
        except SQLAlchemyError as update_err:
            session.rollback()
            logger.warning("Failed to update last_used time for API key", key_id=key_id_str, error=str(update_err))
            # Continue despite error, key is verified

        logger.info("API Key verified successfully via DB", key_id=key_id_str, user_id=str(user.id))

        # --- Cache the valid result --- #
        if redis_client and cache_key:
            try:
                # Cache essential details needed to bypass DB next time
                cache_data = {
                    "key_hash": matched_key_record.key_hash,
                    "user_id": str(user.id),
                    "is_active": matched_key_record.is_active,
                    "user_is_active": user.is_active,
                    "expires_at": matched_key_record.expires_at.isoformat() if matched_key_record.expires_at else None,
                    # Add other fields if needed for cache hit logic (e.g., roles)
                }
                redis_client.setex(cache_key, API_KEY_CACHE_TTL, json.dumps(cache_data))
                logger.debug("Cached valid API key verification result", cache_key=cache_key)
            except (redis.exceptions.RedisError, TypeError, json.JSONDecodeError) as e:
                 logger.warning(f"Failed to cache valid API key result: {e}")

        return user_info

    except InvalidCredentialsError: # Re-raise expected auth errors
         raise
    except Exception as e:
        logger.error("Unexpected error during API key verification", error=str(e), exc_info=True)
        # Cache invalid result on unexpected error during DB phase
        if redis_client and cache_key: _cache_invalid_result(redis_client, cache_key)
        raise DatabaseError(f"API key verification failed: {e}") from e

def _cache_invalid_result(redis_client: redis.Redis, cache_key: str):
    """Helper to cache the invalid marker, ignoring Redis errors."""
    try:
        redis_client.setex(cache_key, API_KEY_CACHE_TTL_INVALID, API_KEY_CACHE_INVALID_MARKER.encode('utf-8')) # Use correct TTL and encode marker
        logger.debug("Cached invalid API key marker", cache_key=cache_key)
    except redis.exceptions.RedisError as e:
        logger.warning(f"Redis SETEX error caching invalid key marker: {e}")

@handle_db_session
def get_user_api_keys(user_id_str: str) -> List[Dict[str, Any]]:
    """
    Get all API keys associated with a user using SQLAlchemy ORM.
    Excludes the key_hash from the result.
    """
    session = _get_session()
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        logger.warning("Invalid UUID format provided for get_user_api_keys", user_id=user_id_str)
        return [] # Return empty list for invalid user ID format

    # Query API keys for the user, ordered by creation time
    keys = session.query(APIKey).filter(APIKey.user_id == user_id).order_by(APIKey.created_at.desc()).all()

    # Format keys into dictionaries, excluding the hash
    key_list = []
    for key in keys:
        key_list.append({
            "key_id": str(key.id),
            "user_id": str(key.user_id),
            "name": key.name,
            "prefix": key.prefix,
            "created_at": key.created_at,
            "expires_at": key.expires_at,
            "last_used": key.last_used,
            "is_active": key.is_active
            # IMPORTANT: Do NOT include key.key_hash
        })

    logger.debug("Retrieved API keys for user", user_id=user_id_str, count=len(key_list))
    return key_list

@handle_db_session
def revoke_api_key(key_id_str: str) -> bool:
    """
    Revoke an API key by setting its is_active flag to False using SQLAlchemy ORM.
    """
    session = _get_session()
    try:
        key_id = uuid.UUID(key_id_str)
    except ValueError:
        logger.warning("Invalid UUID format provided for revoke_api_key", key_id=key_id_str)
        raise ApiKeyNotFoundError("Invalid API key ID format") # Raise specific error

    # Find the key
    api_key = session.query(APIKey).filter(APIKey.id == key_id).first()

    if not api_key:
        logger.warning("API key revocation failed: Key not found", key_id=key_id_str)
        raise ApiKeyNotFoundError(f"API Key with ID {key_id_str} not found")

    if not api_key.is_active:
        logger.info("API key already revoked", key_id=key_id_str)
        return True # Consider already revoked as success

    # Revoke the key
    api_key.is_active = False
    # Consider setting updated_at if the model has it
    if hasattr(api_key, 'updated_at'):
         setattr(api_key, 'updated_at', datetime.now(timezone.utc))

    session.commit()

    # --- Cache Invalidation (Important!) ---
    redis_client = _get_redis_client()
    if redis_client:
        # How to invalidate? Need the original API key string which we don't have here.
        # Option 1: Store hash -> key_id mapping? Complex.
        # Option 2: Add key_id to cache key? `apikey_verify_cache:<key_id>:<hash>`?
        # Option 3: Brute-force delete keys matching prefix? Risky.
        # Option 4: Keep cache TTL short and accept potential staleness on revoke.
        # For now, rely on short TTL. Add explicit invalidation if requirements demand it.
        logger.warning("API key revoked, but cache invalidation is not implemented. Relying on TTL.", key_id=key_id_str)
        # Example (if cache key included key_id):
        # cache_key_pattern = f"{API_KEY_CACHE_PREFIX}{key_id_str}:*"
        # try:
        #     keys_to_delete = redis_client.keys(cache_key_pattern)
        #     if keys_to_delete:
        #         redis_client.delete(*keys_to_delete)
        #         logger.info("Invalidated cache entries for revoked key", key_id=key_id_str, count=len(keys_to_delete))
        # except redis.exceptions.RedisError as e:
        #     logger.error("Redis error during cache invalidation for revoked key", key_id=key_id_str, error=str(e))
        
    logger.info("API key revoked successfully", key_id=key_id_str)
    return True

@handle_db_session
def get_all_users(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Get a list of all users with pagination using SQLAlchemy ORM.
    Includes basic user info and roles.
    """
    session = _get_session()
    users_list = []

    try:
        # Query users with pagination and eager load roles using selectinload
        # Note: Eager loading roles might still be heavy if there are many users/roles.
        # We need to query UserRoleAssociation directly if User.roles is viewonly=True
        # Let's adjust the query strategy:
        
        # 1. Fetch base user info with pagination
        base_query = session.query(User).order_by(User.created_at)
        if limit > 0:
            base_query = base_query.limit(limit)
        if offset > 0:
            base_query = base_query.offset(offset)
        users = base_query.all()
        
        if not users:
            return []
            
        user_ids = [user.id for user in users]

        # 2. Fetch roles for these specific users in a separate query
        roles_query = session.query(UserRoleAssociation).filter(UserRoleAssociation.user_id.in_(user_ids)).all()
        roles_by_user_id = {}
        for role_assoc in roles_query:
            if role_assoc.user_id not in roles_by_user_id:
                roles_by_user_id[role_assoc.user_id] = []
            roles_by_user_id[role_assoc.user_id].append(role_assoc.role)

        # 3. Format the results
        for user in users:
            role_list = roles_by_user_id.get(user.id, [])
            permission_list = [] # Permissions not handled

            users_list.append({
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "is_active": user.is_active,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
                "last_login": user.last_login,
                "roles": role_list,
                "permissions": permission_list
            })

        logger.debug("Retrieved all users", count=len(users_list), limit=limit, offset=offset)
        return users_list

    except SQLAlchemyError as e:
        logger.error("Database error retrieving all users", error=str(e), exc_info=True)
        raise QueryError(f"Failed to get all users: {e}")
    except Exception as e:
        logger.error("Unexpected error retrieving all users", error=str(e), exc_info=True)
        raise DatabaseError(f"An unexpected error occurred getting all users: {e}")

# Optional Cleanup: Remove the old UserDB class and get_connection function if they exist
# Optional Cleanup: Remove unused imports like mysql.connector

# No finally needed 