#!/usr/bin/env python3
"""
User Database Service

Provides database operations for user management, authentication,
and API key management using a connection pool managed by the main application.
"""

# import logging # Removed
# import time # Removed
from typing import Dict, Any, List, Optional # Removed Tuple, Union
from datetime import datetime, timedelta
# import mysql.connector # Removed
# from mysql.connector import pooling, Error # Removed
import uuid
import bcrypt
import structlog # Added
from flask import current_app # Added

# Import SQLAlchemy components
from sqlalchemy.orm import Session # Add Session import
from sqlalchemy.exc import IntegrityError, SQLAlchemyError # Add SQLAlchemy exceptions
from sqlalchemy.orm import joinedload, selectinload # To fetch User efficiently
from sqlalchemy import delete # Import delete statement

# Import Models
from .models.user_models import User, UserRoleAssociation, APIKey # Import User, association, and APIKey

# Import utilities
# from ..utils.auth_utils import hash_password, hash_api_key, verify_password # Assuming these might be deprecated if bcrypt is used directly
from .exceptions import (
    DatabaseError, ConnectionError, QueryError, 
    UserNotFoundError, UserAlreadyExistsError, InvalidCredentialsError, DuplicateUserError, APIKeyNotFoundError
)

# Configure logger
logger = structlog.get_logger(__name__) # Changed

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

# User Operations

def create_user(
    username: str,
    email: str,
    password: str,
    roles: List[str] = None
) -> Dict[str, Any]:
    """
    Create a new user using SQLAlchemy ORM.
    """
    session = _get_session()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    if roles is None:
        roles = ["user"]

    new_user = User(
        username=username,
        email=email,
        password_hash=hashed_password
        # id, created_at, updated_at, is_active have defaults in the model
    )

    try:
        session.add(new_user)
        # Flush to get the new_user.id before adding roles
        session.flush()

        # Assign roles using the association class
        if roles:
            for role_name in roles:
                # Check if role already exists can be added here if needed
                user_role = UserRoleAssociation(user_id=new_user.id, role=role_name)
                session.add(user_role)

        session.commit()
        logger.info("User created and committed successfully", user_id=str(new_user.id), username=username)
        
        # Construct the return dictionary matching the previous format as closely as possible
        # We have the basic user info, roles were handled. Need to fetch created_at etc.?
        # Refreshing loads the defaults like created_at
        session.refresh(new_user)
        # Fetching roles explicitly for the return dictionary if `viewonly=True` is used
        # Or just return the input roles list for simplicity now.
        
        # Get roles from the association table after commit/refresh
        committed_roles = session.query(UserRoleAssociation.role).filter_by(user_id=new_user.id).all()
        role_list = [role[0] for role in committed_roles]

        return {
            "id": str(new_user.id), # Convert UUID to string
            "username": new_user.username,
            "email": new_user.email,
            "is_active": new_user.is_active,
            "created_at": new_user.created_at, # Already datetime
            "updated_at": new_user.updated_at, # Already datetime
            "last_login": new_user.last_login, # Already datetime or None
            "roles": role_list, # Use the roles we added
            "permissions": [] # Permissions not handled in create_user, return empty list
        }

    except IntegrityError as e:
        session.rollback()
        logger.warning("Failed to create user due to integrity error (likely duplicate username/email)", username=username, email=email, error=str(e))
        # Check if it's specifically a duplicate key error
        # The exact error message/code might vary by DB backend
        # For MySQL with unique constraints, this is a common way
        # if "Duplicate entry" in str(e.orig):
        raise DuplicateUserError(f"Username '{username}' or email '{email}' already exists")
        # else:
        #    raise QueryError(f"Database integrity error: {e}")

    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error creating user", username=username, email=email, error=str(e), exc_info=True)
        raise QueryError(f"Failed to create user: {e}")

    except Exception as e:
        session.rollback()
        logger.error("Unexpected error creating user", username=username, email=email, error=str(e), exc_info=True)
        raise DatabaseError(f"An unexpected error occurred: {e}")

def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a user with username and password using SQLAlchemy ORM and bcrypt.
    """
    session = _get_session()
    user_id_for_auth = None # Keep for logging consistency

    try:
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

        # --- Update last login time ---
        try:
            user.last_login = datetime.utcnow() # Use UTC time now
            session.commit() # Commit this specific change
            logger.debug("Updated last_login for user", user_id=str(user_id_for_auth))
        except SQLAlchemyError as update_err:
             session.rollback() # Rollback only the failed last_login update attempt
             # Log failure but don't fail the overall authentication
             logger.warning("Failed to update last_login time after successful authentication",
                              user_id=str(user_id_for_auth), error=str(update_err))
             # Fetch the user again or refresh if needed, as the session might be in a weird state
             # For simplicity, we'll proceed assuming the main user object is still usable
             # Alternatively, re-fetch user = session.query(User).get(user_id_for_auth)

        # --- Get full user info (roles/permissions) ---
        # Construct the return dictionary directly from the user object
        # Query roles explicitly as the relationship might be viewonly or lazy
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

    except InvalidCredentialsError: # Re-raise specific auth errors
        # No rollback needed here as no changes were committed before the error
        raise
    except SQLAlchemyError as e:
        session.rollback() # Rollback any potential changes (e.g., failed last_login)
        logger.error("Database error during authentication", username=username, user_id=str(user_id_for_auth) if user_id_for_auth else None, error=str(e), exc_info=True)
        raise QueryError(f"Database error during authentication: {e}")
    except Exception as e:
         # Catch unexpected errors
         session.rollback()
         logger.error("Unexpected error during authentication", username=username, user_id=str(user_id_for_auth) if user_id_for_auth else None, error=str(e), exc_info=True)
         raise DatabaseError(f"An unexpected error occurred during authentication: {e}")

def get_user_by_id(user_id: str) -> Dict[str, Any]:
    """
    Get user information by ID using SQLAlchemy ORM, including roles.
    """
    session = _get_session()
    try:
        # Convert user_id string to UUID if necessary (model uses UUID type)
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            logger.warning("Invalid UUID format provided for get_user_by_id", user_id=user_id)
            raise UserNotFoundError(f"Invalid user ID format: {user_id}")

        # Query user by ID
        user = session.query(User).filter(User.id == user_uuid).first()

        if not user:
            logger.warning("User not found by ID", user_id=user_id)
            raise UserNotFoundError(f"User with ID {user_id} not found")

        # Query roles explicitly
        committed_roles = session.query(UserRoleAssociation.role).filter_by(user_id=user.id).all()
        role_list = [role[0] for role in committed_roles]

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

        logger.debug("User retrieved by ID", user_id=user_id, username=user.username)
        return user_info

    except UserNotFoundError:
        raise # Re-raise specific error
    except SQLAlchemyError as e:
        # Don't rollback reads
        logger.error("Database error retrieving user by ID", user_id=user_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to get user by ID: {e}")
    except Exception as e:
         logger.error("Unexpected error retrieving user by ID", user_id=user_id, error=str(e), exc_info=True)
         raise DatabaseError(f"An unexpected error occurred retrieving user by ID: {e}")

# Define fields allowed for update via this generic function
ALLOWED_UPDATE_FIELDS = {'username', 'email', 'is_active'}

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
        raise DuplicateUserError(f"Update failed: New username or email might already exist.")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error updating user", user_id=user_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to update user: {e}")
    except Exception as e:
        session.rollback()
        logger.error("Unexpected error updating user", user_id=user_id, error=str(e), exc_info=True)
        raise DatabaseError(f"An unexpected error occurred updating user: {e}")

def update_user_roles(
    user_id: str,
    roles: List[str]
) -> Dict[str, Any]:
    """
    Update user roles. Replaces existing roles with the provided list.
    
    Args:
        user_id: User ID (UUID string)
        roles: List of role names
        
    Returns:
        Updated user information dictionary
        
    Raises:
        UserNotFoundError: If user not found
        QueryError: If database query fails
    """
    user_exists_flag = False # Flag to check if user was found
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:

                # Check if user exists
                cursor.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (user_id,)) # Use string user_id, lock row
                if cursor.fetchone():
                    user_exists_flag = True
                else:
                    raise UserNotFoundError(f"User with ID {user_id} not found")

                # Delete existing roles from user_roles table
                cursor.execute("DELETE FROM user_roles WHERE user_id = %s", (user_id,))

                # Insert new roles directly into user_roles table
                if roles:
                    role_values = [(user_id, role_name) for role_name in roles]
                    cursor.executemany(
                        "INSERT INTO user_roles (user_id, role) VALUES (%s, %s)",
                        role_values
                    )

                conn.commit()
                logger.info("User roles updated successfully in DB", user_id=user_id, new_roles=roles)

    except UserNotFoundError:
         raise # Re-raise
    except Error as e:
        # Rollback handled automatically
        logger.error("Database error updating user roles", user_id=user_id, roles=roles, error=str(e), exc_info=True)
        raise QueryError(f"Failed to update user roles: {str(e)}")
    except Exception as e:
        logger.error("Unexpected error updating user roles", user_id=user_id, error=str(e), exc_info=True)
        raise QueryError(f"Unexpected error updating user roles: {str(e)}")

    # Fetch updated user details outside transaction context
    try:
         return get_user_by_id(user_id)
    except Exception as fetch_err:
         logger.error("Failed to fetch user details after successful role update", user_id=user_id, error=str(fetch_err))
         return {"id": user_id, "roles": roles, "warning": "Roles updated but failed to fetch full details."}

def update_user_password(user_id: str, new_password: str) -> bool:
    """
    Update user password using bcrypt.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Hash the new password using bcrypt (salt is generated internally)
                new_hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
                
                cursor.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (new_hashed_password.decode('utf-8'), user_id)
                )
                
                if cursor.rowcount == 0:
                    logger.warning("Password update failed: User not found", user_id=user_id)
                    raise UserNotFoundError(f"User with ID {user_id} not found for password update.")
            
                conn.commit()
                logger.info("User password updated successfully", user_id=user_id)
                return True
    except UserNotFoundError: # Re-raise specific error
        raise
    except Error as e:
        # Rollback handled automatically
        logger.error("Database error updating password", user_id=user_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to update password: {str(e)}")
    except Exception as e:
        logger.error("Unexpected error updating password", user_id=user_id, error=str(e), exc_info=True)
        raise QueryError(f"Unexpected error updating password: {str(e)}")

# API Key Operations

def create_api_key(user_id: str, name: str, expires_at: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Create a new API key for a user.
    """
    key_id = str(uuid.uuid4()) 
    # Generate a secure API key (consider using secrets module)
    api_key = str(uuid.uuid4()) # Simple UUID for now, use more secure generation
    hashed_key = bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') # Hash the key
    
    try:
        # Use context managers
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO api_keys (id, user_id, name, key_hash, expires_at, is_active) VALUES (%s, %s, %s, %s, %s, TRUE)",
                    (key_id, user_id, name, hashed_key, expires_at)
                )
                conn.commit()
                logger.info("API key created successfully in DB", user_id=user_id, key_id=key_id, key_name=name) # Changed

    except Error as e:
        # Rollback handled automatically
        # Check for specific foreign key constraint violation (user_id not found)
        if hasattr(e, 'errno') and e.errno == 1452:
            logger.warning("Failed to create API key: User not found", user_id=user_id, key_name=name, error_code=e.errno)
            raise UserNotFoundError(f"Cannot create API key: User with ID {user_id} not found")
        else:
            # Handle other database errors
            logger.error("Database error creating API key", user_id=user_id, key_name=name, error=str(e), exc_info=True) # Changed
            raise QueryError(f"Failed to create API key: {str(e)}")
    except Exception as e:
        logger.error("Unexpected error creating API key", user_id=user_id, key_name=name, error=str(e), exc_info=True)
        raise QueryError(f"Unexpected error creating API key: {str(e)}")

    # Return the *unhashed* key only after successful commit
    return {
        'id': key_id,
        'user_id': user_id,
        'name': name,
        'api_key': api_key, # Important: Return the actual key here
        'created_at': datetime.utcnow(), # Approximate
        'expires_at': expires_at,
        'is_active': True
    }

def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Verify an API key by checking it against stored hashes.
    Returns the associated user info if valid, active, and not expired.
    Updates last_used timestamp on successful verification.

    Note: This iterates through keys, which might be slow with many keys.
    """
    session = _get_session()
    matched_key_object = None
    user_info = None

    try:
        # Fetch all potentially relevant API keys (active ones)
        # Eagerly load the associated user to check user.is_active efficiently
        candidate_keys = session.query(APIKey).options(joinedload(APIKey.user)).filter(APIKey.is_active == True).all()

        for key_obj in candidate_keys:
            stored_hash_bytes = key_obj.key_hash.encode('utf-8')
            # Check if the provided key matches the stored hash
            if bcrypt.checkpw(api_key.encode('utf-8'), stored_hash_bytes):
                # Found a potential match, now perform checks
                matched_key_object = key_obj
                logger.debug("API key hash matched", key_id=str(matched_key_object.id))

                # Check expiry
                if matched_key_object.expires_at and matched_key_object.expires_at < datetime.utcnow():
                    logger.warning("API key verification failed: Key expired", key_id=str(matched_key_object.id), user_id=str(matched_key_object.user_id))
                    # Optionally deactivate the key here:
                    # matched_key_object.is_active = False
                    # session.commit()
                    matched_key_object = None # Reset match
                    continue # Check next key if any

                # Check if key is active (redundant due to initial filter, but safe)
                if not matched_key_object.is_active:
                     logger.warning("API key verification failed: Key inactive", key_id=str(matched_key_object.id), user_id=str(matched_key_object.user_id))
                     matched_key_object = None # Reset match
                     continue # Check next key

                # Check if the associated user is active
                if not matched_key_object.user or not matched_key_object.user.is_active:
                    logger.warning("API key verification failed: Associated user inactive or not found", key_id=str(matched_key_object.id), user_id=str(matched_key_object.user_id))
                    matched_key_object = None # Reset match
                    continue # Check next key

                # --- All checks passed ---
                logger.info("API key verified successfully", key_id=str(matched_key_object.id), user_id=str(matched_key_object.user_id))

                # Update last_used time
                try:
                    matched_key_object.last_used = datetime.utcnow()
                    session.commit()
                    logger.debug("Updated last_used for API key", key_id=str(matched_key_object.id))
                except SQLAlchemyError as update_err:
                     session.rollback()
                     logger.warning("Failed to update last_used for API key after verification",
                                      key_id=str(matched_key_object.id), error=str(update_err))
                     # Proceed with returning user info despite failing to update last_used

                # Fetch full user details (including roles/permissions)
                # Use the user object we already loaded via joinedload
                user = matched_key_object.user
                # Query roles explicitly
                committed_roles = session.query(UserRoleAssociation.role).filter_by(user_id=user.id).all()
                role_list = [role[0] for role in committed_roles]
                permission_list = [] # Permissions not handled

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
                break # Found valid key, stop iterating

        if user_info:
             return user_info
        else:
             logger.warning("API key verification failed: No valid matching key found", provided_key_partial=api_key[:10] + "...") # Log partial key for debugging
             return None # No valid, active, unexpired key matched

    except SQLAlchemyError as e:
        # Don't rollback reads typically, but rollback if last_used update failed mid-process
        # It's complex to track, safer to rollback on any DB error during verification
        session.rollback()
        logger.error("Database error during API key verification", error=str(e), exc_info=True)
        raise QueryError(f"Database error verifying API key: {e}")
    except Exception as e:
        session.rollback()
        logger.error("Unexpected error during API key verification", error=str(e), exc_info=True)
        raise DatabaseError(f"An unexpected error occurred verifying API key: {e}")

def get_user_api_keys(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all API keys associated with a user using SQLAlchemy ORM.
    Excludes the key_hash from the result.
    """
    session = _get_session()
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        logger.warning("Invalid UUID format provided for get_user_api_keys", user_id=user_id)
        # Return empty list or raise UserNotFoundError depending on desired behavior
        return []
        # Or: raise UserNotFoundError(f"Invalid user ID format: {user_id}")

    try:
        # Query API keys for the user
        keys = session.query(APIKey).filter(APIKey.user_id == user_uuid).all()

        # Format keys into dictionaries, excluding the hash
        key_list = []
        for key in keys:
            key_list.append({
                "id": str(key.id),
                "user_id": str(key.user_id),
                "name": key.name,
                "created_at": key.created_at,
                "expires_at": key.expires_at,
                "last_used": key.last_used,
                "is_active": key.is_active
                # IMPORTANT: Do NOT include key.key_hash
            })

        logger.debug("Retrieved API keys for user", user_id=user_id, count=len(key_list))
        return key_list

    except SQLAlchemyError as e:
        logger.error("Database error retrieving API keys for user", user_id=user_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to get API keys: {e}")
    except Exception as e:
        logger.error("Unexpected error retrieving API keys for user", user_id=user_id, error=str(e), exc_info=True)
        raise DatabaseError(f"An unexpected error occurred getting API keys: {e}")

def revoke_api_key(key_id: str) -> bool:
    """
    Revoke an API key by setting its is_active flag to False using SQLAlchemy ORM.
    """
    session = _get_session()
    try:
        key_uuid = uuid.UUID(key_id)
    except ValueError:
        logger.warning("Invalid UUID format provided for revoke_api_key", key_id=key_id)
        return False # Or raise an appropriate error

    try:
        # Find the key
        api_key = session.query(APIKey).filter(APIKey.id == key_uuid).first()

        if not api_key:
            logger.warning("API key revocation failed: Key not found", key_id=key_id)
            return False # Key doesn't exist

        if not api_key.is_active:
            logger.info("API key already revoked", key_id=key_id)
            return True # Consider already revoked as success

        # Revoke the key
        api_key.is_active = False
        # updated_at timestamp on User won't be triggered, only APIKey if it had one

        session.commit()
        logger.info("API key revoked successfully", key_id=key_id)
        return True

    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error revoking API key", key_id=key_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to revoke API key: {e}")
    except Exception as e:
        session.rollback()
        logger.error("Unexpected error revoking API key", key_id=key_id, error=str(e), exc_info=True)
        raise DatabaseError(f"An unexpected error occurred revoking API key: {e}")

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