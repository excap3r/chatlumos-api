#!/usr/bin/env python3
"""
User Database Service

Provides database operations for user management, authentication,
and API key management.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import pooling

# Import utilities
from ..utils.auth_utils import get_password_hash, hash_api_key
from ..utils.error_utils import DatabaseError, ValidationError, NotFoundError

# Set up logging
logger = logging.getLogger("user_db")

# Connection pool
user_db_pool = None

def init_db_pool(
    host: str,
    user: str,
    password: str,
    database: str,
    pool_size: int = 5
) -> bool:
    """
    Initialize database connection pool for user operations.
    
    Args:
        host: Database host
        user: Database user
        password: Database password
        database: Database name
        pool_size: Connection pool size
        
    Returns:
        True if successful, False otherwise
    """
    global user_db_pool
    
    try:
        user_db_pool = pooling.MySQLConnectionPool(
            pool_name="user_db_pool",
            pool_size=pool_size,
            host=host,
            user=user,
            password=password,
            database=database
        )
        logger.info("User database connection pool initialized successfully")
        return True
    except mysql.connector.Error as e:
        logger.error(f"Failed to initialize user database connection pool: {e}")
        return False

def get_connection():
    """Get a connection from the pool."""
    if not user_db_pool:
        raise DatabaseError("Database connection pool not initialized")
    
    return user_db_pool.get_connection()

# User Operations

def create_user(
    username: str,
    email: str,
    password: str,
    full_name: str = None,
    roles: List[str] = None
) -> Dict[str, Any]:
    """
    Create a new user.
    
    Args:
        username: Username
        email: Email address
        password: Password (will be hashed)
        full_name: User's full name
        roles: List of role names to assign
        
    Returns:
        Dictionary with user information
        
    Raises:
        ValidationError: If username or email already exists
        DatabaseError: If database operation fails
    """
    # Hash the password
    password_hash = get_password_hash(password)
    
    # Default roles
    if roles is None:
        roles = ["user"]
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if username or email already exists
        cursor.execute(
            "SELECT id FROM users WHERE username = %s OR email = %s",
            (username, email)
        )
        
        if cursor.fetchone():
            raise ValidationError("Username or email already exists")
        
        # Insert user
        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash, full_name, is_active, is_verified)
            VALUES (%s, %s, %s, %s, TRUE, FALSE)
            """,
            (username, email, password_hash, full_name)
        )
        
        user_id = cursor.lastrowid
        
        # Assign roles
        for role_name in roles:
            cursor.execute(
                """
                INSERT INTO user_roles (user_id, role_id)
                SELECT %s, id FROM roles WHERE name = %s
                """,
                (user_id, role_name)
            )
        
        conn.commit()
        
        # Retrieve created user with roles and permissions
        user = get_user_by_id(user_id)
        
        return user
    
    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error creating user: {e}")
        raise DatabaseError(f"Failed to create user: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a user with username and password.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        Dictionary with user information if authentication succeeds
        
    Raises:
        ValidationError: If authentication fails
        DatabaseError: If database operation fails
    """
    password_hash = get_password_hash(password)
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get user with matching username and password
        cursor.execute(
            """
            SELECT id, username, email, full_name, is_active, is_verified
            FROM users
            WHERE username = %s AND password_hash = %s
            """,
            (username, password_hash)
        )
        
        user = cursor.fetchone()
        
        if not user:
            raise ValidationError("Invalid username or password")
        
        if not user["is_active"]:
            raise ValidationError("User account is not active")
        
        # Update last login time
        cursor.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
            (user["id"],)
        )
        
        conn.commit()
        
        # Get user roles and permissions
        user_with_roles = get_user_by_id(user["id"])
        
        return user_with_roles
    
    except mysql.connector.Error as e:
        logger.error(f"Database error authenticating user: {e}")
        raise DatabaseError(f"Failed to authenticate user: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_user_by_id(user_id: int) -> Dict[str, Any]:
    """
    Get user information by ID, including roles and permissions.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary with user information
        
    Raises:
        NotFoundError: If user not found
        DatabaseError: If database operation fails
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get user
        cursor.execute(
            """
            SELECT id, username, email, full_name, is_active, is_verified,
                   created_at, updated_at, last_login
            FROM users
            WHERE id = %s
            """,
            (user_id,)
        )
        
        user = cursor.fetchone()
        
        if not user:
            raise NotFoundError(f"User with ID {user_id} not found")
        
        # Get user roles
        cursor.execute(
            """
            SELECT r.name, r.description
            FROM roles r
            JOIN user_roles ur ON r.id = ur.role_id
            WHERE ur.user_id = %s
            """,
            (user_id,)
        )
        
        roles = [role["name"] for role in cursor.fetchall()]
        
        # Get user permissions based on roles
        cursor.execute(
            """
            SELECT DISTINCT p.name as permission_name
            FROM permissions p
            JOIN role_permissions rp ON p.id = rp.permission_id
            JOIN user_roles ur ON rp.role_id = ur.role_id
            WHERE ur.user_id = %s
            """,
            (user_id,)
        )
        
        permissions = [perm["permission_name"] for perm in cursor.fetchall()]
        
        # Combine data
        user_data = {
            **user,
            "roles": roles,
            "permissions": permissions
        }
        
        return user_data
    
    except mysql.connector.Error as e:
        logger.error(f"Database error getting user: {e}")
        raise DatabaseError(f"Failed to get user: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def update_user(
    user_id: int,
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update user information.
    
    Args:
        user_id: User ID
        data: Dictionary with fields to update
        
    Returns:
        Updated user information
        
    Raises:
        NotFoundError: If user not found
        ValidationError: If validation fails
        DatabaseError: If database operation fails
    """
    valid_fields = {"email", "full_name", "is_active", "is_verified", "password"}
    update_fields = {k: v for k, v in data.items() if k in valid_fields}
    
    # Special handling for password
    if "password" in update_fields:
        update_fields["password_hash"] = get_password_hash(update_fields.pop("password"))
    
    if not update_fields:
        raise ValidationError("No valid fields to update")
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if user exists
        cursor.execute(
            "SELECT id FROM users WHERE id = %s",
            (user_id,)
        )
        
        if not cursor.fetchone():
            raise NotFoundError(f"User with ID {user_id} not found")
        
        # Update user
        set_clause = ", ".join(f"{field} = %s" for field in update_fields)
        values = list(update_fields.values())
        values.append(user_id)
        
        cursor.execute(
            f"UPDATE users SET {set_clause} WHERE id = %s",
            values
        )
        
        conn.commit()
        
        # Get updated user
        updated_user = get_user_by_id(user_id)
        
        return updated_user
    
    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error updating user: {e}")
        raise DatabaseError(f"Failed to update user: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def update_user_roles(
    user_id: int,
    roles: List[str]
) -> Dict[str, Any]:
    """
    Update user roles.
    
    Args:
        user_id: User ID
        roles: List of role names
        
    Returns:
        Updated user information
        
    Raises:
        NotFoundError: If user not found
        ValidationError: If roles are invalid
        DatabaseError: If database operation fails
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if user exists
        cursor.execute(
            "SELECT id FROM users WHERE id = %s",
            (user_id,)
        )
        
        if not cursor.fetchone():
            raise NotFoundError(f"User with ID {user_id} not found")
        
        # Check if roles exist
        placeholders = ", ".join(["%s"] * len(roles))
        cursor.execute(
            f"SELECT name FROM roles WHERE name IN ({placeholders})",
            roles
        )
        
        valid_roles = [row["name"] for row in cursor.fetchall()]
        
        if len(valid_roles) != len(roles):
            invalid_roles = set(roles) - set(valid_roles)
            raise ValidationError(f"Invalid roles: {', '.join(invalid_roles)}")
        
        # Remove all current roles
        cursor.execute(
            "DELETE FROM user_roles WHERE user_id = %s",
            (user_id,)
        )
        
        # Add new roles
        for role_name in roles:
            cursor.execute(
                """
                INSERT INTO user_roles (user_id, role_id)
                SELECT %s, id FROM roles WHERE name = %s
                """,
                (user_id, role_name)
            )
        
        conn.commit()
        
        # Get updated user
        updated_user = get_user_by_id(user_id)
        
        return updated_user
    
    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error updating user roles: {e}")
        raise DatabaseError(f"Failed to update user roles: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# API Key Operations

def create_api_key(
    user_id: int,
    key_name: str,
    api_key: str,
    expires_at: datetime = None,
    daily_rate_limit: int = None
) -> Dict[str, Any]:
    """
    Create a new API key for a user.
    
    Args:
        user_id: User ID
        key_name: Name for the API key
        api_key: Generated API key
        expires_at: Expiration date
        daily_rate_limit: Daily rate limit
        
    Returns:
        Dictionary with API key information
        
    Raises:
        NotFoundError: If user not found
        ValidationError: If validation fails
        DatabaseError: If database operation fails
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if user exists
        cursor.execute(
            "SELECT id FROM users WHERE id = %s",
            (user_id,)
        )
        
        if not cursor.fetchone():
            raise NotFoundError(f"User with ID {user_id} not found")
        
        # Check if key name already exists for this user
        cursor.execute(
            "SELECT id FROM api_keys WHERE user_id = %s AND key_name = %s",
            (user_id, key_name)
        )
        
        if cursor.fetchone():
            raise ValidationError(f"API key name '{key_name}' already exists for this user")
        
        # Hash the API key
        key_hash = hash_api_key(api_key)
        
        # Insert API key
        cursor.execute(
            """
            INSERT INTO api_keys (user_id, key_hash, key_name, expires_at, daily_rate_limit)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, key_hash, key_name, expires_at, daily_rate_limit)
        )
        
        api_key_id = cursor.lastrowid
        
        conn.commit()
        
        # Return API key information
        return {
            "id": api_key_id,
            "user_id": user_id,
            "key_name": key_name,
            "api_key": api_key,  # Return the plain API key (only time it's available)
            "expires_at": expires_at,
            "daily_rate_limit": daily_rate_limit,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error creating API key: {e}")
        raise DatabaseError(f"Failed to create API key: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def verify_api_key(api_key: str) -> Dict[str, Any]:
    """
    Verify an API key and get associated user information.
    
    Args:
        api_key: API key to verify
        
    Returns:
        Dictionary with API key and associated user information
        
    Raises:
        ValidationError: If API key is invalid
        DatabaseError: If database operation fails
    """
    # Hash the API key
    key_hash = hash_api_key(api_key)
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get API key and user information
        cursor.execute(
            """
            SELECT k.id, k.user_id, k.key_name, k.expires_at, k.daily_rate_limit,
                   k.last_used_at, k.is_active,
                   u.username, u.email, u.is_active as user_is_active
            FROM api_keys k
            JOIN users u ON k.user_id = u.id
            WHERE k.key_hash = %s
            """,
            (key_hash,)
        )
        
        key_info = cursor.fetchone()
        
        if not key_info:
            raise ValidationError("Invalid API key")
        
        if not key_info["is_active"]:
            raise ValidationError("API key is not active")
        
        if not key_info["user_is_active"]:
            raise ValidationError("User account is not active")
        
        if key_info["expires_at"] and key_info["expires_at"] < datetime.now():
            raise ValidationError("API key has expired")
        
        # Update last used time
        cursor.execute(
            "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE id = %s",
            (key_info["id"],)
        )
        
        # Get user permissions
        cursor.execute(
            """
            SELECT DISTINCT p.name as permission_name
            FROM permissions p
            JOIN api_key_permissions akp ON p.id = akp.permission_id
            WHERE akp.api_key_id = %s
            UNION
            SELECT DISTINCT p.name as permission_name
            FROM permissions p
            JOIN role_permissions rp ON p.id = rp.permission_id
            JOIN user_roles ur ON rp.role_id = ur.role_id
            WHERE ur.user_id = %s
            """,
            (key_info["id"], key_info["user_id"])
        )
        
        permissions = [perm["permission_name"] for perm in cursor.fetchall()]
        
        # Get user roles
        cursor.execute(
            """
            SELECT r.name
            FROM roles r
            JOIN user_roles ur ON r.id = ur.role_id
            WHERE ur.user_id = %s
            """,
            (key_info["user_id"],)
        )
        
        roles = [role["name"] for role in cursor.fetchall()]
        
        conn.commit()
        
        # Combine data
        api_key_data = {
            **key_info,
            "permissions": permissions,
            "roles": roles
        }
        
        return api_key_data
    
    except mysql.connector.Error as e:
        logger.error(f"Database error verifying API key: {e}")
        raise DatabaseError(f"Failed to verify API key: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_user_api_keys(user_id: int) -> List[Dict[str, Any]]:
    """
    Get all API keys for a user.
    
    Args:
        user_id: User ID
        
    Returns:
        List of API key information dictionaries
        
    Raises:
        NotFoundError: If user not found
        DatabaseError: If database operation fails
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if user exists
        cursor.execute(
            "SELECT id FROM users WHERE id = %s",
            (user_id,)
        )
        
        if not cursor.fetchone():
            raise NotFoundError(f"User with ID {user_id} not found")
        
        # Get API keys
        cursor.execute(
            """
            SELECT id, key_name, created_at, expires_at, last_used_at,
                   is_active, daily_rate_limit
            FROM api_keys
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        )
        
        api_keys = cursor.fetchall()
        
        return api_keys
    
    except mysql.connector.Error as e:
        logger.error(f"Database error getting API keys: {e}")
        raise DatabaseError(f"Failed to get API keys: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def revoke_api_key(api_key_id: int, user_id: int = None) -> bool:
    """
    Revoke an API key.
    
    Args:
        api_key_id: API key ID
        user_id: User ID (for permission check)
        
    Returns:
        True if successful
        
    Raises:
        NotFoundError: If API key not found
        ValidationError: If user doesn't own the API key
        DatabaseError: If database operation fails
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if API key exists
        cursor.execute(
            "SELECT user_id FROM api_keys WHERE id = %s",
            (api_key_id,)
        )
        
        api_key = cursor.fetchone()
        
        if not api_key:
            raise NotFoundError(f"API key with ID {api_key_id} not found")
        
        # Check if user owns the API key
        if user_id and api_key["user_id"] != user_id:
            raise ValidationError("You don't have permission to revoke this API key")
        
        # Revoke API key
        cursor.execute(
            "UPDATE api_keys SET is_active = FALSE WHERE id = %s",
            (api_key_id,)
        )
        
        conn.commit()
        
        return True
    
    except mysql.connector.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error revoking API key: {e}")
        raise DatabaseError(f"Failed to revoke API key: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close() 