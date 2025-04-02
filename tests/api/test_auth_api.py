import pytest
from flask import Flask, jsonify, g
from unittest.mock import patch, MagicMock
import uuid
from functools import wraps
from services.user_service import register_new_user, login_user # Assume user service import
from datetime import datetime
# Import custom exceptions needed for testing side effects
from services.db.exceptions import UserAlreadyExistsError, InvalidCredentialsError
# Import data models
from services.models.api_key import APIKeyData, APIKeyDataWithSecret

# Assuming client fixture and mock_auth fixture are available from conftest.py

# --- Removed Helper for Direct Patching --- 
# def mock_require_auth_factory(...): ...

@patch('services.user_service.register_new_user')
@patch('services.utils.auth_utils.create_token') # Patched at source
def test_register_success(mock_create_token_source, mock_register_service, client):
    """Test successful user registration."""
    username = "newuser"
    email = "new@example.com"
    password = "ValidPass123!"
    mock_user_id = str(uuid.uuid4())
    mock_token = "mock.register.token"

    mock_register_service.return_value = mock_user_id
    mock_create_token_source.return_value = mock_token

    response = client.post('/api/v1/auth/register', json={
        'username': username,
        'email': email,
        'password': password
    })

    assert response.status_code == 201
    data = response.json
    assert data['message'] == 'User registered successfully'
    assert data['user_id'] == mock_user_id
    assert data['token'] == mock_token
    mock_register_service.assert_called_once_with(username, email, password)
    mock_create_token_source.assert_called_once_with(user_id=mock_user_id, username=username)

@patch('services.user_service.register_new_user')
def test_register_duplicate(mock_register_service, client):
    """Test registration failure due to duplicate user/email."""
    username = "existinguser"
    email = "existing@example.com"
    password = "ValidPass123!"
    mock_register_service.side_effect = UserAlreadyExistsError("User already exists.")

    response = client.post('/api/v1/auth/register', json={
        'username': username,
        'email': email,
        'password': password
    })

    assert response.status_code == 409
    assert 'error' in response.json
    assert 'already exists' in response.json['error']
    mock_register_service.assert_called_once_with(username, email, password)

@patch('services.user_service.register_new_user')
def test_register_bad_input(mock_register_service, client):
    """Test registration failure due to invalid input (e.g., weak password)."""
    response = client.post('/api/v1/auth/register', json={
        'username': 'user', # Too short
        'email': 'bad-email', # Invalid format
        'password': 'weak' # Too short
    })

    assert response.status_code == 400
    assert 'error' in response.json
    assert 'Input validation failed' in response.json['error']
    mock_register_service.assert_not_called()

@patch('services.user_service.login_user')
@patch('services.utils.auth_utils.create_token') # Patched at source
def test_login_success(mock_create_token_source, mock_login_service, client):
    """Test successful login."""
    username = 'testuser'
    password = 'correctpassword'
    mock_user_data = {
        'id': str(uuid.uuid4()), 'username': username,
        'email': 'test@example.com', 'roles': ['user'], 'permissions': []
    }
    mock_login_service.return_value = mock_user_data

    mock_access_token = "mock.access.token"
    mock_refresh_token = "mock.refresh.token"
    # Simulate the two calls to create_token by the route
    mock_create_token_source.side_effect = [mock_access_token, mock_refresh_token]

    response = client.post('/api/v1/auth/login', json={
        'username': username, 'password': password
    })

    assert response.status_code == 200
    data = response.json
    assert data['access_token'] == mock_access_token
    assert data['refresh_token'] == mock_refresh_token
    assert data['user']['id'] == mock_user_data['id']

    mock_login_service.assert_called_once_with(username=username, password=password)
    # Check calls to create_token
    assert mock_create_token_source.call_count == 2
    mock_create_token_source.assert_any_call(
        user_id=mock_user_data['id'], username=mock_user_data['username'],
        type='access', roles=mock_user_data['roles'], permissions=mock_user_data['permissions']
    )
    mock_create_token_source.assert_any_call(
        user_id=mock_user_data['id'], username=mock_user_data['username'],
        type='refresh'
    )

@patch('services.user_service.login_user')
def test_login_invalid_credentials(mock_login_service, client):
    """Test login failure with invalid credentials."""
    username = 'testuser'
    password = 'wrongpassword'
    mock_login_service.side_effect = InvalidCredentialsError()
    response = client.post('/api/v1/auth/login', json={
        'username': username, 'password': password
    })
    assert response.status_code == 401
    assert 'Invalid username or password' in response.json['error']
    mock_login_service.assert_called_once_with(username=username, password=password)

@patch('services.user_service.login_user')
def test_login_bad_input(mock_login_service, client):
    """Test login failure due to missing input."""
    response = client.post('/api/v1/auth/login', json={
        'username': 'testuser' # Missing password
    })

    assert response.status_code == 400
    assert 'Input validation failed' in response.json['error']
    mock_login_service.assert_not_called()

# TODO: Add tests for GET /auth/keys
# TODO: Add tests for POST /auth/keys
# TODO: Add tests for DELETE /auth/keys/<key_id>
# TODO: Add tests for GET /auth/validate

# Example structure (will be filled in)
# def test_login_success(client):
#     pass 

# --- API Key Tests (Using mock_auth fixture) --- 

@patch('services.db.user_db.get_user_api_keys') 
def test_get_api_keys_success(mock_get_keys, client, mock_auth, mocker):
    """Test successfully retrieving API keys for an authenticated user."""
    mock_user_id = "user_with_keys"
    # Mock the database call
    mock_db_keys = [
        APIKeyData(key_id="key1", user_id=mock_user_id, name="Key One", created_at=datetime.utcnow(), last_used_at=None, is_active=True),
        APIKeyData(key_id="key2", user_id=mock_user_id, name="Key Two", created_at=datetime.utcnow(), last_used_at=datetime.utcnow(), is_active=True)
    ]
    mocker.patch('services.api.routes.auth.list_api_keys_for_user', return_value=mock_db_keys)

    # Use the mock_auth factory
    with mock_auth(user_id=mock_user_id):
        response = client.get('/api/v1/auth/api-keys')

    assert response.status_code == 200
    assert len(response.json) == 2
    assert response.json[0]['name'] == "Key One"
    assert 'key_prefix' in response.json[0] # Should only contain prefix
    assert 'full_key' not in response.json[0]

def test_get_api_keys_unauthenticated(client):
    """Test retrieving API keys fails without authentication."""
    response = client.get('/api/v1/auth/keys')
    assert response.status_code == 401 # Expect Unauthorized
    assert 'error' in response.json


# TODO: Add tests for POST /auth/keys
# TODO: Add tests for DELETE /auth/keys/<key_id>
# TODO: Add tests for GET /auth/validate

# Example structure (will be filled in)
# def test_login_success(client):
#     pass 

@patch('services.db.user_db.create_api_key') 
def test_create_api_key_success(mock_create_key, client, mock_auth, mocker):
    """Test successfully creating a new API key."""
    mock_user_id = "create-key-user"
    key_name = "My New Test Key"
    # Mock the DB function to return the key details including the full key
    mock_full_key = "testprefix_testsecretkeyvalue"
    mock_created_key = APIKeyDataWithSecret(
        key_id="newkey123",
        user_id=mock_user_id,
        name=key_name,
        created_at=datetime.utcnow(),
        last_used_at=None,
        is_active=True,
        key_prefix="testprefix",
        hashed_key="hashed_secret", # Not returned in API, but stored
        full_key=mock_full_key # Returned only on creation
    )
    mock_create_key.return_value = mock_created_key

    with mock_auth(user_id=mock_user_id):
        response = client.post('/api/v1/auth/api-keys', json={'name': key_name})

    assert response.status_code == 201
    assert response.json['name'] == key_name
    assert response.json['key_id'] == "newkey123"
    assert response.json['key_prefix'] == "testprefix"
    assert response.json['full_key'] == mock_full_key # Verify full key is returned
    mock_create_key.assert_called_once_with(user_id=mock_user_id, name=key_name)


def test_create_api_key_bad_input(client, mock_auth, mocker):
    """Test API key creation fails with missing name."""
    with mock_auth(): # Default user
        response = client.post('/api/v1/auth/api-keys', json={}) # Missing 'name'
    assert response.status_code == 400
    assert 'error' in response.json


def test_create_api_key_unauthenticated(client):
    """Test creating API key fails without authentication."""
    response = client.post('/api/v1/auth/keys', json={'name': 'some key'})
    assert response.status_code == 401
    assert 'error' in response.json


# TODO: Add tests for DELETE /auth/keys/<key_id>
# TODO: Add tests for GET /auth/validate

# Example structure (will be filled in)
# def test_login_success(client):
#     pass 

@patch('services.db.user_db.revoke_api_key') 
def test_delete_api_key_success(mock_revoke_key, client, mock_auth, mocker):
    """Test successfully deleting an API key."""
    mock_user_id = "delete-key-user"
    key_id_to_delete = "key_to_go"
    # Mock the DB function - Correct the target path
    mock_delete = mocker.patch('services.api.auth_routes.delete_api_key_for_user', return_value=True)

    with mock_auth(user_id=mock_user_id):
        response = client.delete(f'/api/v1/auth/api-keys/{key_id_to_delete}')

    assert response.status_code == 204 # No content on successful deletion
    # Verify the correct function was called with user_id and key_id
    mock_delete.assert_called_once_with(user_id=mock_user_id, key_id=key_id_to_delete)

@patch('services.db.user_db.revoke_api_key')
def test_delete_api_key_not_found(mock_revoke_key, client, mock_auth, mocker):
    """Test deleting a non-existent or unauthorized API key."""
    mock_user_id = "delete-key-user-fail"
    non_existent_key_id = "key_does_not_exist"
    # Mock the DB function to indicate failure - Correct the target path
    mock_delete = mocker.patch('services.api.auth_routes.delete_api_key_for_user', return_value=False)

    with mock_auth(user_id=mock_user_id):
        response = client.delete(f'/api/v1/auth/api-keys/{non_existent_key_id}')

    assert response.status_code == 404 # Not Found
    mock_delete.assert_called_once_with(user_id=mock_user_id, key_id=non_existent_key_id)


def test_delete_api_key_unauthenticated(client):
    """Test deleting API key fails without authentication."""
    key_id_to_delete = str(uuid.uuid4())
    response = client.delete(f'/api/v1/auth/keys/{key_id_to_delete}')
    assert response.status_code == 401
    assert 'error' in response.json


# TODO: Add tests for GET /auth/validate

# Example structure (will be filled in)
# def test_login_success(client):
#     pass 

# --- Token Validation Test (Using mock_auth fixture) --- 

def test_validate_token_success(client, mock_auth, mocker):
    """Test successfully validating a token using mock_auth fixture."""
    mock_user_info = {
        'id': str(uuid.uuid4()), 
        'username': 'validateduser',
        'roles': ['user', 'reader'],
        'auth_type': 'mocked'
    }
    
    with mock_auth(user_id=mock_user_info['id'], roles=mock_user_info['roles']):
        response = client.get('/api/v1/auth/validate')

    assert response.status_code == 200
    data = response.json
    assert data['id'] == mock_user_info['id']
    assert data['username'] == f"user_{mock_user_info['id']}" # Verify synthesized username
    assert data['roles'] == mock_user_info['roles']

def test_validate_token_unauthenticated(client):
    """Test validating token fails without authentication."""
    response = client.get('/api/v1/auth/validate')
    assert response.status_code == 401
    assert 'error' in response.json


# Example structure (will be filled in)
# def test_login_success(client):
#     pass