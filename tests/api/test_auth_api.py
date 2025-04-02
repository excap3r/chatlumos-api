import pytest
from flask import Flask, jsonify, g
from unittest.mock import patch, MagicMock, ANY
import uuid
import time
from functools import wraps
# Assume these imports are correct based on project structure
from services.user_service import register_new_user, login_user, get_keys_for_user, add_api_key, remove_api_key 
from datetime import datetime
from typing import NamedTuple, Optional, List, Any
from services.db.exceptions import UserAlreadyExistsError, InvalidCredentialsError
from unittest.mock import Mock
User = Mock()
APIKeyData = Mock() # Placeholder, adjust if real class is available
APIKeyDataWithSecret = Mock() # Placeholder

# Assuming client fixture and mock_auth fixture are available from conftest.py
API_BASE_URL = "/api/v1/auth"

# --- Registration Tests --- #

@patch('services.api.auth_routes.register_new_user') 
def test_register_success(mock_register_service, client):
    """Test successful user registration."""
    mock_user_data = {'username': 'newuser', 'email': 'new@example.com', 'password': 'ValidPassword123!'}
    mock_user_id = str(uuid.uuid4())
    mock_register_service.return_value = mock_user_id
    mock_token = "mock.register.token"
    with patch('services.api.auth_routes.create_token', return_value=mock_token) as mock_create_token:
        response = client.post(f'{API_BASE_URL}/register', json=mock_user_data)

    assert response.status_code == 201
    assert 'message' in response.json
    assert response.json['message'] == "User registered successfully"
    assert response.json['user_id'] == mock_user_id
    assert response.json['token'] == mock_token
    mock_register_service.assert_called_once_with(
        mock_user_data['username'], 
        mock_user_data['email'], 
        mock_user_data['password']
    )
    mock_create_token.assert_called_once_with(user_id=mock_user_id, username=mock_user_data['username'])

@patch('services.api.auth_routes.register_new_user')
def test_register_duplicate(mock_register_service, client):
    """Test registration fails for duplicate username/email."""
    mock_user_data = {'username': 'existinguser', 'email': 'existing@example.com', 'password': 'ValidPassword123!'}
    mock_register_service.side_effect = UserAlreadyExistsError("User or email already exists")
    response = client.post(f'{API_BASE_URL}/register', json=mock_user_data)

    assert response.status_code == 409
    assert 'error' in response.json
    assert "User or email already exists" in response.json['error']
    mock_register_service.assert_called_once_with(
        mock_user_data['username'], 
        mock_user_data['email'], 
        mock_user_data['password']
    )

def test_register_invalid_password(client):
    """Test registration fails with an invalid password."""
    mock_user_data = {'username': 'userpass', 'email': 'user@pass.com', 'password': 'weak'}
    response = client.post(f'{API_BASE_URL}/register', json=mock_user_data)
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert isinstance(json_data['error'], str)
    assert "Password must be at least 8 characters long" in json_data['error']

def test_register_missing_fields(client):
    """Test registration fails with missing fields."""
    response = client.post(f'{API_BASE_URL}/register', json={'username': 'onlyuser'})
    assert response.status_code == 400
    assert 'error' in response.json
    # Assuming Pydantic validation errors might be returned differently now
    # Check if the error string mentions the missing fields
    assert 'email' in response.json['error']
    assert 'password' in response.json['error']

# --- Login Tests --- #

@patch('services.api.auth_routes.login_user')
def test_login_success(mock_login_service, client):
    """Test successful user login."""
    login_data = {'username': 'testuser', 'password': 'password123'}
    mock_user_info = {
        'id': 'user-id-123',
        'username': 'testuser',
        'roles': ['user'],
        'permissions': []
    }
    mock_login_service.return_value = mock_user_info
    mock_access_token = "mock.jwt.access.token"
    mock_refresh_token = "mock.jwt.refresh.token"
    with patch('services.api.auth_routes.create_token') as mock_create_token:
        def create_token_side_effect(*args, **kwargs):
            if kwargs.get('type') == 'refresh':
                return mock_refresh_token
            return mock_access_token
        mock_create_token.side_effect = create_token_side_effect
        response = client.post(f'{API_BASE_URL}/login', json=login_data)

    assert response.status_code == 200
    assert 'access_token' in response.json
    assert response.json['access_token'] == mock_access_token
    assert 'refresh_token' in response.json
    assert response.json['refresh_token'] == mock_refresh_token
    assert 'user' in response.json
    assert response.json['user']['username'] == login_data['username']
    assert response.json['user']['id'] == mock_user_info['id']
    mock_login_service.assert_called_once_with(login_data['username'], login_data['password'])
    assert mock_create_token.call_count == 2
    mock_create_token.assert_any_call(
        user_id=mock_user_info['id'],
        username=mock_user_info['username'],
        type='access',
        roles=ANY, # Use ANY for flexible matching if needed
        permissions=ANY
    )
    mock_create_token.assert_any_call(
        user_id=mock_user_info['id'],
        username=mock_user_info['username'],
        type='refresh'
    )

@patch('services.api.auth_routes.login_user')
def test_login_invalid_credentials(mock_login_service, client):
    """Test login fails with invalid credentials."""
    login_data = {'username': 'testuser', 'password': 'wrongpassword'}
    mock_login_service.side_effect = InvalidCredentialsError("Invalid credentials provided")
    response = client.post(f'{API_BASE_URL}/login', json=login_data)

    assert response.status_code == 401
    assert 'error' in response.json
    assert response.json['error'] == "Invalid username or password"
    mock_login_service.assert_called_once_with(login_data['username'], login_data['password'])

def test_login_missing_fields(client):
    """Test login fails with missing fields."""
    response = client.post(f'{API_BASE_URL}/login', json={'username': 'testuser'})
    assert response.status_code == 400
    assert 'error' in response.json
    assert 'password' in response.json['error']

# --- API Key Tests --- #

@patch('services.api.auth_routes.get_keys_for_user') 
def test_get_api_keys_success(mock_get_keys_service, client, mock_auth):
    """Test successfully retrieving API keys for an authenticated user."""
    mock_user_id = "user_with_keys"
    # Simulate DB returning datetime objects
    now = datetime.utcnow()
    mock_db_keys = [
        {'key_id': "key1", 'user_id': mock_user_id, 'name': "Key One", 'created_at': now, 'last_used_at': None, 'is_active': True, 'key_prefix': 'pref1'},
        {'key_id': "key2", 'user_id': mock_user_id, 'name': "Key Two", 'created_at': now, 'last_used_at': now, 'is_active': True, 'key_prefix': 'pref2'}
    ]
    mock_get_keys_service.return_value = mock_db_keys

    with mock_auth(user_id=mock_user_id):
        headers = {'Authorization': 'Bearer dummy'} 
        response = client.get(f'{API_BASE_URL}/api-keys', headers=headers)

    assert response.status_code == 200
    json_response = response.get_json()
    assert isinstance(json_response, list)
    assert len(json_response) == len(mock_db_keys)
    assert json_response[0]['key_id'] == mock_db_keys[0]['key_id']
    assert json_response[0]['name'] == mock_db_keys[0]['name']
    # Check that datetimes were serialized
    assert isinstance(json_response[0]['created_at'], str)
    assert json_response[0]['last_used_at'] is None
    assert isinstance(json_response[1]['last_used_at'], str)
    assert 'api_key' not in json_response[0] 
    assert 'hashed_key' not in json_response[0]
    mock_get_keys_service.assert_called_once_with(mock_user_id)

@patch('services.api.auth_routes.add_api_key')
def test_create_api_key_success(mock_create_key_service, client, mock_auth):
    """Test successfully creating a new API key."""
    mock_user_id = "create-key-user"
    key_name = "My New Test Key"
    mock_key_id = "newkey-xyz"
    mock_full_key = "sk_testsecretkeyvaluegenerated" 
    mock_created_ts = time.time() # Using timestamp as returned by service
    mock_create_key_service.return_value = (mock_key_id, mock_full_key, mock_created_ts)

    with mock_auth(user_id=mock_user_id):
        headers = {'Authorization': 'Bearer dummy'}
        response = client.post(f'{API_BASE_URL}/api-keys', headers=headers, json={'name': key_name})

    assert response.status_code == 201 
    assert 'key_id' in response.json
    assert response.json['key_id'] == mock_key_id
    assert 'key_name' in response.json
    assert response.json['key_name'] == key_name
    assert 'api_key' in response.json 
    assert response.json['api_key'] == mock_full_key
    assert 'created_at_ts' in response.json
    # Compare timestamps with tolerance for float precision
    assert abs(response.json['created_at_ts'] - mock_created_ts) < 0.001 
    mock_create_key_service.assert_called_once_with(mock_user_id, key_name)

def test_create_api_key_bad_input(client, mock_auth):
    """Test API key creation fails with missing name."""
    with mock_auth(): 
        headers = {'Authorization': 'Bearer dummy'}
        response = client.post(f'{API_BASE_URL}/api-keys', headers=headers, json={})

    assert response.status_code == 400
    assert 'error' in response.json
    assert 'name' in response.json['error']

@patch('services.api.auth_routes.remove_api_key')
def test_delete_api_key_success(mock_remove_key_service, client, mock_auth):
    """Test successfully deleting an API key."""
    mock_user_id = "delete-key-user"
    key_id_to_delete = "key_to_go"
    mock_remove_key_service.return_value = True 

    with mock_auth(user_id=mock_user_id):
        headers = {'Authorization': 'Bearer dummy'} 
        response = client.delete(f'{API_BASE_URL}/api-keys/{key_id_to_delete}', headers=headers)

    assert response.status_code == 204 
    mock_remove_key_service.assert_called_once_with(mock_user_id, key_id_to_delete)

@patch('services.api.auth_routes.remove_api_key')
def test_delete_api_key_not_found(mock_remove_key_service, client, mock_auth):
    """Test deleting a non-existent or unauthorized API key."""
    mock_user_id = "delete-key-user-fail"
    non_existent_key_id = "key_does_not_exist"
    mock_remove_key_service.return_value = False 

    with mock_auth(user_id=mock_user_id):
        headers = {'Authorization': 'Bearer dummy'} 
        response = client.delete(f'{API_BASE_URL}/api-keys/{non_existent_key_id}', headers=headers)

    assert response.status_code == 404 
    assert 'error' in response.json
    assert "API key not found or you do not have permission" in response.json['error']
    mock_remove_key_service.assert_called_once_with(mock_user_id, non_existent_key_id)

# --- Token Validation Test --- #

def test_validate_token_success(client, mock_auth):
    """Test successfully validating a token using mock_auth fixture."""
    mock_user_info = {
        'id': str(uuid.uuid4()),
        'username': 'validateduser',
        'roles': ['user', 'reader'],
    }

    with mock_auth(user_id=mock_user_info['id'], username=mock_user_info['username'], roles=mock_user_info['roles']):
        headers = {'Authorization': 'Bearer dummy'}
        response = client.get(f'{API_BASE_URL}/validate', headers=headers)

    assert response.status_code == 200
    response_data = response.get_json()
    assert response_data['id'] == mock_user_info['id']
    assert response_data['username'] == mock_user_info['username']
    assert response_data['roles'] == mock_user_info['roles']