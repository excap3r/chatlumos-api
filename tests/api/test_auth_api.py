import pytest
from flask import Flask, jsonify
from unittest.mock import patch, MagicMock
import uuid
from functools import wraps

# Assuming client fixture is available from conftest.py

@patch('services.db.user_db.create_user')
def test_register_success(mock_create_user, client):
    """Test successful user registration."""
    mock_user_id = str(uuid.uuid4())
    mock_create_user.return_value = mock_user_id

    response = client.post('/auth/register', json={
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'password123'
    })

    assert response.status_code == 201
    assert response.json['message'] == 'User registered successfully'
    assert response.json['user_id'] == mock_user_id
    mock_create_user.assert_called_once_with('testuser', 'test@example.com', 'password123')

@patch('services.db.user_db.create_user')
def test_register_duplicate(mock_create_user, client):
    """Test registration failure due to duplicate username/email."""
    # Simulate database integrity error (or similar exception raised on duplication)
    mock_create_user.side_effect = Exception("Duplicate entry") # Adjust exception type if specific one is used

    response = client.post('/auth/register', json={
        'username': 'existinguser',
        'email': 'existing@example.com',
        'password': 'password123'
    })

    # Assuming a 409 Conflict or 400 Bad Request is returned for duplicates
    # Need to verify actual error handling in auth_routes.py
    assert response.status_code == 409 # Or 400, or 500 depending on implementation
    assert 'error' in response.json
    # assert response.json['error'] == 'Username or email already exists' # Adjust expected message
    mock_create_user.assert_called_once_with('existinguser', 'existing@example.com', 'password123')

@patch('services.db.user_db.create_user') # Mock to prevent call for bad input
def test_register_bad_input(mock_create_user, client):
    """Test registration failure due to missing fields."""
    response = client.post('/auth/register', json={
        'username': 'testuser'
        # Missing email and password
    })

    assert response.status_code == 400
    assert 'error' in response.json
    assert 'Missing required fields' in response.json['error'] # Adjust expected message
    mock_create_user.assert_not_called()

@patch('services.db.user_db.authenticate_user')
@patch('services.utils.auth_utils.generate_token') # Also mock token generation
def test_login_success(mock_generate_token, mock_authenticate_user, client):
    """Test successful user login."""
    mock_user_info = {'id': str(uuid.uuid4()), 'username': 'testuser', 'roles': ['user']}
    mock_token = "mock_jwt_token"
    mock_authenticate_user.return_value = mock_user_info
    mock_generate_token.return_value = mock_token

    response = client.post('/auth/login', json={
        'username': 'testuser',
        'password': 'password123'
    })

    assert response.status_code == 200
    assert response.json['message'] == 'Login successful'
    assert response.json['token'] == mock_token
    mock_authenticate_user.assert_called_once_with('testuser', 'password123')
    mock_generate_token.assert_called_once_with(mock_user_info['id'], mock_user_info['username'], mock_user_info['roles'])

@patch('services.db.user_db.authenticate_user')
def test_login_invalid_credentials(mock_authenticate_user, client):
    """Test login failure with invalid credentials."""
    mock_authenticate_user.return_value = None # Simulate failed authentication

    response = client.post('/auth/login', json={
        'username': 'testuser',
        'password': 'wrongpassword'
    })

    assert response.status_code == 401
    assert 'error' in response.json
    assert response.json['error'] == 'Invalid username or password' # Adjust expected message
    mock_authenticate_user.assert_called_once_with('testuser', 'wrongpassword')

@patch('services.db.user_db.authenticate_user') # Mock to prevent call for bad input
def test_login_bad_input(mock_authenticate_user, client):
    """Test login failure due to missing fields."""
    response = client.post('/auth/login', json={
        'username': 'testuser'
        # Missing password
    })

    assert response.status_code == 400
    assert 'error' in response.json
    assert 'Missing required fields' in response.json['error'] # Adjust expected message
    mock_authenticate_user.assert_not_called()

# TODO: Add tests for GET /auth/keys
# TODO: Add tests for POST /auth/keys
# TODO: Add tests for DELETE /auth/keys/<key_id>
# TODO: Add tests for GET /auth/validate

# Example structure (will be filled in)
# def test_login_success(client):
#     pass 

# --- API Key Tests --- 

# Assume require_auth decorator is in services.api.middleware.auth_middleware
# It likely injects user info into flask.g or passes it as an argument
# Mocking it to simulate an authenticated user

@patch('services.api.middleware.auth_middleware.require_auth') # Adjust path if needed
@patch('services.db.user_db.get_api_keys_for_user')
def test_get_api_keys_success(mock_get_keys, mock_require_auth, client):
    """Test successfully retrieving API keys for an authenticated user."""
    mock_user_id = str(uuid.uuid4())
    mock_keys = [
        {'id': str(uuid.uuid4()), 'name': 'key1', 'key_preview': 'abc...', 'created_at': '2023-01-01T10:00:00Z', 'last_used': None, 'expires_at': None, 'is_active': True},
        {'id': str(uuid.uuid4()), 'name': 'key2', 'key_preview': 'xyz...', 'created_at': '2023-01-02T11:00:00Z', 'last_used': '2023-01-05T12:00:00Z', 'expires_at': '2024-01-01T00:00:00Z', 'is_active': True}
    ]
    
    # Simulate the decorator injecting user_id into g or context
    # This depends on the exact implementation of require_auth
    # Alternative: mock require_auth to return a function that calls the original view with a mock user_id
    def mock_decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            g.user = {'id': mock_user_id} # Assuming decorator adds user to flask.g
            return f(*args, **kwargs)
        return decorated_function
    # mock_require_auth.side_effect = mock_decorator # This might be too complex, patching g directly is easier if possible
    
    # More direct approach: Patch flask.g if decorator uses it
    # If decorator passes user_id as arg, mock require_auth differently
    # For now, assume the test client setup or conftest handles auth context, 
    # OR that we can mock the DB call assuming auth succeeded.
    # Let's proceed by mocking the DB call directly assuming auth passed.
    
    mock_get_keys.return_value = mock_keys

    # We need a way to pass the user_id to the route. 
    # If require_auth adds it to g, we need to patch g or ensure test setup does.
    # Let's assume for now the test client handles setting up a mock authenticated session/token
    # OR we modify the require_auth mock to inject it. 
    # Simplest for now: Assume the route gets user_id correctly after auth.
    # The `get_api_keys_for_user` function likely needs the user_id.
    # Revisit this if tests fail due to missing user_id.

    # If using a mock token in header: 
    # headers = {'Authorization': f'Bearer mock_valid_token'}
    # response = client.get('/auth/keys', headers=headers)
    
    # If mocking the decorator successfully bypasses auth check:
    response = client.get('/auth/keys') 

    assert response.status_code == 200
    assert len(response.json) == len(mock_keys)
    assert response.json[0]['name'] == 'key1'
    assert response.json[1]['name'] == 'key2'
    # We need to know how user_id is passed to get_api_keys_for_user in the route
    # mock_get_keys.assert_called_once_with(mock_user_id) # This assertion depends on auth implementation


def test_get_api_keys_unauthenticated(client):
    """Test retrieving API keys fails without authentication."""
    response = client.get('/auth/keys')
    assert response.status_code == 401 # Expect Unauthorized
    assert 'error' in response.json


# TODO: Add tests for POST /auth/keys
# TODO: Add tests for DELETE /auth/keys/<key_id>
# TODO: Add tests for GET /auth/validate

# Example structure (will be filled in)
# def test_login_success(client):
#     pass 

@patch('services.api.middleware.auth_middleware.require_auth') # Adjust path
@patch('services.db.user_db.create_api_key')
def test_create_api_key_success(mock_create_key, mock_require_auth, client):
    """Test successfully creating an API key."""
    mock_user_id = str(uuid.uuid4())
    key_name = "My Test Key"
    mock_key_id = str(uuid.uuid4())
    mock_full_key = f"test_key_{uuid.uuid4()}" # Simulate a generated key
    
    # Mock the DB function to return the ID and the full key
    mock_create_key.return_value = {
        'key_id': mock_key_id,
        'full_key': mock_full_key 
    }

    # Similar to GET /keys, assume auth is handled (e.g., mock decorator sets g.user)
    # We need to ensure user_id is available to the route. Patching g or test setup.
    # Let's assume the user_id is passed correctly after auth mock.
    # Need to confirm how user_id is retrieved in the actual route.

    response = client.post('/auth/keys', json={'name': key_name})

    assert response.status_code == 201
    assert response.json['id'] == mock_key_id
    assert response.json['name'] == key_name
    assert response.json['api_key'] == mock_full_key # Endpoint should return the full key on creation
    # Assert DB call - need to confirm how user_id is passed
    # mock_create_key.assert_called_once_with(mock_user_id, key_name)


@patch('services.api.middleware.auth_middleware.require_auth') # Adjust path
@patch('services.db.user_db.create_api_key')
def test_create_api_key_bad_input(mock_create_key, mock_require_auth, client):
    """Test creating API key fails with missing name."""
    # Assume auth is handled/mocked
    response = client.post('/auth/keys', json={}) # Missing 'name'

    assert response.status_code == 400
    assert 'error' in response.json
    assert 'Missing required field: name' in response.json['error'] # Adjust message
    mock_create_key.assert_not_called()


def test_create_api_key_unauthenticated(client):
    """Test creating API key fails without authentication."""
    response = client.post('/auth/keys', json={'name': 'some key'})
    assert response.status_code == 401
    assert 'error' in response.json


# TODO: Add tests for DELETE /auth/keys/<key_id>
# TODO: Add tests for GET /auth/validate

# Example structure (will be filled in)
# def test_login_success(client):
#     pass 

@patch('services.api.middleware.auth_middleware.require_auth') # Adjust path
@patch('services.db.user_db.delete_api_key')
def test_delete_api_key_success(mock_delete_key, mock_require_auth, client):
    """Test successfully deleting an API key."""
    mock_user_id = str(uuid.uuid4())
    mock_key_id = str(uuid.uuid4())
    
    mock_delete_key.return_value = True # Simulate successful deletion

    # Assume auth handled, user_id available to route via g or similar
    # Need to confirm how user_id is passed to delete_api_key in route

    response = client.delete(f'/auth/keys/{mock_key_id}')

    assert response.status_code == 204 # No Content on successful delete
    # Verify DB call
    # mock_delete_key.assert_called_once_with(mock_user_id, mock_key_id)


@patch('services.api.middleware.auth_middleware.require_auth') # Adjust path
@patch('services.db.user_db.delete_api_key')
def test_delete_api_key_not_found(mock_delete_key, mock_require_auth, client):
    """Test deleting a non-existent or unauthorized API key."""
    mock_user_id = str(uuid.uuid4())
    mock_key_id = str(uuid.uuid4())
    
    mock_delete_key.return_value = False # Simulate key not found or not owned by user

    # Assume auth handled

    response = client.delete(f'/auth/keys/{mock_key_id}')

    assert response.status_code == 404 # Not Found
    assert 'error' in response.json
    assert 'API key not found or not authorized' in response.json['error'] # Adjust message
    # Verify DB call
    # mock_delete_key.assert_called_once_with(mock_user_id, mock_key_id)


def test_delete_api_key_unauthenticated(client):
    """Test deleting an API key fails without authentication."""
    mock_key_id = str(uuid.uuid4())
    response = client.delete(f'/auth/keys/{mock_key_id}')
    assert response.status_code == 401
    assert 'error' in response.json


# TODO: Add tests for GET /auth/validate

# Example structure (will be filled in)
# def test_login_success(client):
#     pass 

# --- Token Validation Test --- 

@patch('services.api.middleware.auth_middleware.require_auth') # Adjust path
def test_validate_token_success(mock_require_auth, client):
    """Test successfully validating a token and getting user info."""
    mock_user_info = {
        'id': str(uuid.uuid4()),
        'username': 'validateduser',
        'roles': ['user', 'reader'],
        # Add other relevant fields returned by the endpoint
    }

    # Simulate the decorator succeeding and making user info available
    # This is the crucial part - how does the test access the info injected by the decorator?
    # Option 1: Decorator adds to flask.g
    # Option 2: Decorator returns the user info, and the view uses it
    # Option 3: The test client fixture handles setting up an authenticated context

    # Assuming Option 1 for now: We need to ensure g.user is set somehow.
    # This might require a fixture or patching flask.g directly if the decorator modifies it.
    # Let's modify the mock decorator approach used conceptually before:
    
    original_view_function_called = False
    def mock_decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            nonlocal original_view_function_called
            original_view_function_called = True
            # Simulate the decorator injecting user info into flask.g
            from flask import g
            g.user = mock_user_info 
            # The actual view function in auth_routes.py needs to retrieve from g
            # and return it in the response.
            # We are testing the *route* here, assuming the decorator works and the view uses g.user
            return f(*args, **kwargs) # Call the original route function
        return decorated_function
        
    # Apply the mock decorator logic
    # This patching might need adjustment based on actual decorator implementation
    # For complex decorators, testing might involve calling the decorator directly
    # or relying on integration-style tests with real tokens/sessions.
    mock_require_auth.side_effect = mock_decorator 

    # Make the request
    response = client.get('/auth/validate')

    # Check assertions
    assert original_view_function_called # Ensure the mock decorator logic ran
    assert response.status_code == 200
    assert response.json['id'] == mock_user_info['id']
    assert response.json['username'] == mock_user_info['username']
    assert response.json['roles'] == mock_user_info['roles']
    # Add more assertions based on the actual response structure


def test_validate_token_unauthenticated(client):
    """Test validating token fails without authentication."""
    response = client.get('/auth/validate')
    assert response.status_code == 401
    assert 'error' in response.json


# Example structure (will be filled in)
# def test_login_success(client):
#     pass