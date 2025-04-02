import pytest
import fakeredis
from app import create_app # Assuming your Flask app factory is here
from celery_app import celery_app # Import celery instance
from unittest.mock import patch, MagicMock
from flask import Flask, g
from functools import wraps # Added for wraps
# from services.config import TestingConfig # REMOVED import
import os

# Removed global mock helpers - rely on fixture again
# def mock_pass_through_decorator(f): ...
# def mock_require_auth_factory_global(*args, **kwargs): ...

# Set environment for testing BEFORE creating app
os.environ['FLASK_ENV'] = 'testing'

@pytest.fixture(scope='session')
def app():
    """Session-wide test `Flask` application."""
    
    # Removed the global patch context manager
    # with patch('services.api.middleware.auth_middleware.require_auth', mock_require_auth_factory_global):
        
    _app = create_app() 
    _app.config['TESTING'] = True
    _app.config['WTF_CSRF_ENABLED'] = False 
    
    # --- Configure Celery for Eager Execution --- 
    celery_app.conf.update(task_always_eager=True)

    # --- Configure Fake Redis and add to app --- 
    _app.redis_client = fakeredis.FakeStrictRedis(decode_responses=True)

    # --- Add Mock API Gateway client to app for health check --- 
    mock_gateway = MagicMock(name="MockApiGateway")
    mock_gateway.health.return_value = {
        "status": "ok",
        "services": {
            "llm": {"status": "healthy"}, 
            "vector": {"status": "healthy"}, 
            "db": {"status": "healthy"}
        }
    }
    _app.api_gateway = mock_gateway

    # Establish an application context before running tests
    ctx = _app.app_context()
    ctx.push()

    yield _app
    
    # --- Teardown --- 
    celery_app.conf.update(task_always_eager=False) 
    ctx.pop()

# --- Restore mock_auth fixture --- 
@pytest.fixture
def mock_auth():
    """Fixture to mock the @require_auth decorator factory.

    Returns a function that, when called, patches the decorator factory.
    The patch replaces the factory with one that returns a pass-through decorator
    which adds specified user info to flask.g.

    Example usage in a test:
        def test_protected_route(client, mock_auth):
            auth_patcher = mock_auth(user_id="test-user", roles=["admin"])
            with auth_patcher: # Enter the patch context
                response = client.get('/protected-route')
                assert response.status_code == 200
    """

    def _patch_require_auth_factory(user_id="test-user-id", roles=None, permissions=None):
        if roles is None:
            roles = ["user"]
        if permissions is None:
            permissions = []

        mock_user_data = {
            'id': user_id,
            'username': f"user_{user_id}", # Synthesize a username
            'roles': roles,
            'permissions': permissions,
            'auth_type': 'mocked' # Indicate the auth source
        }

        # Define the decorator that the mock factory will return
        def mock_decorator_inner(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                try:
                    g.user = mock_user_data # Inject user into g
                except RuntimeError:
                    print("Warning: flask.g not available when setting mock user. Ensure request context.")
                    pass 
                return f(*args, **kwargs)
            return decorated_function

        # Define the mock factory function that replaces the original require_auth
        def mock_factory(*factory_args, **factory_kwargs):
            # Return the simple decorator, ignoring original factory args
            return mock_decorator_inner

        # Patch the original factory where it's defined
        patcher = patch('services.api.middleware.auth_middleware.require_auth', mock_factory)
        return patcher # Return the patcher object to be used in a 'with' statement

    # Return the function that creates the patcher
    return _patch_require_auth_factory

@pytest.fixture()
def client(app: Flask):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app: Flask):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()

@pytest.fixture(scope='function')
def redis_client(app):
    """Provides the app's configured (fake) Redis client and clears it before each test."""
    client = app.redis_client 
    assert isinstance(client, fakeredis.FakeStrictRedis), "App redis_client not configured with FakeRedis!"
    client.flushall()
    yield client
    client.flushall()

# Remove the old fake_redis fixture as it's replaced by redis_client
# @pytest.fixture(scope='function') 
# def fake_redis(app):
#     ...

@pytest.fixture(scope='session')
def configured_celery_app():
    """Returns the configured Celery app instance."""
    return celery_app

# --- Mocking Fixtures --- #

# Removed old mock_auth fixture
# @pytest.fixture
# def mock_auth():
#    ...

# Removed mock_redis fixture as redis_client provides a functional fake
# @pytest.fixture
# def mock_redis(mocker):
#    ...

# Removed unused celery_config fixture
# @pytest.fixture(scope='session')
# def celery_config():
#    ...

# Keep other fixtures like celery_app if they are used

# Note: The celery_app fixture depends on your app structure.
# If Celery is initialized within create_app, you might need to adapt.
# Example assumes a standalone celery_app object might be accessible or needed.
# If your tasks use app context, ensure the test app context is active.

# @pytest.fixture(scope='session')
# def celery_app(app): # Assuming celery app is tied to flask app
#     app.config.update(CELERY_BROKER_URL='memory://', CELERY_RESULT_BACKEND='rpc://')
#     celery = create_celery(app) # Your celery factory
#     celery.conf.task_always_eager = True
#     return celery

# Add other shared fixtures below as needed, e.g., mock services 