import pytest
import fakeredis
from app import create_app # Assuming your Flask app factory is here
from celery_app import celery_app # Import celery instance
from unittest.mock import patch, MagicMock
from flask import Flask, g, current_app
from functools import wraps # Added for wraps
# from services.config import TestingConfig # REMOVED import
import os
import uuid
import time
import logging
import json
from io import BytesIO
from celery import Celery
from services.utils.log_utils import setup_logger # For logger fixture
# Import the actual decorator to be wrapped/patched
from services.api.middleware.auth_middleware import require_auth

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
def mock_auth(mocker):
    """Provides a context manager to mock authentication for a test request."""

    # Modify the AuthContextManager to accept mocker as the first argument
    class AuthContextManager:
        def __init__(self, mocker_fixture, user_id=None, username=None, roles=None):
            self.mocker = mocker_fixture
            self.user_id = user_id or "test-user-id"
            self.username = username or "testuser"
            self.roles = roles or ["user"]
            self.mock_user_data = {
                "id": self.user_id,
                "username": self.username,
                "roles": self.roles,
            }
            self.patcher = None # Initialize patcher

        def __enter__(self):
            # --- REVISED STRATEGY: Patch the DECORATOR FUNCTION ITSELF ---
            @wraps(require_auth)
            def mock_decorator_factory(*factory_args, **factory_kwargs):
                def mock_decorator_inner(f):
                    @wraps(f)
                    def decorated_function(*args, **kwargs):
                        g.user = self.mock_user_data
                        g.auth_method = 'mocked'
                        current_app.logger.debug("Mock auth applied", user=g.user)
                        return f(*args, **kwargs)
                    return decorated_function
                return mock_decorator_inner

            # Use the passed-in mocker fixture
            self.patcher = self.mocker.patch(
                'services.api.middleware.auth_middleware.require_auth',
                new=mock_decorator_factory
            )
            self.patcher.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.patcher:
                self.patcher.stop()

    # The fixture now returns a *function* that creates an instance of the context manager,
    # passing the mocker fixture along.
    def _context_manager_factory(**kwargs):
        return AuthContextManager(mocker, **kwargs)

    return _context_manager_factory

@pytest.fixture()
def client(app: Flask):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def request_context(app: Flask):
    """Fixture to provide request context for each test."""
    with app.test_request_context():
        yield

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