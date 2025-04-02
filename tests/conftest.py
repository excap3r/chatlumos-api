import pytest
import fakeredis
from app import app as flask_app  # Import the Flask app instance from app.py
from celery_app import celery_app # Import celery instance

@pytest.fixture(scope='session')
def app():
    """Session-wide test Flask application configured for integration tests."""
    
    # Set Flask config for testing
    flask_app.config.update({
        "TESTING": True,
        # Add other test-specific configurations here if needed
    })

    # --- Configure Celery for Eager Execution --- 
    celery_app.conf.update(task_always_eager=True) 

    # --- Configure Fake Redis --- 
    # Create a single fake redis instance for the session
    # Note: decode_responses=True is important for matching real redis behavior
    fake_redis_instance = fakeredis.FakeStrictRedis(decode_responses=True)
    # Inject this instance into the Flask app so it's used consistently
    flask_app.redis_client = fake_redis_instance

    # If API Gateway client is used, consider mocking it
    # if flask_app.api_gateway:
    #     # Example: Replace with mock
    #     pass

    yield flask_app
    
    # --- Teardown --- 
    # Reset Celery config if necessary (though session scope might make this less critical)
    celery_app.conf.update(task_always_eager=False) 
    # No need to clear fake_redis_instance here if we manage it in a function-scoped fixture


@pytest.fixture()
def client(app):
    """A test client for the app."""
    return app.test_client()

# Modify fake_redis to use the instance from the app fixture
# and clear it for each test function
@pytest.fixture(scope='function')
def redis_client(app):
    """Provides the app's configured (fake) Redis client and clears it before each test."""
    # Get the instance configured in the session-scoped app fixture
    client = app.redis_client 
    assert isinstance(client, fakeredis.FakeStrictRedis), "App redis_client not configured with FakeRedis!"
    
    # Clear before test execution
    client.flushall()
    
    yield client
    
    # Optional: Clear after test execution (good practice)
    client.flushall()


# Remove the old fake_redis fixture as it's replaced by redis_client
# @pytest.fixture(scope='function') 
# def fake_redis(app):
#     ...

# Optional: Fixture for the celery app itself, if needed
@pytest.fixture(scope='session')
def configured_celery_app():
    """Returns the configured Celery app instance."""
    return celery_app

# --- Authentication Mocking Fixture --- #
@pytest.fixture
def mock_auth_user(mocker, app):
    """Fixture factory to mock flask.g.user for authenticated routes."""
    
    def _mock_user(user_id="test-user-123", roles=None, permissions=None, auth_method="JWT"):
        """Mocks flask.g.user with specified details."""
        if roles is None:
            roles = ["user"]
        if permissions is None:
            permissions = []
            
        mock_user_data = {
            'id': user_id,
            'username': f"user_{user_id}", # Generate a username
            'roles': roles,
            'permissions': permissions,
            'auth_type': auth_method
            # Add other fields if the auth middleware/decorators expect them
        }
        
        # Patch flask.g within the app context for the duration of the test
        # This requires the test to run within the app context, which fixtures like `client` provide.
        # We patch 'flask.g' which is the typical way to access it.
        # Using mocker.patch ensures it's automatically cleaned up.
        patcher = mocker.patch('flask.g', return_value=mocker.MagicMock())
        g_mock = patcher.start()
        g_mock.user = mock_user_data
        g_mock.auth_method = auth_method
        
        # Store the patcher to stop it later (pytest might handle this automatically with mocker, but explicit is safer)
        # If not using mocker fixture, would need to manually call patcher.stop()
        # pytest.addfinalizer(patcher.stop) # Let pytest handle cleanup if mocker isn't used explicitly
        
        return g_mock # Return the mocked g object if needed, or just rely on side effect

    return _mock_user # Return the inner function so tests can call it with params


# Add other shared fixtures below as needed, e.g., mock services 