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

# Add other shared fixtures below as needed, e.g., mock services 