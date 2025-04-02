import pytest
import fakeredis
from app import create_app # Assuming your Flask app factory is here
# from celery_app import celery_app # Import celery instance - Let's rely on app.celery
from celery_app import celery_app # Corrected: Import from root
from unittest.mock import patch, MagicMock, Mock
from flask import Flask, g, current_app, has_request_context
from functools import wraps
import os
import uuid
import time
import logging
import json
from io import BytesIO
# from celery import Celery # Removed, use imported celery_app
from services.utils.log_utils import setup_logger # For logger fixture
# No longer need auth_middleware here for patching
# from services.api.middleware import auth_middleware
from datetime import datetime
# import threading # REMOVED threading import

# Import services to mock during app creation if needed
from services.vector_search.vector_search import VectorSearchService
from services.llm_service.llm_service import LLMService
from services.analytics.analytics_service import AnalyticsService
from services.analytics.webhooks.webhook_service import WebhookService
import redis # Import redis for patching from_url


# Use mocks for models if needed (prevents issues if models change)
User = Mock()
APIKeyData = Mock()
APIKeyDataWithSecret = Mock()

# Configure basic logging for fixture setup/teardown visibility
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Set environment for testing BEFORE creating app
os.environ['FLASK_ENV'] = 'testing'

# --- Mock Auth Fixture (Patches middleware checks) ---

# Targets for patching auth checks within the middleware
DECODE_TOKEN_TARGET = 'services.api.middleware.auth_middleware.decode_token'
VERIFY_API_KEY_TARGET = 'services.api.middleware.auth_middleware.verify_api_key'

@pytest.fixture
def mock_auth():
    """
    Context manager factory to mock authentication.
    It patches the token decoding and API key verification functions
    used by the auth middleware to return predefined user data.
    It also sets g.user for convenience in the test function itself.
    """
    class AuthContextManager:
        def __init__(self, user_id=None, username=None, roles=None, permissions=None, auth_type='jwt', api_key_id=None):
            self.user_id = user_id or str(uuid.uuid4())
            self.username = username or f"testuser-{self.user_id[:4]}"
            self.roles = roles if roles is not None else ['user']
            self.permissions = permissions if permissions is not None else []
            self.auth_type = auth_type # 'jwt' or 'api_key'
            self.api_key_id = api_key_id or (str(uuid.uuid4()) if auth_type == 'api_key' else None)

            # Data to be returned by patched functions or set on g
            self.mock_user_data = {
                'id': self.user_id,
                'username': self.username,
                'roles': self.roles,
                'permissions': self.permissions,
                'auth_type': self.auth_type, # Include auth_type here
            }
            # Add JWT specific field if applicable
            if self.auth_type == 'jwt':
                self.mock_user_data['jti'] = str(uuid.uuid4())
            # Add API Key specific fields if applicable
            elif self.auth_type == 'api_key':
                 self.mock_user_data['api_key_id'] = self.api_key_id
                 # Adapt structure returned by verify_api_key mock
                 self.mock_verify_api_key_return = {
                     'user_id': self.user_id,
                     'username': self.username,
                     'roles': self.roles,
                     'permissions': self.permissions,
                     'id': self.api_key_id # key id
                 }
            else:
                self.mock_verify_api_key_return = None # Ensure it's defined

            # Create patchers but don't start them yet
            self.decode_patcher = patch(DECODE_TOKEN_TARGET)
            self.verify_key_patcher = patch(VERIFY_API_KEY_TARGET)

        def __enter__(self):
            self.mock_decode_token = self.decode_patcher.start()
            self.mock_verify_api_key = self.verify_key_patcher.start()

            # Configure mocks based on desired auth_type
            if self.auth_type == 'jwt':
                self.mock_decode_token.return_value = {
                    'sub': self.user_id,
                    'username': self.username,
                    'roles': self.roles,
                    'permissions': self.permissions,
                    'jti': self.mock_user_data['jti']
                }
                self.mock_verify_api_key.return_value = None # API key check should fail if JWT is present
            elif self.auth_type == 'api_key':
                self.mock_decode_token.return_value = None # JWT check should fail
                self.mock_verify_api_key.return_value = self.mock_verify_api_key_return
            else: # Default to failing both if type is unclear
                self.mock_decode_token.return_value = None
                self.mock_verify_api_key.return_value = None

            # Set g.user for convenience in the test body AFTER middleware would have run
            # This part requires an active request context, which should exist
            # when the test client makes a request before this context manager is used.
            try:
                 if has_request_context():
                      g.user = self.mock_user_data
                      g.auth_method = self.auth_type
                      log.debug(f"(mock_auth __enter__): Patched auth & set g.user = {g.user}")
                 else:
                     # If no request context, middleware wouldn't run anyway,
                     # so just log. This might happen in unit tests not involving client.
                     log.debug("(mock_auth __enter__): No request context, patches active but g.user not set.")
            except Exception as e:
                 log.error(f"(mock_auth __enter__): Failed during setup: {e}", exc_info=True)
                 # Stop patchers to prevent issues if setup failed
                 self.decode_patcher.stop()
                 self.verify_key_patcher.stop()
                 raise RuntimeError(f"Failed to setup mock_auth: {e}") from e

            return self # Return the manager instance

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Stop the patchers
            self.decode_patcher.stop()
            self.verify_key_patcher.stop()

            # Clean up g if context still exists
            try:
                 if has_request_context():
                      g.user = None
                      g.auth_method = None
                      log.debug("(mock_auth __exit__): Stopped patches & cleaned up g.user")
                 else:
                     log.debug("(mock_auth __exit__): Stopped patches (no request context)")
            except Exception as e:
                 log.warning(f"(mock_auth __exit__): Error during cleanup: {e}", exc_info=False)

    # Return the factory function
    def _context_manager_factory(**kwargs):
        return AuthContextManager(**kwargs)
    return _context_manager_factory

# --- Core App Fixtures --- #

class TestingConfig:
    TESTING = True
    SECRET_KEY = 'test-secret-key'
    JWT_SECRET_KEY = 'test-jwt-secret-key-for-testing' # Added JWT specific key
    # Make sure DB_* vars are set for testing
    # **IMPORTANT**: Use a dedicated test database to avoid data loss!
    DB_DRIVER = os.getenv('TEST_DB_DRIVER', 'mysql+pymysql')
    DB_USER = os.getenv('TEST_DB_USER', 'test_db_user')
    DB_PASSWORD = os.getenv('TEST_DB_PASSWORD', 'test_db_password')
    DB_HOST = os.getenv('TEST_DB_HOST', '127.0.0.1') # Use 127.0.0.1 instead of localhost sometimes helps
    DB_PORT = os.getenv('TEST_DB_PORT', '3306')
    DB_NAME = os.getenv('TEST_DB_NAME', 'test_db_pdf_wisdom') # Dedicated test DB name

    # Check if core DB config is present, skip DB setup if not fully configured
    DB_CONFIGURED = all([DB_USER, DB_PASSWORD, DB_NAME, DB_HOST])
    if DB_CONFIGURED:
        SQLALCHEMY_DATABASE_URI = f"{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        log.info(f"TestingConfig: Database configured: {SQLALCHEMY_DATABASE_URI}")
    else:
        SQLALCHEMY_DATABASE_URI = None # Signal DB is not configured
        log.warning("TestingConfig: Database environment variables (TEST_DB_USER, TEST_DB_PASSWORD, TEST_DB_HOST, TEST_DB_NAME) not fully set. Database tests will be skipped.")

    # Configure Redis for testing (use fakeredis automatically via fixture below if URL present)
    REDIS_URL = os.getenv('TEST_REDIS_URL', 'redis://localhost:6379/1') # Use a different DB index (e.g., /1)
    log.info(f"TestingConfig: Redis URL set to: {REDIS_URL}")

    # Celery config for testing (eager execution)
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    CELERY_TASK_ALWAYS_EAGER = False # Disable eager execution for debugging hangs
    CELERY_TASK_EAGER_PROPAGATES = True # Exceptions propagate in tests

    # Other app configs needed for testing
    API_VERSION = 'v1'
    FRONTEND_URL = 'http://localhost:3000'
    LOG_LEVEL = 'DEBUG' # More verbose logging for tests
    REDIS_TASK_TTL_SECONDS = 3600

    # Add dummy API keys for services to prevent init errors if they check keys
    # Ensure these env vars are set in your test environment or CI
    OPENAI_API_KEY = os.getenv('TEST_OPENAI_API_KEY', 'test-openai-key-required')
    # PINECONE_API_KEY = os.getenv('TEST_PINECONE_API_KEY', 'test-pinecone-key-required') # Removed
    # PINECONE_ENVIRONMENT = os.getenv('TEST_PINECONE_ENVIRONMENT', 'gcp-starter') # Removed
    # PINECONE_INDEX_NAME = os.getenv('TEST_PINECONE_INDEX_NAME', 'test-pdf-wisdom-index') # Removed
    
    # Annoy config for testing (use a temp path)
    # Consider using pytest's tmp_path fixture dynamically if possible, 
    # but for simplicity in config class, using a fixed relative path for tests.
    ANNOY_INDEX_PATH = './test_data/test_vector_index.ann' 
    ANNOY_METRIC = 'angular'
    ANNOY_NUM_TREES = 10 # Use fewer trees for faster test builds
    VECTOR_SEARCH_DIMENSION = None # Let the service determine from model
    DEFAULT_TOP_K = 5
    
    # Add other API keys as needed
    if OPENAI_API_KEY == 'test-openai-key-required':
        log.warning("TestingConfig: TEST_OPENAI_API_KEY not set, using placeholder.")
    # if PINECONE_API_KEY == 'test-pinecone-key-required': # Removed check
    #     log.warning("TestingConfig: TEST_PINECONE_API_KEY not set, using placeholder.")


@pytest.fixture(scope='session')
def app():
    """Session-wide test Flask application with TestingConfig and mocked services."""
    log.info("--- Creating Test App with TestingConfig and Mocked Services --- ")
    # Patch services *before* create_app is called
    # Use strings for targets matching where they are imported/used in 'app.py' or service modules
    with patch('app.VectorSearchService', autospec=True) as mock_vss, \
         patch('app.LLMService', autospec=True) as mock_llms, \
         patch('app.AnalyticsService', autospec=True) as mock_as, \
         patch('app.WebhookService', autospec=True) as mock_ws, \
         patch('redis.from_url') as mock_redis_from_url: # Patch redis connection creator globally

        # Configure the mock redis client that from_url will return
        # Ensure this happens BEFORE create_app tries to connect to Redis
        mock_redis_instance = fakeredis.FakeStrictRedis(decode_responses=True)
        mock_redis_from_url.return_value = mock_redis_instance
        log.info("Patched redis.from_url to return FakeStrictRedis instance.")

        # Create app *after* patches are active, passing TestingConfig
        _app = create_app(config_object=TestingConfig)

        # Explicitly set the redis_client on the app instance AFTER creation
        # This ensures tests using app.redis_client get the fake one, overriding
        # any potential direct connection attempt during create_app.
        _app.redis_client = mock_redis_instance
        log.info("Explicitly set app.redis_client to mock redis instance.")

        # Add a mock api_gateway attribute for the health check
        _app.api_gateway = MagicMock() 
        log.info("Added mock api_gateway attribute to the app instance.")

        # Store mocks on the app instance for potential access in tests (optional)
        _app.mock_vector_search_service = mock_vss.return_value
        _app.mock_llm_service = mock_llms.return_value
        _app.mock_analytics_service = mock_as.return_value
        _app.mock_webhook_service = mock_ws.return_value
        _app.mock_redis_client = mock_redis_instance # Reference for redis_client fixture

        # Verification: Check if DB session factory was created
        if not _app.config.get('SQLALCHEMY_DATABASE_URI'):
             log.warning("App Fixture: DB not configured (SQLALCHEMY_DATABASE_URI is None in app.config). DB setup skipped.")
        elif not hasattr(_app, 'db_session_factory') or _app.db_session_factory is None:
            log.warning("App Fixture: db_session_factory not found on app even with DB URI configured. Check create_app DB init logic.")
        else:
            log.info("App Fixture: db_session_factory found on app instance.")

        yield _app
        log.info("--- Tearing Down Test App --- ")

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
    """Provides the app's *mocked* Redis client (FakeRedis) and clears it."""
    # Get the client that was explicitly set on the app instance in the app fixture
    client_instance = getattr(app, 'redis_client', None)

    if client_instance is None or not isinstance(client_instance, (fakeredis.FakeStrictRedis, fakeredis.FakeRedis)):
         # This indicates a problem with the app fixture setup
         pytest.fail(f"Mock Redis client (FakeRedis) not found or is wrong type on app instance. Found: {type(client_instance)}")
         return None # Should not be reached

    assert isinstance(client_instance, (fakeredis.FakeStrictRedis, fakeredis.FakeRedis)), \
           f"Redis client is not a FakeRedis instance! Type: {type(client_instance)}"

    log.debug("Using fakeredis client for test.")
    client_instance.flushall() # Clear before test
    yield client_instance
    client_instance.flushall() # Clear after test
    log.debug("Cleared fakeredis client after test.")


@pytest.fixture(scope='session')
def configured_celery_app(app): # Depend on app to ensure config is loaded
    """Returns the configured Celery app instance (forces eager tasks)."""
    log.info("Configuring Celery app for eager testing.")
    # Ensure Celery uses the test config from the app fixture
    celery_app.conf.update(app.config)
    # These might already be set by TestingConfig, but ensures they are applied
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    return celery_app

@pytest.fixture(scope='function')
def db_session(app):
    """Provides a SQLAlchemy session for a test, handling setup/teardown."""
    # Check config first before looking for the factory
    if not app.config.get('SQLALCHEMY_DATABASE_URI'):
        pytest.skip("DB not configured (SQLALCHEMY_DATABASE_URI is None), skipping test.")
        return None

    # Now check if the factory was successfully created in create_app
    if not hasattr(app, 'db_session_factory') or app.db_session_factory is None:
        log.error("db_session fixture: db_session_factory not found on app. Check DB initialization in create_app and TestingConfig.")
        pytest.skip("Database session factory could not be found, skipping test.")
        return None

    # If factory exists
    session = app.db_session_factory()
    log.debug("DB session provided to test.")
    try:
        yield session
    finally:
        # Ensure session is cleaned up even if test fails
        log.debug("Cleaning up DB session after test.")
        try:
            session.rollback() # Rollback any changes made during the test
            if hasattr(app, 'db_session_factory') and app.db_session_factory:
                 app.db_session_factory.remove() # Remove the scoped session
                 log.debug("DB session rolled back and removed after test.")
            else:
                 log.warning("db_session_factory not found on app during teardown, cannot remove session.")
        except Exception as e:
             log.error(f"Error during DB session cleanup: {e}", exc_info=True)

# Remove the old fake_redis fixture as it's replaced by redis_client
# @pytest.fixture(scope='function') 
# def fake_redis(app):
#     ...

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

# Define TestingConfig here or import from config.py if preferred
# class TestingConfig:
#     TESTING = True
#     SECRET_KEY = 'test-secret-key'
#     # Ensure DB vars are set, e.g., via env or defaults
#     DB_DRIVER = os.getenv('TEST_DB_DRIVER', 'mysql+pymysql') 
#     DB_USER = os.getenv('TEST_DB_USER', 'test_db_user') # Use distinct test user
#     DB_PASSWORD = os.getenv('TEST_DB_PASSWORD', 'test_db_password')
#     DB_HOST = os.getenv('TEST_DB_HOST', '127.0.0.1') # Often localhost or test container
#     DB_PORT = os.getenv('TEST_DB_PORT', '3306')
#     DB_NAME = os.getenv('TEST_DB_NAME', 'test_db_pdf_wisdom') # **Use a separate test DB**
#     SQLALCHEMY_DATABASE_URI = f"{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
#     
#     REDIS_URL = os.getenv('TEST_REDIS_URL', 'redis://localhost:6379/1') # Use test Redis DB
#     CELERY_BROKER_URL = REDIS_URL 
#     CELERY_RESULT_BACKEND = REDIS_URL
#     CELERY_TASK_ALWAYS_EAGER = True
#     CELERY_TASK_EAGER_PROPAGATES = True
#
# # Use mocks for models if needed
# User = Mock()
# APIKeyData = Mock()
# APIKeyDataWithSecret = Mock() 