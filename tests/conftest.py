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

# --- Auto-clearing Redis fixture --- #

@pytest.fixture(scope='function', autouse=True)
def clear_redis(redis_client):
    """Automatically clears the Redis database after each test function."""
    yield # Run the test
    # Teardown: Flush the database
    try:
        log.debug("--- Clearing Redis DB after test ---")
        redis_client.flushdb()
    except Exception as e:
        log.error(f"Error flushing Redis DB after test: {e}", exc_info=True)

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
    CELERY_TASK_ALWAYS_EAGER = True # Re-enable eager execution
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

        # --- Clean up after session ---
        yield _app
        log.info("--- Tearing Down Test App ---")
        # Cleanup actions if necessary

@pytest.fixture()
def client(app: Flask):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def request_context(app: Flask):
    """Fixture to provide request context for each test."""
    with app.test_request_context() as context:
        yield context

@pytest.fixture
def runner(app: Flask):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()

# --- Header Fixture for Authenticated Requests --- #

@pytest.fixture
def auth_headers(app, mock_auth):
    """Generates JWT auth headers for a default test user."""
    from flask_jwt_extended import create_access_token

    # Use the mock_auth context manager to define the user context
    with mock_auth() as mocker:
        # Need an application context to create token
        with app.app_context(): # Ensure app context is active
            # The identity for the JWT should typically be the user ID string
            identity = mocker.user_id
            # The full user info is available via g.user thanks to mock_auth
            # identity_dict = {
            #     "id": mocker.user_id,
            #     "username": mocker.username,
            #     "roles": mocker.roles,
            #     "permissions": mocker.permissions
            # }
            access_token = create_access_token(identity=identity)
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        log.debug(f"Generated auth_headers for user_id: {mocker.user_id}")
        return headers

# --- Redis Fixtures --- #

@pytest.fixture(scope='function')
def redis_client(app):
    """Provides a (mocked) Redis client instance cleared for each function."""
    # The main app fixture already patches redis.from_url and sets app.redis_client
    # We just need to ensure it's clean for each test.
    client = app.redis_client
    if client is None:
        pytest.fail("Mock Redis client not found on app. Check app fixture setup.")

    # Clear the fake redis instance before each test
    client.flushdb()
    log.info("Flushed mock Redis database for new test.")
    yield client
    # Optional: Clear again after test if needed
    # client.flushdb()

# --- Celery Fixtures --- #

@pytest.fixture(scope='session')
def configured_celery_app(app): # Depend on app to ensure config is loaded
    """Provides the Celery app instance configured for testing (eager)."""
    # Import celery_app directly to avoid current_app dependency
    from celery_app import celery_app as celery_instance

    # Configure Celery for testing
    celery_instance.conf.update(
        task_always_eager=True,  # Tasks run synchronously
        task_eager_propagates=True,  # Exceptions propagate
        broker_url='memory://',  # In-memory broker
        backend='cache',  # In-memory backend
        result_backend='cache',  # In-memory results
        cache_backend='memory',  # In-memory cache
        worker_hijack_root_logger=False,  # Don't hijack root logger
        worker_log_color=False,  # No colors in logs
    )

    # Ensure the app has the celery instance
    app.celery = celery_instance

    return celery_instance

# --- Database Fixtures (Optional, depending on TestingConfig) --- #

@pytest.fixture(scope='function')
def db_session(app):
    """
    Provides a SQLAlchemy session with rollback for isolated tests.
    Requires TestingConfig.DB_CONFIGURED to be True.
    Uses the session factory attached directly to the app instance by create_app.
    """
    if not TestingConfig.DB_CONFIGURED:
        pytest.skip("Database tests skipped: Database not configured in TestingConfig.")

    # Ensure the session factory is available directly on the app object
    if not hasattr(app, 'db_session_factory') or not app.db_session_factory:
         pytest.fail("SQLAlchemy session factory not found on app.db_session_factory. Check create_app DB initialization.")

    session_factory = app.db_session_factory # Use the factory attached by create_app

    # Start a transaction
    session = session_factory()
    log.info("DB Session: Starting transaction for test.")
    # Optional: Setup schema or initial data if needed for all DB tests
    # You might need app.app_context() here if creating schema
    # with app.app_context():
    #     db.create_all() # If using Flask-SQLAlchemy db object

    yield session

    # Rollback the transaction after the test
    log.info("DB Session: Rolling back transaction after test.")
    session.rollback()
    session.close()
    # Optional: Tear down schema if created per-function
    # with app.app_context():
    #     db.drop_all()

# --- Logging Fixture --- #

@pytest.fixture(scope="function", autouse=True)
def setup_test_logging(request):
    """Sets up logging level for each test function."""
    test_name = request.node.name
    log.info(f"--- Starting test: {test_name} ---")

    # Get the root logger and set the desired level for this test
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG) # Set to DEBUG for tests
    log.debug(f"Set root logger level to DEBUG for {test_name}")

    yield

    # Restore original level after test
    root_logger.setLevel(original_level)
    log.debug(f"Restored root logger level to {logging.getLevelName(original_level)} after {test_name}")
    log.info(f"--- Finished test: {test_name} ---")

# --- Example Mock Dependency Fixture --- #

@pytest.fixture
def mock_dependencies(redis_client, mocker): # Example dependencies
    """Provides a dictionary of common mocked dependencies."""
    return {
        "redis_client": redis_client,
        "redis_pipeline": mocker.patch.object(redis_client, 'pipeline', return_value=MagicMock()),
        # Add other common mocks needed by tasks/services
        "llm_service": mocker.patch('services.llm_service.llm_service.LLMService'),
        "vector_search_service": mocker.patch('services.vector_search.vector_search.VectorSearchService'),
    }

# --- Sample Data Fixtures --- #

@pytest.fixture
def sample_pdf_path():
    """Provides the path to a sample PDF file for testing uploads."""
    # Create a dummy PDF file or use a small existing one
    # Ensure this path is relative to the project root or accessible
    pdf_dir = "./test_data"
    pdf_path = os.path.join(pdf_dir, "dummy_document.pdf")
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    if not os.path.exists(pdf_path):
        # Create a minimal valid PDF (or copy one)
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            c = canvas.Canvas(pdf_path, pagesize=letter)
            c.drawString(100, 750, "This is a dummy PDF for testing.")
            c.save()
            log.info(f"Created dummy PDF for testing at {pdf_path}")
        except ImportError:
            # Fallback: create an empty file, might not work for all tests
            with open(pdf_path, 'wb') as f:
                f.write(b'%') # Minimal PDF needs at least %
            log.warning(f"reportlab not found. Created minimal/empty PDF at {pdf_path}")
    return pdf_path

@pytest.fixture
def sample_pdf_bytes(sample_pdf_path):
     """Provides the content of the sample PDF as bytes."""
     with open(sample_pdf_path, "rb") as f:
         return f.read()

@pytest.fixture
def sample_event():
    """Provides a sample analytics event dictionary."""
    return {
        'id': str(uuid.uuid4()),
        'event_type': 'api_request',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'user_id': 'test-user-123',
        'endpoint': '/api/v1/some/endpoint',
        'method': 'POST',
        'status_code': 200,
        'duration_ms': 150.5,
        'details': {'param1': 'value1'}
    }

@pytest.fixture
def sample_event_data():
    """Provides a sample event data for webhook tests."""
    return {
        'id': str(uuid.uuid4()),
        'event_type': 'document_processed',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'user_id': 'test-user-456',
        'document_id': 'doc-123',
        'status': 'completed',
        'details': {'pages': 5, 'title': 'Test Document'}
    }

@pytest.fixture
def sample_subscription_dict():
    """Provides a sample webhook subscription dictionary."""
    return {
        'id': str(uuid.uuid4()),
        'url': 'https://example.com/webhook',
        'event_types': ['document_processed', 'question_answered'],
        'secret': 'test-webhook-secret-123',
        'user_id': 'test-user-456',
        'owner_id': 'test-user-456',
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'active': True,
        'description': None,
        'last_triggered': None,
        'last_success': None,
        'last_failure': None,
        'success_count': 0,
        'failure_count': 0,
        'last_error': None
    }

@pytest.fixture
def mock_webhook_subscription_class(mocker):
    """Provides a mocked WebhookSubscription class."""
    mock_class = mocker.patch('services.analytics.webhooks.schemas.WebhookSubscription')
    mock_instance = MagicMock()
    mock_instance.id = 'test-webhook-123'
    mock_instance.url = 'https://example.com/webhook'
    mock_instance.secret = 'test-webhook-secret-123'
    mock_instance.enabled = True
    mock_class.return_value = mock_instance
    mock_class.from_dict.return_value = mock_instance
    return (mock_class, mock_instance)

@pytest.fixture
def mock_logging(mocker):
    """Provides a mocked structlog logger instance."""
    # Create a mock logger
    mock_logger = MagicMock()

    # Patch the get_logger function to return our mock
    mocker.patch('structlog.get_logger', return_value=mock_logger)

    # Return the mock logger instance itself
    return mock_logger

@pytest.fixture
def fake_redis(mocker):
    """Provides a fake Redis client for testing."""
    redis_mock = MagicMock()

    # Mock the key-value store with a dictionary
    redis_data = {}

    # Mock Redis methods
    def mock_exists(key):
        return key in redis_data

    def mock_get(key):
        return redis_data.get(key)

    def mock_set(key, value):
        redis_data[key] = value
        return True

    def mock_setex(key, expiry, value):
        redis_data[key] = value
        return True

    def mock_delete(key):
        if key in redis_data:
            del redis_data[key]
            return 1
        return 0

    # Assign the mocked methods
    redis_mock.exists = mock_exists
    redis_mock.get = mock_get
    redis_mock.set = mock_set
    redis_mock.setex = mock_setex
    redis_mock.delete = mock_delete

    # Only patch the Redis client in the Flask app if there's an application context
    try:
        if has_request_context():
            mocker.patch.object(current_app, 'redis_client', redis_mock)
    except RuntimeError:
        # Working outside of application context, which is fine for some tests
        pass

    # Create a pipeline mock
    pipeline_mock = MagicMock()
    redis_mock.pipeline.return_value = pipeline_mock

    return redis_mock

@pytest.fixture
def mock_dependencies(fake_redis, mocker):
    """Provides mocked dependencies for task tests."""
    # Create a pipeline mock
    pipeline_mock = MagicMock()
    fake_redis.pipeline.return_value = pipeline_mock

    # Mock the requests module for webhook tests
    requests_post_mock = mocker.patch('requests.post')
    generate_signature_mock = mocker.patch('services.tasks.webhook_tasks._generate_signature')

    # Create LLM and Vector Search service mocks
    llm_service_mock = MagicMock()
    vector_search_service_mock = MagicMock()

    # Create AppConfig mock for webhook tests
    app_config_mock = MagicMock()
    app_config_mock.WEBHOOK_MAX_RETRIES = 3
    app_config_mock.WEBHOOK_USER_AGENT = "Test-Webhook-Agent/1.0"
    app_config_mock.WEBHOOK_TIMEOUT = 5
    app_config_mock.ANALYTICS_TTL_SECONDS = 3600 # Add missing attribute for analytics tests

    # Mock the update_webhook_stats_in_redis function
    update_stats_mock = mocker.patch('services.tasks.webhook_tasks._update_webhook_stats_in_redis')

    return {
        'llm_service': llm_service_mock,
        'vector_search_service': vector_search_service_mock,
        'redis_client': fake_redis,
        'redis_pipeline': pipeline_mock,
        'requests_post': requests_post_mock,
        'generate_signature': generate_signature_mock,
        'AppConfig': app_config_mock,
        'update_stats': update_stats_mock
    }