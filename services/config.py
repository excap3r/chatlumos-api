"""
Centralized configuration management for the PDF Wisdom Extractor application.

Loads settings from environment variables with sensible defaults.
"""

import os
import multiprocessing
from dotenv import load_dotenv

# Load .env file if present (especially useful for local development)
load_dotenv()

def get_bool_env(var_name: str, default: bool = False) -> bool:
    """Helper to get boolean value from environment variable."""
    value = os.getenv(var_name, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

class AppConfig:
    """Application configuration class."""

    # --- Flask App Settings ---
    SECRET_KEY = os.getenv('SECRET_KEY', 'a-very-secret-key-that-should-be-changed')
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    DEBUG = get_bool_env('FLASK_DEBUG', default=(FLASK_ENV == 'development'))
    API_VERSION = os.getenv('API_VERSION', 'v1')
    FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000') # For CORS

    # --- Logging ---
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

    # --- Database (MySQL - User DB & Main DB share for now) ---
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    MYSQL_USER = os.getenv('MYSQL_USER', 'pdf_user')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'password')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'pdf_wisdom_db')
    # Pool size used by both user_db and db_service pools
    DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', 10))

    # --- Redis ---
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    # TTL for task result keys in Redis (used in ask.py, pdf.py)
    REDIS_TASK_TTL_SECONDS = int(os.getenv('REDIS_TASK_TTL_SECONDS', 86400)) # 24 hours

    # --- JWT Authentication ---
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'another-secret-key-please-change')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    # Consider adding JWT_ACCESS_TOKEN_EXPIRES, JWT_REFRESH_TOKEN_EXPIRES

    # --- Celery ---
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL) # Use Redis by default
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL) # Use Redis by default

    # --- LLM Providers ---
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    # Add other provider keys as needed (e.g., GOOGLE_API_KEY)
    DEFAULT_LLM_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'openai') # Default provider if none specified
    # LLM_CAPABILITIES_URL = os.getenv('LLM_CAPABILITIES_URL') # Optional external capabilities definition

    # --- Vector Search (Pinecone & Embeddings) ---
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    # Option 1: Serverless
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    # Option 2: Pod-based (set PINECONE_ENVIRONMENT instead of CLOUD/REGION)
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., 'gcp-starter' or your specific env
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'pdf-wisdom-index')
    PINECONE_POD_TYPE = os.getenv("PINECONE_POD_TYPE", "s1.x1") # Default pod type if using pod spec

    EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
    # Determine dimension automatically in VectorSearchService or set explicitly
    VECTOR_SEARCH_DIMENSION = os.getenv('VECTOR_SEARCH_DIMENSION') # If set, must be int
    DEFAULT_TOP_K = int(os.getenv('DEFAULT_TOP_K', 10))

    # --- Gunicorn Settings ---
    # These are primarily read by gunicorn_config.py but defined here centrally
    GUNICORN_BIND = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
    # Default workers based on CPU count
    _default_workers = multiprocessing.cpu_count() * 2 + 1
    GUNICORN_WORKERS = int(os.getenv("GUNICORN_WORKERS", _default_workers))
    GUNICORN_WORKER_CLASS = os.getenv("GUNICORN_WORKER_CLASS", "gevent")
    GUNICORN_TIMEOUT = int(os.getenv("GUNICORN_TIMEOUT", 60))
    GUNICORN_KEEPALIVE = int(os.getenv("GUNICORN_KEEPALIVE", 5)) # Increased from 2
    GUNICORN_MAX_REQUESTS = int(os.getenv("GUNICORN_MAX_REQUESTS", 1000))
    GUNICORN_MAX_REQUESTS_JITTER = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", 50))

    # --- Webhook Settings ---
    # Example: could add WEBHOOK_SECRET_SALT if needed globally

    # --- PDF Processor Settings ---
    # Example: Could add chunk size/overlap defaults if needed
    # PDF_CHUNK_SIZE = int(os.getenv('PDF_CHUNK_SIZE', 1000))
    # PDF_CHUNK_OVERLAP = int(os.getenv('PDF_CHUNK_OVERLAP', 200))

    # --- Validation ---
    # Add validation logic here if needed, e.g., check required keys
    @classmethod
    def validate(cls):
        required_keys = [
            'SECRET_KEY', 'JWT_SECRET_KEY',
            'MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DATABASE',
            'REDIS_URL'
            # Add more required keys, e.g., PINECONE_API_KEY if pinecone is essential
        ]
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        if missing_keys:
             # In a real app, might raise an exception or log critical error
             print(f"CRITICAL WARNING: Missing required configuration keys: {', '.join(missing_keys)}")
             # raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")

        if cls.VECTOR_SEARCH_DIMENSION:
            try:
                 cls.VECTOR_SEARCH_DIMENSION = int(cls.VECTOR_SEARCH_DIMENSION)
            except (ValueError, TypeError):
                 print(f"WARNING: Invalid VECTOR_SEARCH_DIMENSION '{cls.VECTOR_SEARCH_DIMENSION}'. Should be an integer. Ignoring.")
                 cls.VECTOR_SEARCH_DIMENSION = None # Reset to None if invalid

    # --- Analytics Settings ---
    ANALYTICS_TTL_SECONDS = int(os.getenv("ANALYTICS_TTL_SECONDS", 60 * 60 * 24 * 30)) # 30 days default
    ANALYTICS_EXCLUDE_PATHS = [p.strip() for p in os.getenv("ANALYTICS_EXCLUDE_PATHS", "/health, /metrics").split(',')]
    ANALYTICS_DEFAULT_LIMIT = int(os.getenv("ANALYTICS_DEFAULT_LIMIT", 100))
    ANALYTICS_EXPORT_LIMIT = int(os.getenv("ANALYTICS_EXPORT_LIMIT", 10000))
    ANALYTICS_USER_STATS_LIMIT = int(os.getenv("ANALYTICS_USER_STATS_LIMIT", 5000))

    # --- Progress Streaming (SSE) ---
    PROGRESS_STREAM_TIMEOUT = int(os.getenv('PROGRESS_STREAM_TIMEOUT', 300)) # Overall timeout in seconds
    PROGRESS_PUBSUB_LISTEN_TIMEOUT = float(os.getenv('PROGRESS_PUBSUB_LISTEN_TIMEOUT', 1.0)) # Pubsub listen timeout

    # --- Rate Limiting ---
    # Default rate limit (applied globally unless overridden)

# Run validation on import (optional, might be better in app factory)
# AppConfig.validate()

# Example usage (in other modules):
# from .config import AppConfig
# db_host = AppConfig.MYSQL_HOST

# Add any other necessary imports and code here 