import os
from celery import Celery
from dotenv import load_dotenv

# Import AppConfig
from services.config import AppConfig

# Load environment variables from .env file
# load_dotenv() # Removed

# Get Redis URL from environment, default to localhost if not set
# Ensure your .env file has REDIS_URL='redis://localhost:6379/0' or similar
# redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0') # Replaced

# Get Redis URL from AppConfig
# Note: Assumes AppConfig loads environment variables itself
redis_url = AppConfig.REDIS_URL
if not redis_url:
    # Fallback or error if REDIS_URL is critical and not set in AppConfig
    print("Error: REDIS_URL not found in AppConfig. Celery requires a broker URL.")
    # Optionally raise an error or use a default for development
    redis_url = 'redis://localhost:6379/0' # Example fallback
    print(f"Warning: Using default Redis URL: {redis_url}")

# Initialize Celery
# The first argument is the name of the current module, used for naming tasks.
# The 'broker' and 'backend' arguments specify the URLs for the message broker and result backend.
celery_app = Celery(
    'tasks', # Name of the main module where tasks might be defined or imported
    broker=redis_url,
    backend=redis_url,
    include=[
        'services.tasks.question_processing', 
        'services.tasks.pdf_processing',
        'services.tasks.webhook_tasks', # Added webhook task module
        'services.tasks.analytics_tasks' # Added analytics task module
    ] # List of modules to import when the worker starts
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer='json', # Use json for task serialization
    result_serializer='json', # Use json for result serialization
    accept_content=['json'],  # Accept json content
    timezone='UTC', # Use UTC timezone
    enable_utc=True, # Enable UTC
    # Optional: Set task result expiration time (e.g., 1 day)
    result_expires=86400, 
    # Optional: Configure task tracking and state storage
    task_track_started=True, 
    # Optional: Broker connection pool limits (adjust based on needs)
    broker_pool_limit=10, 
    # Configure Redis-specific settings if needed
    # broker_transport_options = {'visibility_timeout': 3600}, # Example: 1 hour visibility timeout
    # result_backend_transport_options = {'retry_policy': {'max_retries': 3}}, # Example retry policy
)

# Example: Add Flask app context for tasks if needed later
# Although tasks should ideally be self-contained, sometimes context is useful.
# This setup assumes you might create the Flask app instance elsewhere (e.g., in app.py)
# and potentially pass it or its config to tasks if absolutely required.
# For now, tasks will access Flask context if run via Flask request, 
# but direct Celery worker execution won't have Flask context unless configured.

if __name__ == '__main__':
    # To run the worker: celery -A celery_app worker --loglevel=info
    # Replace 'celery_app' with the actual name of this file if different.
    celery_app.start() 