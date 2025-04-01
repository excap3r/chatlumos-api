#!/usr/bin/env python
"""
Gunicorn configuration file for production deployment.

Reads settings from the central AppConfig class.

Usage:
    gunicorn -c gunicorn_config.py app:create_app()
"""

# Removed: import multiprocessing
# Removed: import os

# Import the centralized configuration
from services.config import AppConfig

# --- Server Socket ---
bind = AppConfig.GUNICORN_BIND
backlog = 2048  # Standard backlog size

# --- Worker Processes ---
# Number of workers based on CPU count or environment variable via AppConfig
workers = AppConfig.GUNICORN_WORKERS
# Worker class (e.g., gevent, sync, uvicorn.workers.UvicornWorker)
worker_class = AppConfig.GUNICORN_WORKER_CLASS
# Max concurrent requests per worker (adjust based on worker_class and load)
worker_connections = 1000
# Worker timeout in seconds
timeout = AppConfig.GUNICORN_TIMEOUT
# Seconds to wait for requests on a Keep-Alive connection
keepalive = AppConfig.GUNICORN_KEEPALIVE

# --- Process Naming ---
proc_name = "pdf_wisdom_api"
# Ensure the app module can be found
pythonpath = "."

# --- Logging ---
# Gunicorn logger class
logger_class = 'gunicorn.glogging.Logger'
# Log level from AppConfig (e.g., INFO, DEBUG, WARNING)
loglevel = AppConfig.LOG_LEVEL
# Log locations (None lets Gunicorn log to stderr/stdout, suitable for containers)
errorlog = None
accesslog = None
# Access log format (example, customize as needed)
# access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# --- Server Mechanics ---
# Load application code before worker processes are forked (improves startup time)
preload_app = True
# Run in the foreground (standard for containerized environments)
daemon = False
# User/Group to run workers as (set to None to run as current user)
user = None
group = None
umask = 0 # Default file mode creation mask
# Directory to store temporary request data (None uses system default)
tmp_upload_dir = None

# --- Worker Lifecycle ---
# Restart workers after this many requests (with jitter) to prevent memory leaks
max_requests = AppConfig.GUNICORN_MAX_REQUESTS
max_requests_jitter = AppConfig.GUNICORN_MAX_REQUESTS_JITTER

# --- SSL configuration (if terminating SSL at Gunicorn) ---
# keyfile = AppConfig.SSL_KEYFILE # Example if defined in AppConfig
# certfile = AppConfig.SSL_CERTFILE # Example if defined in AppConfig

# --- Statsd configuration (if using Statsd) ---
# statsd_host = AppConfig.STATSD_HOST # Example if defined in AppConfig
# statsd_prefix = 'pdf_wisdom_api'

# --- Hooks (Optional) ---
def on_starting(server):
    """Log when the server is starting."""
    # Ensure logging works even before app fully loaded if needed
    # server.log uses Gunicorn's logger
    server.log.info(f"Starting PDF Wisdom API server on {bind}")
    server.log.info(f"Using {workers} workers ({worker_class})")
    server.log.info(f"Log level set to {loglevel}")

def on_exit(server):
    """Clean up resources when shutting down."""
    server.log.info("Shutting down PDF Wisdom API server")

def post_fork(server, worker):
    """Setup worker after it's been forked."""
    # Example: Initialize worker-specific resources if needed
    server.log.debug(f"Worker spawned (pid: {worker.pid})")

def worker_exit(server, worker):
    """Clean up when a worker exits."""
    server.log.info(f"Worker exited (pid: {worker.pid}, exit_code: {worker.exit_code})") 