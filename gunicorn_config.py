#!/usr/bin/env python
"""
Gunicorn configuration file for production deployment.

This configuration is optimized for high load and performance with the
PDF Wisdom Extractor API server.

Usage:
    gunicorn -c gunicorn_config.py app:app
"""

import multiprocessing
import os

# Server socket
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:5000")
backlog = 2048

# Worker processes
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"  # Use gevent for async capability with low overhead
worker_connections = 1000
timeout = 60
keepalive = 2

# Process naming
proc_name = "pdf_wisdom_api"
pythonpath = "."

# Logging
errorlog = "-"  # stdout
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
accesslog = "-"  # stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# Server mechanics
preload_app = True  # Load application code before the worker processes are forked
daemon = False  # Don't daemonize in Docker
user = None
group = None
umask = 0
tmp_upload_dir = None

# Max requests and jitter to avoid memory leaks
max_requests = 1000
max_requests_jitter = 50

# SSL configuration (if needed)
# keyfile = '/path/to/ssl/key.key'
# certfile = '/path/to/ssl/cert.crt'

# Statsd configuration (if using statsd for metrics)
# statsd_host = 'localhost:8125'
# statsd_prefix = 'pdf_wisdom_api'

# Hook functions for application startup/shutdown
def on_starting(server):
    """Log when the server is starting."""
    server.log.info("Starting PDF Wisdom API server")

def on_exit(server):
    """Clean up resources when shutting down."""
    server.log.info("Shutting down PDF Wisdom API server")

def post_fork(server, worker):
    """Setup worker after it's been forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def worker_exit(server, worker):
    """Clean up when a worker exits."""
    server.log.info(f"Worker exited (pid: {worker.pid})") 