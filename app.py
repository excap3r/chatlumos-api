#!/usr/bin/env python3
"""
Wisdom API Server - REST API for Semantic Search with Vector Database

This Flask-based API server provides endpoints for question answering, 
semantic search, and knowledge retrieval using a microservices architecture
with API Gateway. Optimized for high load and Next.js frontend integration.
"""

import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import redis

# Import API Gateway interface
from services.api_gateway import APIGateway

# Import authentication blueprint
from services.api.auth_routes import auth_bp

# Import analytics blueprints and middleware
from services.api.analytics_routes import analytics_bp
from services.api.webhook_routes import webhook_bp
from services.analytics.analytics_middleware import setup_analytics_tracking

# Import route blueprints
from services.api.routes.health import health_bp
from services.api.routes.ask import ask_bp
from services.api.routes.progress import progress_bp
from services.api.routes.translate import translate_bp
from services.api.routes.search import search_bp
from services.api.routes.pdf import pdf_bp
from services.api.routes.question import question_bp
from services.api.routes.docs import docs_bp
from services.api.routes.root import root_bp

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# API version
API_VERSION = "v1"
app.config['API_VERSION'] = API_VERSION # Store in app config
app.config['DEFAULT_TOP_K'] = 10 # Added DEFAULT_TOP_K to config
app.config['DEFAULT_INDEX'] = "wisdom-embeddings" # Added DEFAULT_INDEX

# Configure CORS for Next.js frontend
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv("FRONTEND_URL", "http://localhost:3000"),
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Register existing blueprints
app.register_blueprint(auth_bp, url_prefix=f'/api/{API_VERSION}/auth')
app.register_blueprint(analytics_bp, url_prefix=f'/api/{API_VERSION}/analytics')
app.register_blueprint(webhook_bp, url_prefix=f'/api/{API_VERSION}/webhooks')

# Register new blueprints
app.register_blueprint(health_bp, url_prefix=f'/api/{API_VERSION}')
app.register_blueprint(ask_bp, url_prefix=f'/api/{API_VERSION}')
app.register_blueprint(progress_bp, url_prefix=f'/api/{API_VERSION}')
app.register_blueprint(translate_bp, url_prefix=f'/api/{API_VERSION}')
app.register_blueprint(search_bp, url_prefix=f'/api/{API_VERSION}')
app.register_blueprint(pdf_bp, url_prefix=f'/api/{API_VERSION}')
app.register_blueprint(question_bp, url_prefix=f'/api/{API_VERSION}')
app.register_blueprint(docs_bp, url_prefix=f'/api/{API_VERSION}')
app.register_blueprint(root_bp)

# Global variables

# Progress event storage - maps session IDs to event queues
app.progress_events = {}

# Initialize API Gateway client and attach to app
app.api_gateway = APIGateway(os.getenv("GATEWAY_URL"))

# Initialize Redis for caching if available and attach to app
redis_url = os.getenv("REDIS_URL")
app.redis_client = None
if redis_url:
    try:
        app.redis_client = redis.from_url(redis_url)
        logging.info(f"Connected to Redis at {redis_url}")
    except Exception as e:
        logging.error(f"Failed to connect to Redis: {e}")

# Set up analytics tracking
setup_analytics_tracking(app)

if __name__ == "__main__":
    # Configure logging basic setup if running directly
    logging.basicConfig(level=logging.INFO)
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run the Wisdom API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes (only used with gunicorn)')
    args = parser.parse_args()
    
    # Check if running with gunicorn
    if "gunicorn" in os.environ.get("SERVER_SOFTWARE", ""):
        # Gunicorn will handle the server setup
        logging.info(f"Running with gunicorn with {os.environ.get('WEB_CONCURRENCY', '?')} workers")
    else:
        # Run with Flask's built-in server (not recommended for production)
        logging.info(f"Starting server on {args.host}:{args.port} (debug={args.debug})")
        logging.warning("Using Flask's built-in server. For production, use gunicorn.")
        app.run(host=args.host, port=args.port, debug=args.debug) 