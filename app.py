#!/usr/bin/env python3
"""
Wisdom API Server - REST API for Semantic Search with Vector Database

This Flask-based API server provides endpoints for question answering, 
semantic search, and knowledge retrieval using a microservices architecture
with API Gateway. Optimized for high load and Next.js frontend integration.
"""

import os
from flask import Flask, jsonify, request, g
from flask_cors import CORS
import redis
import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

# Import configuration first
from services.config import AppConfig

# Import logging utilities
from services.utils.log_utils import initialize_logging, setup_request_logging

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

# Import error handling utilities
from services.utils.error_utils import APIError, format_error_response
from werkzeug.exceptions import NotFound

# Import Service Classes for Initialization
# from services.db.user_db import UserDB # Removed obsolete import
from services.llm_service.llm_service import LLMService
from services.vector_search.vector_search import VectorSearchService
from services.analytics.analytics_service import AnalyticsService
from services.analytics.webhooks.webhook_service import WebhookService

# Initialize logger early for potential issues during import or setup
logger = structlog.get_logger(__name__)

def create_app(config_object=AppConfig):
    """Factory function to create and configure the Flask application."""
    app = Flask(__name__)

    # --- Load Configuration from Config Object --- #
    app.config.from_object(config_object)
    logger.info("Flask application configuration loaded.", config_env=os.getenv('FLASK_ENV', 'default'))

    # --- Initialize Structured Logging --- #
    initialize_logging(log_level_name=app.config.get('LOG_LEVEL', 'INFO')) 
    setup_request_logging(app) # Set up before/after request hooks
    logger.info("Structured logging initialized.")

    # --- CORS Configuration --- #
    CORS(app, resources={
        r"/api/*": {
            "origins": app.config.get('FRONTEND_URL', '*'), # Use get with default
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    logger.info("CORS configured.", allowed_origins=app.config.get('FRONTEND_URL', '*'))

    # --- SQLAlchemy Setup --- #
    db_driver = app.config.get('DB_DRIVER', 'mysql+pymysql')
    db_user = app.config.get('DB_USER')
    db_password = app.config.get('DB_PASSWORD')
    db_host = app.config.get('DB_HOST', 'localhost')
    db_port = app.config.get('DB_PORT', '3306')
    db_name = app.config.get('DB_NAME')

    if not all([db_user, db_password, db_name]):
        logger.error("Database credentials (DB_USER, DB_PASSWORD, DB_NAME) not fully configured. SQLAlchemy engine not created.")
        app.db_engine = None
        app.db_session_factory = None
        # Optionally make db_session a stub that raises an error if used
        app.db_session = None # No session available
    else:
        try:
            database_uri = f"{db_driver}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            app.db_engine = create_engine(database_uri, pool_pre_ping=True) # Add pool_pre_ping
            # Create a configured "Session" class
            Session = sessionmaker(autocommit=False, autoflush=False, bind=app.db_engine)
            # Create a thread-local session factory
            app.db_session_factory = scoped_session(Session)
            # Make the session property accessible directly for convenience in request handlers if needed
            # Usage: db_session() to get the current session
            app.db_session = app.db_session_factory

            logger.info("SQLAlchemy engine and session factory created successfully.", db_host=db_host, db_name=db_name)

            # Add teardown context to remove session after request/context ends
            @app.teardown_appcontext
            def shutdown_session(exception=None):
                if hasattr(app, 'db_session_factory') and app.db_session_factory:
                    app.db_session_factory.remove()
                    # logger.debug("SQLAlchemy session removed for context.") # Optional debug logging

        except Exception as e:
            logger.error("Error creating SQLAlchemy engine or session factory.",
                         db_host=db_host, db_name=db_name, error=str(e), exc_info=True)
            app.db_engine = None
            app.db_session_factory = None
            app.db_session = None

    # --- End SQLAlchemy Setup --- #

    # --- Register Blueprints --- #
    api_version = app.config.get('API_VERSION', 'v1')
    api_prefix = f"/api/{api_version}"

    app.register_blueprint(auth_bp, url_prefix=f'{api_prefix}/auth')
    app.register_blueprint(analytics_bp, url_prefix=f'{api_prefix}/analytics')
    app.register_blueprint(webhook_bp, url_prefix=f'{api_prefix}/webhooks')
    app.register_blueprint(health_bp, url_prefix=api_prefix)
    app.register_blueprint(ask_bp, url_prefix=api_prefix)
    app.register_blueprint(progress_bp, url_prefix=api_prefix)
    app.register_blueprint(translate_bp, url_prefix=api_prefix)
    app.register_blueprint(search_bp, url_prefix=api_prefix)
    app.register_blueprint(pdf_bp, url_prefix=api_prefix)
    app.register_blueprint(question_bp, url_prefix=api_prefix)
    app.register_blueprint(docs_bp, url_prefix=api_prefix)
    app.register_blueprint(root_bp) # Root blueprint has no prefix
    logger.info("API blueprints registered.", api_prefix=api_prefix)


    # --- Service Initialization --- #
    
    # Initialize Redis client
    redis_url = app.config.get('REDIS_URL')
    app.redis_client = None
    if redis_url:
        try:
            app.redis_client = redis.from_url(redis_url, decode_responses=True)
            app.redis_client.ping() # Test connection
            logger.info("Redis client connected successfully.", redis_url=redis_url)
        except redis.exceptions.ConnectionError as e:
            logger.error("Failed to connect to Redis. Proceeding without Redis.", redis_url=redis_url, error=str(e))
        except Exception as e:
            logger.error("Unexpected error initializing Redis.", redis_url=redis_url, error=str(e))
    else:
        logger.warning("REDIS_URL not configured. Redis client not initialized.")

    # Initialize User Database Pool (Note: This might be replaced by SQLAlchemy session later) - REMOVED Block
    # try:
    #     app.user_db_pool = UserDB.create_pool(app.config)
    #     if app.user_db_pool:
    #         logger.info("UserDB MySQL connection pool created successfully. (Will be replaced by ORM)")
    #     else:
    #         logger.error("Failed to create UserDB MySQL connection pool.")
    # except Exception as e:
    #     logger.error("Error creating UserDB MySQL connection pool", error=str(e), exc_info=True)
    #     app.user_db_pool = None # Ensure it's None if init fails

    # Initialize LLM Service
    try:
        app.llm_service = LLMService(config=app.config)
        logger.info("LLM Service initialized successfully.")
    except Exception as e:
        logger.error("Error initializing LLM Service", error=str(e), exc_info=True)
        app.llm_service = None

    # Initialize Vector Search Service
    try:
        app.vector_search_service = VectorSearchService(config=app.config)
        logger.info("Vector Search Service initialized successfully.")
        # Optional: Add a connection test/ping here if the service provides one
        # app.vector_search_service.test_connection() 
    except Exception as e:
        logger.error("Error initializing Vector Search Service", error=str(e), exc_info=True)
        app.vector_search_service = None

    # Initialize Analytics Service
    try:
        # Pass necessary initialized components if needed
        app.analytics_service = AnalyticsService(config=app.config, redis_client=app.redis_client)
        logger.info("Analytics Service initialized successfully.")
    except Exception as e:
        logger.error("Error initializing Analytics Service", error=str(e), exc_info=True)
        app.analytics_service = None

    # Initialize Webhook Service
    try:
        app.webhook_service = WebhookService(config=app.config, redis_client=app.redis_client)
        logger.info("Webhook Service initialized successfully.")
    except Exception as e:
        logger.error("Error initializing Webhook Service", error=str(e), exc_info=True)
        app.webhook_service = None

    # Set up analytics tracking middleware (after services are initialized)
    setup_analytics_tracking(app)
    logger.info("Analytics tracking middleware setup.")


    # --- Centralized Error Handling --- #
    error_logger = structlog.get_logger("error_handler")

    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        """Handle custom APIErrors and return standardized JSON response."""""
        error_logger.warning("API Error occurred", 
                             error_message=error.message, 
                             status_code=error.status_code, 
                             details=error.details, 
                             exception_type=type(error).__name__,
                             path=request.path,
                             method=request.method)
        response_dict, status_code = format_error_response(error)
        return jsonify(response_dict), status_code

    @app.errorhandler(NotFound) # Handle 404 Not Found
    def handle_not_found(error: NotFound):
        """Handle Flask's default 404 and return JSON."""""
        error_logger.info("Resource not found (404)", path=request.path, method=request.method)
        response_dict = {
            "error": "Not Found",
            "message": "The requested URL was not found on the server."
        }
        return jsonify(response_dict), 404

    @app.errorhandler(Exception) # Catch-all for other exceptions (500)
    def handle_generic_exception(error: Exception):
        """Handle unexpected exceptions and return a generic 500 error."""""
        error_logger.error("Unhandled exception occurred", 
                           error=str(error),
                           exception_type=type(error).__name__,
                           path=request.path,
                           method=request.method,
                           exc_info=True) 
                           
        response_dict = {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred on the server."
        }
        # Optionally hide details in production
        # if not app.config['DEBUG']:
        #     response_dict['message'] = "An unexpected error occurred."
        return jsonify(response_dict), 500
        
    logger.info("Centralized error handlers registered.")

    return app

# --- End Error Handling ---

# Create the Flask app instance using the factory
app = create_app()

if __name__ == "__main__":
    # Use config values for host, port, debug obtained from the app config
    host = app.config.get('HOST', '0.0.0.0') 
    port = app.config.get('PORT', 5000)
    debug_mode = app.config.get('DEBUG', False)

    # Check if running with gunicorn (Gunicorn sets SERVER_SOFTWARE)
    is_gunicorn = "gunicorn" in os.environ.get("SERVER_SOFTWARE", "")

    if is_gunicorn:
        # Gunicorn manages the server process
        # Logging is typically handled by Gunicorn config (gunicorn_config.py)
        logger.info(f"Running with gunicorn workers.")
    else:
        # Run with Flask's built-in server (for development/debugging)
        logger.info(f"Starting Flask development server", host=host, port=port, debug=debug_mode)
        if not debug_mode:
             logger.warning("Running Flask development server in non-debug mode. Use Gunicorn for production.")
        # Pass debug=debug_mode to app.run
        app.run(host=host, port=port, debug=debug_mode)