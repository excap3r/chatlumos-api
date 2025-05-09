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
from celery import Celery
from flask_jwt_extended import JWTManager
from werkzeug.exceptions import HTTPException, NotFound, BadRequest
from pydantic import ValidationError as PydanticValidationError # Import Pydantic's error
import json # Import json

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
from services.utils.error_utils import APIError, ValidationError, format_error_response
# Middleware is applied via decorators in routes, no global init needed here
# from services.api.middleware.auth_middleware import init_auth_middleware 
# Rate limiting applied via decorator in routes, no global init needed
# from services.utils.rate_limit import init_rate_limiter 

# Import Service Classes for Initialization
# from services.db.user_db import UserDB # Removed obsolete import
from services.llm_service.llm_service import LLMService
from services.vector_search.vector_search import VectorSearchService
from services.analytics.analytics_service import AnalyticsService
from services.analytics.webhooks.webhook_service import WebhookService

# Import flask-swagger-ui
from flask_swagger_ui import get_swaggerui_blueprint

# --- Initialize Loggers ---
# Use structlog.get_logger directly where needed
logger = structlog.get_logger("app") # General application logger
request_logger = structlog.get_logger("request_logger") # Used in request hooks
error_logger = structlog.get_logger("error_handler") # Used in error handlers

# --- Swagger UI Setup --- #
SWAGGER_URL = '/api/v1/docs'  # URL for exposing Swagger UI (must be Blueprint prefix + /docs)
API_URL = '/api/v1/swagger.json'  # Our API url (must be Blueprint prefix + /swagger.json)

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "PDF Wisdom Extractor API Docs"
    }
)

def create_app(config_object=AppConfig):
    """Factory function to create and configure the Flask application."""
    app = Flask(__name__)

    # --- Load Configuration from Config Object --- #
    app.config.from_object(config_object)
    logger.info("Flask application configuration loaded.", config_env=os.getenv('FLASK_ENV', 'default'))

    # --- Initialize JWT Manager --- #
    JWTManager(app)
    logger.info("JWTManager initialized.")

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

    # Register the JSON spec endpoint FIRST (from docs.py)
    app.register_blueprint(docs_bp, url_prefix=api_prefix)
    
    # Register the Swagger UI blueprint (serves the HTML page)
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL) # Use SWAGGER_URL as prefix
    
    # Register other API blueprints
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
    app.register_blueprint(root_bp) # Root blueprint has no prefix
    logger.info("API blueprints registered.", api_prefix=api_prefix, swagger_url=SWAGGER_URL)


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

    @app.errorhandler(ValidationError) # Custom ValidationError
    def handle_validation_error(error: ValidationError):
        """Handle custom ValidationErrors and return 400 JSON response."""
        error_logger.warning("Validation Error occurred", 
                             error_message=error.message, 
                             status_code=400, 
                             details=error.details, 
                             exception_type=type(error).__name__,
                             path=request.path,
                             method=request.method)
        response = {"error": error.message, "details": error.details}
        return jsonify(response), 400

    @app.errorhandler(PydanticValidationError) # Pydantic's ValidationError
    def handle_pydantic_validation_error(error: PydanticValidationError):
        """Handle Pydantic ValidationErrors and return 400 JSON response."""
        error_details = error.errors() # Get structured errors from Pydantic
        error_logger.warning("Pydantic Validation Error occurred", 
                             errors=error_details, 
                             status_code=400, 
                             exception_type=type(error).__name__,
                             path=request.path,
                             method=request.method)
        # Format errors similarly to how Flask-Rebar or others might
        formatted_errors = {}
        for err in error_details:
            loc = err.get('loc', ('unknown',))
            field = loc[-1] if loc else 'unknown'
            msg = err.get('msg', 'Invalid input')
            if field not in formatted_errors:
                formatted_errors[field] = []
            formatted_errors[field].append(msg)
            
        response = {"error": formatted_errors} # Return structured validation errors
        return jsonify(response), 400

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
        """Handle unexpected non-HTTP exceptions and return a generic 500 error.
        HTTPExceptions are handled by the dedicated `handle_http_exception`.
        APIError instances should ideally be caught by `handle_api_error`, 
        but if they reach here (e.g., due to middleware re-raising), handle them specifically.
        """
        # Check if the caught exception is actually an APIError
        if isinstance(error, APIError):
            # Delegate to the specific APIError handler logic
            error_logger.warning("API Error caught by generic handler", 
                                 error_message=error.message, 
                                 status_code=error.status_code, 
                                 details=error.details, 
                                 exception_type=type(error).__name__,
                                 path=request.path,
                                 method=request.method)
            response_dict, status_code = format_error_response(error)
            return jsonify(response_dict), status_code
            
        # For other non-HTTP exceptions, log as 500 and return generic message
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

    # Add back the handler for all HTTPExceptions to return JSON
    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """Return JSON instead of HTML for HTTP errors."""
        # Explicitly import redis here as a workaround for a NameError
        # that occurs specifically when Forbidden is raised before streaming.
        # This suggests a context issue during Flask's error handling for certain exceptions.
        import redis
        import json 
        # start with the correct headers and status code from the error
        response = e.get_response()
        # replace the body with JSON
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
            "error": e.name # Adding error field for consistency
        })
        response.content_type = "application/json"
        error_logger.info("HTTP Exception handled", code=e.code, name=e.name, path=request.path)
        return response
        
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