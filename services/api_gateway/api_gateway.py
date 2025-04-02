#!/usr/bin/env python3
"""
API Gateway Service

This service acts as a central gateway for all API requests, routing them
to the appropriate microservices and handling cross-cutting concerns like:
- Authentication
- Rate limiting
- Request logging
- Service discovery
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, request, jsonify, Response, stream_with_context, g
from flask_cors import CORS
from dotenv import load_dotenv
import argparse
import uuid
import redis
from ..utils.api_helpers import rate_limit

# Import utility modules
from ..utils.error_utils import handle_error, APIError, ValidationError
from ..utils.log_utils import setup_logger, setup_request_logging
# Add imports for JWT validation
from ..utils.auth_utils import decode_token, InvalidTokenError, ExpiredTokenError, MissingSecretError
# Add imports for API Key validation (assuming direct access is intended)
from ..db.user_db import verify_api_key
from ..db.exceptions import InvalidCredentialsError as APIKeyInvalidCredentialsError # Alias to avoid naming conflict
from ..db.exceptions import QueryError as DBQueryError, DatabaseError as DBError # Import relevant DB errors

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger('api_gateway')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure request logging
setup_request_logging(app)

# --- Rate Limiting Configuration --- #
# Example: Get from env or use default
RATE_LIMIT_MAX_CALLS = int(os.getenv('RATE_LIMIT_MAX_CALLS', '100'))
RATE_LIMIT_PER_SECONDS = int(os.getenv('RATE_LIMIT_PER_SECONDS', '60'))

# Service registry
service_registry = {}

# --- Redis Client Setup --- #
_redis_client = None
def _get_redis_client() -> Optional[redis.Redis]:
    """Initializes and returns a Redis client instance, caching the connection."""
    global _redis_client
    if _redis_client:
        return _redis_client
    
    redis_url = os.getenv('REDIS_URL', None)
    redis_host = 'localhost'
    redis_port = 6379
    redis_db = 0
    redis_password = None

    if redis_url:
        try:
            # Use redis-py's built-in parser
            conn_kwargs = redis.connection.Connection.parse_url(redis_url)
            redis_host = conn_kwargs.get('host', redis_host)
            redis_port = conn_kwargs.get('port', redis_port)
            redis_db = conn_kwargs.get('db', redis_db)
            redis_password = conn_kwargs.get('password', redis_password)
            logger.info(f"Parsed REDIS_URL: host={redis_host}, port={redis_port}, db={redis_db}")
        except ValueError as e:
            logger.error(f"Failed to parse REDIS_URL '{redis_url}': {e}. Falling back to individual variables.")
            # Fallback to individual env vars if URL parsing fails or URL not set
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            redis_db = int(os.getenv('REDIS_DB', '0'))
            redis_password = os.getenv('REDIS_PASSWORD', None)
    else:
        # Load individual env vars if REDIS_URL is not set
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_db = int(os.getenv('REDIS_DB', '0'))
        redis_password = os.getenv('REDIS_PASSWORD', None)
    
    try:
        logger.info(f"Attempting to connect to Redis at {redis_host}:{redis_port}/{redis_db}")
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True, # Decode responses to strings
            socket_timeout=5,      # Set connection timeout
            socket_connect_timeout=5
        )
        client.ping() # Verify connection
        _redis_client = client
        logger.info("Successfully connected to Redis.")
        return _redis_client
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}. Rate limiting/Caching might be disabled.")
        _redis_client = None # Ensure client is None if connection fails
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Redis connection: {e}", exc_info=True)
        _redis_client = None
        return None

def create_app():
    """Create and configure the Flask application."""
    # Set up before request handlers
    @app.before_request
    def before_request():
        # Generate a unique request ID
        request.request_id = str(uuid.uuid4())
        # Initialize g.user to None
        g.user = None
        
    # Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found", "status_code": 404}), 404
    
    @app.errorhandler(500)
    def server_error(error):
        logger.error(f"Server error: {str(error)}")
        return jsonify({"error": "Internal server error", "status_code": 500}), 500
    
    # Initialize service registry
    init_service_registry()
    
    return app

def init_service_registry():
    """Initialize the service registry from environment variables or configuration."""
    # Get services configuration from environment variables
    services_config_json = os.getenv("SERVICES_CONFIG")
    services_config = None
    if services_config_json:
        try:
            services_config = json.loads(services_config_json)
        except json.JSONDecodeError:
            logger.error("Failed to parse SERVICES_CONFIG JSON from environment variable.")
    
    if services_config and isinstance(services_config, dict):
        # If JSON configuration is provided in environment variables
        for service_name, service_info in services_config.items():
            register_service(service_name, service_info.get("url"), service_info.get("paths", []))
    else:
        # Otherwise, use individual environment variables
        # LLM Service
        llm_service_url = os.getenv("LLM_SERVICE_URL", "http://localhost:5002")
        if llm_service_url:
            register_service("llm", llm_service_url, ["/llm", "/decompose", "/generate", "/embed"])
        
        # Vector Search Service
        vector_service_url = os.getenv("VECTOR_SERVICE_URL", "http://localhost:5003")
        if vector_service_url:
            register_service("vector", vector_service_url, ["/vector", "/search", "/embed"])
        
        # Database Service
        db_service_url = os.getenv("DB_SERVICE_URL", "http://localhost:5001")
        if db_service_url:
            register_service("db", db_service_url, ["/db", "/documents", "/concepts", "/qa_pairs"])
        
        # PDF Processor Service
        pdf_service_url = os.getenv("PDF_SERVICE_URL", "http://localhost:5004")
        if pdf_service_url:
            register_service("pdf", pdf_service_url, ["/pdf", "/extract", "/process"])

def register_service(name: str, url: str, paths: List[str] = None):
    """
    Register a service with the gateway.
    
    Args:
        name: Service name (identifier)
        url: Base URL of the service
        paths: List of path prefixes this service handles
    """
    service_registry[name] = {
        "url": url.rstrip("/"),  # Remove trailing slash
        "paths": paths or [f"/{name}"],
        "status": "unknown"
    }
    logger.info(f"Registered service: {name} at {url}")
    
    # Check service health
    try:
        health_url = f"{url.rstrip('/')}/health"
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            service_registry[name]["status"] = "healthy"
        else:
            service_registry[name]["status"] = "unhealthy"
    except Exception as e:
        logger.warning(f"Could not check health of service {name}: {str(e)}")
        service_registry[name]["status"] = "unknown"

def get_service(path: str) -> Optional[Dict[str, Any]]:
    """
    Get the service that handles a given path.
    
    Args:
        path: Request path
        
    Returns:
        Service information or None if no matching service
    """
    for service_name, service_info in service_registry.items():
        for path_prefix in service_info["paths"]:
            if path.startswith(path_prefix):
                return {
                    "name": service_name,
                    **service_info
                }
    return None

def get_service_url(service_name: str) -> Optional[str]:
    """
    Get the URL for a registered service by name.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Service URL or None if not found
    """
    service = service_registry.get(service_name)
    if service:
        return service["url"]
    return None

def proxy_request(service_info: Dict[str, Any], path: str, include_body: bool = True) -> Response:
    """
    Proxy a request to the appropriate service.
    
    Args:
        service_info: Service information dictionary
        path: Request path
        include_body: Whether to include the request body
        
    Returns:
        Flask response with the service's response
    """
    # Construct target URL
    service_url = service_info["url"]
    target_url = f"{service_url}{path}"
    
    # Copy request headers
    headers = dict(request.headers)
    
    # Don't pass the host header
    if "Host" in headers:
        del headers["Host"]
    
    # Don't pass hop-by-hop headers
    hop_by_hop_headers = [
        "Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization",
        "TE", "Trailers", "Transfer-Encoding", "Upgrade"
    ]
    for header in hop_by_hop_headers:
        if header in headers:
            del headers[header]
    
    # --- Forward User Context --- #
    if hasattr(g, 'user') and g.user:
        user_id = g.user.get('sub') or g.user.get('id') # Get ID from JWT (sub) or API Key verification result (id)
        if user_id:
            headers['X-User-ID'] = str(user_id) # Ensure it's a string
            logger.debug(f"Forwarding X-User-ID: {user_id}", request_id=getattr(request, 'request_id', 'N/A'))
        else:
            logger.warning("User context found in g.user but no 'sub' or 'id' field available to forward.", request_id=getattr(request, 'request_id', 'N/A'))
    else:
        # This shouldn't normally happen if auth checks are passed, but log just in case
        logger.debug("No authenticated user context (g.user) found to forward.", request_id=getattr(request, 'request_id', 'N/A'))
    
    # Prepare request arguments
    kwargs = {
        "headers": headers,
        "params": request.args
    }
    
    # Add request body if needed
    if include_body and request.method in ["POST", "PUT", "PATCH"]:
        if request.is_json:
            kwargs["json"] = request.get_json()
        else:
            kwargs["data"] = request.get_data()
    
    # Make the request to the service
    try:
        # Handle streaming responses
        if request.headers.get("Accept") == "text/event-stream":
            def generate():
                # Stream the response
                with requests.request(
                    method=request.method,
                    url=target_url,
                    stream=True,
                    **kwargs
                ) as resp:
                    for chunk in resp.iter_content(chunk_size=1024):
                        yield chunk
            
            # Create a streaming response
            return Response(
                stream_with_context(generate()),
                content_type=request.headers.get("Accept")
            )
        else:
            # Make a regular request
            response = requests.request(
                method=request.method,
                url=target_url,
                **kwargs
            )
            
            # Create Flask response
            flask_response = Response(
                response.content,
                status=response.status_code,
                content_type=response.headers.get("Content-Type")
            )
            
            # Copy headers from service response to Flask response
            for key, value in response.headers.items():
                if key.lower() not in ["content-length", "transfer-encoding", "connection"]:
                    flask_response.headers[key] = value
                    
            return flask_response
    except requests.RequestException as e:
        logger.error(f"Error proxying request to {target_url}: {str(e)}")
        return jsonify({
            "error": "Service unavailable",
            "service": service_info.get("name", "unknown"),
            "details": str(e)
        }), 503

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the API gateway."""
    # Check all services
    for service_name, service_info in service_registry.items():
        try:
            health_url = f"{service_info['url']}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                service_registry[service_name]["status"] = "healthy"
            else:
                service_registry[service_name]["status"] = "unhealthy"
        except Exception:
            service_registry[service_name]["status"] = "unreachable"
    
    return jsonify({
        "status": "healthy",
        "service": "api-gateway",
        "services": {name: info["status"] for name, info in service_registry.items()}
    })

@app.route('/services', methods=['GET'])
def list_services():
    """List all registered services."""
    return jsonify({
        "services": {
            name: {
                "url": info["url"],
                "paths": info["paths"],
                "status": info["status"]
            }
            for name, info in service_registry.items()
        }
    })

@rate_limit(redis_client_provider=_get_redis_client, max_calls=RATE_LIMIT_MAX_CALLS, per_seconds=RATE_LIMIT_PER_SECONDS)
@handle_error
def route_request(path):
    """Handle all incoming API requests, perform auth (JWT or API Key), and proxy to services."""
    # Prepend slash if missing
    if not path.startswith('/'):
        path = f"/{path}"

    # --- Authentication Check --- #
    auth_header = request.headers.get('Authorization')
    api_key_header = request.headers.get('X-API-Key') # Standard header for API keys
    user_info = None
    auth_method = None

    # 1. Try JWT Authentication
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        try:
            token_payload = decode_token(token)
            if not token_payload:
                raise InvalidTokenError("Token decoding returned None unexpectedly.")
            
            # TODO: Potentially fetch full user profile from user service/DB based on token_payload['sub']?
            # For now, store the payload itself as user_info.
            user_info = token_payload 
            auth_method = "JWT"
            logger.debug("JWT token validated successfully", user_id=user_info.get('sub'), request_id=getattr(request, 'request_id', 'N/A'))

        except ExpiredTokenError:
            logger.warning("Expired JWT token provided", request_id=getattr(request, 'request_id', 'N/A'))
            # Don't return yet, maybe API key is valid
        except InvalidTokenError as e:
            logger.warning(f"Invalid JWT token provided: {e}", request_id=getattr(request, 'request_id', 'N/A'))
            # Don't return yet
        except MissingSecretError as e:
            logger.error(f"JWT validation failed due to configuration error: {e}", request_id=getattr(request, 'request_id', 'N/A'))
            return jsonify({"error": "Server configuration error during authentication"}), 500 # Server error, return immediately
        except Exception as e:
            logger.error(f"Unexpected error during JWT decoding: {e}", exc_info=True, request_id=getattr(request, 'request_id', 'N/A'))
            return jsonify({"error": "Authentication error"}), 500 # Server error, return immediately

    # 2. Try API Key Authentication (if JWT failed or wasn't provided)
    if not user_info and api_key_header:
        try:
            # This call accesses the DB directly via the imported function
            # It returns user dict on success or raises InvalidCredentialsError/QueryError/DatabaseError
            verified_user_info = verify_api_key(api_key_header)
            if verified_user_info:
                user_info = verified_user_info
                auth_method = "API Key"
                logger.info("API Key verified successfully", user_id=user_info.get('id'), request_id=getattr(request, 'request_id', 'N/A'))
            else:
                # Should not happen if verify_api_key raises errors correctly, but good practice
                 logger.warning("verify_api_key returned None unexpectedly", key_prefix=api_key_header[:8]+"...", request_id=getattr(request, 'request_id', 'N/A'))

        except APIKeyInvalidCredentialsError:
            logger.warning("Invalid API Key provided", key_prefix=api_key_header[:8]+"...", request_id=getattr(request, 'request_id', 'N/A'))
            # Don't return yet, final check below handles 401
        except (DBQueryError, DBError) as e:
            # Database issues during key verification are server-side problems
            logger.error(f"Database error during API key verification: {e}", exc_info=True, request_id=getattr(request, 'request_id', 'N/A'))
            return jsonify({"error": "Authentication error due to database issue"}), 500
        except Exception as e:
             # Catch any other unexpected errors during key verification
            logger.error(f"Unexpected error during API key verification: {e}", exc_info=True, request_id=getattr(request, 'request_id', 'N/A'))
            return jsonify({"error": "Authentication error"}), 500

    # 3. Final Check: If no valid authentication method succeeded
    if not user_info:
        logger.warning("Authentication failed: No valid JWT or API Key provided", request_id=getattr(request, 'request_id', 'N/A'))
        return jsonify({"error": "Unauthorized: Valid authentication required (JWT or API Key)"}), 401

    # --- Store validated user info in request context --- #
    g.user = user_info
    g.auth_method = auth_method
    logger.debug(f"Authentication successful via {auth_method}", user_id=user_info.get('sub') or user_info.get('id'), request_id=getattr(request, 'request_id', 'N/A'))

    # --- Service Discovery --- #
    service_info = get_service(path)
    
    if not service_info:
        raise APIError(f"No service found to handle path: {path}", status_code=404)
    
    return proxy_request(service_info, path)

# Main entry point for running as standalone service
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="API Gateway for PDF Wisdom Extractor")
    parser.add_argument('--port', type=int, default=5000, help='Port to run the service on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create the Flask app
    app = create_app()
    
    # Start the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug) 