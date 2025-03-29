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
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
import argparse
import uuid

# Import utility modules
from ..utils.env_utils import load_env_var, get_config
from ..utils.error_utils import handle_error, APIError, ValidationError
from ..utils.log_utils import setup_logger, create_request_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger('api_gateway')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure request logging
create_request_logger(app)

# Service registry
service_registry = {}

def create_app():
    """Create and configure the Flask application."""
    # Set up before request handlers
    @app.before_request
    def before_request():
        # Generate a unique request ID
        request.request_id = str(uuid.uuid4())
        
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
    services_config = get_config("SERVICES_CONFIG")
    
    if services_config:
        # If JSON configuration is provided in environment variables
        for service_name, service_info in services_config.items():
            register_service(service_name, service_info.get("url"), service_info.get("paths", []))
    else:
        # Otherwise, use individual environment variables
        # LLM Service
        llm_service_url = load_env_var("LLM_SERVICE_URL", "http://localhost:5002")
        if llm_service_url:
            register_service("llm", llm_service_url, ["/llm", "/decompose", "/generate", "/embed"])
        
        # Vector Search Service
        vector_service_url = load_env_var("VECTOR_SERVICE_URL", "http://localhost:5003")
        if vector_service_url:
            register_service("vector", vector_service_url, ["/vector", "/search", "/embed"])
        
        # Database Service
        db_service_url = load_env_var("DB_SERVICE_URL", "http://localhost:5001")
        if db_service_url:
            register_service("db", db_service_url, ["/db", "/documents", "/concepts", "/qa_pairs"])
        
        # PDF Processor Service
        pdf_service_url = load_env_var("PDF_SERVICE_URL", "http://localhost:5004")
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

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
@handle_error
def route_request(path):
    """Route requests to the appropriate service."""
    full_path = f"/{path}"
    service_info = get_service(full_path)
    
    if not service_info:
        raise APIError(f"No service found to handle path: {full_path}", status_code=404)
    
    return proxy_request(service_info, full_path)

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