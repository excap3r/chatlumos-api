import logging
from flask import Blueprint, jsonify, current_app

# Assuming ServiceError is defined in api_gateway module
from services.api_gateway import ServiceError 
# Assuming rate_limit is defined in api_helpers
from services.utils.api_helpers import rate_limit

health_bp = Blueprint('health', __name__)

def initialize_services():
    """Check if all required services are available and healthy via API Gateway."""
    # Access api_gateway via current_app
    if not hasattr(current_app, 'api_gateway') or not current_app.api_gateway:
         current_app.logger.error("API Gateway client not initialized on app context.")
         return False, "API Gateway client not available"
         
    try:
        # Check health of all services through API gateway
        health_response = current_app.api_gateway.health()
        if not health_response.get("status") == "healthy":
            return False, health_response.get("error", "Unknown error checking services health")

        # Check that required services are available
        services = health_response.get("services", {})
        required_services = ["llm", "vector", "db"]

        for service in required_services:
            if service not in services:
                return False, f"Required service '{service}' not available"
            if services[service].get("status") != "healthy":
                return False, f"Service '{service}' is not healthy: {services[service].get('error', 'Unknown error')}"

        return True, None
    except ServiceError as e:
        current_app.logger.error(f"ServiceError during service initialization check: {e}")
        return False, f"Error initializing services: {str(e)}"
    except Exception as e:
        current_app.logger.error(f"Unexpected error during service initialization check: {e}", exc_info=True)
        return False, f"Unexpected error during initialization: {str(e)}"

@health_bp.route('/health')
# Apply rate limit using redis_client from current_app
# Note: The decorator needs modification or a way to access current_app's client
# For now, assuming rate_limit can implicitly access it or is adapted.
# A better approach might be to initialize rate_limit with the app context.
# Let's assume the decorator is adapted to accept the client directly and we pass it:
@rate_limit(lambda: current_app.redis_client, max_calls=10, per_seconds=60) 
def health_check():
    """API endpoint to check service health."""
    # Access logger, api_gateway, redis_client via current_app
    logger = current_app.logger
    api_gateway = current_app.api_gateway
    redis_client = current_app.redis_client
    # Access API_VERSION via current_app config or attribute
    api_version = current_app.config.get('API_VERSION', 'v1') # Assuming it's in config

    if not api_gateway:
        logger.error("API Gateway client not available in health_check.")
        return jsonify({"status": "error", "error_message": "API Gateway not configured"}), 500

    try:
        # Check services health
        services_ok, error_message = initialize_services()

        # Get service details
        llm_capabilities = {}
        vector_status = {}
        db_status = {}
        service_details_error = None
        
        try:
            if services_ok: # Only attempt detail fetch if basic check passed
                 # Get LLM service capabilities
                 llm_capabilities = api_gateway.request("llm", "/capabilities")
                 if "error" in llm_capabilities:
                      logger.warning(f"Error fetching LLM capabilities: {llm_capabilities['error']}")
                      service_details_error = f"LLM: {llm_capabilities['error']}"
                      llm_capabilities = {} # Clear on error
                 
                 # Get vector service status
                 vector_status = api_gateway.request("vector", "/health")
                 if "error" in vector_status:
                      logger.warning(f"Error fetching Vector health: {vector_status['error']}")
                      service_details_error = f"Vector: {vector_status['error']}"
                      vector_status = {} # Clear on error
                      
                 # Get DB service status
                 db_status = api_gateway.request("db", "/health")
                 if "error" in db_status:
                      logger.warning(f"Error fetching DB health: {db_status['error']}")
                      service_details_error = f"DB: {db_status['error']}"
                      db_status = {} # Clear on error
                     
        except ServiceError as e:
            logger.error(f"ServiceError while getting service details: {str(e)}")
            service_details_error = f"Service details unavailable: {str(e)}"
            # Keep services_ok status, but note the details error
        except Exception as e:
            logger.error(f"Unexpected error while getting service details: {str(e)}", exc_info=True)
            service_details_error = f"Unexpected error fetching service details: {str(e)}"

        # Determine overall status and construct response
        final_status = "ok" if services_ok else "error"
        final_error_message = error_message
        if services_ok and service_details_error:
             # If services initialized OK but details failed, reflect this
             # Keep status 'ok' but add note about partial failure? Or change status?
             # Let's keep 'ok' but add the details error message.
             final_error_message = service_details_error 

        response_payload = {
            "status": final_status,
            "error_message": final_error_message,
            "version": api_version,
            "services": {
                "llm": {
                    # Use actual health from initialize_services if available, otherwise default/derive
                    "status": "healthy" if "available_providers" in llm_capabilities else "error", 
                    "providers": llm_capabilities.get("available_providers", []),
                    "default_provider": llm_capabilities.get("default_provider")
                },
                "vector": {
                    "status": "healthy" if vector_status.get("status") == "healthy" else "error",
                    "pinecone_initialized": vector_status.get("pinecone_initialized", False),
                    "embedding_model_loaded": vector_status.get("embedding_model_loaded", False)
                },
                "db": {
                    "status": "healthy" if db_status.get("status") == "healthy" else "error",
                    "connection_pool_active": db_status.get("connection_pool_active", False)
                }
            },
            "cache": {
                "enabled": redis_client is not None,
                "redis_connected": False # Default false
            }
        }
        
        # Check Redis connection separately if client exists
        if redis_client:
            try:
                response_payload["cache"]["redis_connected"] = redis_client.ping()
            except Exception as redis_e:
                 logger.warning(f"Could not ping Redis: {redis_e}")
                 response_payload["cache"]["redis_connected"] = False
                 
        status_code = 200 if final_status == "ok" else 503 # Service Unavailable
        return jsonify(response_payload), status_code

    except Exception as e:
        logger.error(f"Critical error in health check: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error_message": f"Unexpected critical error: {str(e)}"
        }), 500 