import logging
from flask import Blueprint, jsonify, current_app
from typing import Dict, Any, Tuple, Optional # Added for type hinting
from services.api_gateway import ServiceError # Import real ServiceError

# Assuming rate_limit is defined in api_helpers
from services.utils.api_helpers import rate_limit
from services.utils.error_utils import APIError # Added

import structlog # Assuming structlog is used based on previous context

health_bp = Blueprint('health', __name__)

# Moved logger initialization here to be available to helpers
logger = structlog.get_logger(__name__)

# --- Helper Functions for Health Check Logic ---

def _get_service_details(api_gateway) -> Tuple[Dict[str, Any], Optional[str]]:
    """Fetches detailed status from LLM, Vector, and DB services via gateway."""
    service_details = {}
    errors = []

    for service_name in ["llm", "vector", "db"]:
        capabilities = None
        health_status = None
        error_message = None
        try:
            # Use a method like 'request' or a specific health check method if available
            # Assuming a generic 'request' method for demonstration
            health_status = api_gateway.request(service=service_name, path="/health", method="GET", timeout=5)
            # Attempt to get capabilities if health is okay (adjust path/method as needed)
            if health_status and health_status.get("status") == "healthy":
                 # Assuming capabilities endpoint exists
                 capabilities = api_gateway.request(service=service_name, path="/capabilities", method="GET", timeout=5)

        except ServiceError as e:
            # Catch specific ServiceError from the gateway
            error_message = f"Error contacting {service_name}: {e.message}"
            logger.warning(error_message, service=service_name, error_details=str(e))
            errors.append(error_message)
            # Clear results for this service on error
            health_status = None
            capabilities = None
        except Exception as e:
            # Catch any other unexpected errors during the request
            error_message = f"Unexpected error checking {service_name}: {str(e)}"
            logger.error(error_message, service=service_name, exc_info=True)
            errors.append(error_message)
            # Clear results for this service on error
            health_status = None
            capabilities = None

        service_details[service_name] = {
            "health": health_status,
            "capabilities": capabilities,
            "error": error_message # Include error message per service
        }

    combined_errors = "; ".join(errors) if errors else None
    return service_details, combined_errors

def _check_redis_status(redis_client) -> Dict[str, Any]:
    """Checks Redis connection status."""
    try:
        if redis_client and hasattr(redis_client, 'ping') and redis_client.ping():
            return {"status": "healthy"}
        else:
            logger.warning("Redis client not available or ping failed.")
            return {"status": "unhealthy", "error": "Redis connection failed or client not configured."}
    except Exception as e:
        logger.error(f"Error checking Redis status: {str(e)}", exc_info=True)
        return {"status": "unhealthy", "error": str(e)}

def _build_health_response(
    service_details: Dict[str, Any],
    service_details_error: Optional[str],
    redis_status: Dict[str, Any],
    api_version: str
) -> Dict[str, Any]:
    """Constructs the final health check response payload."""

    # Determine overall status based ONLY on whether critical services passed initial check
    # The `initialize_services` function handles raising 503 if critical services fail.
    # If we reach here, the basic check passed.
    # Detailed errors only affect the sub-status in the response.
    final_status = "ok" # Indicates the health endpoint itself executed successfully

    # Use details fetched or mark as unknown/error if fetching failed
    llm_details = service_details.get('llm', {})
    vector_details = service_details.get('vector', {})
    db_details = service_details.get('db', {})

    # --- Status Derivation Logic Explanation ---
    # If service_details_error is set, it means fetching details failed for AT LEAST ONE service.
    # In this case, any service not confirmed as "healthy" is marked "unknown".
    # If service_details_error is NOT set, any service not confirmed as "healthy" is marked "error".
    # This distinguishes between inability to check ("unknown") and a confirmed problem ("error").
    # --- End Status Derivation Logic Explanation ---

    llm_status = "healthy" if not service_details_error and llm_details.get("health") and llm_details["health"].get("status") == "healthy" else ("unknown" if service_details_error else "error")
    vector_status = "healthy" if not service_details_error and vector_details.get("health") and vector_details["health"].get("status") == "healthy" else ("unknown" if service_details_error else "error")
    db_status = "healthy" if not service_details_error and db_details.get("health") and db_details["health"].get("status") == "healthy" else ("unknown" if service_details_error else "error")

    response_payload = {
        "status": final_status,
        "error_message": service_details_error, # Report detail fetch error if any
        "version": api_version,
        "services": {
            "llm": {
                "status": llm_status,
                "providers": llm_details.get("capabilities", {}).get("available_providers", []),
                "default_provider": llm_details.get("capabilities", {}).get("default_provider")
            },
            "vector": {
                "status": vector_status,
                "pinecone_initialized": vector_details.get("capabilities", {}).get("pinecone_initialized", False),
                "embedding_model_loaded": vector_details.get("capabilities", {}).get("embedding_model_loaded", False)
            },
            "db": {
                "status": db_status,
                "connection_pool_active": db_details.get("health", {}).get("connection_pool_active", False)
            }
        },
        "redis": redis_status # Renamed from 'cache' for clarity
    }
    return response_payload

# --- End Helper Functions ---

def initialize_services():
    """Check if all required services are available and healthy via API Gateway.
       Raises APIError if critical components are missing or unhealthy based on gateway health report.
    """
    # Access api_gateway via current_app
    if not hasattr(current_app, 'api_gateway') or not current_app.api_gateway:
        logger.critical("API Gateway client not initialized on app context.")
        # This is a critical configuration error
        raise APIError("Internal configuration error: API Gateway client not available.", status_code=500)

    # Check health of all services through API gateway
    # The api_gateway.health() method handles connection errors internally
    # and returns a dictionary indicating status, including gateway connection errors.
    health_response = current_app.api_gateway.health()

    # Check if the gateway itself reported an error connecting
    if health_response.get("status") == "error":
        gateway_error = health_response.get("error", "Unknown gateway connection error")
        logger.error(f"API Gateway health check failed: {gateway_error}")
        # Treat gateway connection failure as critical
        raise APIError(f"Service health check failed: Cannot reach API Gateway. Details: {gateway_error}", status_code=503)

    # Check that required services are available and healthy in the response
    services = health_response.get("services", {})
    required_services = ["llm", "vector", "db"]
    unhealthy_details = []

    for service in required_services:
        if service not in services:
            msg = f"Required service '{service}' not available in gateway health response."
            logger.error(msg)
            unhealthy_details.append(msg)
            continue # Check other services

        # Service might be present but status details missing or different structure
        service_status_info = services.get(service) # Get the whole info dict for the service
        if not isinstance(service_status_info, dict):
             msg = f"Invalid status format for service '{service}' in gateway health response."
             logger.error(msg, service_info=service_status_info)
             unhealthy_details.append(msg)
             continue

        service_status = service_status_info.get("status") # Now safely get status
        if service_status != "healthy":
            # Use 'details' if available, otherwise default message or raw status info
            error_detail = service_status_info.get('error', service_status_info.get('details', f'Reported status: {service_status}'))
            msg = f"Service '{service}' is not healthy: {error_detail}"
            logger.error(msg)
            unhealthy_details.append(msg)

    if unhealthy_details:
        # Combine error messages and raise a single error
        combined_error = "; ".join(unhealthy_details)
        # Raise 503 Service Unavailable as dependencies are not met
        raise APIError(f"Service health check failed: {combined_error}", status_code=503)

    # If loop completes without raising, all required services are reported healthy by gateway
    logger.debug("All required services reported healthy by gateway.")

@health_bp.route('/health')
@rate_limit(lambda: current_app.redis_client, max_calls=10, per_seconds=60)
def health_check():
    """API endpoint to check service health."""
    # Access logger, api_gateway, redis_client via current_app
    api_gateway = current_app.api_gateway
    redis_client = current_app.redis_client
    # Access API_VERSION via current_app config or attribute
    api_version = current_app.config.get('API_VERSION', 'v1') # Assuming it's in config

    try:
        # 1. Perform basic health check (raises APIError on critical failure)
        initialize_services()
        # 2. Get detailed status from backend services (best effort)
        service_details, service_details_error = _get_service_details(api_gateway)
        # 3. Check Redis status
        redis_status = _check_redis_status(redis_client)
        # 4. Build final response payload
        response_payload = _build_health_response(
            service_details=service_details,
            service_details_error=service_details_error,
            redis_status=redis_status,
            api_version=api_version
        )

        # Health check should return 200 if the API itself is running,
        # even if backend services have issues (indicated in the payload).
        # Exceptions raised by initialize_services will be caught by global handler (503/500).
        return jsonify(response_payload), 200

    except APIError as e:
         # Log APIErrors specifically if needed, otherwise rely on global handler
         logger.error(f"APIError during health check: {e.message}", status_code=e.status_code, exc_info=False) # Log less verbosely for APIError
         # Re-raise to be caught by Flask's error handling or a global handler
         raise e
    except Exception as e:
        logger.critical(f"Critical unexpected error in health check: {str(e)}", exc_info=True)
        # Return a generic 500 for truly unexpected issues
        return jsonify({
            "status": "error",
            "error_message": f"Unexpected critical error during health check."
        }), 500