from flask import Blueprint, request, jsonify, current_app

# Import decorators and helpers
from services.utils.api_helpers import rate_limit, cache_result
from services.analytics.analytics_middleware import track_specific_event
from services.analytics.analytics_service import AnalyticsEvent
from services.utils.error_utils import APIError, ValidationError, ServiceError
from services.config import AppConfig

search_bp = Blueprint('search', __name__)

# Define defaults here or get from config
# DEFAULT_INDEX = "wisdom-embeddings" 
# DEFAULT_TOP_K = 10

@search_bp.route('/search', methods=['POST'])
@rate_limit(lambda: current_app.redis_client, max_calls=60, per_seconds=60)
@cache_result(lambda: current_app.redis_client, ttl=3600)
@track_specific_event(AnalyticsEvent.SEARCH)
def search_route():
    """
    Search for relevant information using vector search via API Gateway.
    
    Request Body:
        query: Search query (string, required)
        index_name: Optional vector index name (string)
        top_k: Optional number of results to return (integer, >0)
        metadata_filter: Optional dictionary for filtering based on metadata
        
    Returns:
        Search results
    """
    logger = current_app.logger
    api_gateway = current_app.api_gateway
    
    # Get request data
    data = request.get_json()
    
    # Validate request data
    if not data:
        raise ValidationError("Request body is required and must be JSON")
    
    query = data.get("query")
    if not query or not isinstance(query, str):
        raise ValidationError("'query' field is required and must be a string")
    
    # --- Get optional parameters and validate --- 
    index_name = data.get("index_name", AppConfig.PINECONE_INDEX_NAME) # Use AppConfig default
    
    try:
        top_k_raw = data.get("top_k")
        if top_k_raw is not None:
            top_k = int(top_k_raw)
            if top_k <= 0:
                raise ValidationError("'top_k' must be a positive integer")
        else:
            top_k = AppConfig.DEFAULT_TOP_K # Use AppConfig default
    except (ValueError, TypeError):
        raise ValidationError("'top_k' must be a valid integer")

    metadata_filter = data.get("metadata_filter") # Optional filter
    if metadata_filter is not None and not isinstance(metadata_filter, dict):
         raise ValidationError("'metadata_filter' must be a dictionary (object) if provided")
    # --- End validation --- 
    
    logger.info(f"Received search request: '{query[:50]}...' (top_k={top_k}, index={index_name})")
    
    if not api_gateway:
        logger.error("API Gateway service not available for search route.")
        raise APIError("Search service temporarily unavailable.", 503)
        
    try:
        # Prepare payload for the vector service search endpoint
        search_payload = {
            "query": query,
            "index_name": index_name,
            "top_k": top_k
        }
        if metadata_filter:
             search_payload["metadata_filter"] = metadata_filter
             
        # Use vector service via API Gateway to search
        results = api_gateway.request(
             "vector",
             "/search",
             method="POST",
             json=search_payload
        )
        
        # Check for errors returned *within* the successful response from the vector service
        # Ideally, the gateway should translate these into HTTP errors, but handle defensively
        if isinstance(results, dict) and "error" in results:
            error_msg = results.get('message', results['error'])
            status_code = 500 # Default internal error
            try:
                 # Try to get status code from response, but default to 500
                 status_code_raw = results.get("status_code")
                 if status_code_raw is not None:
                      status_code = int(status_code_raw)
            except (ValueError, TypeError):
                 logger.warning(f"Invalid status_code '{status_code_raw}' in error response body, using 500.")
            
            logger.warning(f"Vector service search returned an error in response body: {error_msg}", status_code=status_code)
            # Raise APIError based on the service's response - This pattern is fragile.
            raise APIError(f"Search service failed: {error_msg}", status_code=status_code)

        # Success
        # Ensure result is serializable (usually gateway handles this, but double-check)
        try:
            response = jsonify(results)
            response.status_code = 200
            logger.info(f"Search successful, found {len(results.get('results', []))} results.")
            return response
        except (TypeError, ValueError) as json_err:
             logger.error(f"Failed to serialize search results from vector service: {json_err}", results_type=type(results).__name__)
             raise APIError("Search service returned unserializable data.", 500)
    
    except ServiceError as e:
        # Errors during communication *with* the gateway/service
        logger.error(f"ServiceError during search communication: {str(e)}")
        # Raise APIError, using status code from ServiceError if available
        raise APIError(f"Failed to communicate with search service: {e}", status_code=getattr(e, 'status_code', 503))
    except Exception as e:
        # Catchall for unexpected errors in this route handler
        logger.error(f"Unexpected error during search route: {str(e)}", exc_info=True)
        raise APIError(f"An unexpected error occurred during search.", status_code=500) 