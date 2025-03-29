from flask import Blueprint, request, jsonify, current_app

# Import decorators and helpers
from services.utils.api_helpers import rate_limit, cache_result
from services.analytics.analytics_middleware import track_specific_event
from services.analytics.analytics_service import AnalyticsEvent
from services.api_gateway import ServiceError

search_bp = Blueprint('search', __name__)

# Define defaults here or get from config
DEFAULT_INDEX = "wisdom-embeddings" 
DEFAULT_TOP_K = 10

@search_bp.route('/search', methods=['POST'])
@rate_limit(lambda: current_app.redis_client, max_calls=60, per_seconds=60)
@cache_result(lambda: current_app.redis_client, ttl=3600)
@track_specific_event(AnalyticsEvent.SEARCH)
def search_route():
    """
    Search for relevant information using vector search via API Gateway.
    
    Request Body:
        query: Search query
        index_name: Optional vector index name
        top_k: Optional number of results to return
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
        return jsonify({
            "error": "Invalid request",
            "message": "Request body is required and must be JSON"
        }), 400
    
    query = data.get("query")
    if not query:
        return jsonify({
            "error": "Invalid request",
            "message": "'query' field is required"
        }), 400
    
    # Get optional parameters from request or use defaults/config
    index_name = data.get("index_name", current_app.config.get('DEFAULT_INDEX', DEFAULT_INDEX))
    top_k = data.get("top_k", current_app.config.get('DEFAULT_TOP_K', DEFAULT_TOP_K))
    metadata_filter = data.get("metadata_filter") # Optional filter
    
    logger.info(f"Received search request: '{query[:50]}...' (top_k={top_k}, index={index_name})")
    
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
        # Assuming gateway has a dedicated search method or uses generic request
        # results = api_gateway.search(search_payload)
        # OR using generic request:
        results = api_gateway.request(
             "vector",
             "/search",
             method="POST",
             json=search_payload
        )
        
        if "error" in results:
             # If vector service returns an error in the JSON body
             logger.error(f"Vector service search error: {results.get('message', results['error'])}")
             # Map to appropriate HTTP status if possible, else 500/503
             status_code = results.get("status_code", 500) 
             return jsonify(results), status_code
             
        # Success
        logger.info(f"Search successful, found {len(results.get('results', []))} results.")
        return jsonify(results), 200
    
    except ServiceError as e:
        # Errors during communication with the gateway/service
        logger.error(f"ServiceError during search: {str(e)}")
        return jsonify({
            "error": "Search service error",
            "message": str(e)
        }), getattr(e, 'status_code', 503) # Use status from error or default to 503
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Unexpected server error",
            "message": "An unexpected error occurred during search"
        }), 500 