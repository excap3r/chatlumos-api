import logging
import os
from flask import Blueprint, request, jsonify, current_app

# Import necessary components used by the endpoint
# from services.api_gateway import ServiceError # Remove if gateway removed
from services.analytics.analytics_service import AnalyticsEvent
from services.analytics.analytics_middleware import track_specific_event
from services.utils.api_helpers import rate_limit, cache_result
from services.utils.error_utils import APIError, ValidationError # Added

# Define the Blueprint
question_bp = Blueprint('question_bp', __name__)

# Define constants or get from config if preferred
DEFAULT_MODEL = "all-MiniLM-L6-v2" 
# Alternatively, if set in app config:
# DEFAULT_MODEL = current_app.config.get('DEFAULT_MODEL', "all-MiniLM-L6-v2")

@question_bp.route('/question', methods=['POST'])
@rate_limit(lambda: current_app.redis_client, max_calls=30, per_seconds=60) # Use current_app
@cache_result(lambda: current_app.redis_client, ttl=3600) # Use current_app
@track_specific_event(AnalyticsEvent.QUESTION)
def question():
    """API endpoint to answer a question."""
    data = request.json
    
    if not data or not data.get('question'):
        raise ValidationError("Missing required field: question")
    
    question_text = data.get('question')
    context = data.get('context', [])
    model = data.get('model', DEFAULT_MODEL)
    
    try:
        # Access API Gateway via current_app
        result = current_app.api_gateway.request(
            service="llm",
            path="/answer",
            method="POST",
            json={
                "question": question_text,
                "context": context,
                "model": model
            }
        )
        
        # Check for errors returned by the service via the gateway
        if isinstance(result, dict) and "error" in result:
            error_msg = result.get('message', result['error'])
            status_code = result.get("status_code", 500)
            current_app.logger.error(f"LLM service /answer failed: {error_msg}", status_code=status_code)
            raise APIError(f"Failed to get answer: {error_msg}", status_code=status_code)

        return jsonify(result)
    except ServiceError as e:
        current_app.logger.error(f"ServiceError communicating with LLM /answer: {str(e)}")
        raise APIError(f"Failed to communicate with question answering service: {e}", status_code=getattr(e, 'status_code', 503))
    except Exception as e:
        current_app.logger.error(f"Unexpected error in /question route: {str(e)}", exc_info=True)
        raise APIError(f"An unexpected error occurred processing the question: {str(e)}", status_code=500) 