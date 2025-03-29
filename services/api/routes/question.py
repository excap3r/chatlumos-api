import logging
import os
from flask import Blueprint, request, jsonify, current_app

# Import necessary components used by the endpoint
from services.api_gateway import ServiceError
from services.analytics.analytics_service import AnalyticsEvent
from services.analytics.analytics_middleware import track_specific_event
from services.utils.api_helpers import rate_limit, cache_result

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
        return jsonify({"error": "No question provided"}), 400
    
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
        
        return jsonify(result)
    except ServiceError as e:
        # Access logging via current_app logger or import logging
        logging.error(f"ServiceError in /question: {str(e)}") 
        return jsonify({"error": str(e)}), e.status_code
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        return jsonify({"error": "Failed to process question"}), 500 