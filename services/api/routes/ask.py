import uuid
import time
import json
from queue import Queue
from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context, g
from datetime import datetime, timedelta

# Import rate limiting decorator
from services.utils.api_helpers import rate_limit
# Import the background task
from services.tasks.question_processing import process_question_task
# Import ServiceError for error handling
from services.api_gateway import ServiceError

# Import auth_required decorator
from services.decorators import auth_required
# Import handle_error, get_request_data, validate_input, log_api_call
from services.utils.api_helpers import (
    handle_error, get_request_data, validate_input, log_api_call
)
# Import process_question_task Celery task
from services.tasks.question_processing import process_question_task

# Import APIError for error handling
from services.utils.api_helpers import APIError
from services.utils.error_utils import APIError, ValidationError
from services.config import AppConfig

ask_bp = Blueprint('ask', __name__)

# Default values
# DEFAULT_TOP_K = 10 # Removed, logic moved to Celery task
DEFAULT_INDEX_NAME = "wisdom-embeddings"
REDIS_TASK_TTL = 86400 # 24 hours in seconds

@ask_bp.route('/ask', methods=['POST'])
@auth_required(level='user')
@rate_limit(limit=10, per=timedelta(minutes=1)) # Example rate limit
def ask_question():
    """API endpoint to ask a question and get an answer via background task."""
    logger = current_app.logger
    # api_gateway = current_app.api_gateway # No longer used directly here
    redis_client = current_app.redis_client
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    question = data.get("question")
    # stream = data.get("stream", False) # Stream parameter is now ignored, always async
    
    if not question:
        return jsonify({"error": "Missing 'question' parameter"}), 400
        
    logger.info(f"Received question: {question[:50]}... (Processing asynchronously)")
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Get user_id from authenticated context
    if not hasattr(g, 'user') or not g.user or 'id' not in g.user:
        logger.error("User context not found in g after @auth_required decorator in /ask")
        raise APIError("Authentication context missing or invalid", 500)
    user_id = g.user['id']

    # --- Always process asynchronously --- 
    
    # Store initial task state in Redis
    if not redis_client:
        # Use APIError for consistency with other error handling
        # return handle_error("Redis client not configured for streaming", 500)
        logger.error("Redis client not configured for async task processing.")
        raise APIError("Server configuration error: Cannot initiate task.", 500)
    
    redis_key = f"task:{task_id}"
    initial_state = {
        'status': 'Queued',
        'progress': 0,
        'details': 'Task received, waiting for processing',
        'started_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'result': None, # Placeholder for final result
        'user_id': user_id, # Store the user ID
        'error': None # Placeholder for error info
    }
    try:
        # Use HSET to store multiple fields
        redis_client.hset(redis_key, mapping=initial_state)
        redis_client.expire(redis_key, REDIS_TASK_TTL) # Set TTL
        logger.info(f"Stored initial state for task {task_id} in Redis.")
    except Exception as e:
        logger.error(f"Failed to store initial task state in Redis for task {task_id}: {e}")
        # Use APIError
        # return handle_error(f"Failed to initiate task processing: {e}", 500)
        raise APIError(f"Failed to initiate task processing: {e}", 500)

    # Enqueue the background task using Celery
    try:
        # Pass the whole data dict, task will extract needed params
        process_question_task.delay(task_id, data)
        logger.info(f"Enqueued question processing task {task_id} with Celery.")
    except Exception as e:
        logger.error(f"Failed to enqueue Celery task {task_id}: {e}", exc_info=True)
        # Attempt to clean up Redis state if enqueuing fails? Maybe not, user can retry.
        # For now, just return error.
        # Use APIError
        # return handle_error(f"Failed to enqueue processing task: {e}", 500)
        raise APIError(f"Failed to enqueue processing task: {e}", 500)
    
    # Return task ID for client to poll progress endpoint
import uuid
import time
import json
from queue import Queue
from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context, g
from datetime import datetime, timedelta

# Import rate limiting decorator
from services.utils.api_helpers import rate_limit
# Import the background task
from services.tasks.question_processing import process_question_async
# Import ServiceError for error handling
from services.api_gateway import ServiceError

# Import auth_required decorator
from services.decorators import auth_required
# Import handle_error, get_request_data, validate_input, log_api_call
from services.utils.api_helpers import (
    handle_error, get_request_data, validate_input, log_api_call
)
# Import process_question_task Celery task
from services.tasks.question_processing import process_question_task

# Import APIError for error handling
from services.utils.api_helpers import APIError

ask_bp = Blueprint('ask', __name__)

# Default values
# DEFAULT_TOP_K = 10 # Removed, logic moved to Celery task
DEFAULT_INDEX_NAME = "wisdom-embeddings"
REDIS_TASK_TTL = 86400 # 24 hours in seconds

@ask_bp.route('/ask', methods=['POST'])
@auth_required(level='user')
@rate_limit(limit=10, per=timedelta(minutes=1)) # Example rate limit
def ask_question():
    """API endpoint to ask a question and get an answer via background task."""
    logger = current_app.logger
    # api_gateway = current_app.api_gateway # No longer used directly here
    redis_client = current_app.redis_client
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    question = data.get("question")
    # stream = data.get("stream", False) # Stream parameter is now ignored, always async
    
    if not question:
        return jsonify({"error": "Missing 'question' parameter"}), 400
        
    logger.info(f"Received question: {question[:50]}... (Processing asynchronously)")
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Get user_id from authenticated context
    if not hasattr(g, 'user') or not g.user or 'id' not in g.user:
        logger.error("User context not found in g after @auth_required decorator in /ask")
        raise APIError("Authentication context missing or invalid", 500)
    user_id = g.user['id']

    # --- Always process asynchronously --- 
    
    # Store initial task state in Redis
    if not redis_client:
        # Use APIError for consistency with other error handling
        # return handle_error("Redis client not configured for streaming", 500)
        logger.error("Redis client not configured for async task processing.")
        raise APIError("Server configuration error: Cannot initiate task.", 500)
    
    redis_key = f"task:{task_id}"
    initial_state = {
        'status': 'Queued',
        'progress': 0,
        'details': 'Task received, waiting for processing',
        'started_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'result': None, # Placeholder for final result
        'user_id': user_id, # Store the user ID
        'error': None # Placeholder for error info
    }
    try:
        # Use HSET to store multiple fields
        redis_client.hset(redis_key, mapping=initial_state)
        redis_client.expire(redis_key, REDIS_TASK_TTL) # Set TTL
        logger.info(f"Stored initial state for task {task_id} in Redis.")
    except Exception as e:
        logger.error(f"Failed to store initial task state in Redis for task {task_id}: {e}")
        # Use APIError
        # return handle_error(f"Failed to initiate task processing: {e}", 500)
        raise APIError(f"Failed to initiate task processing: {e}", 500)

    # Enqueue the background task using Celery
    try:
        # Pass the whole data dict, task will extract needed params
        process_question_task.delay(task_id, data)
        logger.info(f"Enqueued question processing task {task_id} with Celery.")
    except Exception as e:
        logger.error(f"Failed to enqueue Celery task {task_id}: {e}", exc_info=True)
        # Attempt to clean up Redis state if enqueuing fails? Maybe not, user can retry.
        # For now, just return error.
        # Use APIError
        # return handle_error(f"Failed to enqueue processing task: {e}", 500)
        raise APIError(f"Failed to enqueue processing task: {e}", 500)
    
    # Return task ID for client to poll progress endpoint
    return jsonify({"task_id": task_id, "status": "Processing started"}), 202
    # --- Synchronous processing block removed --- 

# Ensure the APIError exception handler is registered appropriately in app.py
# (Assuming handle_error or a similar mechanism does this) 