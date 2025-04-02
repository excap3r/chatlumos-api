import os
import uuid
# import threading # Removed threading
import time
import json
# from queue import Queue # Removed queue
from flask import Blueprint, request, jsonify, current_app, stream_with_context, Response, g
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from typing import Optional
import base64 # Added for upload route

# Import middleware and utilities
# from services.auth_middleware import auth_required # Assuming it can access user context
from services.api.middleware.auth_middleware import require_auth # Corrected import path
from services.analytics.analytics_middleware import track_specific_event
from services.analytics.analytics_service import AnalyticsEvent
from services.tasks.pdf_processing import process_pdf_task # Import the Celery task
# from services.utils.api_helpers import allowed_file, get_cache_key, handle_error # handle_error is used
# Removed import of allowed_file as it is defined locally
# from services.utils.api_helpers import handle_error # Removed unused get_cache_key, allowed_file 
# Removed unused log_request_start/end import
# from services.utils.log_utils import log_request_start, log_request_end 
# Import handle_error from error_utils
from services.utils.error_utils import APIError, ValidationError, handle_error # Added handle_error import
# Import rate_limit from api_helpers
from services.utils.api_helpers import rate_limit 
# from services.decorators import rate_limit # Adjusted relative import
from services.config import AppConfig # Added for REDIS_TASK_TTL
# from services.utils.error_utils import APIError, ValidationError # Already imported above

pdf_bp = Blueprint('pdf', __name__)

# Configuration (Consider moving to Flask app config) - Removed unused constants
# UPLOAD_FOLDER = '/tmp/pdf_uploads' # WARNING: Using /tmp is generally discouraged for persistent storage
ALLOWED_EXTENSIONS = {'pdf'}
# REDIS_TASK_TTL = 86400 # 24 hours in seconds - Use AppConfig.REDIS_TASK_TTL_SECONDS

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Removed Deprecated /process_pdf Route --- 
# @pdf_bp.route('/process_pdf', methods=['POST'])
# @auth_required() # Ensure user is authenticated
# @track_specific_event(AnalyticsEvent.PDF_PROCESSING, include_payload=False) # Avoid logging file content
# def process_pdf_route():
#     ...
# --- End Removed Route --- 

@pdf_bp.route('/pdf/upload', methods=['POST'])
@require_auth() 
@rate_limit(redis_client_provider=lambda: current_app.redis_client, max_calls=5, per_seconds=300) 
def upload_pdf():
    """
    Upload a PDF document for asynchronous processing via Celery.
    
    Request Body (multipart/form-data):
        file: PDF file (required)
        author: Author name (optional, defaults to None)
        title: Document title (optional, defaults to filename)
        language: Document language (optional, defaults to 'en')
        translate: Translate to English? (string 'true'/'false', optional, defaults to false)
        
    Returns:
        JSON with task_id for tracking progress.
    """
    logger = current_app.logger
    redis_client = current_app.redis_client

    if not redis_client:
         logger.error("Redis client not configured for PDF upload route.")
         return handle_error("Service configuration error.", 500)

    # Check file part
    if 'file' not in request.files:
        return handle_error("No file part in the request", 400)
    
    file = request.files['file']
    
    if file.filename == '':
        return handle_error("No file selected for uploading", 400)
    
    # Check if the file is allowed (PDF)
    if not allowed_file(file.filename):
        return handle_error("Invalid file type, only PDF allowed", 400)

    # Check user auth context 
    if not hasattr(g, 'user') or not g.user or 'id' not in g.user:
         logger.error("User context not found in g after @auth_required decorator")
         return handle_error("Authentication context missing or invalid", 500)
    user_id = g.user['id']

    # Generate secure filename and unique task ID
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    redis_key = f"task:{task_id}"
    
    # Read file content directly into memory and encode
    file_content_b64: Optional[str] = None
    try:
        file_content = file.read()
        if not file_content:
            return handle_error("Uploaded file content is empty.", 400)
        file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        logger.info(f"Read and encoded file {filename} for task {task_id}")
    except Exception as read_err:
        logger.error(f"Failed to read or encode uploaded file {filename}: {read_err}")
        return handle_error(f"Failed to read uploaded file: {read_err}", 500)

    # Get other form data for the task
    author_name = request.form.get('author') # Match task arg name
    title = request.form.get('title', filename) # Default title to filename if not provided
    language = request.form.get('language', 'en')
    translate_to_english = request.form.get('translate', 'false').lower() == 'true' # Match task arg name

    initial_state = {
        'status': 'Queued',
        'progress': 0,
        'details': f"PDF '{filename}' received, preparing for processing...", # Updated detail
        'filename': filename,
        'started_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'result': None,
        'error': None
    }
    try:
        # Use HSET to store multiple fields
        redis_client.hset(redis_key, mapping=initial_state)
        redis_client.expire(redis_key, AppConfig.REDIS_TASK_TTL_SECONDS) # Use AppConfig TTL
        logger.info(f"Stored initial state for PDF task {task_id} in Redis.")
    except Exception as e:
        logger.error(f"Failed to store initial PDF task state in Redis for task {task_id}: {e}")
        return handle_error(f"Failed to initiate PDF processing: {e}", 500)

    # Enqueue the background task using Celery
    try:
        # Pass only the necessary arguments to the Celery task
        process_pdf_task.delay(
            task_id=task_id,
            file_content_b64=file_content_b64,
            filename=filename,
            user_id=user_id,
            author_name=author_name,
            title=title,
            language=language,
            translate_to_english=translate_to_english
        )
        logger.info(f"Enqueued PDF processing task {task_id} for file {filename} with Celery.")
    except Exception as e:
        logger.error(f"Failed to enqueue Celery task {task_id}: {e}", exc_info=True)
        # No file to clean up now, just return error
        return handle_error(f"Failed to enqueue PDF processing task: {e}", 500)

    return jsonify({"task_id": task_id, "status": "PDF processing task queued", "filename": filename}), 202 # Updated status message 