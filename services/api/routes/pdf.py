import os
import uuid
import threading
import time
from queue import Queue
from flask import Blueprint, request, jsonify, current_app, stream_with_context, Response
from werkzeug.utils import secure_filename

# Import middleware and utilities
# from services.auth_middleware import auth_required # Assuming it can access user context
from services.api.middleware.auth_middleware import auth_required # Corrected import path
from services.analytics.analytics_middleware import track_specific_event
from services.analytics.analytics_service import AnalyticsEvent, track_event
from services.tasks.pdf_processing import process_pdf_async
from services.utils.api_helpers import allowed_file, get_cache_key
from services.utils.log_utils import log_request_start, log_request_end

pdf_bp = Blueprint('pdf', __name__)

# Define upload folder - consider making this configurable
UPLOAD_FOLDER = "/tmp"
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@pdf_bp.route('/process_pdf', methods=['POST'])
@auth_required() # Ensure user is authenticated
@track_specific_event(AnalyticsEvent.PDF_PROCESSING, include_payload=False) # Avoid logging file content
def process_pdf_route():
    """
    Upload and process a PDF document asynchronously.
    
    Request Body (multipart/form-data):
        file: PDF file
        author_name: Name of the document author
        title: Optional document title (defaults to filename)
        language: Optional document language (defaults to 'en')
        translate_to_english: Optional boolean (form string 'true'/'false')
        
    Returns:
        Session ID for tracking progress via SSE.
    """
    logger = current_app.logger
    progress_events = current_app.progress_events
    api_gateway = current_app.api_gateway
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser might submit an empty file
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    # Check if the file is allowed (PDF)
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type, only PDF allowed"}), 400
    
    # Get form data
    author_name = request.form.get('author_name')
    if not author_name:
        return jsonify({"error": "Author name is required"}), 400
    
    # Use filename as default title if not provided
    title = request.form.get('title', file.filename.rsplit('.', 1)[0])
    language = request.form.get('language', 'en')
    # Convert form string 'true'/'false' to boolean
    translate_to_english_str = request.form.get('translate_to_english', 'true')
    translate_to_english = translate_to_english_str.lower() == 'true'
    
    # Get user ID from auth context (assuming auth_required adds it)
    user = getattr(request, 'user', None)
    if not user or "id" not in user:
         logger.error("User context not found after @auth_required decorator")
         return jsonify({"error": "Authentication context missing"}), 500
    user_id = user["id"]
    
    try:
        # Create a unique session ID for tracking progress
        session_id = str(uuid.uuid4())
        
        # Create the upload folder if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save the file securely to a temporary location
        # Consider using Werkzeug's secure_filename if needed, though UUID prefix helps
        temp_file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{file.filename}")
        file.save(temp_file_path)
        logger.info(f"PDF saved temporarily to {temp_file_path} for session {session_id}")
        
        # Initialize progress tracking queue for this session
        progress_events[session_id] = Queue()
        
        # Start the background processing task
        thread = threading.Thread(target=process_pdf_async, args=(
            temp_file_path, author_name, title, language, translate_to_english, 
            session_id, user_id, logger, progress_events, api_gateway
        ))
        thread.daemon = True # Allow main app to exit
        thread.start()
        logger.info(f"Started background PDF processing thread for session {session_id}")
        
        # Start a separate thread to clean up the progress queue item after a delay
        # This prevents the queue from growing indefinitely if client disconnects
        def cleanup_progress_queue_item():
             # Wait longer than the SSE timeout + processing time estimate
             time.sleep(900) # 15 minutes delay
             if session_id in progress_events:
                  try:
                       # Check if queue still exists (might be removed by SSE finally block)
                       if session_id in progress_events:
                            del progress_events[session_id]
                            logger.info(f"Cleaned up progress_events entry for session {session_id} after delay.")
                  except KeyError:
                       pass # Already removed, likely by SSE handler
                  except Exception as e:
                       logger.error(f"Error during delayed cleanup of progress_events for session {session_id}: {e}")

        cleanup_thread = threading.Thread(target=cleanup_progress_queue_item)
        cleanup_thread.daemon = True
        cleanup_thread.start()

        # Return session ID for client to track progress
        api_version = current_app.config.get('API_VERSION', 'v1')
        return jsonify({
            "message": "PDF processing started successfully.",
            "session_id": session_id,
            "status": "processing",
            "progress_stream_url": f"/api/{api_version}/progress-stream/{session_id}"
        }), 202 # Accepted
        
    except Exception as e:
        logger.error(f"Error initiating PDF processing: {str(e)}", exc_info=True)
        # Attempt to clean up temp file if created before error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
             try:
                  os.remove(temp_file_path)
             except Exception as cleanup_e:
                  logger.error(f"Failed to cleanup temp file {temp_file_path} after error: {cleanup_e}")
        return jsonify({
            "error": "Server error",
            "message": f"Failed to start PDF processing: {str(e)}"
        }), 500 