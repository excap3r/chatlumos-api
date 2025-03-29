import uuid
import threading
from queue import Queue
from flask import Blueprint, request, jsonify, current_app

# Import rate limiting decorator
from services.utils.api_helpers import rate_limit
# Import the background task
from services.tasks.question_processing import process_question_async
# Import ServiceError for error handling
from services.api_gateway import ServiceError

ask_bp = Blueprint('ask', __name__)

@ask_bp.route('/ask', methods=['POST'])
@rate_limit(lambda: current_app.redis_client, max_calls=20, per_seconds=60)
def ask_question():
    """API endpoint to ask a question and get an answer."""
    logger = current_app.logger
    api_gateway = current_app.api_gateway
    redis_client = current_app.redis_client
    progress_events = current_app.progress_events
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    question = data.get("question")
    stream = data.get("stream", False)  # Check if streaming is requested
    
    if not question:
        return jsonify({"error": "Missing 'question' parameter"}), 400
        
    logger.info(f"Received question: {question[:50]}... (Stream: {stream})")
    
    if stream:
        # --- Asynchronous processing with streaming --- 
        session_id = str(uuid.uuid4())
        progress_events[session_id] = Queue()
        logger.info(f"Starting async processing for question, session: {session_id}")
        
        # Pass required objects to the thread
        thread = threading.Thread(target=process_question_async, args=(
            question, 
            session_id, 
            logger,      # Pass Flask app logger
            progress_events, 
            api_gateway, 
            redis_client
        ))
        thread.daemon = True # Allow app to exit even if background tasks are running
        thread.start()
        
        # Return session ID for client to connect to progress stream
        return jsonify({"session_id": session_id}), 202
    else:
        # --- Synchronous processing (blocking) --- 
        logger.info(f"Starting synchronous processing for question")
        try:
            # Note: Synchronous path does not use progress events/queue
            # This logic might need reimplementation or reuse parts of async logic
            # For now, let's assume a direct call to LLM/Vector services via gateway
            
            # Basic synchronous implementation (example - might need refinement)
            # 1. Decompose (optional, could skip for sync)
            # 2. Search
            search_payload = {"query": question, "top_k": current_app.config.get("DEFAULT_TOP_K", 10)}
            search_result = api_gateway.request("vector", "/search", method="POST", json=search_payload)
            if "error" in search_result:
                raise ServiceError(f"Search failed: {search_result['error']}")
            
            context_results = search_result.get("results", [])
            if not context_results:
                 # Handle case with no results - maybe return specific message?
                 return jsonify({"answer": "No relevant information found.", "context": []})
                 
            context = "\n".join([r.get("text", "") for r in context_results])
            
            # 3. Generate Answer
            answer_payload = {"question": question, "context": context}
            answer_result = api_gateway.request("llm", "/generate_answer", method="POST", json=answer_payload)
            if "error" in answer_result:
                raise ServiceError(f"Answer generation failed: {answer_result['error']}")
            
            answer = answer_result.get("answer", "Could not generate an answer.")
            
            return jsonify({
                "answer": answer,
                "context": context_results # Optionally return context used
            })

        except ServiceError as e:
            logger.error(f"ServiceError during synchronous question processing: {e}")
            return jsonify({"error": f"Failed to process question: {e}"}), 500
        except Exception as e:
            logger.error(f"Unexpected error during synchronous question processing: {e}", exc_info=True)
            return jsonify({"error": "An unexpected error occurred"}), 500 