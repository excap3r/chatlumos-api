#!/usr/bin/env python3
"""
Wisdom Web App - Web Interface for Semantic Search with Vector Database

This Flask-based web application provides a chatbot interface for asking questions,
visualizing vector database searches, and displaying answers in a clean UI.
"""

import os
import traceback
import time
import json
import uuid
import threading
from queue import Queue
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import numpy as np
from dotenv import load_dotenv

# Import wisdom_qa functionality
try:
    from wisdom_qa import (
        decompose_question,
        init_pinecone,
        load_embedding_model,
        batch_search_pinecone,
        generate_answer,
        query_deepseek
    )
    wisdom_qa_imported = True
except ImportError as e:
    print(f"Error importing wisdom_qa: {e}")
    wisdom_qa_imported = False

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INDEX = "wisdom-embeddings"
DEFAULT_TOP_K = 10

# Progress event storage - maps session IDs to event queues
progress_events = {}

# Initialize services
pinecone_initialized = False
model_loaded = False
initialization_error = None

def initialize_services():
    """Initialize Pinecone and embedding model."""
    global pinecone_initialized, model_loaded, initialization_error
    
    if not wisdom_qa_imported:
        initialization_error = "wisdom_qa module could not be imported"
        return False
    
    try:
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if pinecone_api_key and not pinecone_initialized:
            try:
                pinecone_initialized = init_pinecone(pinecone_api_key)
                if not pinecone_initialized:
                    initialization_error = "Failed to initialize Pinecone"
            except Exception as e:
                initialization_error = f"Exception initializing Pinecone: {str(e)}"
                pinecone_initialized = False
        
        # Load embedding model
        if not model_loaded:
            try:
                model_loaded = load_embedding_model(DEFAULT_MODEL)
                if not model_loaded:
                    initialization_error = "Failed to load embedding model"
            except Exception as e:
                initialization_error = f"Exception loading embedding model: {str(e)}"
                model_loaded = False
        
        return pinecone_initialized and model_loaded
    except Exception as e:
        initialization_error = f"Unexpected error during initialization: {str(e)}"
        return False

def translate_text(text, target_lang="en", source_lang=None):
    """
    Translate text using DeepSeek API with improved timeout handling.
    
    Args:
        text: Text to translate
        target_lang: Target language code
        source_lang: Source language code (if known)
    
    Returns:
        Dictionary with translation or error
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return {"error": "DeepSeek API key not found"}
    
    source_lang_prompt = f" from {source_lang}" if source_lang else ""
    
    system_prompt = f"You are a professional translator. Translate the user's text{source_lang_prompt} to {target_lang}. Return only the translated text without explanations or notes."
    
    # Use improved parameters for better reliability
    # - Increased timeout to handle longer texts
    # - More retries for better reliability
    response = query_deepseek(
        text, 
        system_prompt, 
        api_key, 
        max_tokens=1500,  # Increased for longer texts
        timeout=60,       # Increased timeout for full text
        retries=4,        # More retries
        task_type="translation"
    )
    
    if "error" in response:
        return {"error": response["error"]}
    
    return {"translated_text": response["content"]}

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """API endpoint to check service health."""
    try:
        services_ok = initialize_services()
        
        # Only return serializable data
        deepseek_api_available = os.getenv("DEEPSEEK_API_KEY") is not None
        
        return jsonify({
            "status": "ok" if services_ok else "error",
            "pinecone_initialized": pinecone_initialized,
            "embedding_model_loaded": model_loaded,
            "deepseek_api_available": deepseek_api_available,
            "error_message": initialization_error if initialization_error else None
        })
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in health check: {str(e)}\n{error_traceback}")
        return jsonify({
            "status": "error",
            "error_message": str(e)
        }), 500

@app.route('/api/progress-stream/<session_id>')
def progress_stream(session_id):
    """
    Server-Sent Events endpoint to stream question processing progress.
    
    This endpoint streams real-time updates about the processing stages
    of a question being answered.
    """
    if session_id not in progress_events:
        return jsonify({"error": "Invalid or expired session ID"}), 404
    
    def generate_events():
        event_queue = progress_events[session_id]
        try:
            # Send initial event
            yield 'data: {"event": "connected", "message": "Connected to progress stream"}\n\n'
            
            # Listen for events until completion or timeout
            timeout = time.time() + 300  # 5 minute timeout
            while time.time() < timeout:
                try:
                    # Non-blocking get with timeout
                    event = event_queue.get(timeout=1.0)
                    if event is None:  # None signals end of stream
                        yield 'data: {"event": "complete", "message": "Processing complete"}\n\n'
                        break
                    
                    if isinstance(event, dict):
                        event_json = json.dumps(event)
                        yield f'data: {event_json}\n\n'
                except Exception as e:
                    # Queue.Empty exception or other error, just continue
                    continue
                    
            # Send end event
            yield 'data: {"event": "end", "message": "Stream ended"}\n\n'
            
            # Clean up
            if session_id in progress_events:
                del progress_events[session_id]
                
        except GeneratorExit:
            # Client disconnected
            if session_id in progress_events:
                del progress_events[session_id]
    
    return Response(stream_with_context(generate_events()),
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'X-Accel-Buffering': 'no',  # For Nginx
                       'Connection': 'keep-alive'
                   })

def process_question_async(question, session_id):
    """
    Process a question asynchronously, sending progress updates.
    
    Args:
        question: The question to process
        session_id: The session ID for tracking progress
    """
    event_queue = progress_events.get(session_id)
    if not event_queue:
        print(f"Error: No event queue found for session {session_id}")
        return
    
    try:
        # Initialize process steps - removing language detection and separate translation steps
        process_steps = [
            {"id": "decomposition", "name": "Question Analysis & Translation", "status": "pending", "details": None},
            {"id": "search", "name": "Vector Search", "status": "pending", "details": None},
            {"id": "answer", "name": "Answer Generation", "status": "pending", "details": None}
        ]
        
        # Send initial process steps
        print(f"[Session {session_id}] Initializing process steps")
        event_queue.put({
            "event": "process_init",
            "process_steps": process_steps,
            "progress": 0
        })
        
        # Calculate progress increment per step
        progress_increment = 100.0 / len(process_steps)
        current_progress = 0
        
        # Track result data
        result_data = {
            "status": "processing"
        }
        
        # Store the original question for reference
        original_question = question
        
        # Process question for decomposition and translation
        event_queue.put({
            "event": "step_update",
            "step_id": "decomposition",
            "status": "processing"
        })
        
        # First identify language through decomposition
        # Passing empty string for language to let decomposition function detect it
        decomposition = decompose_question(question, language="")
        result_data["decomposition"] = decomposition
        
        # Get translated question and detected language
        search_question = decomposition.get("translated_question", question)
        detected_language = decomposition.get("detected_language", "en")
        result_data["original_language"] = detected_language
        
        if detected_language != "en" and search_question != question:
            print(f"Using translated question from decomposition: {search_question}")
        
        event_queue.put({
            "event": "step_update",
            "step_id": "decomposition",
            "status": "completed",
            "details": decomposition
        })
        current_progress += progress_increment
        event_queue.put({"event": "progress", "value": current_progress})
        
        # Prepare search queries - use translated question for search
        all_queries = []
        all_queries.append(search_question)  # The translated question for better search
        all_queries.extend(decomposition["sub_questions"])
        all_queries.extend(decomposition["concepts"])  # Now concepts are already simple strings
        
        # Add body parts to search queries if present
        if "body_parts" in decomposition and decomposition["body_parts"]:
            # Add specific body part queries
            for part in decomposition["body_parts"]:
                all_queries.append(part)  # Add the body part as a direct search term
        
        all_queries.extend(decomposition["search_queries"])
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in all_queries:
            if isinstance(q, str) and q and q not in seen:
                unique_queries.append(q)
                seen.add(q)
        
        # Search for all queries
        event_queue.put({
            "event": "step_update",
            "step_id": "search",
            "status": "processing"
        })
        
        search_results = batch_search_pinecone(unique_queries, DEFAULT_INDEX, DEFAULT_TOP_K)
        
        event_queue.put({
            "event": "step_update",
            "step_id": "search",
            "status": "completed",
            "details": {"query_count": len(unique_queries), "result_count": sum(len(results) for results in search_results.values())}
        })
        current_progress += progress_increment
        event_queue.put({"event": "progress", "value": current_progress})
        
        # Generate answer
        event_queue.put({
            "event": "step_update",
            "step_id": "answer",
            "status": "processing"
        })
        
        # Generate answer with the original question and include language info
        answer = generate_answer(
            original_question, 
            search_results,
            ensure_relevance=True,
            target_language=detected_language  # Pass target language to generate answer in user's language
        )
        result_data["answer"] = answer
        
        event_queue.put({
            "event": "step_update",
            "step_id": "answer",
            "status": "completed",
            "details": {"length": len(answer)}
        })
        current_progress += progress_increment
        event_queue.put({"event": "progress", "value": current_progress})
        
        # Set to 100% when complete
        current_progress = 100
        event_queue.put({"event": "progress", "value": current_progress})
        
        # Process search results for visualization
        viz_data = []
        
        for query, results in search_results.items():
            query_results = []
            
            for result in results:
                metadata = result["metadata"]
                result_type = metadata.get("type", "unknown")
                score = result["score"]
                
                result_item = {
                    "id": result["id"],
                    "score": score,
                    "type": result_type,
                    "similarity": score,  # For visualization
                }
                
                # Add type-specific data
                if result_type == "concept":
                    result_item["label"] = metadata.get("concept", "Unknown")
                    result_item["content"] = metadata.get("explanation", "No explanation")
                elif result_type in ["qa_pair", "question"]:
                    result_item["label"] = metadata.get("question", "Unknown")
                    result_item["content"] = metadata.get("answer", "No answer")
                else:
                    result_item["label"] = "Text"
                    result_item["content"] = metadata.get("text", "No text")
                
                result_item["source"] = metadata.get("document_title", "Unknown")
                query_results.append(result_item)
            
            # Add notice that query was translated if needed
            display_query = query
            if query == search_question and search_question != original_question:
                display_query = f"{original_question} [Translated: {query}]"
            elif detected_language != "en" and query in decomposition.get("sub_questions", []):
                # This is a sub-question that was already translated by DeepSeek
                display_query = f"{query} [Translated from {detected_language}]"
            
            viz_data.append({
                "query": display_query,
                "results": query_results
            })
        
        # Update final result data
        result_data["visualization"] = viz_data
        result_data["status"] = "success"
        
        # For non-English queries, add a translation note
        if detected_language != "en":
            result_data["translation_note"] = f"Your question was translated from {detected_language} to English for processing."
        
        # Debug the result data before sending
        print(f"[Session {session_id}] Sending result data. Answer present: {'Yes' if 'answer' in result_data else 'No'}")
        if 'answer' in result_data:
            print(f"[Session {session_id}] Answer length: {len(result_data['answer'])}")
        else:
            print(f"[Session {session_id}] WARNING: No answer in result_data")
            # Add a fallback answer if none exists
            result_data["answer"] = "I'm sorry, I couldn't generate an answer based on the available information."
        
        # Make sure visualization exists even if empty
        if 'visualization' not in result_data:
            result_data["visualization"] = []
        
        # Send result data event before completing the stream
        print(f"[Session {session_id}] Putting result event in queue")
        event_queue.put({
            "event": "result",
            "data": result_data
        })
        
        # Small delay to ensure the result event is processed before the stream ends
        time.sleep(0.5)
        
        # Signal end of stream
        print(f"[Session {session_id}] Putting None in queue (end of stream)")
        event_queue.put(None)
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in async processing for session {session_id}: {str(e)}\n{error_traceback}")
        
        # Send error event
        event_queue.put({
            "event": "error",
            "error": str(e)
        })
        
        # Signal end of stream
        event_queue.put(None)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint to process questions."""
    try:
        if not initialize_services():
            return jsonify({
                "error": f"Services not initialized correctly: {initialization_error}"
            }), 500
        
        # Get question from request
        data = request.json
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({
                "error": "Question cannot be empty"
            }), 400
        
        # Generate a session ID for tracking progress
        session_id = str(uuid.uuid4())
        
        # Create a queue for progress events
        progress_events[session_id] = Queue()
        
        # Start processing in a background thread
        threading.Thread(
            target=process_question_async,
            args=(question, session_id),
            daemon=True
        ).start()
        
        # Return the session ID for clients to connect to progress stream
        return jsonify({
            "status": "processing",
            "session_id": session_id
        })
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error processing question: {str(e)}\n{error_traceback}")
        return jsonify({
            "error": str(e),
            "traceback": error_traceback
        }), 500

if __name__ == '__main__':
    # Try to initialize services on startup
    try:
        initialize_services()
        if initialization_error:
            print(f"Warning: {initialization_error}")
    except Exception as e:
        print(f"Error during initialization: {e}")
    
    # Run the app
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    port = int(os.getenv("PORT", 5000))
    print(f"Starting server on port {port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port) 