#!/usr/bin/env python3
"""
Wisdom Web App - Web Interface for Semantic Search with Vector Database

This Flask-based web application provides a chatbot interface for asking questions,
visualizing vector database searches, and displaying answers in a clean UI.
"""

import os
import traceback
import langid
from flask import Flask, render_template, request, jsonify
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

def detect_language(text):
    """Detect the language of the input text."""
    try:
        lang, confidence = langid.classify(text)
        return lang, confidence
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "en", 0.0  # Default to English

def translate_text(text, target_lang="en", source_lang=None):
    """
    Translate text using DeepSeek API.
    
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
    
    response = query_deepseek(text, system_prompt, api_key)
    
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
        
        # Process steps to track and visualize
        process_steps = [
            {"id": "language", "name": "Language Detection", "status": "pending", "details": None},
            {"id": "translation", "name": "Translation", "status": "pending", "details": None},
            {"id": "decomposition", "name": "Question Decomposition", "status": "pending", "details": None},
            {"id": "search", "name": "Vector Search", "status": "pending", "details": None},
            {"id": "answer", "name": "Answer Generation", "status": "pending", "details": None},
            {"id": "translation_back", "name": "Translation to User Language", "status": "pending", "details": None}
        ]
        
        # Detect language
        process_steps[0]["status"] = "processing"
        lang, confidence = detect_language(question)
        original_language = lang
        process_steps[0]["status"] = "completed"
        process_steps[0]["details"] = {"language": lang, "confidence": confidence}
        
        # Translate if not English
        translated_question = question
        if lang != "en" and os.getenv("DEEPSEEK_API_KEY"):
            process_steps[1]["status"] = "processing"
            translation_result = translate_text(question, "en", lang)
            
            if "error" not in translation_result:
                translated_question = translation_result["translated_text"]
                process_steps[1]["status"] = "completed"
                process_steps[1]["details"] = {
                    "original": question, 
                    "translated": translated_question
                }
            else:
                process_steps[1]["status"] = "error"
                process_steps[1]["details"] = {"error": translation_result["error"]}
        else:
            process_steps[1]["status"] = "skipped"
            process_steps[1]["details"] = {"reason": "Language is English or translation API not available"}
        
        # Decompose the question
        process_steps[2]["status"] = "processing"
        decomposition = decompose_question(translated_question)
        process_steps[2]["status"] = "completed"
        process_steps[2]["details"] = decomposition
        
        # Prepare search queries
        all_queries = []
        all_queries.append(translated_question)  # The original question
        all_queries.extend(decomposition["sub_questions"])
        all_queries.extend([c for c in decomposition["concepts"] if isinstance(c, str)])
        all_queries.extend(decomposition["search_queries"])
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in all_queries:
            if isinstance(q, str) and q and q not in seen:
                unique_queries.append(q)
                seen.add(q)
        
        # Search for all queries
        process_steps[3]["status"] = "processing"
        search_results = batch_search_pinecone(unique_queries, DEFAULT_INDEX, DEFAULT_TOP_K)
        process_steps[3]["status"] = "completed"
        process_steps[3]["details"] = {"query_count": len(unique_queries), "result_count": sum(len(results) for results in search_results.values())}
        
        # Generate answer
        process_steps[4]["status"] = "processing"
        answer = generate_answer(translated_question, search_results)
        process_steps[4]["status"] = "completed"
        process_steps[4]["details"] = {"length": len(answer)}
        
        # Translate answer back if needed
        if lang != "en" and os.getenv("DEEPSEEK_API_KEY"):
            process_steps[5]["status"] = "processing"
            back_translation = translate_text(answer, lang, "en")
            
            if "error" not in back_translation:
                answer = back_translation["translated_text"]
                process_steps[5]["status"] = "completed" 
                process_steps[5]["details"] = {"translated_to": lang}
            else:
                process_steps[5]["status"] = "error"
                process_steps[5]["details"] = {"error": back_translation["error"]}
        else:
            process_steps[5]["status"] = "skipped"
            process_steps[5]["details"] = {"reason": "Language is English or translation API not available"}
        
        # Process search results for visualization
        viz_data = []
        
        for query, results in search_results.items():
            query_results = []
            
            for result in results:
                metadata = result["metadata"]
                result_type = metadata.get("type", "unknown")
                score = result["score"]
                
                result_data = {
                    "id": result["id"],
                    "score": score,
                    "type": result_type,
                    "similarity": score,  # For visualization
                }
                
                # Add type-specific data
                if result_type == "concept":
                    result_data["label"] = metadata.get("concept", "Unknown")
                    result_data["content"] = metadata.get("explanation", "No explanation")
                elif result_type in ["qa_pair", "question"]:
                    result_data["label"] = metadata.get("question", "Unknown")
                    result_data["content"] = metadata.get("answer", "No answer")
                else:
                    result_data["label"] = "Text"
                    result_data["content"] = metadata.get("text", "No text")
                
                result_data["source"] = metadata.get("document_title", "Unknown")
                query_results.append(result_data)
            
            viz_data.append({
                "query": query,
                "results": query_results
            })
        
        # Return response with visualization data and process steps
        return jsonify({
            "status": "success",
            "decomposition": decomposition,
            "answer": answer,
            "visualization": viz_data,
            "process": process_steps,
            "original_language": original_language
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