#!/usr/bin/env python3
"""
LLM Service - Microservice for language model operations

This service is responsible for:
1. Decomposing questions into sub-questions
2. Generating answers based on search results
3. Providing an API for LLM operations

Supports multiple LLM providers:
- DeepSeek
- Anthropic (Claude)
- OpenAI (GPT models)
- Groq
- OpenRouter
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple, Callable
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
import logging

# Import provider functionality
from providers import (
    LLMProvider,
    LLMResponse,
    StreamingHandler,
    get_provider,
    list_available_providers
)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('llm_service')

# ----------------- API Endpoints -----------------

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    providers = list_available_providers()
    return jsonify({
        "status": "healthy", 
        "service": "llm-service",
        "available_providers": providers,
        "default_provider": os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    })

@app.route('/capabilities', methods=['GET'])
def get_capabilities():
    """Get capabilities of available providers"""
    provider_name = request.args.get('provider')
    
    # If provider specified, get capabilities for that provider
    if provider_name:
        try:
            provider = get_provider(provider_name)
            return jsonify({
                "provider": provider_name,
                "capabilities": provider.get_capabilities()
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    
    # Otherwise, get capabilities for all providers
    capabilities = {}
    for provider_name in list_available_providers():
        try:
            provider = get_provider(provider_name)
            capabilities[provider_name] = provider.get_capabilities()
        except Exception as e:
            capabilities[provider_name] = {"error": str(e)}
    
    return jsonify(capabilities)

@app.route('/decompose_question', methods=['POST'])
def decompose_question():
    """Decompose a complex question into sub-questions"""
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data['question']
    context = data.get('context', '')
    provider_name = data.get('provider')
    api_key = data.get('api_key')
    model = data.get('model')
    
    # System prompt for question decomposition
    system_prompt = """
    You are an expert at breaking down complex questions into simpler sub-questions.
    Your task is to analyze the given question and decompose it into 2-5 sub-questions that:
    1. Are simpler and more focused than the original question
    2. When answered together, would provide a comprehensive answer to the original question
    3. Are self-contained and can be answered independently
    
    Return your response as a JSON array of sub-questions.
    """
    
    # Create prompt with context if provided
    if context:
        prompt = f"Original question: {question}\n\nRelevant context: {context}\n\nPlease decompose this question into sub-questions."
    else:
        prompt = f"Original question: {question}\n\nPlease decompose this question into sub-questions."
    
    try:
        # Initialize provider
        provider = get_provider(provider_name, api_key=api_key, model=model)
        
        # Generate response
        response = provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2  # Lower temperature for more focused decomposition
        )
        
        # Check for errors
        if response.is_error:
            return jsonify({
                "error": response.error,
                "details": response.details
            }), 500
        
        # Try to parse the response as JSON
        try:
            content = response.content
            # Check if the content is already JSON or needs extraction
            if content.strip().startswith('[') and content.strip().endswith(']'):
                sub_questions = json.loads(content)
            else:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\n(.+?)\n```', content, re.DOTALL)
                if json_match:
                    sub_questions = json.loads(json_match.group(1))
                else:
                    # Fallback: return the raw content
                    return jsonify({
                        "sub_questions": [content],
                        "raw_response": content,
                        "provider": response.provider,
                        "model": response.model
                    })
            
            return jsonify({
                "sub_questions": sub_questions,
                "provider": response.provider,
                "model": response.model,
                "usage": response.usage
            })
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return jsonify({
                "error": "Failed to parse LLM response",
                "raw_response": response.content,
                "provider": response.provider,
                "model": response.model
            }), 500
    except Exception as e:
        logger.error(f"Error in decompose_question: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    """Generate an answer based on search results"""
    data = request.json
    if not data or 'question' not in data or 'search_results' not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    question = data['question']
    search_results = data['search_results']
    provider_name = data.get('provider')
    api_key = data.get('api_key')
    model = data.get('model')
    stream = data.get('stream', False)
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 2000)
    
    # System prompt for answer generation
    system_prompt = """
    You are a helpful assistant that generates accurate, comprehensive answers based on the provided search results.
    Your task is to:
    1. Analyze the search results and extract relevant information
    2. Synthesize a coherent answer that directly addresses the question
    3. Only use information from the provided search results
    4. If the search results don't contain enough information to answer the question, acknowledge this limitation
    5. Provide citations to the source chunks where appropriate
    
    Be concise but thorough in your response.
    """
    
    # Format search results for the prompt
    formatted_results = ""
    for i, result in enumerate(search_results):
        content = result.get('content', result.get('text', ''))
        metadata = result.get('metadata', {})
        source = metadata.get('source', f"Result {i+1}")
        formatted_results += f"\n--- Result {i+1} (Source: {source}) ---\n{content}\n"
    
    prompt = f"Question: {question}\n\nSearch Results:{formatted_results}\n\nPlease generate a comprehensive answer based on these search results."
    
    try:
        # Initialize provider
        provider = get_provider(provider_name, api_key=api_key, model=model)
        
        if stream:
            # Set up streaming response
            def generate():
                # Create streaming handler
                def handle_chunk(chunk):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Create streaming handler
                handler = StreamingHandler(handle_chunk)
                
                # Generate streaming response
                response = provider.stream_generate(
                    prompt=prompt,
                    streaming_handler=handler,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Send the [DONE] message
                yield "data: [DONE]\n\n"
            
            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        else:
            # Non-streaming response
            response = provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Check for errors
            if response.is_error:
                return jsonify({
                    "error": response.error,
                    "details": response.details
                }), 500
            
            return jsonify({
                "answer": response.content,
                "provider": response.provider,
                "model": response.model,
                "usage": response.usage
            })
    except Exception as e:
        logger.error(f"Error in generate_answer: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/complete', methods=['POST'])
def complete():
    """Generic completion endpoint for LLM generation"""
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    prompt = data['prompt']
    system_prompt = data.get('system_prompt')
    provider_name = data.get('provider')
    api_key = data.get('api_key')
    model = data.get('model')
    stream = data.get('stream', False)
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 2000)
    additional_params = data.get('params', {})
    
    try:
        # Initialize provider
        provider = get_provider(provider_name, api_key=api_key, model=model)
        
        if stream:
            # Set up streaming response
            def generate():
                # Create streaming handler
                def handle_chunk(chunk):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Create streaming handler
                handler = StreamingHandler(handle_chunk)
                
                # Generate streaming response
                response = provider.stream_generate(
                    prompt=prompt,
                    streaming_handler=handler,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    params=additional_params
                )
                
                # Send the [DONE] message
                yield "data: [DONE]\n\n"
            
            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        else:
            # Non-streaming response
            response = provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                params=additional_params
            )
            
            # Check for errors
            if response.is_error:
                return jsonify({
                    "error": response.error,
                    "details": response.details
                }), 500
            
            return jsonify({
                "content": response.content,
                "provider": response.provider,
                "model": response.model,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "details": response.details
            })
    except Exception as e:
        logger.error(f"Error in complete: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5002, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Log available providers
    logger.info(f"Available LLM providers: {list_available_providers()}")
    logger.info(f"Default provider: {os.getenv('DEFAULT_LLM_PROVIDER', 'openai')}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)