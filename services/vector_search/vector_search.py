#!/usr/bin/env python3
"""
Vector Search Service - Microservice for vector embeddings and search

This service is responsible for:
1. Generating embeddings for text chunks
2. Storing and retrieving vectors from Pinecone
3. Providing an API for semantic search operations
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import Pinecone
try:
    import pinecone
except ImportError:
    print("Error: Pinecone package not installed. Run 'pip install pinecone'")
    sys.exit(1)

# Import embedding model
try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required ML packages not installed. Run 'pip install torch sentence-transformers'")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Default embedding model
DEFAULT_INDEX = "wisdom-embeddings"   # Default Pinecone index
DEFAULT_TOP_K = 10                   # Default number of results to retrieve

# Global Pinecone instance and embedding model
pc = None
embedding_model = None

# Initialize Pinecone
def init_pinecone(api_key: str = None, environment: str = None) -> bool:
    """Initialize Pinecone with API key."""
    global pc
    
    if not api_key:
        api_key = os.getenv("PINECONE_API_KEY")
    
    if not environment:
        environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    
    if not api_key:
        print("Error: Pinecone API key not provided")
        return False
    
    try:
        # Initialize Pinecone with API key
        pinecone.init(api_key=api_key, environment=environment)
        pc = pinecone
        print(f"Successfully initialized Pinecone in {environment} environment")
        return True
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        return False

# Load embedding model
def load_embedding_model(model_name: str = DEFAULT_MODEL) -> bool:
    """Load the sentence transformer embedding model."""
    global embedding_model
    
    try:
        # Load the embedding model
        embedding_model = SentenceTransformer(model_name)
        print(f"Successfully loaded embedding model: {model_name}")
        return True
    except Exception as e:
        print(f"Error loading embedding model: {str(e)}")
        return False

# Generate embeddings for text
def generate_embedding(text: str) -> List[float]:
    """Generate embedding vector for a text string."""
    if not embedding_model:
        raise ValueError("Embedding model not loaded")
    
    # Generate embedding
    embedding = embedding_model.encode(text)
    
    # Convert to list and return
    return embedding.tolist()

# Search Pinecone index
def search_pinecone(query_embedding: List[float], index_name: str = DEFAULT_INDEX, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """Search Pinecone index with query embedding."""
    if not pc:
        raise ValueError("Pinecone not initialized")
    
    # Get Pinecone index
    try:
        index = pc.Index(index_name)
    except Exception as e:
        raise ValueError(f"Error accessing Pinecone index: {str(e)}")
    
    # Search index
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results
    except Exception as e:
        raise ValueError(f"Error searching Pinecone: {str(e)}")

# Batch search Pinecone index
def batch_search_pinecone(query_embeddings: List[List[float]], index_name: str = DEFAULT_INDEX, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Batch search Pinecone index with multiple query embeddings."""
    if not pc:
        raise ValueError("Pinecone not initialized")
    
    # Get Pinecone index
    try:
        index = pc.Index(index_name)
    except Exception as e:
        raise ValueError(f"Error accessing Pinecone index: {str(e)}")
    
    # Search index in batch
    try:
        results = []
        for embedding in query_embeddings:
            result = index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            results.append(result)
        return results
    except Exception as e:
        raise ValueError(f"Error batch searching Pinecone: {str(e)}")

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "vector-search",
        "pinecone_initialized": pc is not None,
        "embedding_model_loaded": embedding_model is not None
    })

@app.route('/initialize', methods=['POST'])
def initialize_service():
    """Initialize the vector search service"""
    data = request.json or {}
    
    # Initialize Pinecone
    pinecone_api_key = data.get('pinecone_api_key', os.getenv("PINECONE_API_KEY"))
    pinecone_env = data.get('pinecone_environment', os.getenv("PINECONE_ENVIRONMENT"))
    pinecone_initialized = init_pinecone(pinecone_api_key, pinecone_env)
    
    # Load embedding model
    model_name = data.get('model_name', DEFAULT_MODEL)
    model_loaded = load_embedding_model(model_name)
    
    return jsonify({
        "pinecone_initialized": pinecone_initialized,
        "embedding_model_loaded": model_loaded,
        "model_name": model_name
    })

@app.route('/embed', methods=['POST'])
def embed_text():
    """Generate embeddings for text"""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    if not embedding_model:
        return jsonify({"error": "Embedding model not loaded"}), 500
    
    # Handle both single text and batch of texts
    if isinstance(data['text'], str):
        # Single text
        try:
            embedding = generate_embedding(data['text'])
            return jsonify({
                "embedding": embedding,
                "dimensions": len(embedding)
            })
        except Exception as e:
            return jsonify({"error": f"Error generating embedding: {str(e)}"}), 500
    elif isinstance(data['text'], list):
        # Batch of texts
        try:
            embeddings = [generate_embedding(text) for text in data['text']]
            return jsonify({
                "embeddings": embeddings,
                "count": len(embeddings),
                "dimensions": len(embeddings[0]) if embeddings else 0
            })
        except Exception as e:
            return jsonify({"error": f"Error generating embeddings: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid text format"}), 400

@app.route('/search', methods=['POST'])
def search():
    """Search Pinecone index with query text or embedding"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    if not pc:
        return jsonify({"error": "Pinecone not initialized"}), 500
    
    if not embedding_model:
        return jsonify({"error": "Embedding model not loaded"}), 500
    
    # Get parameters
    index_name = data.get('index_name', DEFAULT_INDEX)
    top_k = data.get('top_k', DEFAULT_TOP_K)
    
    # Check if query text or embedding is provided
    if 'query_text' in data:
        # Generate embedding from query text
        try:
            query_embedding = generate_embedding(data['query_text'])
        except Exception as e:
            return jsonify({"error": f"Error generating query embedding: {str(e)}"}), 500
    elif 'query_embedding' in data:
        # Use provided embedding
        query_embedding = data['query_embedding']
    else:
        return jsonify({"error": "No query_text or query_embedding provided"}), 400
    
    # Search Pinecone
    try:
        results = search_pinecone(query_embedding, index_name, top_k)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Error searching Pinecone: {str(e)}"}), 500

@app.route('/batch_search', methods=['POST'])
def batch_search():
    """Batch search Pinecone index with multiple queries"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    if not pc:
        return jsonify({"error": "Pinecone not initialized"}), 500
    
    if not embedding_model:
        return jsonify({"error": "Embedding model not loaded"}), 500
    
    # Get parameters
    index_name = data.get('index_name', DEFAULT_INDEX)
    top_k = data.get('top_k', DEFAULT_TOP_K)
    
    # Check if query texts or embeddings are provided
    if 'query_texts' in data:
        # Generate embeddings from query texts
        try:
            query_embeddings = [generate_embedding(text) for text in data['query_texts']]
        except Exception as e:
            return jsonify({"error": f"Error generating query embeddings: {str(e)}"}), 500
    elif 'query_embeddings' in data:
        # Use provided embeddings
        query_embeddings = data['query_embeddings']
    else:
        return jsonify({"error": "No query_texts or query_embeddings provided"}), 400
    
    # Batch search Pinecone
    try:
        results = batch_search_pinecone(query_embeddings, index_name, top_k)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": f"Error batch searching Pinecone: {str(e)}"}), 500

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vector Search Service")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--port', type=int, default=5002, help='Port to run the service on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--init', action='store_true', help='Initialize Pinecone and embedding model on startup')
    args = parser.parse_args()
    
    # Initialize services if requested
    if args.init:
        init_pinecone()
        load_embedding_model()
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)