#!/usr/bin/env python3
"""
Frontend Service - Standalone frontend for the PDF Wisdom Extractor

This service is responsible for:
1. Serving the UI components
2. Communicating with the API gateway
3. Handling user interactions
"""

import os
import json
import argparse
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
           static_folder="../../static",
           template_folder="../../templates")
CORS(app)

# Configuration
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:5000")

# Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "frontend"})

@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_proxy(path):
    """Proxy API requests to the API gateway"""
    # Forward the request to the API gateway
    url = f"{API_GATEWAY_URL}/{path}"
    
    # Get request data
    headers = {key: value for key, value in request.headers if key != 'Host'}
    data = request.get_data()
    params = request.args
    
    # Forward the request
    try:
        if request.method == 'GET':
            response = requests.get(url, headers=headers, params=params)
        elif request.method == 'POST':
            response = requests.post(url, headers=headers, data=data, params=params)
        elif request.method == 'PUT':
            response = requests.put(url, headers=headers, data=data, params=params)
        elif request.method == 'DELETE':
            response = requests.delete(url, headers=headers, params=params)
        else:
            return jsonify({"error": "Method not supported"}), 405
        
        # Return the response from the API gateway
        return response.content, response.status_code, response.headers.items()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to API gateway: {str(e)}"}), 500

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Frontend Service")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--port', type=int, default=5004, help='Port to run the service on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)