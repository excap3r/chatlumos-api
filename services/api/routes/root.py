from flask import Blueprint, jsonify, current_app

# Define the Blueprint
root_bp = Blueprint('root_bp', __name__)

@root_bp.route('/')
def redirect_to_docs():
    """Return basic API info and redirect hint to docs."""
    # Access API_VERSION from app config
    API_VERSION = current_app.config.get('API_VERSION', 'v1') 
    
    return jsonify({
        "message": "PDF Wisdom Extractor API Server",
        "version": API_VERSION,
        "documentation": f"/api/{API_VERSION}/docs",
        "health": f"/api/{API_VERSION}/health"
    }) 