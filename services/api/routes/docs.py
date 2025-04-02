from flask import Blueprint, jsonify, request, current_app

# Import helper definitions for OpenAPI paths - REVERTED
# from ..docs_helpers import AuthPaths, PdfPaths, AskPaths, HealthPaths, ProgressPaths, SearchPaths, TranslatePaths # Assuming these exist

# Define the Blueprint
docs_bp = Blueprint('docs_bp', __name__)

# Renamed endpoint to serve the JSON spec
@docs_bp.route('/swagger.json')
def api_spec():
    """Return API specification in OpenAPI JSON format."""
    API_VERSION = current_app.config.get('API_VERSION', 'v1')
    
    # Build the OpenAPI spec dictionary
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "PDF Wisdom Extractor API",
            "description": "API for semantic search and question answering on PDF documents",
            "version": API_VERSION
        },
        "servers": [
            {
                "url": request.host_url.rstrip('/') + f"/api/{API_VERSION}", # Base path for API calls
                "description": "Current server"
            }
        ],
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "JWT token obtained from login endpoint"
                },
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key obtained from /api/v1/auth/api-keys endpoint"
                }
            }
        },
        "paths": {
            # TODO: Define paths properly here using helpers or manual definition
            # Example:
             f"/health": { # Note: path relative to server URL above
                 "get": {
                     "summary": "Check API health",
                     "description": "Returns the health status of the API and its dependencies",
                     "responses": {
                         "200": {
                             "description": "Health status"
                         }
                     }
                 }
             },
             # Add other paths here...
        }
    }
    return jsonify(spec)

# The route to serve the HTML UI will be created using flask-swagger-ui in app.py 