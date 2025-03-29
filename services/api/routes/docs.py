from flask import Blueprint, jsonify, request, current_app

# Define the Blueprint
docs_bp = Blueprint('docs_bp', __name__)

@docs_bp.route('/docs')
def api_docs():
    """Return API documentation in OpenAPI format."""
    API_VERSION = current_app.config.get('API_VERSION', 'v1') # Get API version from app config
    # Simplified OpenAPI documentation
    docs = {
        "openapi": "3.0.0",
        "info": {
            "title": "PDF Wisdom Extractor API",
            "description": "API for semantic search and question answering on PDF documents",
            "version": API_VERSION
        },
        "servers": [
            {
                "url": request.host_url.rstrip('/'), # Use request.host_url
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
            # Dynamically generate paths based on registered blueprints/routes if needed
            # For simplicity, keeping the static definition for now
            f"/api/{API_VERSION}/health": {
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
             f"/api/{API_VERSION}/question": {
                 "post": {
                     "summary": "Answer a question",
                     "description": "Sends a question to the LLM service for processing.",
                     "requestBody": {
                         "required": True,
                         "content": {
                             "application/json": {
                                 "schema": {
                                     "type": "object",
                                     "properties": {
                                         "question": {"type": "string"},
                                         "context": {"type": "array", "items": {"type": "string"}},
                                         "model": {"type": "string"}
                                     },
                                     "required": ["question"]
                                 }
                             }
                         }
                     },
                     "responses": {
                         "200": {"description": "Answer received"},
                         "400": {"description": "Bad Request"},
                         "500": {"description": "Internal Server Error"}
                     }
                 }
             },
            f"/api/{API_VERSION}/ask": {
                "post": {
                    "summary": "Ask a question (potentially streaming)",
                    "description": "Ask a question and get an answer based on the knowledge base. Can optionally stream the response.",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "question": {
                                            "type": "string",
                                            "description": "The question to ask"
                                        },
                                        "stream": {
                                            "type": "boolean",
                                            "description": "Whether to stream the response"
                                        }
                                    },
                                    "required": ["question"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Answer to the question (or stream started)"
                        },
                        "400": {
                            "description": "Invalid request"
                        },
                        "500": {
                            "description": "Server error"
                        }
                    }
                }
            },
            f"/api/{API_VERSION}/translate": {
                "post": {
                    "summary": "Translate text",
                    "description": "Translate text from one language to another",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "The text to translate"
                                        },
                                        "target_lang": {
                                            "type": "string",
                                            "description": "Target language code"
                                        },
                                        "source_lang": {
                                            "type": "string",
                                            "description": "Source language code (optional)"
                                        }
                                    },
                                    "required": ["text", "target_lang"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Translated text"
                        },
                        "400": {
                            "description": "Invalid request"
                        },
                        "500": {
                            "description": "Server error"
                        }
                    }
                }
            },
            f"/api/{API_VERSION}/process_pdf": {
                "post": {
                    "summary": "Upload and process a PDF document",
                    "description": "Upload a PDF document, extract knowledge, and store it in the database",
                    "security": [
                        {"bearerAuth": []},
                        {"apiKeyAuth": []}
                    ],
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "PDF file to upload"
                                        },
                                        "author_name": {
                                            "type": "string",
                                            "description": "Name of the document author"
                                        },
                                        "title": {
                                            "type": "string",
                                            "description": "Optional document title (defaults to filename)"
                                        },
                                        "language": {
                                            "type": "string",
                                            "description": "Optional document language (defaults to 'en')"
                                        },
                                        "translate_to_english": {
                                            "type": "boolean",
                                            "description": "Whether to translate content to English (defaults to true)"
                                        }
                                    },
                                    "required": ["file", "author_name"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": {
                            "description": "PDF processing started"
                        },
                        "400": {
                            "description": "Invalid request or file type"
                        },
                         "401": {
                             "description": "Unauthorized"
                         },
                        "500": {
                            "description": "Server error during processing"
                        }
                    }
                }
            },
             f"/api/{API_VERSION}/progress-stream/{{session_id}}": {
                "get": {
                    "summary": "Stream progress updates",
                    "description": "Streams progress updates for a specific session ID (e.g., PDF processing).",
                    "parameters": [
                        {
                            "name": "session_id",
                            "in": "path",
                            "required": True,
                            "description": "The session ID obtained when starting a process.",
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Stream of progress events",
                            "content": {
                                "text/event-stream": {
                                    "schema": {"type": "string"}
                                }
                            }
                        },
                        "404": {"description": "Session ID not found"}
                    }
                }
            },
            # Add other endpoints here (e.g., search, auth, analytics, webhooks)
        }
    }
    return jsonify(docs) 