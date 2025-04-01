#!/usr/bin/env python3
"""
PDF Processor Service - Microservice for PDF text extraction and processing

This service is responsible for:
1. Extracting text from PDF files
2. Chunking text into processable segments
3. Providing an API for other services to request PDF processing
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import datetime
import pytz
import sys
from tqdm import tqdm
from collections import defaultdict
import threading
import io
import logging
import PyPDF2
import structlog

# Try to import tiktoken, but use a fallback if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available, using simple character-based token estimation")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global rate limiting
api_rate_limits = defaultdict(lambda: {"last_call": 0, "backoff": 0})
api_lock = threading.Lock()  # Lock for thread-safe API call scheduling

# Configure logger
logger = structlog.get_logger(__name__)

class PDFProcessor:
    """Handles PDF text extraction."""
    
    def __init__(self):
        # Initialization, if any needed
        logger.info("PDFProcessor initialized.")
        pass

    def extract_text(self, pdf_content: bytes) -> str:
        """
        Extracts text content from PDF bytes.
        
        Args:
            pdf_content: Raw bytes of the PDF file.
            
        Returns:
            Extracted text content as a single string.
            Returns an empty string if extraction fails.
        """
        if not pdf_content:
             logger.warning("Attempted to extract text from empty PDF content.")
             return ""
             
        try:
            # Create a file-like object from the bytes
            pdf_file = io.BytesIO(pdf_content)
            
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from each page
            text_content = []
            num_pages = len(reader.pages)
            logger.debug("Starting text extraction", num_pages=num_pages)
            
            for page_num in range(num_pages):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                    else:
                        logger.debug("No text found on page", page_number=page_num + 1)
                except Exception as page_err:
                    # Log error for specific page but continue if possible
                    logger.error("Error extracting text from page", 
                                 page_number=page_num + 1, 
                                 error=str(page_err), 
                                 exc_info=True)
            
            full_text = "\n".join(text_content)
            logger.info("PDF text extraction completed successfully", 
                        num_pages_processed=num_pages, 
                        extracted_text_length=len(full_text))
            return full_text
            
        except PyPDF2.errors.PdfReadError as e:
            # Handle specific PyPDF2 errors (e.g., encrypted, corrupted)
            logger.error("Failed to read PDF (likely corrupted or encrypted)", error=str(e), exc_info=True)
            return "" # Return empty string on failure
        except Exception as e:
            # Handle other unexpected errors during processing
            logger.error("Unexpected error during PDF text extraction", error=str(e), exc_info=True)
            return "" # Return empty string on failure

# 1. Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.strip()
        
        print(f"Successfully extracted {len(text)} characters from {pdf_path}")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

# Simple token counter that approximates token count for when tiktoken is not available
def estimate_tokens(text: str) -> int:
    """Estimate token count based on simple rules."""
    # Rough approximation: one token is about 4 characters on average
    return len(text) // 4

# 2. Split text into smaller, processable chunks
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split the text into overlapping chunks of specified size."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    i = 0
    while i < len(words):
        # Get chunk of words with specified size
        chunk = words[i:i + chunk_size]
        # Join words back into text
        chunks.append(" ".join(chunk))
        # Move to next chunk, considering overlap
        i += (chunk_size - overlap)
    
    # Ensure minimum meaningful size
    return [chunk for chunk in chunks if len(chunk.split()) > 10]

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "pdf-processor"})

@app.route('/extract', methods=['POST'])
def extract_pdf():
    """Extract text from a PDF file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File must be a PDF"}), 400
    
    # Save the file temporarily
    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)
    
    # Extract text
    text = extract_text_from_pdf(temp_path)
    
    # Remove temporary file
    os.remove(temp_path)
    
    if not text:
        return jsonify({"error": "Failed to extract text from PDF"}), 500
    
    return jsonify({
        "text": text,
        "characters": len(text),
        "estimated_tokens": estimate_tokens(text)
    })

@app.route('/chunk', methods=['POST'])
def chunk_pdf_text():
    """Chunk text into smaller segments"""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    chunk_size = data.get('chunk_size', 1000)
    overlap = data.get('overlap', 200)
    
    chunks = chunk_text(data['text'], chunk_size, overlap)
    
    return jsonify({
        "chunks": chunks,
        "count": len(chunks)
    })

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PDF Processor Service")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the service on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)