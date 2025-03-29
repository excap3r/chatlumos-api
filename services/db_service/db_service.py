#!/usr/bin/env python3
"""
Database Service - Microservice for database operations

This service is responsible for:
1. Managing database connections and pools
2. CRUD operations for documents, concepts, and QA pairs
3. Providing an API for database operations
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import mysql.connector
from mysql.connector import pooling, Error
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('db_service')

# Database configuration
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'database': os.getenv('MYSQL_DATABASE', 'wisdom_db'),
}

# Connection pool configuration
DB_POOL_NAME = "pdf_wisdom_pool"
DB_POOL_SIZE = 10  # Increased pool size for better performance

# Initialize connection pool
try:
    connection_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name=DB_POOL_NAME,
        pool_size=DB_POOL_SIZE,
        **DB_CONFIG
    )
    logger.info(f"Connection pool created successfully with {DB_POOL_SIZE} connections")
except Exception as e:
    connection_pool = None
    logger.error(f"Error creating connection pool: {str(e)}")

def get_connection():
    """Get a connection from the pool with retry logic."""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            if connection_pool:
                return connection_pool.get_connection()
            else:
                # Fallback to direct connection if pool initialization failed
                return mysql.connector.connect(**DB_CONFIG)
        except Exception as e:
            logger.warning(f"Connection attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("All connection attempts failed")
                raise

def initialize_database(force_recreate=False):
    """
    Create database tables if they don't exist.
    If force_recreate is True, drop existing tables first.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if force_recreate:
            # Drop tables if they exist
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            cursor.execute("DROP TABLE IF EXISTS qa_pairs")
            cursor.execute("DROP TABLE IF EXISTS concepts")
            cursor.execute("DROP TABLE IF EXISTS document_chunks")
            cursor.execute("DROP TABLE IF EXISTS documents")
            cursor.execute("DROP TABLE IF EXISTS summaries")
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            logger.info("Existing tables dropped")

        # Create tables
        # Documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            title VARCHAR(255),
            author VARCHAR(255),
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            full_text LONGTEXT,
            total_chunks INT,
            status ENUM('processing', 'completed', 'failed') DEFAULT 'processing',
            file_path VARCHAR(512),
            UNIQUE INDEX idx_filename (filename)
        ) ENGINE=InnoDB
        """)

        # Document chunks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id INT AUTO_INCREMENT PRIMARY KEY,
            document_id INT NOT NULL,
            chunk_index INT NOT NULL,
            chunk_text LONGTEXT NOT NULL,
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
            UNIQUE INDEX idx_doc_chunk (document_id, chunk_index),
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        ) ENGINE=InnoDB
        """)

        # Concepts table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            concept_id INT AUTO_INCREMENT PRIMARY KEY,
            document_id INT NOT NULL,
            chunk_id INT NOT NULL,
            concept_name VARCHAR(255) NOT NULL,
            explanation TEXT NOT NULL,
            UNIQUE INDEX idx_doc_concept (document_id, concept_name),
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES document_chunks(chunk_id) ON DELETE CASCADE
        ) ENGINE=InnoDB
        """)

        # QA pairs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_pairs (
            qa_id INT AUTO_INCREMENT PRIMARY KEY,
            document_id INT NOT NULL,
            chunk_id INT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES document_chunks(chunk_id) ON DELETE CASCADE
        ) ENGINE=InnoDB
        """)

        # Summaries table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id INT AUTO_INCREMENT PRIMARY KEY,
            document_id INT NOT NULL,
            summary_text TEXT NOT NULL,
            generated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        ) ENGINE=InnoDB
        """)

        conn.commit()
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Document operations
def create_document(filename: str, title: str = None, author: str = None, file_path: str = None) -> Optional[int]:
    """
    Create a new document entry in the database.
    Returns the document_id if successful, None otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO documents (filename, title, author, file_path, status)
        VALUES (%s, %s, %s, %s, 'processing')
        ON DUPLICATE KEY UPDATE
        title = IFNULL(%s, title),
        author = IFNULL(%s, author),
        file_path = IFNULL(%s, file_path),
        status = 'processing'
        """
        cursor.execute(query, (filename, title, author, file_path, title, author, file_path))
        
        if cursor.lastrowid:
            document_id = cursor.lastrowid
        else:
            # Get the document_id if the record already existed
            cursor.execute("SELECT document_id FROM documents WHERE filename = %s", (filename,))
            document_id = cursor.fetchone()[0]
            
        conn.commit()
        logger.info(f"Document created/updated: {filename} (ID: {document_id})")
        return document_id
    except Exception as e:
        logger.error(f"Error creating document: {str(e)}")
        if conn:
            conn.rollback()
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def update_document_status(document_id: int, status: str, total_chunks: int = None, full_text: str = None) -> bool:
    """
    Update document status and optional fields.
    Returns True if successful, False otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query_parts = ["UPDATE documents SET status = %s"]
        params = [status]
        
        if total_chunks is not None:
            query_parts.append("total_chunks = %s")
            params.append(total_chunks)
            
        if full_text is not None:
            query_parts.append("full_text = %s")
            params.append(full_text)
            
        query_parts.append("WHERE document_id = %s")
        params.append(document_id)
        
        query = " , ".join(query_parts)
        cursor.execute(query, params)
        conn.commit()
        
        logger.info(f"Document status updated: ID {document_id} -> {status}")
        return True
    except Exception as e:
        logger.error(f"Error updating document status: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Document chunk operations
def create_document_chunk(document_id: int, chunk_index: int, chunk_text: str) -> Optional[int]:
    """
    Create a document chunk entry in the database.
    Returns the chunk_id if successful, None otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO document_chunks (document_id, chunk_index, chunk_text, status)
        VALUES (%s, %s, %s, 'pending')
        ON DUPLICATE KEY UPDATE
        chunk_text = %s,
        status = 'pending'
        """
        cursor.execute(query, (document_id, chunk_index, chunk_text, chunk_text))
        
        if cursor.lastrowid:
            chunk_id = cursor.lastrowid
        else:
            # Get the chunk_id if the record already existed
            cursor.execute(
                "SELECT chunk_id FROM document_chunks WHERE document_id = %s AND chunk_index = %s",
                (document_id, chunk_index)
            )
            chunk_id = cursor.fetchone()[0]
            
        conn.commit()
        return chunk_id
    except Exception as e:
        logger.error(f"Error creating document chunk: {str(e)}")
        if conn:
            conn.rollback()
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def update_chunk_status(chunk_id: int, status: str) -> bool:
    """
    Update chunk processing status.
    Returns True if successful, False otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = "UPDATE document_chunks SET status = %s WHERE chunk_id = %s"
        cursor.execute(query, (status, chunk_id))
        conn.commit()
        
        return True
    except Exception as e:
        logger.error(f"Error updating chunk status: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Concept operations
def store_concepts(document_id: int, chunk_id: int, concepts: List[Dict[str, str]]) -> bool:
    """
    Store concepts extracted from a document chunk.
    Each concept should have 'concept' and 'explanation' keys.
    Returns True if successful, False otherwise.
    """
    if not concepts:
        return True  # Nothing to store
        
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        for concept_data in concepts:
            concept_name = concept_data.get('concept', '')
            explanation = concept_data.get('explanation', '')
            
            if not concept_name or not explanation:
                continue
                
            query = """
            INSERT INTO concepts (document_id, chunk_id, concept_name, explanation)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            explanation = %s
            """
            cursor.execute(query, (document_id, chunk_id, concept_name, explanation, explanation))
        
        conn.commit()
        logger.info(f"Stored {len(concepts)} concepts for document {document_id}, chunk {chunk_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing concepts: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# QA pair operations
def store_qa_pairs(document_id: int, chunk_id: int, qa_pairs: List[Dict[str, str]]) -> bool:
    """
    Store QA pairs extracted from a document chunk.
    Each QA pair should have 'question' and 'answer' keys.
    Returns True if successful, False otherwise.
    """
    if not qa_pairs:
        return True  # Nothing to store
        
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        for qa_data in qa_pairs:
            question = qa_data.get('question', '')
            answer = qa_data.get('answer', '')
            
            if not question or not answer:
                continue
                
            query = """
            INSERT INTO qa_pairs (document_id, chunk_id, question, answer)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (document_id, chunk_id, question, answer))
        
        conn.commit()
        logger.info(f"Stored {len(qa_pairs)} QA pairs for document {document_id}, chunk {chunk_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing QA pairs: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Summary operations
def store_summary(document_id: int, summary_text: str) -> bool:
    """
    Store a summary for a document.
    Returns True if successful, False otherwise.
    """
    if not summary_text:
        return False
        
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO summaries (document_id, summary_text)
        VALUES (%s, %s)
        """
        cursor.execute(query, (document_id, summary_text))
        conn.commit()
        
        return True
    except Exception as e:
        logger.error(f"Error storing summary: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Query operations
def get_all_concepts(document_id: Optional[int] = None) -> List[Dict]:
    """Get all concepts from database, optionally filtered by document_id."""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        if document_id:
            query = """
            SELECT 
                c.concept_id,
                c.document_id,
                d.title as document_title,
                d.author as document_author,
                c.concept_name as concept,
                c.explanation
            FROM concepts c
            JOIN documents d ON c.document_id = d.document_id
            WHERE c.document_id = %s
            """
            cursor.execute(query, (document_id,))
        else:
            query = """
            SELECT 
                c.concept_id,
                c.document_id,
                d.title as document_title,
                d.author as document_author,
                c.concept_name as concept,
                c.explanation
            FROM concepts c
            JOIN documents d ON c.document_id = d.document_id
            """
            cursor.execute(query)
            
        concepts = cursor.fetchall()
        return concepts
    except Error as e:
        logger.error(f"Error fetching concepts: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_all_qa_pairs(document_id: Optional[int] = None) -> List[Dict]:
    """Get all QA pairs from database, optionally filtered by document_id."""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        if document_id:
            query = """
            SELECT 
                q.qa_id,
                q.document_id,
                d.title as document_title,
                d.author as document_author,
                q.question,
                q.answer
            FROM qa_pairs q
            JOIN documents d ON q.document_id = d.document_id
            WHERE q.document_id = %s
            """
            cursor.execute(query, (document_id,))
        else:
            query = """
            SELECT 
                q.qa_id,
                q.document_id,
                d.title as document_title,
                d.author as document_author,
                q.question,
                q.answer
            FROM qa_pairs q
            JOIN documents d ON q.document_id = d.document_id
            """
            cursor.execute(query)
            
        qa_pairs = cursor.fetchall()
        return qa_pairs
    except Error as e:
        logger.error(f"Error fetching QA pairs: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_document_summary(document_id: int) -> Optional[str]:
    """Get the summary for a document if available."""
    conn = get_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        query = """
        SELECT summary_text
        FROM summaries
        WHERE document_id = %s
        ORDER BY generated_date DESC
        LIMIT 1
        """
        cursor.execute(query, (document_id,))
        result = cursor.fetchone()
        
        if result:
            return result[0]  # Summary text
        return None
    except Error as e:
        logger.error(f"Error fetching document summary: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_document_info(document_id: int) -> Optional[Dict]:
    """Get detailed information about a document."""
    conn = get_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
        SELECT 
            d.document_id,
            d.filename,
            d.title,
            d.author,
            d.processed_date,
            d.status,
            d.total_chunks,
            COUNT(DISTINCT c.concept_id) as concept_count,
            COUNT(DISTINCT q.qa_id) as qa_count
        FROM 
            documents d
        LEFT JOIN 
            concepts c ON d.document_id = c.document_id
        LEFT JOIN 
            qa_pairs q ON d.document_id = q.document_id
        WHERE 
            d.document_id = %s
        GROUP BY 
            d.document_id
        """
        cursor.execute(query, (document_id,))
        document = cursor.fetchone()
        
        if not document:
            return None
            
        # Get summary if available
        document['summary'] = get_document_summary(document_id)
        
        return document
    except Error as e:
        logger.error(f"Error fetching document info: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_document_statistics() -> Dict[str, Any]:
    """Get statistics about the documents in the database."""
    conn = get_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        
        # Total documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_documents = cursor.fetchone()[0]
        
        # Total concepts
        cursor.execute("SELECT COUNT(*) FROM concepts")
        total_concepts = cursor.fetchone()[0]
        
        # Total QA pairs
        cursor.execute("SELECT COUNT(*) FROM qa_pairs")
        total_qa_pairs = cursor.fetchone()[0]
        
        # Documents by status
        cursor.execute("SELECT status, COUNT(*) FROM documents GROUP BY status")
        status_counts = {status: count for status, count in cursor.fetchall()}
        
        # Authors
        cursor.execute("SELECT DISTINCT author FROM documents")
        authors = [author[0] for author in cursor.fetchall() if author[0]]
        
        return {
            "total_documents": total_documents,
            "total_concepts": total_concepts,
            "total_qa_pairs": total_qa_pairs,
            "document_status": status_counts,
            "authors": authors
        }
    except Error as e:
        logger.error(f"Error fetching document statistics: {e}")
        return {}
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_all_documents() -> List[Dict]:
    """Get list of all documents in the database."""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
        SELECT 
            d.document_id, 
            d.filename, 
            d.title, 
            d.author, 
            d.processed_date, 
            d.status,
            COUNT(DISTINCT c.concept_id) as concept_count,
            COUNT(DISTINCT q.qa_id) as qa_count
        FROM 
            documents d
        LEFT JOIN 
            concepts c ON d.document_id = c.document_id
        LEFT JOIN 
            qa_pairs q ON d.document_id = q.document_id
        GROUP BY 
            d.document_id
        ORDER BY 
            d.processed_date DESC
        """
        cursor.execute(query)
        documents = cursor.fetchall()
        return documents
    except Error as e:
        logger.error(f"Error fetching documents: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    connected = False
    
    try:
        conn = get_connection()
        if conn:
            connected = True
            conn.close()
    except:
        connected = False
        
    return jsonify({
        "status": "healthy" if connected else "unhealthy", 
        "service": "db-service",
        "database_connected": connected
    })

@app.route('/initialize', methods=['POST'])
def initialize_db():
    """Initialize the database"""
    data = request.json or {}
    force_recreate = data.get('force_recreate', False)
    
    success = initialize_database(force_recreate)
    
    return jsonify({
        "status": "success" if success else "error",
        "message": "Database initialized successfully" if success else "Failed to initialize database",
        "force_recreate": force_recreate
    })

@app.route('/documents', methods=['GET'])
def get_documents():
    """Get all documents"""
    documents = get_all_documents()
    
    return jsonify({
        "documents": documents,
        "total": len(documents)
    })

@app.route('/documents/<int:document_id>', methods=['GET'])
def get_document(document_id):
    """Get document by ID"""
    document = get_document_info(document_id)
    
    if not document:
        return jsonify({"error": "Document not found"}), 404
        
    return jsonify(document)

@app.route('/documents/<int:document_id>/concepts', methods=['GET'])
def get_document_concepts(document_id):
    """Get concepts for a document"""
    concepts = get_all_concepts(document_id)
    
    return jsonify({
        "document_id": document_id,
        "concepts": concepts,
        "total": len(concepts)
    })

@app.route('/documents/<int:document_id>/qa_pairs', methods=['GET'])
def get_document_qa_pairs(document_id):
    """Get QA pairs for a document"""
    qa_pairs = get_all_qa_pairs(document_id)
    
    return jsonify({
        "document_id": document_id,
        "qa_pairs": qa_pairs,
        "total": len(qa_pairs)
    })

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get database statistics"""
    stats = get_document_statistics()
    
    return jsonify(stats)

@app.route('/documents', methods=['POST'])
def create_new_document():
    """Create a new document"""
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({"error": "Filename is required"}), 400
        
    filename = data['filename']
    title = data.get('title')
    author = data.get('author')
    file_path = data.get('file_path')
    
    document_id = create_document(filename, title, author, file_path)
    
    if not document_id:
        return jsonify({"error": "Failed to create document"}), 500
        
    return jsonify({
        "status": "success",
        "document_id": document_id,
        "message": f"Document created with ID: {document_id}"
    })

# Main entry point for running as standalone service
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Database Service for PDF Wisdom Extractor")
    parser.add_argument('--port', type=int, default=5001, help='Port to run the service on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Start the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug) 