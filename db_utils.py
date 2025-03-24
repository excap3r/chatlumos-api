#!/usr/bin/env python3
"""
Database utilities for PDF Wisdom Extractor.
Handles connections, table creation, and data operations.
"""

import os
import mysql.connector
from mysql.connector import pooling
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_utils')

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'database': os.getenv('MYSQL_DATABASE', 'IAAI'),
}

# Connection pool configuration
DB_POOL_NAME = "pdf_wisdom_pool"
DB_POOL_SIZE = 5

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

def create_document(filename: str, title: str = None, author: str = "Iva Adamcová", file_path: str = None) -> Optional[int]:
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

def update_document_fulltext(document_id: int, full_text: str, total_chunks: int) -> bool:
    """
    Update the full text and total chunks for a document.
    Returns True if successful, False otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        UPDATE documents
        SET full_text = %s, total_chunks = %s
        WHERE document_id = %s
        """
        cursor.execute(query, (full_text, total_chunks, document_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error updating document full text: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def create_document_chunk(document_id: int, chunk_index: int, chunk_text: str) -> Optional[int]:
    """
    Create a new document chunk entry.
    Returns the chunk_id if successful, None otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO document_chunks (document_id, chunk_index, chunk_text, status)
        VALUES (%s, %s, %s, 'pending')
        ON DUPLICATE KEY UPDATE
        chunk_text = %s, status = 'pending'
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
    Update the status of a document chunk.
    Status can be 'pending', 'processing', 'completed', or 'failed'.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        UPDATE document_chunks
        SET status = %s
        WHERE chunk_id = %s
        """
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

def store_concepts(chunk_id: int, document_id: int, concepts: List[Dict[str, str]]) -> bool:
    """
    Store concepts extracted from a chunk.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        for concept in concepts:
            if not isinstance(concept, dict) or 'concept' not in concept or 'explanation' not in concept:
                continue
                
            query = """
            INSERT INTO concepts (document_id, chunk_id, concept_name, explanation)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            explanation = CASE 
                WHEN LENGTH(%s) > LENGTH(explanation) THEN %s
                ELSE explanation
            END
            """
            cursor.execute(
                query, 
                (
                    document_id, 
                    chunk_id, 
                    concept['concept'], 
                    concept['explanation'],
                    concept['explanation'],
                    concept['explanation']
                )
            )
        
        conn.commit()
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

def store_qa_pairs(chunk_id: int, document_id: int, qa_pairs: List[Dict[str, str]]) -> bool:
    """
    Store Q&A pairs extracted from a chunk.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        for qa in qa_pairs:
            if not isinstance(qa, dict) or 'question' not in qa or 'answer' not in qa:
                continue
                
            query = """
            INSERT INTO qa_pairs (document_id, chunk_id, question, answer)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (document_id, chunk_id, qa['question'], qa['answer']))
        
        conn.commit()
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

def store_summary(document_id: int, summary_text: str) -> bool:
    """
    Store document summary.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        INSERT INTO summaries (document_id, summary_text)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE summary_text = %s
        """
        cursor.execute(query, (document_id, summary_text, summary_text))
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

def update_document_status(document_id: int, status: str) -> bool:
    """
    Update the status of a document.
    Status can be 'processing', 'completed', or 'failed'.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
        UPDATE documents
        SET status = %s
        WHERE document_id = %s
        """
        cursor.execute(query, (status, document_id))
        conn.commit()
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

def get_all_concepts(document_id: int) -> List[Dict]:
    """
    Get all concepts for a document.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT concept_name as concept, explanation 
        FROM concepts 
        WHERE document_id = %s
        """
        cursor.execute(query, (document_id,))
        concepts = cursor.fetchall()
        return concepts
    except Exception as e:
        logger.error(f"Error getting concepts: {str(e)}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_all_qa_pairs(document_id: int) -> List[Dict]:
    """
    Get all Q&A pairs for a document.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT question, answer 
        FROM qa_pairs 
        WHERE document_id = %s
        """
        cursor.execute(query, (document_id,))
        qa_pairs = cursor.fetchall()
        return qa_pairs
    except Exception as e:
        logger.error(f"Error getting QA pairs: {str(e)}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_document_summary(document_id: int) -> Optional[str]:
    """
    Get the summary for a document.
    """
    try:
        conn = get_connection()
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
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error getting document summary: {str(e)}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_document_info(document_id: int) -> Optional[Dict]:
    """
    Get basic information about a document.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT document_id, filename, title, author, processed_date, 
               total_chunks, status, file_path 
        FROM documents 
        WHERE document_id = %s
        """
        cursor.execute(query, (document_id,))
        document = cursor.fetchone()
        return document
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_document_id_by_filename(filename: str) -> Optional[int]:
    """
    Get document ID by filename.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = "SELECT document_id FROM documents WHERE filename = %s"
        cursor.execute(query, (filename,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error getting document ID: {str(e)}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_document_statistics(document_id: int) -> Dict:
    """
    Get statistics about a document processing.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get chunk statistics
        query = """
        SELECT 
            COUNT(*) as total_chunks,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_chunks,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_chunks,
            SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing_chunks,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_chunks
        FROM document_chunks
        WHERE document_id = %s
        """
        cursor.execute(query, (document_id,))
        chunk_stats = cursor.fetchone()
        
        # Get concept count
        cursor.execute("SELECT COUNT(*) FROM concepts WHERE document_id = %s", (document_id,))
        concept_count = cursor.fetchone()[0]
        
        # Get QA pair count
        cursor.execute("SELECT COUNT(*) FROM qa_pairs WHERE document_id = %s", (document_id,))
        qa_count = cursor.fetchone()[0]
        
        # Check if summary exists
        cursor.execute("SELECT COUNT(*) FROM summaries WHERE document_id = %s", (document_id,))
        has_summary = cursor.fetchone()[0] > 0
        
        return {
            "document_id": document_id,
            "total_chunks": chunk_stats[0],
            "completed_chunks": chunk_stats[1],
            "failed_chunks": chunk_stats[2],
            "processing_chunks": chunk_stats[3],
            "pending_chunks": chunk_stats[4],
            "total_concepts": concept_count,
            "total_qa_pairs": qa_count,
            "has_summary": has_summary
        }
    except Exception as e:
        logger.error(f"Error getting document statistics: {str(e)}")
        return {
            "document_id": document_id,
            "error": str(e)
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_all_documents() -> List[Dict]:
    """
    Get a list of all documents in the database.
    """
    try:
        conn = get_connection()
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
            COUNT(DISTINCT q.qa_id) as qa_count,
            COUNT(DISTINCT s.summary_id) as summary_count,
            d.total_chunks
        FROM 
            documents d
        LEFT JOIN 
            concepts c ON d.document_id = c.document_id
        LEFT JOIN 
            qa_pairs q ON d.document_id = q.document_id
        LEFT JOIN 
            summaries s ON d.document_id = s.document_id
        GROUP BY 
            d.document_id
        ORDER BY 
            d.processed_date DESC
        """
        cursor.execute(query)
        documents = cursor.fetchall()
        return documents
    except Exception as e:
        logger.error(f"Error getting all documents: {str(e)}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    # Test database connection and initialization
    print("Testing database connection and initialization...")
    initialize_database()
    print("Database initialization complete.") 