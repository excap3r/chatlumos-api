#!/usr/bin/env python3
"""
MySQL to Vector Database Converter

This script extracts QA pairs and concepts from the MySQL database
and converts them to vector embeddings for storage in Pinecone.
"""

import os
import sys
import argparse
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

# Import Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    print("Error: Pinecone package not installed. Run 'pip install pinecone-client'")
    sys.exit(1)

# Import embedding model
try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required ML packages not installed. Run 'pip install torch sentence-transformers'")
    sys.exit(1)
    
# Optional module for efficient queue processing
try:
    from concurrent.futures import ThreadPoolExecutor
    CONCURRENT_PROCESSING = True
except ImportError:
    CONCURRENT_PROCESSING = False

# Global Pinecone instance
pc = None

# ------------------ Database Connection ------------------

def get_mysql_connection():
    """Create and return a connection to MySQL database."""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "IAAI")
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

# ------------------ Data Extraction ------------------

def get_all_documents() -> List[Dict]:
    """Get list of all documents in the database."""
    conn = get_mysql_connection()
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
        print(f"Error fetching documents: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_all_concepts(document_id: Optional[int] = None) -> List[Dict]:
    """Get all concepts from database, optionally filtered by document_id."""
    conn = get_mysql_connection()
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
        print(f"Error fetching concepts: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_all_qa_pairs(document_id: Optional[int] = None) -> List[Dict]:
    """Get all QA pairs from database, optionally filtered by document_id."""
    conn = get_mysql_connection()
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
        print(f"Error fetching QA pairs: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# ------------------ Translation Functions ------------------

def translate_text(text: str, target_language: str = "en") -> str:
    """
    Translation function stub that returns the original text.
    Since all data is already in English, no translation is needed.
    """
    return text

def batch_translate(texts: List[str], target_language: str = "en", 
                   batch_size: int = 10) -> List[str]:
    """
    Batch translation stub that returns the original texts.
    Since all data is already in English, no translation is needed.
    """
    return texts

# ------------------ Embedding Generation ------------------

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and return the sentence transformer embedding model."""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

def generate_embeddings(texts: List[str], model) -> List[np.ndarray]:
    """Generate embeddings for a list of texts."""
    try:
        # Process in batches to avoid memory issues with large datasets
        batch_size = 32
        all_embeddings = []
        max_retries = 3
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i+batch_size]
            
            # Implement retry logic for API failures
            for retry in range(max_retries):
                try:
                    batch_embeddings = model.encode(batch)
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"Error in batch {i//batch_size + 1}, retrying ({retry+1}/{max_retries}): {e}")
                        time.sleep(2)  # Wait before retrying
                    else:
                        print(f"Failed to process batch {i//batch_size + 1} after {max_retries} attempts: {e}")
                        # Return empty embeddings as fallback
                        return []
            
        return all_embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# ------------------ Pinecone Vector Database ------------------

def init_pinecone(api_key: str) -> bool:
    """Initialize Pinecone client."""
    global pc
    try:
        pc = Pinecone(api_key=api_key)
        # Verify connection by listing indexes
        try:
            pc.list_indexes()
            print("Successfully connected to Pinecone")
            return True
        except Exception as e:
            print(f"Error verifying Pinecone connection: {e}")
            print("Please check your Pinecone API key and network connection")
            return False
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        print("Please check if your Pinecone API key is valid")
        return False

def create_pinecone_index(index_name: str, dimension: int, metric: str = "cosine") -> bool:
    """Create a Pinecone index if it doesn't exist."""
    global pc
    try:
        # Check if the index already exists
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists")
            return True
            
        # Create a new index with required cloud and region parameters for free tier
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Created new Pinecone index: {index_name}")
        # Wait for index to be initialized
        time.sleep(10)
        return True
    except Exception as e:
        print(f"Error creating Pinecone index: {e}")
        return False

def upsert_to_pinecone(index_name: str, items: List[Dict], batch_size: int = 100) -> bool:
    """
    Upsert items to Pinecone index.
    Each item should have: id, values (embedding), metadata
    """
    global pc
    try:
        # Connect to the index
        index = pc.Index(index_name)
        
        # Split items into batches
        total_items = len(items)
        successful_items = 0
        failed_batches = 0
        
        for i in tqdm(range(0, len(items), batch_size), desc=f"Upserting to {index_name}"):
            batch = items[i:i+batch_size]
            try:
                index.upsert(vectors=batch)
                successful_items += len(batch)
                
                # Rate limiting to avoid API limitations
                if i + batch_size < len(items):
                    time.sleep(0.5)
            except Exception as e:
                failed_batches += 1
                print(f"Error upserting batch {i//batch_size + 1}: {e}")
                # Continue with next batch instead of failing completely
                continue
                
        print(f"Successfully upserted {successful_items}/{total_items} items to Pinecone index '{index_name}'")
        if failed_batches > 0:
            print(f"Warning: {failed_batches} batches failed during upsert")
            
        # Consider operation successful if at least 90% of items were upserted
        return successful_items >= total_items * 0.9
    except Exception as e:
        print(f"Error upserting to Pinecone: {e}")
        return False

# ------------------ Main Processing Functions ------------------

def prepare_concept_items(concepts: List[Dict], model) -> List[Dict]:
    """Prepare concept items for Pinecone with embeddings."""
    if not concepts:
        return []
        
    # Extract texts to embed
    concept_texts = []
    for concept in concepts:
        # Combine concept name and explanation for better semantic search
        combined_text = f"{concept['concept']}: {concept['explanation']}"
        concept_texts.append(combined_text)
    
    # Generate embeddings (no translation needed as all data is in English)
    print("Generating embeddings for concepts...")
    embeddings = generate_embeddings(concept_texts, model)
    
    if len(embeddings) != len(concepts):
        print(f"Error: Mismatch between concepts ({len(concepts)}) and embeddings ({len(embeddings)})")
        return []
    
    # Prepare items for Pinecone
    items = []
    for i, (concept, embedding) in enumerate(zip(concepts, embeddings)):
        item = {
            "id": f"concept-{concept['concept_id']}",
            "values": embedding.tolist(),
            "metadata": {
                "type": "concept",
                "concept_id": concept['concept_id'],
                "document_id": concept['document_id'],
                "document_title": concept['document_title'],
                "document_author": concept['document_author'],
                "concept": concept['concept'],
                "explanation": concept['explanation'],
                "text": concept_texts[i]  # Include the text that was embedded
            }
        }
        items.append(item)
    
    return items

def prepare_qa_items(qa_pairs: List[Dict], model) -> List[Dict]:
    """Prepare QA items for Pinecone with embeddings."""
    if not qa_pairs:
        return []
        
    # Extract texts to embed
    qa_texts = []
    for qa in qa_pairs:
        # Combine question and answer for better semantic search
        combined_text = f"Q: {qa['question']} A: {qa['answer']}"
        qa_texts.append(combined_text)
    
    # Also create separate embeddings for questions for better search precision
    question_texts = [qa['question'] for qa in qa_pairs]
    
    # Generate embeddings (no translation needed as all data is in English)
    print("Generating embeddings for QA pairs...")
    qa_embeddings = generate_embeddings(qa_texts, model)
    question_embeddings = generate_embeddings(question_texts, model)
    
    if len(qa_embeddings) != len(qa_pairs) or len(question_embeddings) != len(qa_pairs):
        print(f"Error: Mismatch between QA pairs ({len(qa_pairs)}) and embeddings")
        return []
    
    # Prepare items for Pinecone
    items = []
    
    # Add combined QA embeddings
    for i, (qa, embedding) in enumerate(zip(qa_pairs, qa_embeddings)):
        item = {
            "id": f"qa-{qa['qa_id']}",
            "values": embedding.tolist(),
            "metadata": {
                "type": "qa_pair",
                "qa_id": qa['qa_id'],
                "document_id": qa['document_id'],
                "document_title": qa['document_title'],
                "document_author": qa['document_author'],
                "question": qa['question'],
                "answer": qa['answer'],
                "text": qa_texts[i]  # Include the text that was embedded
            }
        }
        items.append(item)
    
    # Add question-only embeddings for more precise question matching
    for i, (qa, embedding) in enumerate(zip(qa_pairs, question_embeddings)):
        item = {
            "id": f"question-{qa['qa_id']}",
            "values": embedding.tolist(),
            "metadata": {
                "type": "question",
                "qa_id": qa['qa_id'],
                "document_id": qa['document_id'],
                "document_title": qa['document_title'],
                "document_author": qa['document_author'],
                "question": qa['question'],
                "answer": qa['answer'],
                "text": question_texts[i]  # Include the text that was embedded
            }
        }
        items.append(item)
    
    return items

def process_document_to_vectors(document_id: int, 
                               embedding_model_name: str = "all-MiniLM-L6-v2",
                               pinecone_index_name: str = "wisdom-embeddings") -> bool:
    """Process a single document to vectors and store in Pinecone."""
    # Load embedding model
    model = load_embedding_model(embedding_model_name)
    if not model:
        return False
    
    # Get embedding dimension for Pinecone index
    test_embedding = model.encode(["Test text"])[0]
    embedding_dimension = len(test_embedding)
    
    # Create Pinecone index if it doesn't exist
    if not create_pinecone_index(pinecone_index_name, embedding_dimension):
        return False
    
    # Get concepts and QA pairs for this document
    print(f"Fetching concepts and QA pairs for document ID {document_id}...")
    concepts = get_all_concepts(document_id)
    qa_pairs = get_all_qa_pairs(document_id)
    
    print(f"Found {len(concepts)} concepts and {len(qa_pairs)} QA pairs")
    
    # Prepare data for Pinecone
    if concepts:
        concept_items = prepare_concept_items(concepts, model)
        if concept_items:
            if not upsert_to_pinecone(pinecone_index_name, concept_items):
                print("Failed to upsert concepts to Pinecone")
                
    if qa_pairs:
        qa_items = prepare_qa_items(qa_pairs, model)
        if qa_items:
            if not upsert_to_pinecone(pinecone_index_name, qa_items):
                print("Failed to upsert QA pairs to Pinecone")
    
    return True

def process_all_documents_to_vectors(embedding_model_name: str = "all-MiniLM-L6-v2",
                                    pinecone_index_name: str = "wisdom-embeddings") -> bool:
    """Process all documents to vectors and store in Pinecone."""
    documents = get_all_documents()
    if not documents:
        print("No documents found in database")
        return False
    
    print(f"Found {len(documents)} documents to process")
    
    # Process each document
    success_count = 0
    for doc in documents:
        print(f"\nProcessing document ID {doc['document_id']}: {doc['title']}")
        if process_document_to_vectors(
            doc['document_id'], 
            embedding_model_name, 
            pinecone_index_name
        ):
            success_count += 1
    
    print(f"\nSuccessfully processed {success_count} out of {len(documents)} documents")
    return success_count > 0

# ------------------ Command Line Interface ------------------

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert MySQL data to Pinecone vector database")
    
    parser.add_argument("--document_id", type=int, help="Process only a specific document ID")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                        help="Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)")
    parser.add_argument("--index_name", type=str, default="wisdom-embeddings",
                        help="Name for the Pinecone index (default: wisdom-embeddings)")
    parser.add_argument("--list_documents", action="store_true", help="List all documents in the database")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Load environment variables
    load_dotenv()
    
    # Get Pinecone credentials
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not pinecone_api_key:
        print("Error: PINECONE_API_KEY must be set in .env file")
        sys.exit(1)
    
    # Check if we're in production environment
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    if is_production:
        print("Running in production mode")
        # In production, verify all required environment variables
        required_vars = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE", "PINECONE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            sys.exit(1)
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize Pinecone
    if not init_pinecone(pinecone_api_key):
        print("Failed to initialize Pinecone")
        sys.exit(1)
    
    # List documents if requested
    if args.list_documents:
        documents = get_all_documents()
        if not documents:
            print("No documents found in database")
            sys.exit(0)
        
        print("\nDocuments in the database:")
        print("-" * 80)
        print(f"{'ID':5} | {'Title':<30} | {'Author':<20} | {'Concepts':<8} | {'QA Pairs':<8}")
        print("-" * 80)
        for doc in documents:
            print(f"{doc['document_id']:<5} | {doc['title'][:30]:<30} | {doc['author'][:20]:<20} | {doc['concept_count']:<8} | {doc['qa_count']:<8}")
        sys.exit(0)
    
    # Process documents
    if args.document_id:
        # Process single document
        if process_document_to_vectors(
            args.document_id, 
            args.model,
            args.index_name
        ):
            print(f"Successfully processed document ID {args.document_id}")
        else:
            print(f"Failed to process document ID {args.document_id}")
    else:
        # Process all documents
        if process_all_documents_to_vectors(
            args.model,
            args.index_name
        ):
            print("Successfully processed all documents")
        else:
            print("Failed to process all documents")

if __name__ == "__main__":
    main() 