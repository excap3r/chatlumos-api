#!/usr/bin/env python3
"""
Database Service Module

Provides database operations for documents, chunks, concepts, QA pairs, etc.,
using SQLAlchemy session management.
"""

import os
import json
# import time # Keep removed
from typing import Dict, List, Any, Optional, Sequence
# import mysql.connector # Keep removed
# from mysql.connector import pooling, Error # Keep removed
import structlog
from flask import current_app
from datetime import datetime
from sqlalchemy import desc, func # For ordering and func

# SQLAlchemy imports
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Import custom exceptions
from services.db.exceptions import QueryError, NotFoundError, DuplicateEntryError, DatabaseError, ConnectionError

# Import Models (adjust path if needed)
from services.db.models.document_models import Document, DocumentChunk, Concept, QAPair, Summary
# Import DB session handler decorator
from services.db.db_utils import handle_db_session

# Configure logger
logger = structlog.get_logger(__name__)

# Removed DBService class and create_pool method

# Removed get_connection function

# --- Helper function to get session --- (Copied from user_db.py)
def _get_session() -> Session:
    """Get the SQLAlchemy session from Flask's current_app."""
    if not hasattr(current_app, 'db_session') or current_app.db_session is None:
        logger.error("SQLAlchemy session not initialized in current_app.")
        raise ConnectionError("Database session not available.")
    # db_session is a scoped_session factory, call it to get the actual session
    return current_app.db_session()


# --- Document Operations (now regular functions) --- #

@handle_db_session
def create_document(filename: str, title: str = None, author: str = None, file_path: str = None) -> Optional[int]:
    """
    Creates or updates a document entry using SQLAlchemy ORM with MySQL upsert.
    Returns the document_id.
    """
    session = _get_session()
    try:
        # Prepare data for insert/update
        insert_data = {
            "filename": filename,
            "title": title,
            "author": author,
            "file_path": file_path,
            "status": "processing" # Set initial status
        }
        # Filter out None values for the initial insert part
        values_to_insert = {k: v for k, v in insert_data.items() if v is not None}
        values_to_insert['filename'] = filename # Ensure filename is always included

        # Define fields to update on duplicate. Use func.values() for MySQL upsert.
        update_data = {}
        if title is not None:
            update_data["title"] = func.values(Document.title)
        if author is not None:
            update_data["author"] = func.values(Document.author)
        if file_path is not None:
            update_data["file_path"] = func.values(Document.file_path)
        update_data["status"] = "processing" # Always reset status on upsert
        

        # Construct the upsert statement
        stmt = insert(Document).values(values_to_insert)
        upsert_stmt = stmt.on_duplicate_key_update(**update_data)

        # Execute the statement
        result = session.execute(upsert_stmt)
        session.commit()

        # Get the document_id
        document_id = None
        if result.inserted_primary_key:
             document_id = result.inserted_primary_key[0]
             logger.info("Document created via INSERT", filename=filename, document_id=document_id)
        else:
             # If no insert occurred (update happened), query the ID
             doc = session.query(Document.document_id).filter(Document.filename == filename).scalar()
             if doc is not None:
                 document_id = doc
                 logger.info("Document updated via ON DUPLICATE KEY UPDATE", filename=filename, document_id=document_id)
             else:
                 # This case is unlikely but possible if commit failed silently or another issue
                 logger.error("Failed to retrieve document_id after upsert", filename=filename)
                 raise QueryError(f"Could not determine document_id for {filename} after upsert.")

        return document_id

    except IntegrityError as e: # Should be caught by on_duplicate_key_update mostly
         session.rollback()
         logger.error("Integrity error during document upsert (unexpected)", filename=filename, error=str(e), exc_info=True)
         raise QueryError(f"Database integrity error for document: {e}")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error creating/updating document", filename=filename, error=str(e), exc_info=True)
        raise QueryError(f"Failed to create/update document: {e}")
    except Exception as e:
         session.rollback()
         logger.error("Unexpected error creating/updating document", filename=filename, error=str(e), exc_info=True)
         raise DatabaseError(f"Unexpected error processing document: {e}")

@handle_db_session
def update_document_status(document_id: int, status: str, total_chunks: Optional[int] = None, full_text: Optional[str] = None) -> bool:
    """Update the status and optionally other fields of a document using ORM."""
    session = _get_session()
    try:
        # Find the document
        doc = session.query(Document).filter(Document.document_id == document_id).first()

        if not doc:
            logger.warning("Document not found for status update", document_id=document_id)
            raise NotFoundError(f"Document with ID {document_id} not found.")

        # Update fields
        doc.status = status
        update_fields = {'status': status} # Keep track of what was updated for logging
        if total_chunks is not None:
            doc.total_chunks = total_chunks
            update_fields['total_chunks'] = total_chunks
        if full_text is not None:
            doc.full_text = full_text
            # Not logging full_text value itself for brevity/security
            update_fields['full_text'] = '(updated)'

        # Note: updated_at on Document model should handle timestamp automatically if configured

        session.commit()
        logger.info("Document status/fields updated", document_id=document_id, updated_fields=update_fields)
        return True

    except NotFoundError:
         # No rollback needed for not found error
         raise
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error updating document status", document_id=document_id, status=status, error=str(e), exc_info=True)
        raise QueryError(f"Failed to update document status: {e}")
    except Exception as e:
         session.rollback()
         logger.error("Unexpected error updating document status", document_id=document_id, error=str(e), exc_info=True)
         raise DatabaseError(f"Unexpected error updating document status: {e}")

@handle_db_session
def create_document_chunk(document_id: int, chunk_index: int, chunk_text: str) -> int:
    """Creates a new document chunk using ORM and returns its ID."""
    session = _get_session()
    new_chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        status='pending' # Default status
        # chunk_id is auto-incrementing PK
        # processed_date has server_default
    )
    try:
        session.add(new_chunk)
        # Flush to get the new chunk_id before commit (optional but good practice)
        session.flush()
        chunk_id = new_chunk.chunk_id # Get the auto-generated ID

        if chunk_id is None:
             # This shouldn't happen if flush() worked and PK is auto-increment
             logger.error("Chunk ID is None after flush", document_id=document_id, chunk_index=chunk_index)
             raise QueryError("Failed to get ID for newly created chunk after flush.")

        session.commit()
        logger.info("Document chunk created", document_id=document_id, chunk_index=chunk_index, chunk_id=chunk_id)
        return chunk_id

    except IntegrityError as e:
        session.rollback()
        # Check the error details if possible (depends on DBAPI driver)
        # e.g., str(e.orig) might contain constraint name or duplicate key info
        logger.warning("Integrity error creating chunk (duplicate index or invalid document_id?)",
                     document_id=document_id, chunk_index=chunk_index, error_info=str(e.orig))
        # Raise specific errors based on heuristics or keep generic
        if "FOREIGN KEY" in str(e.orig).upper():
             raise NotFoundError(f"Document with ID {document_id} not found.") from e
        elif "uq_doc_chunk_idx" in str(e.orig) or "Duplicate entry" in str(e.orig): # Check specific constraint name or generic message
             raise DuplicateEntryError(f"Chunk with index {chunk_index} already exists for document {document_id}.") from e
        else:
             raise QueryError(f"Database integrity error creating chunk: {e}") from e

    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error creating document chunk", document_id=document_id, chunk_index=chunk_index, error=str(e), exc_info=True)
        raise QueryError(f"Failed to create document chunk: {e}")
    except Exception as e:
        session.rollback()
        logger.error("Unexpected error creating document chunk", document_id=document_id, error=str(e), exc_info=True)
        raise DatabaseError(f"Unexpected error creating document chunk: {e}")

@handle_db_session
def update_chunk_status(chunk_id: int, status: str) -> bool:
    """Updates the status of a specific chunk using ORM."""
    session = _get_session()
    try:
        # Find the chunk
        chunk = session.query(DocumentChunk).filter(DocumentChunk.chunk_id == chunk_id).first()

        if not chunk:
            logger.warning("Chunk not found for status update", chunk_id=chunk_id)
            raise NotFoundError(f"Chunk with ID {chunk_id} not found.")

        # Update status
        chunk.status = status
        # Note: processed_date on chunk model should handle timestamp automatically if configured

        session.commit()
        logger.info("Chunk status updated", chunk_id=chunk_id, status=status)
        return True

    except NotFoundError:
         raise
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error updating chunk status", chunk_id=chunk_id, status=status, error=str(e), exc_info=True)
        raise QueryError(f"Failed to update chunk status: {e}")
    except Exception as e:
         session.rollback()
         logger.error("Unexpected error updating chunk status", chunk_id=chunk_id, error=str(e), exc_info=True)
         raise DatabaseError(f"Unexpected error updating chunk status: {e}")

@handle_db_session
def store_concepts(document_id: int, chunk_id: int, concepts: List[Dict[str, str]]) -> bool:
    """Stores multiple concepts for a given chunk using ORM."""
    if not concepts:
        logger.debug("No concepts provided to store", document_id=document_id, chunk_id=chunk_id)
        return True # Nothing to store is considered success

    session = _get_session()
    concept_objects = []
    for concept_data in concepts:
        concept_text = concept_data.get('concept')
        explanation = concept_data.get('explanation')
        if not concept_text or not explanation:
            logger.warning("Skipping concept due to missing 'concept' or 'explanation'", concept_data=concept_data)
            continue
        
        concept_objects.append(Concept(
            document_id=document_id,
            chunk_id=chunk_id,
            concept=concept_text,
            explanation=explanation
        ))

    if not concept_objects:
        logger.debug("No valid concept objects to save after filtering.", document_id=document_id, chunk_id=chunk_id)
        return True # Still considered success if all inputs were invalid

    try:
        session.bulk_save_objects(concept_objects)
        session.commit()
        logger.info(f"Stored {len(concept_objects)} concepts for chunk", document_id=document_id, chunk_id=chunk_id)
        return True

    except IntegrityError as e:
        session.rollback()
        logger.warning("Integrity error storing concepts (invalid doc/chunk ID?)",
                     document_id=document_id, chunk_id=chunk_id, error_info=str(e.orig))
        # Determine if it's a foreign key violation
        if "FOREIGN KEY" in str(e.orig).upper():
             raise NotFoundError(f"Document ID {document_id} or Chunk ID {chunk_id} not found.") from e
        else:
             raise QueryError(f"Database integrity error storing concepts: {e}") from e
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error storing concepts", document_id=document_id, chunk_id=chunk_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to store concepts: {e}")
    except Exception as e:
        session.rollback()
        logger.error("Unexpected error storing concepts", document_id=document_id, error=str(e), exc_info=True)
        raise DatabaseError(f"Unexpected error storing concepts: {e}")

@handle_db_session
def store_qa_pairs(document_id: int, chunk_id: int, qa_pairs: List[Dict[str, str]]) -> bool:
    """Stores multiple question-answer pairs for a given chunk using ORM."""
    if not qa_pairs:
        logger.debug("No QA pairs provided to store", document_id=document_id, chunk_id=chunk_id)
        return True # Nothing to store is considered success

    session = _get_session()
    qa_objects = []
    for qa_data in qa_pairs:
        question = qa_data.get('question')
        answer = qa_data.get('answer')
        if not question or not answer:
            logger.warning("Skipping QA pair due to missing 'question' or 'answer'", qa_data=qa_data)
            continue
        
        qa_objects.append(QAPair(
            document_id=document_id,
            chunk_id=chunk_id,
            question=question,
            answer=answer
        ))

    if not qa_objects:
        logger.debug("No valid QA pair objects to save after filtering.", document_id=document_id, chunk_id=chunk_id)
        return True

    try:
        session.bulk_save_objects(qa_objects)
        session.commit()
        logger.info(f"Stored {len(qa_objects)} QA pairs for chunk", document_id=document_id, chunk_id=chunk_id)
        return True

    except IntegrityError as e:
        session.rollback()
        logger.warning("Integrity error storing QA pairs (invalid doc/chunk ID?)",
                     document_id=document_id, chunk_id=chunk_id, error_info=str(e.orig))
        # Determine if it's a foreign key violation
        if "FOREIGN KEY" in str(e.orig).upper():
             raise NotFoundError(f"Document ID {document_id} or Chunk ID {chunk_id} not found.") from e
        else:
             raise QueryError(f"Database integrity error storing QA pairs: {e}") from e
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error storing QA pairs", document_id=document_id, chunk_id=chunk_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to store QA pairs: {e}")
    except Exception as e:
        session.rollback()
        logger.error("Unexpected error storing QA pairs", document_id=document_id, error=str(e), exc_info=True)
        raise DatabaseError(f"Unexpected error storing QA pairs: {e}")

@handle_db_session
def store_summary(document_id: int, summary_text: str) -> bool:
    """Stores a summary for a given document using ORM."""
    session = _get_session()
    new_summary = Summary(
        document_id=document_id,
        summary_text=summary_text
        # summary_id is auto-incrementing PK
        # generated_date has server_default
    )
    try:
        session.add(new_summary)
        session.commit()
        logger.info("Summary stored successfully", document_id=document_id)
        return True
    except IntegrityError as e:
        session.rollback()
        logger.warning("Integrity error storing summary (invalid document ID or duplicate summary?) ",
                     document_id=document_id, error_info=str(e.orig))
        if "FOREIGN KEY" in str(e.orig).upper():
            raise NotFoundError(f"Document with ID {document_id} not found.") from e
        elif "Duplicate entry" in str(e.orig): # Check if document_id has a unique constraint
            # Option 1: Update existing summary if duplicate found
            logger.info("Summary already exists for document, updating.", document_id=document_id)
            try:
                existing_summary = session.query(Summary).filter(Summary.document_id == document_id).first()
                if existing_summary:
                    existing_summary.summary_text = summary_text
                    existing_summary.generated_date = datetime.utcnow() # Manually update timestamp if no onupdate
                    session.commit()
                    return True
                else:
                    # Should not happen if Duplicate entry error occurred, but handle defensively
                    raise QueryError("Failed to find existing summary after duplicate entry error.")
            except SQLAlchemyError as update_e:
                 session.rollback()
                 logger.error("Error updating existing summary", document_id=document_id, error=str(update_e))
                 raise QueryError(f"Failed to update existing summary: {update_e}") from update_e
            # Option 2: Raise DuplicateEntryError
            # raise DuplicateEntryError(f"Summary already exists for document ID {document_id}.") from e
        else:
            raise QueryError(f"Database integrity error storing summary: {e}") from e
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error storing summary", document_id=document_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to store summary: {e}")
    except Exception as e:
        session.rollback()
        logger.error("Unexpected error storing summary", document_id=document_id, error=str(e), exc_info=True)
        raise DatabaseError(f"Unexpected error storing summary: {e}")

@handle_db_session
def get_all_concepts(document_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves all concepts, optionally filtered by document ID, using ORM."""
    session = _get_session()
    try:
        query = session.query(Concept)
        if document_id:
            # Ensure the filter uses the correct column name
            query = query.filter(Concept.document_id == document_id)

        concepts: Sequence[Concept] = query.order_by(Concept.concept_id).all() # Add ordering if desired

        # Convert Concept objects to dictionaries
        result_list = [
            {
                "concept_id": c.concept_id,
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "concept_name": c.concept_name,
                "explanation": c.explanation
            }
            for c in concepts
        ]

        logger.debug("Retrieved concepts", document_id=document_id or "all", count=len(result_list))
        return result_list

    except SQLAlchemyError as e:
        # No rollback needed for reads
        logger.error("Database error retrieving concepts", document_id=document_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to retrieve concepts: {e}")
    except Exception as e:
         logger.error("Unexpected error retrieving concepts", document_id=document_id, error=str(e), exc_info=True)
         raise DatabaseError(f"Unexpected error retrieving concepts: {e}")

@handle_db_session
def get_all_qa_pairs(document_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieves all QA pairs, optionally filtered by document ID, using ORM."""
    session = _get_session()
    try:
        query = session.query(QAPair)
        if document_id:
            query = query.filter(QAPair.document_id == document_id)

        qa_pairs: Sequence[QAPair] = query.order_by(QAPair.qa_id).all() # Add ordering if desired

        # Convert QAPair objects to dictionaries
        result_list = [
            {
                "qa_id": qa.qa_id,
                "document_id": qa.document_id,
                "chunk_id": qa.chunk_id,
                "question": qa.question,
                "answer": qa.answer
            }
            for qa in qa_pairs
        ]

        logger.debug("Retrieved QA pairs", document_id=document_id or "all", count=len(result_list))
        return result_list

    except SQLAlchemyError as e:
        logger.error("Database error retrieving QA pairs", document_id=document_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to retrieve QA pairs: {e}")
    except Exception as e:
         logger.error("Unexpected error retrieving QA pairs", document_id=document_id, error=str(e), exc_info=True)
         raise DatabaseError(f"Unexpected error retrieving QA pairs: {e}")

@handle_db_session
def get_document_summary(document_id: int) -> Optional[str]:
    """Retrieves the latest summary text for a given document ID using ORM."""
    session = _get_session()
    try:
        # Query the summary, order by date to get the latest if multiple exist
        summary = session.query(Summary.summary_text)\
            .filter(Summary.document_id == document_id)\
            .order_by(desc(Summary.generated_date))\
            .first() # Fetch only the first result (latest)

        if summary:
            logger.debug("Retrieved document summary", document_id=document_id)
            return summary[0] # .first() returns a tuple/row when querying specific columns
        else:
            logger.debug("No summary found for document", document_id=document_id)
            return None

    except SQLAlchemyError as e:
        logger.error("Database error retrieving document summary", document_id=document_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to retrieve document summary: {e}")
    except Exception as e:
         logger.error("Unexpected error retrieving summary", document_id=document_id, error=str(e), exc_info=True)
         raise DatabaseError(f"Unexpected error retrieving summary: {e}")

@handle_db_session
def get_document_info(document_id: int) -> Dict[str, Any]:
    """Retrieves comprehensive information for a document using ORM."""
    session: Session = _get_session()
    try:
        # Query the document and eagerly load related concepts and QA pairs
        doc = session.query(Document)\
            .options(
                selectinload(Document.concepts),
                selectinload(Document.qa_pairs)
            )\
            .filter(Document.id == document_id)\
            .first() # Use first() as filter is applied

        if not doc:
            logger.warning("Document not found", document_id=document_id)
            raise NotFoundError(f"Document with ID {document_id} not found.")

        # Get the latest summary separately
        # Re-use the existing function. Ensure get_document_summary handles its own session.
        summary_text = get_document_summary(document_id)

        # Format concepts
        concepts_list = [
            {
                'concept_id': concept.id,
                'concept_name': concept.concept_name,
                'explanation': concept.explanation,
                'generated_date': concept.generated_date.isoformat() if concept.generated_date else None
            } for concept in doc.concepts
        ]

        # Format QA pairs
        qa_pairs_list = [
            {
                'qa_pair_id': qa.id,
                'question': qa.question,
                'answer': qa.answer,
                'generated_date': qa.generated_date.isoformat() if qa.generated_date else None
            } for qa in doc.qa_pairs
        ]

        document_info = {
            "document_id": doc.id,
            "filename": doc.filename,
            "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
            "status": doc.status.value if doc.status else None, # Assuming status is an Enum
            "user_id": doc.user_id,
            "summary": summary_text,
            "concepts": concepts_list,
            "qa_pairs": qa_pairs_list
        }
        logger.debug("Retrieved document info", document_id=document_id)
        return document_info

    except NotFoundError: # Re-raise NotFoundError
        raise
    except SQLAlchemyError as e:
        logger.error("Database error retrieving document info", document_id=document_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to retrieve document info: {e}")
    except Exception as e:
        logger.error("Unexpected error retrieving document info", document_id=document_id, error=str(e), exc_info=True)
        raise DatabaseError(f"Unexpected error retrieving document info: {e}")

@handle_db_session
def get_document_chunks(document_id: int) -> List[Dict[str, Any]]:
    """Retrieves all chunks for a given document ID, ordered by index."""
    session = _get_session()
    try:
        chunks = session.query(DocumentChunk)\
            .filter(DocumentChunk.document_id == document_id)\
            .order_by(DocumentChunk.chunk_index)\
            .all()

        chunk_list = [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content_hash": chunk.content_hash, # Assuming content is not stored directly or too large
                "status": chunk.status.value if chunk.status else None, # Assuming status is an Enum
                "processing_start_time": chunk.processing_start_time.isoformat() if chunk.processing_start_time else None,
                "processing_end_time": chunk.processing_end_time.isoformat() if chunk.processing_end_time else None
                # Add other relevant fields from the DocumentChunk model if needed
            } for chunk in chunks
        ]
        logger.debug("Retrieved document chunks", document_id=document_id, count=len(chunk_list))
        return chunk_list

    except SQLAlchemyError as e:
        logger.error("Database error retrieving document chunks", document_id=document_id, error=str(e), exc_info=True)
        raise QueryError(f"Failed to retrieve document chunks: {e}")
    except Exception as e:
        logger.error("Unexpected error retrieving document chunks", document_id=document_id, error=str(e), exc_info=True)
        raise DatabaseError(f"Unexpected error retrieving document chunks: {e}")

@handle_db_session
def get_document_statistics() -> Dict[str, Any]:
    """Retrieves various statistics about documents, chunks, and generated content using ORM."""
    session = _get_session()
    try:
        # Total documents
        total_documents = session.query(func.count(Document.id)).scalar()

        # Documents by status
        docs_by_status_raw = session.query(
            Document.status, func.count(Document.id)
        ).group_by(Document.status).all()
        docs_by_status = {status.value if status else 'UNKNOWN': count for status, count in docs_by_status_raw}

        # Total chunks
        total_chunks = session.query(func.count(DocumentChunk.id)).scalar()

        # Chunks by status
        chunks_by_status_raw = session.query(
             DocumentChunk.status, func.count(DocumentChunk.id)
        ).group_by(DocumentChunk.status).all()
        chunks_by_status = {status.value if status else 'UNKNOWN': count for status, count in chunks_by_status_raw}

        # Total concepts
        total_concepts = session.query(func.count(Concept.id)).scalar()

        # Total QA pairs
        total_qa_pairs = session.query(func.count(QAPair.id)).scalar()

        # Total summaries
        total_summaries = session.query(func.count(Summary.id)).scalar()

        stats = {
            "total_documents": total_documents or 0,
            "documents_by_status": docs_by_status,
            "total_chunks": total_chunks or 0,
            "chunks_by_status": chunks_by_status,
            "total_concepts": total_concepts or 0,
            "total_qa_pairs": total_qa_pairs or 0,
            "total_summaries": total_summaries or 0
        }
        logger.debug("Retrieved document statistics", stats=stats)
        return stats

    except SQLAlchemyError as e:
        logger.error("Database error retrieving document statistics", error=str(e), exc_info=True)
        raise QueryError(f"Failed to retrieve document statistics: {e}")
    except Exception as e:
        logger.error("Unexpected error retrieving document statistics", error=str(e), exc_info=True)
        raise DatabaseError(f"Unexpected error retrieving document statistics: {e}")

@handle_db_session
def get_all_documents() -> List[Dict]:
    # ... (Existing code) ...
    pass # Placeholder

# Removed all Flask @app.route definitions and associated functions
# (health_check, initialize_db, get_documents, get_document, 
#  get_document_concepts, get_document_qa_pairs, get_statistics, create_new_document)

# Removed __main__ block if it existed for running this as a standalone service 