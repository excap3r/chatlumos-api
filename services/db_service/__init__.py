"""
Database Service Package

This package provides database operations for the PDF Wisdom Extractor:
- Connection management
- CRUD operations for documents, concepts, and QA pairs
- Query utilities
"""

from .db_service import (
    # initialize_database, # REMOVED: Initialization handled in create_app
    # get_connection, # REMOVED: Session handled via app context
    create_document,
    update_document_status,
    create_document_chunk,
    update_chunk_status,
    store_concepts,
    store_qa_pairs,
    store_summary,
    get_all_concepts,
    get_all_qa_pairs,
    get_document_summary,
    get_document_info,
    get_document_statistics,
    get_all_documents
)

__all__ = [
    # 'initialize_database', # REMOVED
    # 'get_connection', # REMOVED
    'create_document',
    'update_document_status',
    'create_document_chunk',
    'update_chunk_status',
    'store_concepts',
    'store_qa_pairs',
    'store_summary',
    'get_all_concepts',
    'get_all_qa_pairs',
    'get_document_summary',
    'get_document_info',
    'get_document_statistics',
    'get_all_documents'
] 