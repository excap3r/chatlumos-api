#!/usr/bin/env python3
"""
Vector Search Service Module

Provides vector embedding generation and search capabilities using Pinecone
and Sentence Transformers, initialized via main application configuration.
"""

import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union # Added Union
import requests
import numpy as np
# Removed Flask imports: Flask, request, jsonify, current_app
# Removed CORS
from pinecone import Pinecone, ServerlessSpec, PodSpec
import structlog
from sentence_transformers import SentenceTransformer

# Configure logger
logger = structlog.get_logger(__name__)

class VectorSearchService:
    """Service for vector search operations using Pinecone."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the service with configuration.
        Args:
            config: Dictionary containing configuration values like
                    PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_REGION,
                    PINECONE_INDEX_NAME, EMBEDDING_MODEL_NAME, VECTOR_SEARCH_DIMENSION etc.
        """
        self.config = config
        self.api_key = self.config.get("PINECONE_API_KEY")
        self.cloud = self.config.get("PINECONE_CLOUD", "aws")
        self.region = self.config.get("PINECONE_REGION", "us-east-1")
        self.environment = self.config.get("PINECONE_ENVIRONMENT") # For pod-based
        self.model_name = self.config.get("EMBEDDING_MODEL_NAME", 'all-MiniLM-L6-v2')
        self.index_name = self.config.get("PINECONE_INDEX_NAME", "wisdom-embeddings")
        self.dimension = self.config.get("VECTOR_SEARCH_DIMENSION") # Expected dimension

        if not self.api_key:
            logger.error("Pinecone API key not found in configuration (PINECONE_API_KEY).")
            raise ValueError("PINECONE_API_KEY not configured.")

        # Check essential packages (optional but helpful)
        # ... (package check logic can remain) ...
        try:
            __import__('pinecone')
        except ImportError:
            logger.critical("Pinecone package not installed. Run 'pip install pinecone-client'")
            raise ImportError("Pinecone package not installed.")
        try:
            __import__('torch')
            __import__('sentence_transformers')
        except ImportError:
             logger.critical("Required ML packages not installed. Run 'pip install torch sentence-transformers'")
             raise ImportError("Required ML packages (torch, sentence-transformers) not installed.")

        try:
            # Initialize Pinecone client
            self.pinecone = Pinecone(api_key=self.api_key)
            logger.info("Pinecone client initialized.")

            # Load Sentence Transformer model
            self.model = SentenceTransformer(self.model_name)
            logger.info("Sentence Transformer model loaded.", model_name=self.model_name)

            # Determine or verify dimension
            model_dim = self.model.get_sentence_embedding_dimension()
            if self.dimension is None:
                self.dimension = model_dim
                logger.info("Embedding dimension determined from model.", dimension=self.dimension)
            elif model_dim != self.dimension:
                 logger.warning("Configured VECTOR_SEARCH_DIMENSION mismatches model dimension. Using model dimension.",
                                  config_dim=self.dimension, model_dim=model_dim, model_name=self.model_name)
                 self.dimension = model_dim # Override with actual model dimension

            # Initialize Pinecone index connection
            self._init_index()

        except Exception as e:
            logger.error("Failed to initialize VectorSearchService", error=str(e), exc_info=True)
            # Ensure partial initializations don't cause issues later
            self.pinecone = None
            self.model = None
            self.index = None
            raise RuntimeError(f"Failed to initialize VectorSearchService: {e}")

    def _init_index(self):
        """Initialize Pinecone index connection, creating it if it doesn't exist."""
        if self.index_name not in self.pinecone.list_indexes().names:
            logger.warning("Pinecone index not found, attempting creation.", index_name=self.index_name)
            try:
                # Determine spec based on configuration
                if self.cloud and self.region:
                    spec = ServerlessSpec(cloud=self.cloud, region=self.region)
                    logger.info("Using ServerlessSpec for index creation", cloud=self.cloud, region=self.region)
                elif self.environment:
                    pod_type = self.config.get("PINECONE_POD_TYPE", "s1.x1")
                    spec = PodSpec(environment=self.environment, pod_type=pod_type)
                    logger.info("Using PodSpec for index creation", environment=self.environment, pod_type=pod_type)
                else:
                    raise ValueError("Pinecone index configuration insufficient. Need cloud/region (Serverless) or environment (Pod)." )

                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine", # Or config-driven: self.config.get("PINECONE_METRIC", "cosine")
                    spec=spec,
                    timeout=self.config.get("PINECONE_CREATE_TIMEOUT", 300) # Configurable timeout
                )
                # Basic wait, consider checking index status actively in production
                wait_time = self.config.get("PINECONE_CREATE_WAIT_SECONDS", 15)
                logger.info(f"Pinecone index creation initiated, waiting {wait_time}s for potential readiness...", index_name=self.index_name)
                time.sleep(wait_time)

            except Exception as e:
                logger.error("Failed to create Pinecone index", index_name=self.index_name, error=str(e), exc_info=True)
                raise RuntimeError(f"Failed to create Pinecone index '{self.index_name}': {e}")
        else:
            logger.info("Pinecone index already exists.", index_name=self.index_name)

        # Connect to the index
        self.index = self.pinecone.Index(self.index_name)
        try:
             # Verify connection with describe_index_stats
             index_stats = self.index.describe_index_stats()
             logger.info("Successfully connected to Pinecone index.", index_name=self.index_name, stats=index_stats)
        except Exception as e:
             logger.error("Failed to connect to or describe Pinecone index after creation/check.", index_name=self.index_name, error=str(e), exc_info=True)
             # This might be critical, raise an error
             raise RuntimeError(f"Could not connect to or describe index '{self.index_name}': {e}")

    def _get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embedding(s) for the given text or list of texts."""
        if not self.model:
            raise RuntimeError("SentenceTransformer model is not initialized.")
        try:
            embeddings = self.model.encode(text, show_progress_bar=False) # Disable progress bar for non-interactive use
            logger.debug("Generated embedding(s)", input_type=type(text).__name__, count=1 if isinstance(text, str) else len(text))
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1 else [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error("Failed to generate embedding(s)", error=str(e), exc_info=True)
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def add_documents(self, user_id: str, documents: List[Tuple[str, Dict[str, Any]]]) -> Tuple[int, int]:
        """Add a batch of documents (text and metadata) to the index."""
        if not self.index:
            raise RuntimeError("Pinecone index is not initialized.")
        if not documents:
            logger.warning("Attempted to add empty batch of documents.", user_id=user_id)
            return 0, 0

        texts_to_embed = [doc[0] for doc in documents if doc[0]]
        original_indices = [i for i, doc in enumerate(documents) if doc[0]]

        if not texts_to_embed:
            logger.warning("No valid text found in the document batch.", user_id=user_id, total_docs=len(documents))
            return 0, len(documents)

        success_count = 0
        failure_count = len(documents) - len(texts_to_embed)

        try:
            embeddings = self._get_embedding(texts_to_embed)
            vectors_to_upsert = []

            for i, embedding in enumerate(embeddings):
                original_doc_index = original_indices[i]
                text, metadata = documents[original_doc_index]
                # Use chunk_id if present, fallback to generated ID
                doc_id = metadata.get("chunk_id") or metadata.get("id") or str(uuid.uuid4())

                # Prepare metadata for Pinecone (simple types, stringify complex ones)
                pinecone_metadata = {
                    k: str(v) if isinstance(v, (dict, list, uuid.UUID)) else v
                    for k, v in metadata.items() 
                    if v is not None and k not in ['id', 'chunk_id'] # Exclude ID fields used as vector ID
                }
                pinecone_metadata['user_id'] = str(user_id) # Ensure user_id is string
                # Store reference keys if available
                if metadata.get('document_id'): pinecone_metadata['document_id'] = str(metadata['document_id'])
                if metadata.get('chunk_index') is not None: pinecone_metadata['chunk_index'] = int(metadata['chunk_index']) # Store as int if possible
                if metadata.get('filename'): pinecone_metadata['filename'] = str(metadata['filename'])

                vectors_to_upsert.append((str(doc_id), embedding, pinecone_metadata))

            if vectors_to_upsert:
                # Consider batching upserts if vectors_to_upsert is very large
                batch_size = self.config.get('PINECONE_UPSERT_BATCH_SIZE', 100)
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i+batch_size]
                    upsert_response = self.index.upsert(vectors=batch)
                    upserted_count_batch = upsert_response.get('upserted_count', 0)
                    success_count += upserted_count_batch
                    failure_count += (len(batch) - upserted_count_batch)
                    if upserted_count_batch < len(batch):
                         logger.warning("Partial success in Pinecone upsert batch", 
                                          user_id=user_id, attempted=len(batch), succeeded=upserted_count_batch)
                
                logger.info("Documents batch processed for Pinecone upsert.",
                             user_id=user_id,
                             total_attempted=len(vectors_to_upsert),
                             total_succeeded=success_count)
            else:
                 logger.warning("No vectors generated for upsert after embedding.", user_id=user_id)

            return success_count, failure_count

        except Exception as e:
            logger.error("Failed during add_documents batch processing", user_id=user_id, batch_size=len(documents), error=str(e), exc_info=True)
            return success_count, failure_count + (len(documents) - success_count - failure_count) # Assume remaining failed

    def query(self, user_id: str, query_text: str, top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict]:
        """Query the index for relevant documents, automatically filtering by user_id."""
        if not self.index:
            raise RuntimeError("Pinecone index is not initialized.")
        if not query_text:
            logger.warning("Attempted query with empty text.", user_id=user_id)
            return []

        try:
            query_embedding = self._get_embedding(query_text)

            # Build final Pinecone filter, ensuring user_id is included
            final_filter = {'user_id': str(user_id)} # User ID must be string for filter
            if filter:
                # Merge external filters, ensuring values are suitable types
                for k, v in filter.items():
                    if k != 'user_id': # Don't allow overriding user_id filter
                         # Basic type check/conversion for common filter scenarios
                         if isinstance(v, (str, int, float, bool)):
                              final_filter[k] = v
                         elif isinstance(v, list):
                              # Assuming list filter is for $in operator, check elements
                              final_filter[k] = {"$in": [str(item) for item in v]} # Example: convert all to string
                         else:
                              final_filter[k] = str(v) # Fallback: convert to string
            
            logger.debug("Performing Pinecone query", user_id=user_id, top_k=top_k, filter=final_filter)

            # Execute the query
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=final_filter,
                include_metadata=True
            )

            # Format results
            results = []
            for match in query_response.get('matches', []):
                # Try to reconstruct original format somewhat
                result_item = {
                    'id': match.get('id'),
                    'score': match.get('score'),
                    'metadata': match.get('metadata', {})
                    # Optionally add 'page_content' if stored in metadata, or fetch separately if needed
                }
                results.append(result_item)
            
            logger.info("Pinecone query successful", user_id=user_id, query_preview=query_text[:50]+"...", results_count=len(results))
            return results

        except Exception as e:
            logger.error("Failed to execute Pinecone query", user_id=user_id, query_preview=query_text[:50]+"...", error=str(e), exc_info=True)
            return [] # Return empty list on error

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all vectors associated with a specific user_id."""
        if not self.index:
            raise RuntimeError("Pinecone index is not initialized.")
        logger.warning("Attempting to delete all vector data for user", user_id=user_id)
        try:
            # Use the delete operation with a filter for the user_id
            delete_response = self.index.delete(filter={'user_id': str(user_id)})
            # The response might be empty or {} on success, check docs if specific confirmation needed
            logger.info("Pinecone delete operation for user data completed.", user_id=user_id, response=delete_response)
            # We assume success if no exception was raised, delete is often async
            return True
        except Exception as e:
            logger.error("Failed to delete user data from Pinecone", user_id=user_id, error=str(e), exc_info=True)
            return False

    # Maybe add other utility methods like: get_index_stats, delete_by_ids, etc.

# Removed Flask app, CORS, create_vector_app function, and all @app.route definitions
# Removed __main__ block