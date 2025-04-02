#!/usr/bin/env python3
"""
Vector Search Service Module

Provides local vector embedding generation and search capabilities using Annoy
and Sentence Transformers.
"""

import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import threading

# Configure logger
logger = structlog.get_logger(__name__)

# Constants
DEFAULT_ANNOY_METRIC = 'angular' # Cosine similarity equivalent
DEFAULT_ANNOY_NUM_TREES = 50

class VectorSearchService:
    """Service for local vector search operations using Annoy."""
    # Class-level lock for thread safety during index build/save/load
    _lock = threading.Lock()

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the service with configuration.
        Args:
            config: Dictionary containing configuration values like
                    EMBEDDING_MODEL_NAME, VECTOR_SEARCH_DIMENSION,
                    ANNOY_INDEX_PATH, ANNOY_METRIC, ANNOY_NUM_TREES etc.
        """
        self.config = config
        self.model_name = self.config.get("EMBEDDING_MODEL_NAME", 'all-MiniLM-L6-v2')
        self.dimension = self.config.get("VECTOR_SEARCH_DIMENSION") # Expected dimension
        self.index_path = self.config.get("ANNOY_INDEX_PATH") # Path for Annoy index file
        self.metadata_path = f"{self.index_path}.metadata.json" # Path for metadata
        self.metric = self.config.get("ANNOY_METRIC", DEFAULT_ANNOY_METRIC)
        self.num_trees = self.config.get("ANNOY_NUM_TREES", DEFAULT_ANNOY_NUM_TREES)

        self.index = None
        self.metadata_map: Dict[int, Dict[str, Any]] = {} # Maps Annoy item index to metadata
        self._index_loaded = False

        if not self.index_path:
            logger.error("Annoy index path not configured (ANNOY_INDEX_PATH).")
            raise ValueError("ANNOY_INDEX_PATH not configured.")

        # Check required packages
        try:
            __import__('torch')
            __import__('sentence_transformers')
            __import__('annoy')
        except ImportError:
            logger.critical("Required packages not installed. Run 'pip install torch sentence-transformers annoy'")
            raise ImportError("Required ML packages (torch, sentence-transformers, annoy) not installed.")

        try:
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

            # Initialize Annoy index (lazy loading)
            self.index = AnnoyIndex(self.dimension, self.metric)
            logger.info("Annoy index structure initialized (will load/build on demand)", 
                         dimension=self.dimension, metric=self.metric)

            # Attempt to load existing index and metadata
            self._load_index()

        except Exception as e:
            logger.error("Failed to initialize VectorSearchService", error=str(e), exc_info=True)
            self.model = None
            self.index = None
            raise RuntimeError(f"Failed to initialize VectorSearchService: {e}")

    def _load_index(self):
        """Load Annoy index and metadata from files if they exist."""
        with VectorSearchService._lock:
            if self._index_loaded:
                return # Already loaded
                
            index_file_exists = os.path.exists(self.index_path)
            metadata_file_exists = os.path.exists(self.metadata_path)

            if index_file_exists and metadata_file_exists:
                try:
                    self.index.load(self.index_path)
                    with open(self.metadata_path, 'r') as f:
                        self.metadata_map = {int(k): v for k, v in json.load(f).items()} # Ensure keys are int
                    self._index_loaded = True
                    logger.info("Loaded existing Annoy index and metadata", 
                                 index_path=self.index_path, 
                                 metadata_path=self.metadata_path,
                                 item_count=self.index.get_n_items())
                except Exception as e:
                    logger.error("Failed to load existing Annoy index/metadata. Re-initializing.", 
                                 index_path=self.index_path, metadata_path=self.metadata_path, error=str(e), exc_info=True)
                    # Reset state if loading failed
                    self.index.unload() # Ensure unloaded
                    self.metadata_map = {}
                    self._index_loaded = False 
            elif index_file_exists or metadata_file_exists:
                 logger.warning("Annoy index file or metadata file exists, but not both. Re-initializing index.", 
                                  index_exists=index_file_exists, meta_exists=metadata_file_exists)
                 # Clean up potentially orphaned file
                 if index_file_exists: os.remove(self.index_path)
                 if metadata_file_exists: os.remove(self.metadata_path)
                 self.metadata_map = {}
                 self._index_loaded = False
            else:
                logger.info("No existing Annoy index found. Will build upon adding documents.", index_path=self.index_path)
                self._index_loaded = True # Mark as "loaded" (empty)

    def _save_index(self):
        """Save Annoy index and metadata to files."""
        if not self.index:
             raise RuntimeError("Annoy index is not initialized.")
        
        with VectorSearchService._lock:
            try:
                # Create directory if it doesn't exist
                index_dir = os.path.dirname(self.index_path)
                if index_dir and not os.path.exists(index_dir):
                    os.makedirs(index_dir, exist_ok=True)
                    
                self.index.save(self.index_path)
                with open(self.metadata_path, 'w') as f:
                    # Ensure keys in metadata_map are strings for JSON compatibility
                    json.dump({str(k): v for k, v in self.metadata_map.items()}, f)
                logger.info("Saved Annoy index and metadata", 
                             index_path=self.index_path, 
                             metadata_path=self.metadata_path, 
                             item_count=self.index.get_n_items())
            except Exception as e:
                logger.error("Failed to save Annoy index/metadata", error=str(e), exc_info=True)
                raise RuntimeError(f"Failed to save Annoy index/metadata: {e}")

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
            raise RuntimeError("Annoy index is not initialized.")
        if not documents:
            logger.warning("Attempted to add empty batch of documents.", user_id=user_id)
            return 0, 0
        
        # Ensure index is loaded
        self._load_index()

        texts_to_embed = [doc[0] for doc in documents if doc[0]]
        original_indices = [i for i, doc in enumerate(documents) if doc[0]]

        if not texts_to_embed:
            logger.warning("No valid text found in the document batch.", user_id=user_id, total_docs=len(documents))
            return 0, len(documents)

        added_count = 0
        failure_count = len(documents) - len(texts_to_embed)
        needs_rebuild = False

        try:
            embeddings = self._get_embedding(texts_to_embed)
            
            with VectorSearchService._lock: # Lock during index modification
                current_item_index = self.index.get_n_items()
                
                for i, embedding in enumerate(embeddings):
                    original_doc_index = original_indices[i]
                    text, metadata = documents[original_doc_index]
                    # Store metadata keyed by the Annoy item index
                    # Add user_id to metadata
                    metadata_with_user = metadata.copy()
                    metadata_with_user['user_id'] = str(user_id)
                    metadata_with_user['_text_preview'] = text[:100] # Store preview for debugging
                    
                    self.metadata_map[current_item_index] = metadata_with_user
                    self.index.add_item(current_item_index, embedding)
                    current_item_index += 1
                    added_count += 1
                    needs_rebuild = True # Mark that index needs rebuild after adding

            logger.info("Documents added to Annoy index structure", added_count=added_count, user_id=user_id)
            
            # Rebuild and save the index if items were added
            if needs_rebuild:
                logger.info(f"Building Annoy index with {self.num_trees} trees...")
                self.index.build(self.num_trees)
                logger.info("Annoy index build complete.")
                self._save_index() # Save index and metadata
                
            return added_count, failure_count

        except Exception as e:
            logger.error("Failed during add_documents batch processing", user_id=user_id, batch_size=len(documents), error=str(e), exc_info=True)
            # Return counts based on what was successfully added before the error
            return added_count, failure_count + (len(texts_to_embed) - added_count)

    def query(self, user_id: str, query_text: str, top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict]:
        """Query the index for relevant documents, filtering by metadata (incl user_id)."""
        if not self.index:
            raise RuntimeError("Annoy index is not initialized.")
        if not query_text:
            logger.warning("Attempted query with empty text.", user_id=user_id)
            return []

        # Ensure index is loaded
        self._load_index()

        if self.index.get_n_items() == 0:
            logger.warning("Query attempted on empty Annoy index.", user_id=user_id)
            return []

        try:
            query_embedding = self._get_embedding(query_text)

            # Annoy search - gets indices and distances
            # Increase search_k for better recall accuracy with filtering
            search_k = -1 # Default search_k (-1 means N*num_trees where N=top_k)
            if filter:
                # If filtering, search more candidates to increase chance of finding matches
                search_k_multiplier = self.config.get('ANNOY_FILTER_SEARCH_K_MULTIPLIER', 10) 
                search_k = top_k * search_k_multiplier * self.num_trees
            
            annoy_indices, distances = self.index.get_nns_by_vector(
                query_embedding, 
                top_k * 10, # Fetch more initially if filtering later
                search_k=search_k, 
                include_distances=True
            )

            # Filter results based on metadata (user_id and optional filter)
            results = []
            target_user_id = str(user_id)

            for idx, dist in zip(annoy_indices, distances):
                if len(results) >= top_k:
                    break # Stop once we have enough results
                    
                metadata = self.metadata_map.get(idx)
                if not metadata:
                    logger.warning(f"Metadata not found for Annoy index {idx}")
                    continue
                
                # --- Apply Filtering --- #
                # 1. Check user_id
                if metadata.get('user_id') != target_user_id:
                    continue # Skip if user_id doesn't match
                
                # 2. Apply additional filter if provided
                if filter:
                    match = True
                    for filter_key, filter_value in filter.items():
                        if filter_key == 'user_id': continue # Already checked
                        
                        metadata_value = metadata.get(filter_key)
                        # Simple equality check for now, could be extended
                        # Handle potential type mismatches (e.g., filter value is int, metadata is str)
                        try:
                             if isinstance(filter_value, list): # Handle $in operator style
                                 if metadata_value not in filter_value and str(metadata_value) not in filter_value:
                                      match = False
                                      break
                             elif metadata_value != filter_value and str(metadata_value) != str(filter_value):
                                 match = False
                                 break
                        except Exception:
                             match = False # Error during comparison means no match
                             break
                    if not match:
                        continue # Skip if any filter condition fails
                # --- End Filtering --- #

                # Convert Annoy distance (angular) to similarity score (cosine)
                # Cosine similarity = (2 - distance^2) / 2 
                # For angular distance: similarity = cos(theta) = (2 - d^2) / 2
                # For euclidean: similarity depends, maybe 1 / (1 + distance)
                # For dot: similarity = distance
                # Adjust based on the metric used!
                if self.metric == 'angular':
                    similarity_score = (2 - (dist**2)) / 2
                elif self.metric == 'dot':
                     similarity_score = dist
                else: # Euclidean or others, use distance directly or inverse
                     similarity_score = 1.0 / (1.0 + dist) # Example for Euclidean
                
                result_item = {
                    'id': metadata.get('chunk_id') or metadata.get('document_id') or idx, # Use stored ID if available
                    'score': similarity_score,
                    'metadata': metadata
                }
                results.append(result_item)

            logger.info("Annoy query successful", user_id=user_id, query_preview=query_text[:50]+"...", results_count=len(results))
            return results

        except Exception as e:
            logger.error("Failed to execute Annoy query", user_id=user_id, query_preview=query_text[:50]+"...", error=str(e), exc_info=True)
            return [] # Return empty list on error

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all vectors and metadata associated with a specific user_id."""
        if not self.index:
            raise RuntimeError("Annoy index is not initialized.")
        
        target_user_id = str(user_id)
        logger.warning("Attempting to delete all vector data for user", user_id=target_user_id)
        
        with VectorSearchService._lock: # Lock for modification
            # Ensure index is loaded
            self._load_index()
            
            items_to_remove = [idx for idx, meta in self.metadata_map.items() if meta.get('user_id') == target_user_id]
            
            if not items_to_remove:
                logger.info("No data found for user to delete.", user_id=target_user_id)
                return True # Nothing to delete
                
            logger.info(f"Found {len(items_to_remove)} items to remove for user {target_user_id}")
            
            # Annoy doesn't directly support item deletion. The standard approach is 
            # to rebuild the index excluding the items.
            
            # 1. Create a new temporary index
            new_index = AnnoyIndex(self.dimension, self.metric)
            new_metadata_map = {}
            new_item_counter = 0
            
            # 2. Iterate through old items and add only those NOT belonging to the user
            for old_idx in range(self.index.get_n_items()):
                if old_idx not in items_to_remove:
                    vector = self.index.get_item_vector(old_idx)
                    metadata = self.metadata_map.get(old_idx)
                    if metadata:
                        new_index.add_item(new_item_counter, vector)
                        new_metadata_map[new_item_counter] = metadata
                        new_item_counter += 1
            
            # 3. Build the new index
            logger.info(f"Rebuilding index to remove user data. New item count: {new_item_counter}")
            new_index.build(self.num_trees)
            
            # 4. Replace old index and metadata with new ones
            self.index.unload() # Unload old index memory
            self.index = new_index
            self.metadata_map = new_metadata_map
            self._index_loaded = True # Mark the new index as loaded
            
            # 5. Save the new index and metadata
            try:
                self._save_index()
                logger.info("Successfully rebuilt and saved index after deleting user data", user_id=target_user_id)
                return True
            except Exception as e:
                 logger.error("Failed to save rebuilt index after deleting user data", user_id=target_user_id, error=str(e))
                 # State might be inconsistent here, potential data loss?
                 # Consider recovery strategies or raising a critical error.
                 return False # Indicate save failed

    # Maybe add other utility methods like: get_index_stats, etc.

# Removed Flask app, CORS, create_vector_app function, and all @app.route definitions
# Removed __main__ block