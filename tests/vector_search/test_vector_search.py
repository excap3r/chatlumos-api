import pytest
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, ANY

from services.vector_search.vector_search import VectorSearchService

# --- Fixtures ---

@pytest.fixture
def mock_config():
    """Provides a basic configuration for VectorSearchService tests."""
    return {
        "EMBEDDING_MODEL_NAME": "mock-model",
        "VECTOR_SEARCH_DIMENSION": 128, # Specify dimension
        "ANNOY_INDEX_PATH": "/test/index/mock.ann",
        "ANNOY_METRIC": "angular",
        "ANNOY_NUM_TREES": 10
    }

@pytest.fixture
def mock_sentence_transformer():
    """Mocks the SentenceTransformer class."""
    with patch('services.vector_search.vector_search.SentenceTransformer') as MockST:
        mock_instance = MockST.return_value
        mock_instance.get_sentence_embedding_dimension.return_value = 128 # Match config
        # Mock encode to return numpy arrays of the correct dimension
        def mock_encode(text, show_progress_bar=False):
             dim = mock_instance.get_sentence_embedding_dimension()
             if isinstance(text, str):
                 return np.random.rand(dim).astype(np.float32)
             elif isinstance(text, list):
                 return np.random.rand(len(text), dim).astype(np.float32)
             else:
                 raise TypeError("Unsupported input type for mock encode")
        mock_instance.encode.side_effect = mock_encode
        yield MockST

@pytest.fixture
def mock_annoy_index():
    """Mocks the AnnoyIndex class."""
    with patch('services.vector_search.vector_search.AnnoyIndex') as MockAnnoy:
        mock_instance = MockAnnoy.return_value
        # Set up mock methods needed
        mock_instance.load = MagicMock()
        mock_instance.unload = MagicMock()
        mock_instance.save = MagicMock()
        mock_instance.build = MagicMock()
        mock_instance.add_item = MagicMock()
        mock_instance.get_n_items = MagicMock(return_value=0) # Start empty
        mock_instance.get_nns_by_vector = MagicMock(return_value=([], [])) # (ids, distances)
        mock_instance.get_item_vector = MagicMock(return_value=np.random.rand(128).tolist()) # Example vector
        yield MockAnnoy

@pytest.fixture
def mock_filesystem():
    """Mocks filesystem operations (os.path, open, os.makedirs, os.remove)."""
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('os.makedirs') as mock_makedirs, \
         patch('os.remove') as mock_remove:
        # Default: assume files don't exist initially
        mock_exists.return_value = False
        yield {
            'exists': mock_exists,
            'open': mock_file,
            'makedirs': mock_makedirs,
            'remove': mock_remove
        }

# --- Initialization Tests ---

def test_vector_service_init_success_no_existing_index(
    mock_config,
    mock_sentence_transformer,
    mock_annoy_index,
    mock_filesystem
):
    """Test successful initialization when index files don't exist."""
    mock_filesystem['exists'].return_value = False # Ensure files don't exist

    service = VectorSearchService(mock_config)

    assert service.model_name == "mock-model"
    assert service.dimension == 128
    assert service.index_path == "/test/index/mock.ann"
    assert service.metadata_path == "/test/index/mock.ann.metadata.json"
    assert service.metric == "angular"
    assert service.num_trees == 10

    # Check model and index initialized
    mock_sentence_transformer.assert_called_once_with("mock-model")
    mock_annoy_index.assert_called_once_with(128, "angular")
    assert service.model is mock_sentence_transformer.return_value
    assert service.index is mock_annoy_index.return_value

    # Check index loading logic (should not load/unload, should be marked loaded)
    mock_filesystem['exists'].assert_any_call("/test/index/mock.ann")
    mock_filesystem['exists'].assert_any_call("/test/index/mock.ann.metadata.json")
    service.index.load.assert_not_called()
    service.index.unload.assert_not_called()
    assert service._index_loaded
    assert service.metadata_map == {}

def test_vector_service_init_success_load_existing_index(
    mock_config,
    mock_sentence_transformer,
    mock_annoy_index,
    mock_filesystem
):
    """Test successful initialization loading existing index and metadata files."""
    mock_filesystem['exists'].return_value = True # Assume both files exist
    mock_metadata = {10: {"id": "doc1", "user_id": "userA"}, 15: {"id": "doc2", "user_id": "userB"}}
    # Use mock_open to simulate reading the metadata file
    mock_filesystem['open'] = mock_open(read_data=json.dumps({str(k): v for k, v in mock_metadata.items()}))

    mock_annoy_instance = mock_annoy_index.return_value
    mock_annoy_instance.get_n_items.return_value = 2 # Simulate items in loaded index

    service = VectorSearchService(mock_config)

    # Check index loading
    mock_filesystem['exists'].assert_any_call("/test/index/mock.ann")
    mock_filesystem['exists'].assert_any_call("/test/index/mock.ann.metadata.json")
    mock_annoy_instance.load.assert_called_once_with("/test/index/mock.ann")
    mock_filesystem['open'].assert_called_once_with("/test/index/mock.ann.metadata.json", 'r')
    assert service._index_loaded
    # Check metadata keys are correctly converted to int
    assert service.metadata_map == mock_metadata # Keys should be int after loading

def test_vector_service_init_dimension_mismatch(
    mock_config,
    mock_sentence_transformer,
    mock_annoy_index,
    mock_filesystem
):
    """Test init handles dimension mismatch between config and model, uses model dim."""
    mock_config["VECTOR_SEARCH_DIMENSION"] = 64 # Mismatched config
    mock_model_instance = mock_sentence_transformer.return_value
    mock_model_instance.get_sentence_embedding_dimension.return_value = 128 # Actual model dim

    service = VectorSearchService(mock_config)

    assert service.dimension == 128 # Should use the model's dimension
    mock_annoy_index.assert_called_once_with(128, "angular") # Annoy index uses correct dim

def test_vector_service_init_dimension_not_configured(
    mock_config,
    mock_sentence_transformer,
    mock_annoy_index,
    mock_filesystem
):
    """Test init determines dimension from model if not in config."""
    del mock_config["VECTOR_SEARCH_DIMENSION"] # Remove from config
    mock_model_instance = mock_sentence_transformer.return_value
    mock_model_instance.get_sentence_embedding_dimension.return_value = 768 # Model dim

    service = VectorSearchService(mock_config)

    assert service.dimension == 768 # Should use the model's dimension
    mock_annoy_index.assert_called_once_with(768, "angular")

def test_vector_service_init_load_fails( # Renamed to be more specific
    mock_config,
    mock_sentence_transformer,
    mock_annoy_index,
    mock_filesystem
):
    """Test init handles failure during index/metadata load, resets state."""
    mock_filesystem['exists'].return_value = True # Assume files exist
    mock_annoy_instance = mock_annoy_index.return_value
    mock_annoy_instance.load.side_effect = Exception("Corrupted index file")

    service = VectorSearchService(mock_config)

    mock_annoy_instance.load.assert_called_once_with("/test/index/mock.ann")
    mock_filesystem['open'].assert_not_called() # Should fail before opening metadata
    mock_annoy_instance.unload.assert_called_once() # Should unload after failed load
    assert service._index_loaded is True # It initializes to an empty state if load fails
    assert service.metadata_map == {}

def test_vector_service_init_partial_files_exist( # Added test case
    mock_config,
    mock_sentence_transformer,
    mock_annoy_index,
    mock_filesystem
):
    """Test init cleans up and resets if only one of index/metadata file exists."""
    # Scenario 1: Only index exists
    mock_filesystem['exists'].side_effect = lambda p: p == mock_config["ANNOY_INDEX_PATH"]

    service1 = VectorSearchService(mock_config)

    mock_filesystem['remove'].assert_called_once_with(mock_config["ANNOY_INDEX_PATH"])
    assert service1._index_loaded
    assert service1.metadata_map == {}
    mock_annoy_index.return_value.load.assert_not_called()
    mock_filesystem['remove'].reset_mock()
    # Reset exists mock side effect for next scenario
    mock_filesystem['exists'].side_effect = None


    # Scenario 2: Only metadata exists
    mock_filesystem['exists'].side_effect = lambda p: p == f"{mock_config['ANNOY_INDEX_PATH']}.metadata.json"

    service2 = VectorSearchService(mock_config)

    mock_filesystem['remove'].assert_called_once_with(f"{mock_config['ANNOY_INDEX_PATH']}.metadata.json")
    assert service2._index_loaded
    assert service2.metadata_map == {}
    mock_annoy_index.return_value.load.assert_not_called()

def test_vector_service_init_no_index_path_config(
    mock_config, # Use fixture but modify
    mock_sentence_transformer,
    mock_annoy_index,
    mock_filesystem
):
    """Test init raises ValueError if ANNOY_INDEX_PATH is missing."""
    del mock_config["ANNOY_INDEX_PATH"]
    with pytest.raises(ValueError, match="ANNOY_INDEX_PATH not configured"):
        VectorSearchService(mock_config)

def test_vector_service_init_model_load_failure(
    mock_config,
    mock_sentence_transformer,
    mock_annoy_index,
    mock_filesystem
):
    """Test init raises RuntimeError if SentenceTransformer fails to load."""
    mock_sentence_transformer.side_effect = Exception("Model download failed")
    with pytest.raises(RuntimeError, match="Failed to initialize VectorSearchService: Model download failed"):
        VectorSearchService(mock_config)

# --- _get_embedding Tests ---

def test_get_embedding_single_string(
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test getting embedding for a single string."""
    # Initialize service (mocks are already set up by fixtures)
    service = VectorSearchService(mock_config)
    mock_model = service.model
    text = "This is a test sentence."
    # Use the mock_encode defined in the fixture
    expected_embedding = mock_model.encode(text) # Get the expected format from mock

    embedding = service._get_embedding(text)

    mock_model.encode.assert_called_once_with(text, show_progress_bar=False)
    assert isinstance(embedding, list) # Should return list
    # Compare content - mock encode returns ndarray, _get_embedding returns list
    assert np.array_equal(embedding, expected_embedding.tolist())


def test_get_embedding_list_of_strings(
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test getting embeddings for a list of strings."""
    service = VectorSearchService(mock_config)
    mock_model = service.model
    texts = ["Sentence one.", "Sentence two."]
    # Use the mock_encode defined in the fixture
    expected_embeddings = mock_model.encode(texts) # Get the expected format from mock

    embeddings = service._get_embedding(texts)

    # Reset call count before the call we are testing
    mock_model.encode.reset_mock()
    embeddings = service._get_embedding(texts)


    mock_model.encode.assert_called_once_with(texts, show_progress_bar=False)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert isinstance(embeddings[0], list) # Should be list of lists
    # Compare content - mock encode returns ndarray, _get_embedding returns list of lists
    assert np.array_equal(embeddings, expected_embeddings.tolist())


def test_get_embedding_model_not_initialized(
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test embedding raises RuntimeError if model failed to initialize."""
    # Simulate model init failure
    mock_sentence_transformer.side_effect = Exception("Model init failed")
    with pytest.raises(RuntimeError):
        # This __init__ call itself should raise the RuntimeError
        VectorSearchService(mock_config)

    # Reset side effect for the next part of the test
    mock_sentence_transformer.side_effect = None

    # To test the state *after* a failed init where service.model might be None
    # (though the current init logic raises before assignment),
    # let's manually set model to None after a *successful* init.
    # This isolates the check within _get_embedding.
    mock_sentence_transformer.side_effect = None # Ensure no error on this init
    mock_sentence_transformer.return_value.get_sentence_embedding_dimension.return_value=128 # Ensure mock is valid

    service = VectorSearchService(mock_config) # Create a valid service first
    service.model = None # Manually break it

    with pytest.raises(RuntimeError, match="SentenceTransformer model is not initialized."):
        service._get_embedding("test")

def test_get_embedding_encode_fails(
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test embedding raises RuntimeError if model.encode fails."""
    service = VectorSearchService(mock_config)
    mock_model = service.model
    error_message = "Encoding process failed"
    # Ensure the side effect is set *after* successful init
    mock_model.encode.side_effect = Exception(error_message)

    with pytest.raises(RuntimeError, match=f"Embedding generation failed: {error_message}"):
        service._get_embedding("test")

# --- add_documents Tests ---

@patch('services.vector_search.vector_search.VectorSearchService._save_index') # Mock save
@patch('services.vector_search.vector_search.VectorSearchService._get_embedding') # Mock embedding
def test_add_documents_success(
    mock_get_embedding, mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test adding a batch of documents successfully."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "user123"
    docs_to_add = [
        ("First document text.", {"id": "doc1", "source": "fileA"}),
        ("Second document, slightly longer.", {"id": "doc2", "source": "fileB"})
    ]
    texts = [d[0] for d in docs_to_add]
    # Mock embeddings returned by _get_embedding
    mock_embeddings = np.random.rand(len(texts), service.dimension).tolist()
    mock_get_embedding.return_value = mock_embeddings

    # Simulate starting with an empty index
    mock_annoy_instance.get_n_items.return_value = 0

    added_count, failure_count = service.add_documents(user_id, docs_to_add)

    assert added_count == 2
    assert failure_count == 0
    mock_get_embedding.assert_called_once_with(texts)

    # Verify items added to Annoy and metadata map
    assert mock_annoy_instance.add_item.call_count == 2
    mock_annoy_instance.add_item.assert_any_call(0, mock_embeddings[0])
    mock_annoy_instance.add_item.assert_any_call(1, mock_embeddings[1])
    assert len(service.metadata_map) == 2
    assert 0 in service.metadata_map
    assert 1 in service.metadata_map
    assert service.metadata_map[0]["id"] == "doc1"
    assert service.metadata_map[0]["user_id"] == user_id
    assert service.metadata_map[0]["_text_preview"] == "First document text."
    assert service.metadata_map[1]["id"] == "doc2"
    assert service.metadata_map[1]["user_id"] == user_id

    # Verify index build and save were called
    mock_annoy_instance.build.assert_called_once_with(service.num_trees)
    mock_save_index.assert_called_once()

@patch('services.vector_search.vector_search.VectorSearchService._save_index') # Mock save
@patch('services.vector_search.vector_search.VectorSearchService._get_embedding') # Mock embedding
def test_add_documents_empty_list(
    mock_get_embedding, mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test adding an empty list of documents."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "user123"

    added_count, failure_count = service.add_documents(user_id, [])

    assert added_count == 0
    assert failure_count == 0
    mock_get_embedding.assert_not_called()
    mock_annoy_instance.add_item.assert_not_called()
    mock_annoy_instance.build.assert_not_called()
    mock_save_index.assert_not_called()

@patch('services.vector_search.vector_search.VectorSearchService._save_index') # Mock save
@patch('services.vector_search.vector_search.VectorSearchService._get_embedding') # Mock embedding
def test_add_documents_invalid_only(
    mock_get_embedding, mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test adding documents where all have invalid (empty) text."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "user456"
    docs_to_add = [
        ("", {"id": "doc1"}),
        (None, {"id": "doc2"}) # None should also be skipped
    ]

    added_count, failure_count = service.add_documents(user_id, docs_to_add)

    assert added_count == 0
    assert failure_count == 2 # Both failed pre-embedding
    mock_get_embedding.assert_not_called()
    mock_annoy_instance.add_item.assert_not_called()
    mock_annoy_instance.build.assert_not_called()
    mock_save_index.assert_not_called()
    assert service.metadata_map == {}

@patch('services.vector_search.vector_search.VectorSearchService._save_index') # Mock save
@patch('services.vector_search.vector_search.VectorSearchService._get_embedding') # Mock embedding
def test_add_documents_partial_invalid(
    mock_get_embedding, mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test adding documents where some are invalid."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "user789"
    docs_to_add = [
        ("Valid doc 1", {"id": "docA"}),
        ("", {"id": "docB"}), # Invalid
        ("Valid doc 2", {"id": "docC"})
    ]
    valid_texts = ["Valid doc 1", "Valid doc 2"]
    mock_embeddings = np.random.rand(len(valid_texts), service.dimension).tolist()
    mock_get_embedding.return_value = mock_embeddings

    mock_annoy_instance.get_n_items.return_value = 0

    added_count, failure_count = service.add_documents(user_id, docs_to_add)

    assert added_count == 2
    assert failure_count == 1 # One was invalid
    mock_get_embedding.assert_called_once_with(valid_texts)
    assert mock_annoy_instance.add_item.call_count == 2
    # Check metadata using the correct expected indices (0, 1)
    assert 0 in service.metadata_map
    assert 1 in service.metadata_map
    assert service.metadata_map[0]["id"] == "docA"
    assert service.metadata_map[1]["id"] == "docC"
    mock_annoy_instance.build.assert_called_once()
    mock_save_index.assert_called_once()

@patch('services.vector_search.vector_search.VectorSearchService._save_index')
@patch('services.vector_search.vector_search.VectorSearchService._get_embedding')
def test_add_documents_embedding_error(
    mock_get_embedding, mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test handling when embedding fails during add_documents."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "userErr"
    docs_to_add = [("Doc 1", {"id": "d1"}), ("Doc 2", {"id": "d2"})]
    texts = [d[0] for d in docs_to_add]

    # Simulate embedding failure
    error_message = "Embedding generator broke"
    mock_get_embedding.side_effect = RuntimeError(error_message)

    added_count, failure_count = service.add_documents(user_id, docs_to_add)

    assert added_count == 0
    # Failure count should reflect the total number of docs attempted before error
    assert failure_count == 2
    mock_get_embedding.assert_called_once_with(texts)
    mock_annoy_instance.add_item.assert_not_called()
    mock_annoy_instance.build.assert_not_called()
    mock_save_index.assert_not_called()

@patch('services.vector_search.vector_search.VectorSearchService._save_index')
@patch('services.vector_search.vector_search.VectorSearchService._get_embedding')
def test_add_documents_build_error(
    mock_get_embedding, mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test handling when Annoy index build fails."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "userBuildErr"
    docs_to_add = [("Good Doc", {"id": "g1"})]
    mock_embeddings = np.random.rand(1, service.dimension).tolist()
    mock_get_embedding.return_value = mock_embeddings
    mock_annoy_instance.get_n_items.return_value = 0

    # Simulate build failure
    build_error_message = "Cannot build index"
    mock_annoy_instance.build.side_effect = Exception(build_error_message)

    # The current implementation logs the error but returns counts before build error
    added_count, failure_count = service.add_documents(user_id, docs_to_add)

    # Check state *before* build failure
    assert added_count == 1 # Item was added to structure
    assert failure_count == 0
    mock_get_embedding.assert_called_once()
    mock_annoy_instance.add_item.assert_called_once()
    assert 0 in service.metadata_map

    # Check build was called, but save was not
    mock_annoy_instance.build.assert_called_once_with(service.num_trees)
    mock_save_index.assert_not_called() # Build failed before save

@patch('services.vector_search.vector_search.VectorSearchService._save_index')
@patch('services.vector_search.vector_search.VectorSearchService._get_embedding')
def test_add_documents_save_error(
    mock_get_embedding, mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test handling when saving index/metadata fails after build."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "userSaveErr"
    docs_to_add = [("Another Good Doc", {"id": "g2"})]
    mock_embeddings = np.random.rand(1, service.dimension).tolist()
    mock_get_embedding.return_value = mock_embeddings
    mock_annoy_instance.get_n_items.return_value = 0

    # Simulate save failure (mocking the internal _save_index method)
    save_error_message = "Disk full"
    mock_save_index.side_effect = RuntimeError(save_error_message)

    # The error from _save_index propagates up in the current implementation
    with pytest.raises(RuntimeError, match=save_error_message):
      service.add_documents(user_id, docs_to_add)

    # Verify state before the error
    mock_get_embedding.assert_called_once()
    mock_annoy_instance.add_item.assert_called_once()
    assert 0 in service.metadata_map
    mock_annoy_instance.build.assert_called_once_with(service.num_trees)
    mock_save_index.assert_called_once() # Save was attempted

# --- query Tests ---

@patch('services.vector_search.vector_search.VectorSearchService._get_embedding')
def test_query_success_no_filter(
    mock_get_embedding,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test successful query without filters, matching user_id implicitly."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "userQuery1"
    query = "Search for this"
    query_embedding = np.random.rand(service.dimension).tolist()
    mock_get_embedding.return_value = query_embedding

    # Setup existing index state
    service.metadata_map = {
        0: {"id": "docA", "user_id": user_id, "content": "Content A"},
        1: {"id": "docB", "user_id": "otherUser", "content": "Content B"},
        2: {"id": "docC", "user_id": user_id, "content": "Content C"}
    }
    service._index_loaded = True
    mock_annoy_instance.get_n_items.return_value = 3
    # Annoy returns indices and distances
    mock_annoy_instance.get_nns_by_vector.return_value = ([0, 2, 1], [0.1, 0.3, 0.5]) # userQuery1, userQuery1, otherUser

    results = service.query(user_id, query, top_k=2)

    # Verify embedding was called
    mock_get_embedding.assert_called_once_with(query)
    # Verify Annoy search was called (search_k=-1 by default)
    mock_annoy_instance.get_nns_by_vector.assert_called_once_with(query_embedding, 20, search_k=-1, include_distances=True) # Default top_k * 10 = 20

    # Verify results (should only include docs for user_id and be sorted by score)
    assert len(results) == 2
    assert results[0]["id"] == "docA"
    assert results[0]["user_id"] == user_id
    assert results[0]["score"] == pytest.approx(1.0 - 0.1 / 2) # Convert angular distance to similarity score (0-1)
    assert results[1]["id"] == "docC"
    assert results[1]["user_id"] == user_id
    assert results[1]["score"] == pytest.approx(1.0 - 0.3 / 2)

@patch('services.vector_search.vector_search.VectorSearchService._get_embedding')
def test_query_success_with_filter(
    mock_get_embedding,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test successful query with additional metadata filters."""
    mock_config['ANNOY_FILTER_SEARCH_K_MULTIPLIER'] = 5 # Lower multiplier for easier testing
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "userFilter"
    query = "Search for type:special"
    query_embedding = np.random.rand(service.dimension).tolist()
    mock_get_embedding.return_value = query_embedding

    # Setup index state
    service.metadata_map = {
        10: {"id": "doc10", "user_id": user_id, "type": "normal", "value": 1},
        11: {"id": "doc11", "user_id": user_id, "type": "special", "value": 2},
        12: {"id": "doc12", "user_id": "other", "type": "special", "value": 3},
        13: {"id": "doc13", "user_id": user_id, "type": "special", "value": 4},
        14: {"id": "doc14", "user_id": user_id, "type": "normal", "value": 5},
        15: {"id": "doc15", "user_id": user_id, "type": "special", "value": 6}, # Further away but matches filter
    }
    service._index_loaded = True
    mock_annoy_instance.get_n_items.return_value = 6
    # Annoy returns candidates: some match user/filter, some don't
    # Indices:       11, 13, 10,      12,      14, 15
    # User Match:    Y,  Y,  Y,       N,       Y,  Y
    # Filter Match:  Y,  Y,  N,       Y,       N,  Y
    # Distances:    0.1, 0.2, 0.3,    0.4,     0.5, 0.6
    mock_annoy_instance.get_nns_by_vector.return_value = ([11, 13, 10, 12, 14, 15], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    results = service.query(user_id, query, top_k=2, filter={"type": "special"})

    # Verify Annoy search used larger search_k due to filter
    # search_k = top_k * multiplier = 2 * 5 = 10
    mock_annoy_instance.get_nns_by_vector.assert_called_once_with(query_embedding, 10, search_k=-1, include_distances=True)

    # Verify results match user AND filter, limited by top_k, sorted by score
    assert len(results) == 2
    assert results[0]["id"] == "doc11"
    assert results[0]["user_id"] == user_id
    assert results[0]["type"] == "special"
    assert results[0]["score"] == pytest.approx(1.0 - 0.1 / 2)
    assert results[1]["id"] == "doc13"
    assert results[1]["user_id"] == user_id
    assert results[1]["type"] == "special"
    assert results[1]["score"] == pytest.approx(1.0 - 0.2 / 2)
    # Doc 15 also matches but is excluded by top_k

def test_query_empty_index(
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test querying when the index is empty."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    mock_annoy_instance.get_n_items.return_value = 0 # Ensure index is empty
    service._index_loaded = True

    results = service.query("userX", "query text")

    assert results == []
    # Should not attempt embedding or Annoy search
    service.model.encode.assert_not_called()
    mock_annoy_instance.get_nns_by_vector.assert_not_called()

def test_query_empty_query_string(
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test querying with an empty query string."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    # Give index some items so it's not an empty index check
    service.metadata_map = {0: {"id": "docA", "user_id": "userY"}}
    service._index_loaded = True
    mock_annoy_instance.get_n_items.return_value = 1

    results = service.query("userY", "") # Empty query

    assert results == []
    # Should not attempt embedding or Annoy search
    service.model.encode.assert_not_called()
    mock_annoy_instance.get_nns_by_vector.assert_not_called()

@patch('services.vector_search.vector_search.VectorSearchService._get_embedding')
def test_query_embedding_error(
    mock_get_embedding,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test query handling when embedding fails."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    # Give index items
    service.metadata_map = {0: {"id": "docA", "user_id": "userZ"}}
    service._index_loaded = True
    mock_annoy_instance.get_n_items.return_value = 1

    error_message = "Failed to embed query"
    mock_get_embedding.side_effect = RuntimeError(error_message)

    with pytest.raises(RuntimeError, match=f"Embedding generation failed: {error_message}"):
        service.query("userZ", "a query")

    mock_annoy_instance.get_nns_by_vector.assert_not_called()

@patch('services.vector_search.vector_search.VectorSearchService._get_embedding')
def test_query_annoy_search_error(
    mock_get_embedding,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test query handling when Annoy search itself fails."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_id = "userAnnErr"
    query = "Search this"
    query_embedding = np.random.rand(service.dimension).tolist()
    mock_get_embedding.return_value = query_embedding

    # Setup index state
    service.metadata_map = {0: {"id": "docE", "user_id": user_id}}
    service._index_loaded = True
    mock_annoy_instance.get_n_items.return_value = 1

    # Simulate Annoy search failure
    error_message = "Annoy internal error"
    mock_annoy_instance.get_nns_by_vector.side_effect = Exception(error_message)

    with pytest.raises(RuntimeError, match=f"Annoy search failed: {error_message}"):
        service.query(user_id, query)

    mock_get_embedding.assert_called_once_with(query)
    mock_annoy_instance.get_nns_by_vector.assert_called_once()

# --- delete_user_data Tests ---

@patch('services.vector_search.vector_search.VectorSearchService._save_index')
def test_delete_user_data_success(
    mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test successfully deleting data for a specific user."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    mock_new_annoy_instance = MagicMock(spec=service.index)
    mock_new_annoy_instance.add_item = MagicMock()
    mock_new_annoy_instance.build = MagicMock()
    mock_annoy_index.return_value = mock_new_annoy_instance # Make AnnoyIndex() return the new mock

    user_to_delete = "userToDelete"
    user_to_keep = "userToKeep"

    # Setup initial state with data from both users
    original_metadata = {
        0: {"id": "delDoc1", "user_id": user_to_delete, "content": "Delete me"},
        1: {"id": "keepDoc1", "user_id": user_to_keep, "content": "Keep me"},
        2: {"id": "delDoc2", "user_id": user_to_delete, "content": "Delete me too"},
        3: {"id": "keepDoc2", "user_id": user_to_keep, "content": "Keep me too"}
    }
    service.metadata_map = original_metadata.copy()
    service._index_loaded = True
    # Mock get_item_vector to return predictable vectors
    def get_vector_side_effect(item_id):
        return np.array([item_id * 0.1] * service.dimension).tolist()
    mock_annoy_instance.get_item_vector.side_effect = get_vector_side_effect
    mock_annoy_instance.get_n_items.return_value = 4

    deleted_count = service.delete_user_data(user_to_delete)

    assert deleted_count == 2

    # Verify a new AnnoyIndex was created with the correct parameters
    # This check is tricky because the mock fixture replaces the class
    # We rely on the fact that `service.index` was replaced inside the method
    assert service.index is mock_new_annoy_instance

    # Verify only the kept user's data was added to the new index
    assert mock_new_annoy_instance.add_item.call_count == 2
    mock_new_annoy_instance.add_item.assert_any_call(0, get_vector_side_effect(1)) # keepDoc1 -> new index 0
    mock_new_annoy_instance.add_item.assert_any_call(1, get_vector_side_effect(3)) # keepDoc2 -> new index 1
    # Check get_item_vector was called for the kept items
    assert mock_annoy_instance.get_item_vector.call_count == 2
    mock_annoy_instance.get_item_vector.assert_any_call(1)
    mock_annoy_instance.get_item_vector.assert_any_call(3)

    # Verify the service metadata map now only contains the kept user's data
    assert len(service.metadata_map) == 2
    assert 0 in service.metadata_map # New index
    assert 1 in service.metadata_map # New index
    assert service.metadata_map[0]["id"] == "keepDoc1"
    assert service.metadata_map[0]["user_id"] == user_to_keep
    assert service.metadata_map[1]["id"] == "keepDoc2"
    assert service.metadata_map[1]["user_id"] == user_to_keep

    # Verify the new index was built and saved
    mock_new_annoy_instance.build.assert_called_once_with(service.num_trees)
    mock_save_index.assert_called_once()

@patch('services.vector_search.vector_search.VectorSearchService._save_index')
def test_delete_user_data_user_not_found(
    mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test deleting data for a user who has no data in the index."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    user_to_delete = "nonExistentUser"
    user_to_keep = "existingUser"

    # Setup initial state
    original_metadata = {
        0: {"id": "keep1", "user_id": user_to_keep},
        1: {"id": "keep2", "user_id": user_to_keep}
    }
    service.metadata_map = original_metadata.copy()
    service._index_loaded = True
    mock_annoy_instance.get_n_items.return_value = 2
    # Mock get_item_vector
    def get_vector_side_effect(item_id):
        return np.array([item_id * 0.1] * service.dimension).tolist()
    mock_annoy_instance.get_item_vector.side_effect = get_vector_side_effect

    # Spy on AnnoyIndex constructor to ensure it's NOT called
    annoy_constructor_spy = mock_annoy_index

    deleted_count = service.delete_user_data(user_to_delete)

    assert deleted_count == 0
    # Verify the index was NOT rebuilt
    assert service.index is mock_annoy_instance # Index should not have been replaced
    annoy_constructor_spy.assert_not_called() # No new AnnoyIndex instance needed
    mock_annoy_instance.add_item.assert_not_called()
    mock_annoy_instance.build.assert_not_called()
    mock_save_index.assert_not_called()
    # Metadata map should remain unchanged
    assert service.metadata_map == original_metadata

@patch('services.vector_search.vector_search.VectorSearchService._save_index')
def test_delete_user_data_empty_index(
    mock_save_index,
    mock_config, mock_sentence_transformer, mock_annoy_index, mock_filesystem
):
    """Test deleting from an empty index."""
    service = VectorSearchService(mock_config)
    mock_annoy_instance = service.index
    service.metadata_map = {}
    service._index_loaded = True
    mock_annoy_instance.get_n_items.return_value = 0

    annoy_constructor_spy = mock_annoy_index

    deleted_count = service.delete_user_data("anyUser")

    assert deleted_count == 0
    assert service.index is mock_annoy_instance
    annoy_constructor_spy.assert_not_called()
    mock_annoy_instance.build.assert_not_called()
    mock_save_index.assert_not_called() 