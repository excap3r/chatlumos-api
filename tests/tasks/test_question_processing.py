# Placeholder for question processing task tests 

import pytest
from unittest.mock import patch, MagicMock, ANY
import json
from datetime import datetime

# Module to test
from services.tasks import question_processing

# Mock the Celery task decorator for direct function testing
# This avoids needing a full Celery worker setup for unit tests
mock_celery_task = MagicMock()
question_processing.celery_app.task = MagicMock(return_value=lambda f: f) # Decorator returns the function itself

# Mock structlog
@pytest.fixture(autouse=True)
def mock_logging():
    with patch('services.tasks.question_processing.structlog.get_logger') as mock_get_logger:
        mock_logger = MagicMock()
        # Configure all logger instances to return the same mock
        mock_get_logger.return_value = mock_logger 
        yield mock_logger

@pytest.fixture
def mock_redis_client():
    """Fixture for a mocked Redis client."""
    client = MagicMock()
    client.ping.return_value = True
    client.hmset.return_value = True
    client.publish.return_value = 1
    return client

@pytest.fixture
def mock_llm_service():
    """Fixture for a mocked LLMService."""
    service = MagicMock()
    # Mock the response object structure based on usage in the task
    mock_response = MagicMock()
    mock_response.is_error = False
    mock_response.content = "This is the generated answer."
    service.generate_answer.return_value = mock_response
    return service

@pytest.fixture
def mock_vector_service():
    """Fixture for a mocked VectorSearchService."""
    service = MagicMock()
    # Example successful search result structure
    service.query.return_value = [
        {'metadata': {'text': 'Context snippet 1'}, 'id': 'doc1'},
        {'metadata': {'text': 'Context snippet 2'}, 'id': 'doc2'}
    ]
    return service

@pytest.fixture(autouse=True) # Apply this fixture automatically to all tests
def mock_dependencies(mock_redis_client, mock_llm_service, mock_vector_service):
    """Patch dependencies used within the task."""
    with patch('services.tasks.question_processing.get_redis_client', return_value=mock_redis_client) as mock_get_redis, \
         patch('services.tasks.question_processing.LLMService', return_value=mock_llm_service) as mock_LLMService_class, \
         patch('services.tasks.question_processing.VectorSearchService', return_value=mock_vector_service) as mock_VectorService_class, \
         patch('services.tasks.question_processing.AppConfig') as mock_AppConfig, \
         patch('services.tasks.question_processing._update_task_progress') as mock_update_progress: # Also mock the helper

        # Configure mock AppConfig defaults
        mock_AppConfig.REDIS_URL = "redis://mock-redis:6379"
        mock_AppConfig.PINECONE_INDEX_NAME = "mock-index"
        mock_AppConfig.DEFAULT_TOP_K = 5
        # Add other necessary AppConfig attributes used by services if any
        # Example: Mocking config dict creation
        mock_AppConfig_instance = MagicMock()
        # Simplification: Assume config dict creation works, focus on service mocks
        # We can refine this if needed by mocking getattr(AppConfig, attr) behavior

        yield {
            "redis_client": mock_redis_client,
            "get_redis_client": mock_get_redis,
            "LLMService": mock_LLMService_class,
            "llm_instance": mock_llm_service,
            "VectorSearchService": mock_VectorService_class,
            "vector_instance": mock_vector_service,
            "AppConfig": mock_AppConfig,
            "update_progress": mock_update_progress
        }

# --- Test Cases ---

def test_process_question_task_success(mock_dependencies, mock_logging):
    """Test the successful execution path of the question processing task."""
    task_id = "test-task-123"
    user_id = "user-abc"
    question = "What is the meaning of life?"
    data = {
        "question": question,
        "user_id": user_id,
        "index_name": "custom-index", # Override default
        "top_k": 3 # Override default
    }

    # Mock the task context object (self)
    mock_self = MagicMock()
    mock_self.request.id = task_id # Simulate Celery task request ID
    mock_self.update_state = MagicMock() # Mock the state update method

    # --- Call the task function ---
    result = question_processing.process_question_task(mock_self, task_id, data)

    # --- Assertions ---
    # 1. Service Initialization
    mock_dependencies["LLMService"].assert_called_once()
    mock_dependencies["VectorSearchService"].assert_called_once()

    # 2. Vector Search Call
    mock_dependencies["vector_instance"].query.assert_called_once_with(
        user_id=user_id,
        query_text=question,
        top_k=3, # Should use value from data
        index_name="custom-index" # Should use value from data
    )

    # 3. LLM Call
    expected_search_results = mock_dependencies["vector_instance"].query.return_value
    mock_dependencies["llm_instance"].generate_answer.assert_called_once_with(
        question=question,
        search_results=expected_search_results
        # Ensure other params match if the LLMService call signature changes
    )

    # 4. Progress Updates (check key stages)
    mock_dependencies["update_progress"].assert_any_call(
        mock_dependencies["redis_client"], f"task:{task_id}", f"progress:{task_id}", task_id, ANY, # logger can be ANY
        status="Processing", progress=10, details="Starting analysis", result=None, error=None
    )
    mock_dependencies["update_progress"].assert_any_call(
        mock_dependencies["redis_client"], f"task:{task_id}", f"progress:{task_id}", task_id, ANY,
        status="Processing", progress=30, details="Searching index 'custom-index' using top_k=3...", result=None, error=None
    )
    mock_dependencies["update_progress"].assert_any_call(
        mock_dependencies["redis_client"], f"task:{task_id}", f"progress:{task_id}", task_id, ANY,
        status="Processing", progress=70, details="Generating answer...", result=None, error=None
    )
    mock_dependencies["update_progress"].assert_any_call(
        mock_dependencies["redis_client"], f"task:{task_id}", f"progress:{task_id}", task_id, ANY,
        status="Completed", progress=100, details="Answer generated successfully.",
        result={"answer": "This is the generated answer."}, # Check final result passed
        error=None
    )
    # Check total calls if needed (might be brittle)
    assert mock_dependencies["update_progress"].call_count == 4 # Start, Search, Generate, Complete

    # 5. Celery State Update (should not be called on success)
    mock_self.update_state.assert_not_called()

    # 6. Return Value
    assert result == {"answer": "This is the generated answer."}

    # 7. Logging (Optional: Check for specific log messages)
    mock_logging.info.assert_any_call(f"Starting question processing task", passed_task_id=task_id, data=data)
    mock_logging.info.assert_any_call("LLM and Vector Search services initialized for task.")
    mock_logging.info.assert_any_call("Search completed.", num_results=2, context_length=ANY)
    mock_logging.info.assert_any_call("Answer generation completed.")
    mock_logging.info.assert_any_call("Question processing task completed successfully.")

# Placeholder for more tests
def test_process_question_task_missing_question(mock_dependencies, mock_logging):
    """Test task failure when 'question' is missing from input data."""
    task_id = "test-task-no-question"
    user_id = "user-xyz"
    data = {
        # "question": "This is missing",
        "user_id": user_id
    }

    mock_self = MagicMock()
    mock_self.request.id = task_id
    mock_self.update_state = MagicMock()

    # Expect the task to raise ValueError, which is caught and re-raised
    with pytest.raises(ValueError, match="Question is required in task data"):
        question_processing.process_question_task(mock_self, task_id, data)

    # Assertions
    # 1. Progress updated to failure
    mock_dependencies["update_progress"].assert_called_with(
        mock_dependencies["redis_client"], f"task:{task_id}", f"progress:{task_id}", task_id, ANY,
        status="Failed", progress=ANY, # Progress might be 10 or 100 depending on exact flow
        details=ANY, # Details might vary slightly
        error=ANY # Error message should be present
    )

    # 2. Celery state updated to FAILURE
    mock_self.update_state.assert_called_once_with(
        state='FAILURE',
        meta=ANY # Check for specific keys if needed: {'exc_type': 'ValueError', 'exc_message': ANY}
    )

    # 3. Ensure VectorSearch and LLM services were NOT called
    mock_dependencies["vector_instance"].query.assert_not_called()
    mock_dependencies["llm_instance"].generate_answer.assert_not_called()

    # 4. Logging
    mock_logging.error.assert_any_call("Unhandled error during question processing task", error=ANY, exc_info=True)

def test_process_question_task_no_context_found(mock_dependencies, mock_logging):
    """Test the case where vector search finds no relevant context."""
    task_id = "test-task-no-context"
    user_id = "user-def"
    question = "Find documents about obscure topic X."
    data = {
        "question": question,
        "user_id": user_id
        # Use default index and top_k from mock_AppConfig
    }

    # Configure VectorSearchService mock to return no results
    mock_dependencies["vector_instance"].query.return_value = []

    mock_self = MagicMock()
    mock_self.request.id = task_id
    mock_self.update_state = MagicMock()

    # --- Call the task function ---
    result = question_processing.process_question_task(mock_self, task_id, data)

    # --- Assertions ---
    # 1. Vector Search was called (using default config)
    mock_dependencies["vector_instance"].query.assert_called_once_with(
        user_id=user_id,
        query_text=question,
        top_k=mock_dependencies["AppConfig"].DEFAULT_TOP_K,
        index_name=mock_dependencies["AppConfig"].PINECONE_INDEX_NAME
    )

    # 2. LLM Service was NOT called
    mock_dependencies["llm_instance"].generate_answer.assert_not_called()

    # 3. Progress updated to Completion with specific details
    mock_dependencies["update_progress"].assert_called_with( # Check the final call
        mock_dependencies["redis_client"], f"task:{task_id}", f"progress:{task_id}", task_id, ANY,
        status="Completed", progress=100,
        details="No relevant context found.",
        result={"answer": "I couldn't find relevant information to answer your question."},
        error=None
    )
    # Ensure it was called multiple times (start, search, complete)
    assert mock_dependencies["update_progress"].call_count >= 3

    # 4. Celery State Update (should not be called for this 'success' case)
    mock_self.update_state.assert_not_called()

    # 5. Return Value
    assert result == {"answer": "I couldn't find relevant information to answer your question."}

    # 6. Logging
    mock_logging.info.assert_any_call("Search completed.", num_results=0, context_length=0)
    mock_logging.info.assert_any_call("Question processing task completed successfully.") # Task still completes

def test_process_question_task_vector_search_error(mock_dependencies, mock_logging):
    """Test task failure when VectorSearchService.query raises an error."""
    task_id = "test-task-vec-err"
    user_id = "user-ghi"
    question = "What happens on error?"
    data = {"question": question, "user_id": user_id}

    # Configure VectorSearchService mock to raise an exception
    search_error_message = "Vector DB connection failed"
    mock_dependencies["vector_instance"].query.side_effect = Exception(search_error_message)

    mock_self = MagicMock()
    mock_self.request.id = task_id
    mock_self.update_state = MagicMock()

    # Expect the task to raise the specific ServiceError, which is caught and re-raised
    # The underlying exception type might differ depending on exact wrapping
    with pytest.raises(Exception) as excinfo:
        question_processing.process_question_task(mock_self, task_id, data)

    # Check if the final raised exception is the one from the service
    # Note: The task wraps it in ServiceError, which is then caught and re-raised
    # The exact type re-raised depends on the outer try/except block.
    # Let's check the error message propagated.
    assert search_error_message in str(excinfo.value)

    # Assertions
    # 1. Vector Search was called
    mock_dependencies["vector_instance"].query.assert_called_once()

    # 2. LLM Service was NOT called
    mock_dependencies["llm_instance"].generate_answer.assert_not_called()

    # 3. Progress updated to failure
    # Check the *last* call to update_progress before the exception is re-raised
    # This happens inside the main exception handler
    mock_dependencies["update_progress"].assert_called_with(
        mock_dependencies["redis_client"], f"task:{task_id}", f"progress:{task_id}", task_id, ANY,
        status="Failed", progress=ANY, # Progress might be 30 or 100
        details=ANY, # Details will mention the error
        error=ANY # Error message should contain the root cause
    )

    # 4. Celery state updated to FAILURE
    mock_self.update_state.assert_called_once_with(
        state='FAILURE',
        meta=ANY # Should contain exception info
    )

    # 5. Logging
    mock_logging.error.assert_any_call("Error during vector search", error=search_error_message, exc_info=True)
    mock_logging.error.assert_any_call("Unhandled error during question processing task", error=ANY, exc_info=True)

def test_process_question_task_llm_error(mock_dependencies, mock_logging):
    """Test task failure when LLMService.generate_answer raises an error."""
    task_id = "test-task-llm-err"
    user_id = "user-jkl"
    question = "What happens on LLM error?"
    data = {"question": question, "user_id": user_id}

    # Configure LLMService mock to raise an exception
    llm_error_message = "LLM API rate limit exceeded"
    # Option 1: Raise directly
    # mock_dependencies["llm_instance"].generate_answer.side_effect = Exception(llm_error_message)
    # Option 2: Simulate error response object (matches current code structure better)
    mock_error_response = MagicMock()
    mock_error_response.is_error = True
    mock_error_response.error = llm_error_message
    mock_dependencies["llm_instance"].generate_answer.return_value = mock_error_response

    mock_self = MagicMock()
    mock_self.request.id = task_id
    mock_self.update_state = MagicMock()

    # Expect the task to raise ServiceError, which is caught and re-raised
    with pytest.raises(Exception) as excinfo:
        question_processing.process_question_task(mock_self, task_id, data)

    # Check the error message propagated
    assert f"LLM service error: {llm_error_message}" in str(excinfo.value)

    # Assertions
    # 1. Vector Search was called successfully
    mock_dependencies["vector_instance"].query.assert_called_once()

    # 2. LLM Service was called
    mock_dependencies["llm_instance"].generate_answer.assert_called_once()

    # 3. Progress updated to failure
    mock_dependencies["update_progress"].assert_called_with(
        mock_dependencies["redis_client"], f"task:{task_id}", f"progress:{task_id}", task_id, ANY,
        status="Failed", progress=ANY, # Progress might be 70 or 100
        details=ANY,
        error=ANY
    )

    # 4. Celery state updated to FAILURE
    mock_self.update_state.assert_called_once_with(
        state='FAILURE',
        meta=ANY
    )

    # 5. Logging
    mock_logging.error.assert_any_call("Error during answer generation", error=ANY, exc_info=True)
    mock_logging.error.assert_any_call("Unhandled error during question processing task", error=ANY, exc_info=True)

def test_process_question_task_redis_unavailable(mock_dependencies, mock_logging):
    """Test task failure when the Redis client cannot be obtained."""
    task_id = "test-task-redis-fail"
    data = {"question": "Does this work without Redis?", "user_id": "user-mno"}

    # Configure get_redis_client mock to return None
    mock_dependencies["get_redis_client"].return_value = None

    mock_self = MagicMock()
    mock_self.request.id = task_id
    mock_self.update_state = MagicMock()

    # --- Call the task function ---
    # Expect it to return an error dict, not raise, after setting state
    result = question_processing.process_question_task(mock_self, task_id, data)

    # Assertions
    # 1. get_redis_client was called
    mock_dependencies["get_redis_client"].assert_called_once()

    # 2. Core services (LLM, Vector) were NOT initialized or called
    mock_dependencies["LLMService"].assert_not_called()
    mock_dependencies["VectorSearchService"].assert_not_called()
    mock_dependencies["vector_instance"].query.assert_not_called()
    mock_dependencies["llm_instance"].generate_answer.assert_not_called()

    # 3. Progress update was NOT called (as redis client is None)
    mock_dependencies["update_progress"].assert_not_called()

    # 4. Celery state updated to FAILURE
    mock_self.update_state.assert_called_once_with(
        state='FAILURE',
        meta={'exc_type': 'ConnectionError', 'exc_message': 'Redis client unavailable'}
    )

    # 5. Return value indicates error
    assert result == {"error": "Redis client unavailable"}

    # 6. Logging
    mock_logging.critical.assert_called_once_with("Cannot proceed without Redis connection.")

def test_process_question_task_service_init_error(mock_dependencies, mock_logging):
    """Test task failure when a core service (LLM or Vector) fails to initialize."""
    task_id = "test-task-init-fail"
    data = {"question": "Will this init?", "user_id": "user-pqr"}

    # Configure LLMService constructor mock to raise an exception
    init_error_message = "Failed to load LLM model"
    mock_dependencies["LLMService"].side_effect = Exception(init_error_message)

    mock_self = MagicMock()
    mock_self.request.id = task_id
    mock_self.update_state = MagicMock()

    # --- Call the task function ---
    # Expect the task to re-raise the initialization exception
    with pytest.raises(Exception) as excinfo:
        question_processing.process_question_task(mock_self, task_id, data)

    # Check if the final raised exception is the one from init
    assert init_error_message in str(excinfo.value)

    # Assertions
    # 1. get_redis_client was called and succeeded (mocked)
    mock_dependencies["get_redis_client"].assert_called_once()

    # 2. LLMService constructor was called (and raised error)
    mock_dependencies["LLMService"].assert_called_once()

    # 3. VectorSearchService constructor might not be called if LLMService failed first
    # mock_dependencies["VectorSearchService"].assert_called_once() # This depends on order

    # 4. Core methods (query, generate_answer) were NOT called
    mock_dependencies["vector_instance"].query.assert_not_called()
    mock_dependencies["llm_instance"].generate_answer.assert_not_called()

    # 5. Progress updated to failure (called before raising)
    mock_dependencies["update_progress"].assert_called_once_with(
        mock_dependencies["redis_client"], f"task:{task_id}", f"progress:{task_id}", task_id, ANY,
        status="Failed", progress=0,
        details="Service initialization failed.",
        error=f"Service init error: {init_error_message}"
    )

    # 6. Celery state updated to FAILURE
    mock_self.update_state.assert_called_once_with(
        state='FAILURE',
        meta={'exc_type': 'Exception', 'exc_message': f'Service init failed: {init_error_message}'}
    )

    # 7. Logging
    mock_logging.error.assert_called_once_with(
        "Failed to initialize services within task",
        error=init_error_message,
        exc_info=True
    )

def test_process_question_task_service_init_error():
    pass # TODO 