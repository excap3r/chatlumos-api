import pytest
import uuid
import json
from unittest.mock import patch, MagicMock
from io import BytesIO

# Assuming client, app, configured_celery_app, redis_client fixtures are available from conftest.py
# These fixtures will need to be set up to handle integration testing specifics.

# Test the full flow: POST /api/pdf -> Celery Task -> Redis Update -> GET /api/progress

def test_pdf_processing_integration(client, app, configured_celery_app, redis_client):
    """Integration test for the PDF processing workflow."""
    mock_user_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 fake pdf for integration test"
    file_name = "integration_test.pdf"
    task_id = None # Will be set from the API response

    # --- Step 1: Mock External Dependencies (within the task) ---
    # Mock services called by process_pdf_task (e.g., PDF parsing, vector DB, etc.)
    # These mocks ensure the test focuses on the integration, not external systems.
    
    # Example: Mocking vector search interactions if process_pdf_task uses it
    with patch('services.vector_search.vector_search.add_document', return_value=True) as mock_add_doc, \
         patch('services.pdf_processor.pdf_processor.extract_text', return_value="Extracted text content.") as mock_extract_text:
        
        # --- Step 2: Authenticate the Request ---
        # We need a way to simulate an authenticated user for the API call.
        # This might involve a fixture that generates a test token or mocks the require_auth decorator.
        # For now, let's assume a fixture or direct patch handles this.
        # Patching require_auth to inject g.user:
        with patch('services.api.middleware.auth_middleware.require_auth') as mock_require_auth:
            from functools import wraps
            from flask import g
            def mock_decorator(f):
                @wraps(f)
                def decorated_function(*args, **kwargs):
                    g.user = {'id': mock_user_id}
                    return f(*args, **kwargs)
                return decorated_function
            mock_require_auth.side_effect = mock_decorator

            # --- Step 3: Make the API Request to trigger the workflow ---
            data = {'file': (BytesIO(file_content), file_name)}
            response = client.post('/api/pdf', data=data, content_type='multipart/form-data')

            # --- Step 4: Assert Initial API Response ---
            assert response.status_code == 202
            assert 'task_id' in response.json
            task_id = response.json['task_id']
            assert task_id is not None

        # --- Step 5: Assert Task Execution and Mocks Called ---
        # Because Celery is configured as eager, the task should have run synchronously.
        mock_extract_text.assert_called_once()
        # Ensure the text extraction was called with the correct file content
        # This might need adjustment if the task saves the file temporarily
        call_args, call_kwargs = mock_extract_text.call_args
        assert call_args[0] == file_content 
        
        mock_add_doc.assert_called_once()
        # Ensure add_document was called with expected arguments (user_id, content, metadata)
        call_args, call_kwargs = mock_add_doc.call_args
        assert call_args[0] == mock_user_id
        assert call_args[1] == "Extracted text content." # The mocked extracted text
        assert call_args[2] == {"source": file_name, "user_id": mock_user_id, "task_id": task_id} # Example metadata
        # Add more assertions based on the specific calls within process_pdf_task

        # --- Step 6: Verify Final State in Redis ---
        # Check Redis directly to confirm the task updated the status.
        redis_key = f"task:{task_id}"
        final_status_bytes = redis_client.get(redis_key)
        assert final_status_bytes is not None
        final_status = json.loads(final_status_bytes.decode('utf-8'))
        
        assert final_status['status'] == 'SUCCESS' # Or whatever the final success status is
        assert final_status['progress'] == 100
        assert final_status['filename'] == file_name
        # Check for other expected fields like results or messages
        assert 'Document processed and added successfully' in final_status['message']

    # --- (Optional) Step 7: Verify SSE Endpoint --- 
    # This is more complex to test reliably in an integration test.
    # Checking Redis directly (Step 6) is often sufficient.

# TODO: Add integration test for error handling during PDF processing
# TODO: Add integration test for the /ask workflow

def test_pdf_processing_integration_task_error(client, app, configured_celery_app, redis_client):
    """Integration test for error handling during PDF processing task."""
    mock_user_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 error case"
    file_name = "error_test.pdf"
    task_id = None

    # --- Step 1: Mock Dependencies to Simulate Failure ---
    # Simulate vector_search.add_document raising an exception
    simulated_error_message = "Vector DB connection failed"
    with patch('services.vector_search.vector_search.add_document', side_effect=Exception(simulated_error_message)) as mock_add_doc, \
         patch('services.pdf_processor.pdf_processor.extract_text', return_value="Some text") as mock_extract_text:
        
        # --- Step 2: Authenticate ---
        with patch('services.api.middleware.auth_middleware.require_auth') as mock_require_auth:
            from functools import wraps
            from flask import g
            def mock_decorator(f):
                @wraps(f)
                def decorated_function(*args, **kwargs):
                    g.user = {'id': mock_user_id}
                    return f(*args, **kwargs)
                return decorated_function
            mock_require_auth.side_effect = mock_decorator

            # --- Step 3: Make API Request ---
            data = {'file': (BytesIO(file_content), file_name)}
            response = client.post('/api/pdf', data=data, content_type='multipart/form-data')

            # --- Step 4: Assert Initial API Response ---
            assert response.status_code == 202
            assert 'task_id' in response.json
            task_id = response.json['task_id']
            assert task_id is not None

        # --- Step 5: Assert Task Execution and Mocks Called ---
        # Task still runs eagerly
        mock_extract_text.assert_called_once()
        mock_add_doc.assert_called_once() # Assert it was called, even though it failed

        # --- Step 6: Verify Error State in Redis ---
        redis_key = f"task:{task_id}"
        final_status_bytes = redis_client.get(redis_key)
        assert final_status_bytes is not None
        final_status = json.loads(final_status_bytes.decode('utf-8'))
        
        assert final_status['status'] == 'FAILURE'
        assert final_status['progress'] == 0 # Or progress might be partial if error occurs mid-way
        assert final_status['filename'] == file_name
        assert 'error' in final_status
        # Check if the specific error message is included
        assert simulated_error_message in final_status['error'] 

# TODO: Add integration test for the /ask workflow

def test_ask_processing_integration(client, app, configured_celery_app, redis_client):
    """Integration test for the question answering workflow (/ask)."""
    mock_user_id = str(uuid.uuid4())
    question = "What is the meaning of life?"
    pdf_id = "processed_pdf_123" # Assume this PDF ID exists or is mocked
    task_id = None
    mock_answer = "The answer is 42."
    mock_context = "Some relevant context from vector search."

    # --- Step 1: Mock External Dependencies (within the task) ---
    # Mock vector search query and LLM call used by process_question_task
    with patch('services.vector_search.vector_search.query', return_value=mock_context) as mock_vector_query, \
         patch('services.llm_service.llm_service.generate_answer', return_value=mock_answer) as mock_llm_call:

        # --- Step 2: Authenticate ---
        with patch('services.api.middleware.auth_middleware.require_auth') as mock_require_auth:
            from functools import wraps
            from flask import g
            def mock_decorator(f):
                @wraps(f)
                def decorated_function(*args, **kwargs):
                    g.user = {'id': mock_user_id}
                    return f(*args, **kwargs)
                return decorated_function
            mock_require_auth.side_effect = mock_decorator

            # --- Step 3: Make API Request ---
            response = client.post('/api/ask', json={
                'question': question,
                'pdf_id': pdf_id
            })

            # --- Step 4: Assert Initial API Response ---
            assert response.status_code == 202
            assert 'task_id' in response.json
            task_id = response.json['task_id']
            assert task_id is not None

        # --- Step 5: Assert Task Execution and Mocks Called ---
        # Task runs eagerly due to config
        mock_vector_query.assert_called_once()
        # Assert vector query args - user_id, question, potentially pdf_id filter
        call_args, call_kwargs = mock_vector_query.call_args
        assert call_args[0] == mock_user_id
        assert call_args[1] == question
        # Check filter if applicable, e.g., call_kwargs.get('filter') == {'pdf_id': pdf_id}
        
        mock_llm_call.assert_called_once()
        # Assert LLM call args - question, context
        call_args, call_kwargs = mock_llm_call.call_args
        assert call_args[0] == question
        assert call_args[1] == mock_context

        # --- Step 6: Verify Final State in Redis ---
        redis_key = f"task:{task_id}"
        final_status_bytes = redis_client.get(redis_key)
        assert final_status_bytes is not None
        final_status = json.loads(final_status_bytes.decode('utf-8'))

        assert final_status['status'] == 'SUCCESS'
        assert final_status['progress'] == 100
        assert 'result' in final_status
        assert final_status['result'] == {
            'question': question,
            'answer': mock_answer,
            'context': mock_context # Or however the result is structured
        }
        assert 'Task completed successfully' in final_status['message'] 