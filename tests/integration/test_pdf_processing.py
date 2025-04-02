import pytest
import uuid
import json
from unittest.mock import patch, MagicMock
from io import BytesIO
from flask import g
from functools import wraps

# Assuming client, app, configured_celery_app, redis_client fixtures are available from conftest.py
# These fixtures will need to be set up to handle integration testing specifics.

# Test the full flow: POST /api/pdf -> Celery Task -> Redis Update -> GET /api/progress

def test_pdf_processing_integration(client, app, configured_celery_app, redis_client, mock_auth):
    """Integration test for the PDF processing workflow."""
    mock_user_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 fake pdf for integration test"
    file_name = "integration_test.pdf"
    task_id = None # Will be set from the API response

    # --- Step 1: Mock External Dependencies (within the task) ---
    # Mock services called by process_pdf_task (e.g., PDF parsing, vector DB, etc.)
    # These mocks ensure the test focuses on the integration, not external systems.
    
    # Example: Mocking vector search interactions if process_pdf_task uses it
    with patch('services.tasks.pdf_processing.VectorSearchService.add_documents', return_value=True) as mock_add_doc, \
         patch('services.pdf_processor.pdf_processor.PDFProcessor.extract_text', return_value="Extracted text content.") as mock_extract_text:
        
        # --- Step 2: Authenticate the Request using mock_auth fixture --- 
        auth_patcher = mock_auth(user_id=mock_user_id)
        with auth_patcher: # Apply patch context
            # --- Step 3: Make the API Request to trigger the workflow --- 
            data = {'file': (BytesIO(file_content), file_name)}
            # Corrected URL to include API version prefix AND the correct endpoint /upload
            response = client.post('/api/v1/pdf/upload', data=data, content_type='multipart/form-data')

            # --- Step 4: Assert Initial API Response --- 
            assert response.status_code == 202
            assert 'task_id' in response.json
            task_id = response.json['task_id']
            assert task_id is not None

        # --- Step 5: Assert Task Execution and Mocks Called --- 
        # Because Celery is configured as eager, the task should have run synchronously.
        mock_extract_text.assert_called_once()
        # Ensure the text extraction was called with the correct file content
        call_args, call_kwargs = mock_extract_text.call_args
        # Task receives base64 encoded content, not raw bytes
        # assert call_args[0] == file_content # Incorrect
        assert 'file_content_b64' in call_kwargs 
        
        mock_add_doc.assert_called_once()
        # Ensure add_document was called with expected arguments (user_id, content, metadata)
        call_args, call_kwargs = mock_add_doc.call_args
        assert call_args[0] == mock_user_id
        assert call_args[1] == "Extracted text content." # The mocked extracted text
        # Corrected metadata assertion to include pdf_id (which is task_id in this flow)
        assert call_args[2] == {"source": file_name, "user_id": mock_user_id, "task_id": task_id, "pdf_id": task_id} 
        

        # --- Step 6: Verify Final State in Redis --- 
        # Check Redis directly to confirm the task updated the status.
        redis_key = f"task:{task_id}"
        final_status_data = redis_client.hgetall(redis_key) # Use HGETALL
        assert final_status_data is not None
        # final_status = json.loads(final_status_bytes) # No decode needed if redis_client decodes responses
        # Convert HGETALL result (dict of strings) to expected types if necessary
        final_status = { k: v for k, v in final_status_data.items() } # Basic conversion

        assert final_status['status'] == 'SUCCESS' # Or whatever the final success status is
        assert int(final_status['progress']) == 100 # Convert progress to int
        assert final_status['filename'] == file_name
        # Check for other expected fields like results or messages
        assert 'Document processed and added successfully' in final_status['message']
        # Assert pdf_id is included in the final status
        assert final_status['pdf_id'] == task_id 


def test_pdf_processing_integration_task_error(client, app, configured_celery_app, redis_client, mock_auth):
    """Integration test for error handling during PDF processing task."""
    mock_user_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 error case"
    file_name = "error_test.pdf"
    task_id = None

    # --- Step 1: Mock Dependencies to Simulate Failure ---
    simulated_error_message = "Vector DB connection failed"
    with patch('services.tasks.pdf_processing.VectorSearchService.add_documents', side_effect=Exception(simulated_error_message)) as mock_add_doc, \
         patch('services.pdf_processor.pdf_processor.PDFProcessor.extract_text', return_value="Some text") as mock_extract_text:
        
        # --- Step 2: Authenticate using mock_auth fixture --- 
        auth_patcher = mock_auth(user_id=mock_user_id)
        with auth_patcher: # Apply patch context
            # --- Step 3: Make API Request --- 
            data = {'file': (BytesIO(file_content), file_name)}
            # Corrected URL to include API version prefix AND the correct endpoint /upload
            response = client.post('/api/v1/pdf/upload', data=data, content_type='multipart/form-data')

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
        final_status_data = redis_client.hgetall(redis_key) # Use HGETALL
        assert final_status_data is not None
        final_status = { k: v for k, v in final_status_data.items() }
        
        assert final_status['status'] == 'FAILURE'
        assert int(final_status['progress']) == 0 # Convert progress to int
        assert final_status['filename'] == file_name
        assert 'error' in final_status
        # Check if the specific error message is included
        assert simulated_error_message in final_status['error'] 


def test_ask_processing_integration(client, app, configured_celery_app, redis_client, mock_auth):
    """Integration test for the question answering workflow (/ask)."""
    mock_user_id = str(uuid.uuid4())
    question = "What is the meaning of life?"
    pdf_id = "processed_pdf_123" # Assume this PDF ID exists or is mocked
    task_id = None
    mock_answer = "The answer is 42."
    mock_context = "Some relevant context from vector search."

    # --- Step 1: Mock External Dependencies (within the task) ---
    with patch('services.tasks.question_processing.VectorSearchService.query', return_value=mock_context) as mock_vector_query, \
         patch('services.tasks.question_processing.LLMService.generate_answer', return_value=mock_answer) as mock_llm_call:

        # --- Step 2: Authenticate using mock_auth fixture --- 
        auth_patcher = mock_auth(user_id=mock_user_id)
        with auth_patcher: # Apply patch context
            # --- Step 3: Make API Request --- 
            # Corrected URL to include API version prefix
            response = client.post('/api/v1/ask', json={
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
        # Assuming filter is passed via kwargs
        assert call_kwargs.get('filter') == {'pdf_id': pdf_id}
        
        mock_llm_call.assert_called_once()
        # Assert LLM call args - question, context
        call_args, call_kwargs = mock_llm_call.call_args
        assert call_args[0] == question
        assert call_args[1] == mock_context

        # --- Step 6: Verify Final State in Redis --- 
        redis_key = f"task:{task_id}"
        final_status_data = redis_client.hgetall(redis_key) # Use HGETALL
        assert final_status_data is not None
        final_status = { k: v for k, v in final_status_data.items() }

        assert final_status['status'] == 'SUCCESS'
        assert int(final_status['progress']) == 100
        assert 'result' in final_status
        assert final_status['result'] == json.dumps({ # Assuming result is stored as JSON string
            'question': question,
            'answer': mock_answer,
            'context': mock_context # Or however the result is structured
        })
        assert 'Task completed successfully' in final_status['message'] 