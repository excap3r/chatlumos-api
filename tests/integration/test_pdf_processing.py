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

def test_pdf_processing_integration(client, app, configured_celery_app, redis_client, mock_auth, mocker):
    """Integration test for the PDF processing workflow."""
    mock_user_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 fake pdf for integration test"
    file_name = "integration_test.pdf"
    task_id = None # Will be set from the API response

    # --- Step 1: Mock External Dependencies (within the task) ---
    with patch('services.tasks.pdf_processing.VectorSearchService.add_documents', return_value=True) as mock_add_doc, \
         patch('services.pdf_processor.pdf_processor.PDFProcessor.extract_text', return_value="Extracted text content.") as mock_extract_text:

        # --- Step 2: Use mock_auth context manager factory ---
        with mock_auth(user_id=mock_user_id):
            # --- Step 3: Upload PDF via API ---
            response = client.post(
                '/api/v1/upload', # Assuming this is the correct endpoint
                data={'file': (BytesIO(file_content), file_name)},
                content_type='multipart/form-data'
            )
            assert response.status_code == 202
            task_id = response.json['task_id']
            assert task_id is not None

        # --- Step 4: Execute the Celery Task Synchronously ---
        # Import the task function
        from services.tasks.pdf_processing import process_pdf_task
        # Get task state from Redis before execution
        initial_task_state = {k.decode(): v.decode() for k, v in redis_client.hgetall(f"task:{task_id}").items()}
        assert initial_task_state['status'] == 'PENDING'

        # Execute the task directly (synchronously for testing)
        # Need to pass the file content again, or handle file storage/retrieval
        # Assuming the task retrieves content based on task_id/filename if not passed directly
        # For simplicity, let's assume task args include necessary info if not content itself
        # (Adjust based on actual task signature and data flow)
        try:
            # Simulate task execution - actual arguments depend on task signature
            process_pdf_task(task_id=task_id, user_id=mock_user_id, filename=file_name, file_bytes=file_content)
        except Exception as e:
            pytest.fail(f"Celery task execution failed: {e}")

        # --- Step 5: Verify Task Completion and Side Effects ---
        # Check Redis for final task status
        final_task_state = {k.decode(): v.decode() for k, v in redis_client.hgetall(f"task:{task_id}").items()}
        assert final_task_state['status'] == 'SUCCESS'
        assert final_task_state['progress'] == '100'

        # Verify mocks were called
        mock_extract_text.assert_called_once()
        mock_add_doc.assert_called_once()


def test_pdf_processing_integration_task_error(client, app, configured_celery_app, redis_client, mock_auth, mocker):
    """Integration test for error handling during PDF processing task."""
    mock_user_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 error case"
    file_name = "error_test.pdf"
    task_id = None

    # --- Step 1: Mock Dependencies to Simulate Failure ---
    simulated_error_message = "Vector DB connection failed"
    with patch('services.tasks.pdf_processing.VectorSearchService.add_documents', side_effect=Exception(simulated_error_message)) as mock_add_doc, \
         patch('services.pdf_processor.pdf_processor.PDFProcessor.extract_text', return_value="Some text") as mock_extract_text:

        # --- Step 2: Use mock_auth context manager factory ---
        with mock_auth(user_id=mock_user_id):
            # --- Step 3: Upload PDF via API ---
            response = client.post(
                '/api/v1/upload', # Use the correct endpoint
                data={'file': (BytesIO(file_content), file_name)},
                content_type='multipart/form-data'
            )
            assert response.status_code == 202
            task_id = response.json['task_id']
            assert task_id is not None

        # --- Step 4: Execute the Celery Task Synchronously ---
        from services.tasks.pdf_processing import process_pdf_task
        initial_task_state = {k.decode(): v.decode() for k, v in redis_client.hgetall(f"task:{task_id}").items()}
        assert initial_task_state['status'] == 'PENDING'

        # Execute task, expecting it to raise the simulated exception internally
        # and update Redis status to FAILURE
        process_pdf_task(task_id=task_id, user_id=mock_user_id, filename=file_name, file_bytes=file_content)
        # The task itself should handle the exception and update status, not re-raise here usually.

        # --- Step 5: Verify Task Failure State ---
        final_task_state = {k.decode(): v.decode() for k, v in redis_client.hgetall(f"task:{task_id}").items()}
        assert final_task_state['status'] == 'FAILURE'
        assert simulated_error_message in final_task_state.get('error_message', '') # Check error message propagation

        # Verify mocks (extract text might be called before the failing add_documents)
        mock_extract_text.assert_called_once()
        mock_add_doc.assert_called_once()


def test_ask_processing_integration(client, app, configured_celery_app, redis_client, mock_auth, mocker):
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

        # --- Step 2: Use mock_auth context manager factory ---
        with mock_auth(user_id=mock_user_id):
            # --- Step 3: Submit Question via API ---
            response = client.post(
                '/api/v1/ask',
                json={'question': question, 'pdf_id': pdf_id}
            )
            assert response.status_code == 202
            task_id = response.json['task_id']
            assert task_id is not None

        # --- Step 4: Execute the Celery Task Synchronously ---
        from services.tasks.question_processing import process_question_task
        initial_task_state = {k.decode(): v.decode() for k, v in redis_client.hgetall(f"task:{task_id}").items()}
        assert initial_task_state['status'] == 'PENDING'

        # Execute the task
        process_question_task(task_id=task_id, user_id=mock_user_id, question=question, pdf_id=pdf_id)

        # --- Step 5: Verify Task Completion and Side Effects ---
        final_task_state = {k.decode(): v.decode() for k, v in redis_client.hgetall(f"task:{task_id}").items()}
        assert final_task_state['status'] == 'SUCCESS'
        assert final_task_state['result'] == mock_answer # Check if the answer is stored

        # Verify mocks were called
        mock_vector_query.assert_called_once_with(query_text=question, pdf_id=pdf_id, user_id=mock_user_id) # Verify args
        mock_llm_call.assert_called_once_with(context=mock_context, question=question) # Verify args 