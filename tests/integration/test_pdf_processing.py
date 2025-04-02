import pytest
import uuid
import json
from unittest.mock import patch, MagicMock, ANY
from io import BytesIO
from flask import g
from functools import wraps
import time
import base64
from flask import url_for

# Assuming client, app, configured_celery_app, redis_client fixtures are available from conftest.py
# These fixtures will need to be set up to handle integration testing specifics.

# Import Celery app for potential task inspection/mocking
from celery_app import celery_app

# Import models if needed for setup/assertions
from services.db.models.user_models import User
from services.db.models.document_models import Document

API_BASE_URL = "/api/v1"

# Define the patch decorator as a constant for reuse
PATCH_AUTH = patch('services.api.middleware.auth_middleware.require_auth', new=lambda *args, **kwargs: lambda f: f)

# Test the full flow: POST /api/pdf -> Celery Task -> Redis Update -> GET /api/progress

@patch('services.tasks.pdf_processing.VectorSearchService.add_documents', return_value=True)
@patch('services.pdf_processor.pdf_processor.PDFProcessor.extract_text', return_value="Extracted text content.")
def test_pdf_processing_integration(mock_extract_text, mock_add_doc, client, app, configured_celery_app, redis_client, mock_auth):
    """Integration test for the PDF processing workflow."""
    # Note: mock_auth fixture handles patching require_auth
    mock_user_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 fake pdf for integration test"
    file_name = "integration_test.pdf"
    task_id = None

    # Use mock_auth setter context manager
    with mock_auth(user_id=mock_user_id):
        # Upload PDF via API
        response = client.post(
            f'{API_BASE_URL}/pdf/upload', 
            data={'file': (BytesIO(file_content), file_name)},
            content_type='multipart/form-data'
        )
        assert response.status_code == 202
        task_id = response.json['task_id']
        assert task_id is not None

    # Task runs eagerly due to celery_app fixture config
    # Wait a short time for eager task to likely complete and update Redis
    time.sleep(0.1)

    # Check final task state in Redis
    redis_key = f"task:{task_id}"
    task_state = redis_client.hgetall(redis_key)
    assert task_state is not None, f"Task state not found in Redis for key {redis_key}"
    assert task_state.get('status') == 'SUCCESS'
    assert task_state.get('filename') == file_name
    assert task_state.get('user_id') == mock_user_id
    assert int(task_state.get('progress', 0)) == 100
    assert "Processing complete" in task_state.get('details', '')

    # Verify mocked dependencies were called
    mock_extract_text.assert_called_once()
    mock_add_doc.assert_called_once()

@patch('services.tasks.pdf_processing.VectorSearchService.add_documents', side_effect=Exception("Vector DB connection failed"))
@patch('services.pdf_processor.pdf_processor.PDFProcessor.extract_text', return_value="Some text")
def test_pdf_processing_integration_task_error(mock_extract_text, mock_add_doc_error, client, app, configured_celery_app, redis_client, mock_auth):
    """Integration test for error handling during PDF processing task."""
    # Note: mock_auth fixture handles patching require_auth
    mock_user_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 error case"
    file_name = "error_test.pdf"
    task_id = None
    simulated_error_message = "Vector DB connection failed"

    with mock_auth(user_id=mock_user_id):
        headers = {'Authorization': 'Bearer dummy'}
        response = client.post(
            f'{API_BASE_URL}/pdf/upload', 
            data={'file': (BytesIO(file_content), file_name)},
            content_type='multipart/form-data',
            headers=headers
        )
        assert response.status_code == 202
        task_id = response.json['task_id']
        assert task_id is not None

    time.sleep(0.1)

    redis_key = f"task:{task_id}"
    task_state = redis_client.hgetall(redis_key)
    assert task_state is not None
    assert task_state.get('status') == 'FAILURE'
    assert task_state.get('filename') == file_name
    assert task_state.get('user_id') == mock_user_id
    assert task_state.get('error') is not None
    assert simulated_error_message in task_state.get('error', '')

    mock_extract_text.assert_called_once()
    mock_add_doc_error.assert_called_once()

@patch('services.tasks.question_processing.VectorSearchService.query', return_value="Some relevant context from vector search.")
@patch('services.tasks.question_processing.LLMService.generate_answer', return_value="The answer is 42.")
def test_ask_processing_integration(mock_llm_call, mock_vector_query, client, app, configured_celery_app, redis_client, mock_auth):
    """Integration test for the question answering workflow (/ask)."""
    # Note: mock_auth fixture handles patching require_auth
    mock_user_id = str(uuid.uuid4())
    question = "What is the meaning of life?"
    pdf_id = "processed_pdf_123" # Assume this PDF ID exists/is setup if needed
    task_id = None
    mock_answer = "The answer is 42."
    mock_context = "Some relevant context from vector search."

    with mock_auth(user_id=mock_user_id):
        headers = {'Authorization': 'Bearer dummy'}
        response = client.post(
            f'{API_BASE_URL}/ask',
            json={'question': question, 'pdf_id': pdf_id},
            headers=headers
        )
        assert response.status_code == 202
        task_id = response.json['task_id']
        assert task_id is not None

    time.sleep(0.1)

    redis_key = f"task:{task_id}"
    task_state = redis_client.hgetall(redis_key)
    assert task_state is not None
    assert task_state.get('status') == 'SUCCESS'
    assert task_state.get('user_id') == mock_user_id
    assert task_state.get('question') == question
    assert task_state.get('pdf_id') == pdf_id
    assert int(task_state.get('progress', 0)) == 100
    result_data = json.loads(task_state.get('result', '{}')) # Result stored as JSON string
    assert result_data.get('answer') == mock_answer

    mock_vector_query.assert_called_once_with(question)
    mock_llm_call.assert_called_once_with(question, mock_context)

MOCK_USER_ID = "mock-pdf-user-id"
MOCK_TASK_ID = "mock-pdf-task-id-123"

# Basic PDF content for testing
PDF_CONTENT = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>/Contents 4 0 R>>endobj\n4 0 obj<</Length 35>>stream\nBT /F1 12 Tf (Test PDF Content)Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \n0000000112 00000 n \n0000000201 00000 n \ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n278\n%%EOF"
PDF_FILENAME = "test_document.pdf"

def create_dummy_pdf():
    return BytesIO(PDF_CONTENT)

# Test requiring authentication and mocking the PDF processing task
# @patch('services.api.middleware.auth_middleware.require_auth', new_callable=create_mock_auth_decorator)
@PATCH_AUTH
@patch('services.tasks.pdf_processing.process_pdf_task.delay')
def test_pdf_upload_queues_task(mock_delay, client, mock_auth):
    """Integration test for PDF upload verifying Celery task queuing."""
    mock_delay.return_value = MagicMock(id=MOCK_TASK_ID)

    with mock_auth(user_id=MOCK_USER_ID):
        headers = {'Authorization': 'Bearer dummy'}
        # 1. Upload PDF
        pdf_file = create_dummy_pdf()
        data = {
            'file': (pdf_file, PDF_FILENAME)
        }
        response = client.post(
            '/api/v1/pdf/upload',
            content_type='multipart/form-data',
            data=data,
            headers=headers
        )

    assert response.status_code == 202
    assert 'task_id' in response.json
    assert response.json['task_id'] == MOCK_TASK_ID

    # Verify task was called with correct arguments
    mock_delay.assert_called_once()
    args, kwargs = mock_delay.call_args
    assert args[0] == MOCK_TASK_ID # First positional arg is task_id
    assert isinstance(args[1], str) # file_content_b64
    decoded_content = base64.b64decode(args[1].encode('utf-8'))
    assert decoded_content == PDF_CONTENT
    assert args[2] == PDF_FILENAME # filename
    assert args[3] == MOCK_USER_ID # user_id
    # Add checks for other args if necessary

    # 2. Check initial progress (optional, depends on task behavior)
    # Assuming the task immediately sets initial state in Redis
    # You might need a short sleep or a more robust check
    # time.sleep(0.1)
    # with mock_auth(user_id=MOCK_USER_ID):
    #     progress_response = client.get(f'/api/v1/tasks/{MOCK_TASK_ID}/progress')
    # assert progress_response.status_code == 200
    # assert progress_response.json['status'] in ['Pending', 'Received'] # Adjust based on actual initial state

    # 3. Simulate task completion (if testing full flow)
    # This would involve mocking Redis/DB updates done by the task
    # For this example, we just check the task was queued.

    # ... rest of the file ... 

from services.tasks.question_processing import ServiceError # Import ServiceError if needed for mocking

@patch('services.tasks.question_processing._update_task_progress') # Mock the progress update helper
@patch('services.tasks.question_processing.VectorSearchService.query')
@patch('services.tasks.question_processing.LLMService.generate_answer')
def test_ask_streaming_integration(mock_llm_generate, mock_vector_query, mock_update_progress, client, configured_celery_app, redis_client, mock_auth):
    """Integration test for the streaming /ask workflow, verifying progress updates."""
    mock_user_id = str(uuid.uuid4())
    question = "What is streaming?"
    task_id = None
    mock_answer_content = "Streaming is sending data piece by piece."
    mock_search_results = [
        {'metadata': {'text': 'Context 1'}, 'id': 'doc1'},
        {'metadata': {'text': 'Context 2'}, 'id': 'doc2'}
    ]
    # Mock service responses
    mock_vector_query.return_value = mock_search_results
    mock_llm_response = MagicMock()
    mock_llm_response.is_error = False
    mock_llm_response.content = mock_answer_content
    mock_llm_generate.return_value = mock_llm_response

    with mock_auth(user_id=mock_user_id):
        headers = {'Authorization': 'Bearer dummy'}
        response = client.post(
            f'{API_BASE_URL}/ask',
            json={
                'question': question,
                'stream': True # Enable streaming
            },
            headers=headers
        )
        # 1. Verify API response for streaming request
        assert response.status_code == 202
        task_id = response.json['task_id']
        assert task_id is not None

    # Task runs eagerly. Now verify the progress updates.
    # 2. Verify progress updates were called via the mocked helper
    assert mock_update_progress.call_count >= 4 # Expect at least: Start, Search, Generate, Complete

    # Check some key progress calls (arguments: redis_client, redis_key, pubsub_channel, task_id, logger, status, progress, details, result, error)
    redis_key = f"task:{task_id}"
    pubsub_channel = f"progress:{task_id}"

    # Check Starting call
    mock_update_progress.assert_any_call(
        ANY, # redis_client (mocked globally or passed)
        redis_key,
        pubsub_channel,
        task_id,
        ANY, # logger
        status="Processing", progress=10, details="Starting analysis", result=None, error=None
    )
    # Check Searching call
    mock_update_progress.assert_any_call(
        ANY, redis_key, pubsub_channel, task_id, ANY,
        status="Processing", progress=30, details=ANY, result=None, error=None # Details might vary slightly based on index/top_k defaults
    )
    # Check Generating call
    mock_update_progress.assert_any_call(
        ANY, redis_key, pubsub_channel, task_id, ANY,
        status="Processing", progress=70, details="Generating answer...", result=None, error=None
    )
    # Check Completion call
    final_result_data = {"answer": mock_answer_content}
    mock_update_progress.assert_any_call(
        ANY, redis_key, pubsub_channel, task_id, ANY,
        status="Completed", progress=100, details="Answer generated successfully.",
        result=final_result_data,
        error=None
    )

    # 3. Verify external services were called by the task
    mock_vector_query.assert_called_once()
    # We need to check the args passed to query (user_id, question, top_k, index_name)
    # Get defaults from AppConfig mock if necessary (might need to enhance fixture)
    # Example: Assuming default top_k=5, default index='mock-index'
    call_args, call_kwargs = mock_vector_query.call_args
    assert call_kwargs.get('user_id') == mock_user_id
    assert call_kwargs.get('query_text') == question
    # assert call_kwargs.get('top_k') == MOCK_DEFAULT_TOP_K # Need to get this from mock AppConfig
    # assert call_kwargs.get('index_name') == MOCK_DEFAULT_INDEX # Need to get this from mock AppConfig

    mock_llm_generate.assert_called_once_with(
        question=question,
        search_results=mock_search_results
    ) 