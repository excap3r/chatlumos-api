import pytest
from flask import Flask
from unittest.mock import patch, MagicMock
import json
from services.utils.error_utils import APIError
import uuid

# Assuming fixtures like 'client', 'auth_headers' are defined in conftest.py

def test_ask_non_streaming_success(client, auth_headers, request_context):
    """Test successful /ask request with stream=false."""
    # Mock the backend call that ask_route makes
    # This depends on the implementation: does it call LLMService directly? Or API Gateway?
    # Assuming direct call to a service/function for this example
    mock_llm_response = {
        "answer": "This is the answer.",
        "context": [{"text": "Relevant context 1"}]
    }
    
    # Replace 'services.api.routes.ask.ask_question_internal' with the actual function/method called
    with patch('services.api.routes.ask.ask_question_internal', return_value=mock_llm_response) as mock_internal_ask:
        response = client.post('/api/v1/ask', 
                                headers=auth_headers, 
                                json={'question': 'What is the test question?', 'stream': False})
        
        assert response.status_code == 200
        json_data = response.get_json()
        assert 'answer' in json_data
        assert 'context' in json_data
        assert json_data['answer'] == mock_llm_response['answer']
        
        # Verify the internal function was called correctly
        mock_internal_ask.assert_called_once_with(
            question='What is the test question?',
            index_name=None, # Assuming default
            top_k=None,      # Assuming default
            stream=False,
            user_info=pytest.approx({}) # Check if user_info is passed; might need adjustment based on auth setup
        )

@patch('services.api.routes.ask.process_question_task.delay')
def test_ask_streaming_success(mock_celery_delay, client, redis_client, mock_auth):
    """Test successful /ask request with stream=true (queues task)."""
    mock_user_id = "ask-stream-user-fixture"
    question = "What is the streaming test question?"
    pdf_id = "pdf-for-stream-ask"
    task_id_capture = None

    def capture_task_id(*args, **kwargs):
        nonlocal task_id_capture
        if args:
            task_id_capture = args[0]
        mock_task_result = MagicMock()
        mock_task_result.id = task_id_capture or str(uuid.uuid4())
        task_id_capture = mock_task_result.id
        return mock_task_result
    mock_celery_delay.side_effect = capture_task_id

    with mock_auth(user_id=mock_user_id):
        headers = {'Authorization': 'Bearer dummy'} # Header needed for middleware
        response = client.post('/api/v1/ask', 
                                headers=headers, 
                                json={'question': question, 'pdf_id': pdf_id, 'stream': True})
        
    assert response.status_code == 202 # Accepted for async task
    json_data = response.get_json()
    assert 'task_id' in json_data
    assert json_data['task_id'] is not None
    if task_id_capture:
        assert json_data['task_id'] == task_id_capture
        
    # Verify the Celery task was called correctly
    mock_celery_delay.assert_called_once()
    call_args, call_kwargs = mock_celery_delay.call_args
    assert call_args[0] == task_id_capture # Task ID
    expected_data = {
        'question': question,
        'pdf_id': pdf_id,
        'user_id': mock_user_id,
        'stream': True # Ensure stream flag is passed
        # Include other expected fields like index_name, top_k if defaults are tested
    }
    assert call_args[1] == expected_data # Data dict

    # Check Redis state for the queued task
    redis_key = f"task:{task_id_capture}"
    task_state = redis_client.hgetall(redis_key)
    assert task_state['status'] == 'Queued'
    assert task_state['user_id'] == mock_user_id
    assert task_state['question'] == question
    assert task_state['pdf_id'] == pdf_id
    assert task_state['stream'] == 'True' # Check stream flag in Redis (likely stored as string)

def test_ask_missing_question(client, mock_auth):
    """Test /ask request with missing 'question' field."""
    with mock_auth():
        headers = {'Authorization': 'Bearer dummy'} # Header needed
        response = client.post('/api/v1/ask', 
                                headers=headers, 
                                json={'stream': False}) # Missing 'question'
    
    assert response.status_code == 400 # Bad Request
    json_data = response.get_json()
    assert 'error' in json_data
    # Adjusted based on expected Marshmallow/validation error format
    assert 'question' in json_data['error']
    assert 'Missing data for required field' in json_data['error']['question'][0]

def test_ask_unauthorized(client):
    """Test /ask request without authentication."""
    response = client.post('/api/v1/ask', 
                            json={'question': 'This should fail', 'stream': False})
    
    assert response.status_code == 401 # Unauthorized
    json_data = response.get_json()
    assert 'error' in json_data
    # Check message from the actual middleware
    assert 'Authentication required' in json_data['error']

def test_ask_internal_service_error(client, mock_auth):
    """Test /ask handles errors raised by the internal ask function/task."""
    error_message = "Internal LLM service exploded"
    # Patch the Celery task delay to simulate failure
    with patch('services.api.routes.ask.process_question_task.delay', side_effect=APIError(error_message, status_code=503)) as mock_delay:
        with mock_auth():
            headers = {'Authorization': 'Bearer dummy'} # Header needed
            response = client.post('/api/v1/ask', 
                                    headers=headers, 
                                    json={'question': 'Trigger an error', 'stream': False})
            
    # The API should still return 202 as the task was submitted (even if it fails immediately)
    # Error handling happens within the task or is reported via progress stream
    # Let's adjust assertion based on API behavior (assuming it returns 500 if delay fails?)
    # Re-evaluating: If .delay() itself fails, Flask might catch it and return 500.
    assert response.status_code == 503 # Assuming APIError propagates if delay fails sync
    json_data = response.get_json()
    assert 'error' in json_data
    assert error_message in json_data['error']
    mock_delay.assert_called_once() # Verify the attempt to call delay

def test_ask_with_index_and_topk(client, mock_auth):
    """Test /ask request with optional index_name and top_k parameters."""
    # Patch the celery task delay
    with patch('services.api.routes.ask.process_question_task.delay') as mock_delay:
        mock_task_id = "task-with-params"
        mock_delay.return_value = MagicMock(id=mock_task_id)
        
        with mock_auth() as auth_manager:
            headers = {'Authorization': 'Bearer dummy'} # Header needed
            user_id = auth_manager.user_id
            response = client.post('/api/v1/ask', 
                                    headers=headers, 
                                    json={
                                        'question': 'What about this index?', 
                                        'stream': False,
                                        'pdf_id': 'pdf-for-params', # Added pdf_id
                                        'index_name': 'my-custom-index',
                                        'top_k': 5
                                    })
            
        assert response.status_code == 202 # Task submitted
        json_data = response.get_json()
        assert 'task_id' in json_data
        assert json_data['task_id'] == mock_task_id
        
        # Verify the Celery task was called with the specific parameters
        mock_delay.assert_called_once()
        call_args, call_kwargs = mock_delay.call_args
        assert call_args[0] == mock_task_id # Task ID
        expected_data = {
            'question': 'What about this index?',
            'pdf_id': 'pdf-for-params',
            'user_id': user_id,
            'stream': False,
            'index_name': 'my-custom-index',
            'top_k': 5
        }
        assert call_args[1] == expected_data # Data dict

# Add more tests below:
# - test_ask_with_index_and_topk 