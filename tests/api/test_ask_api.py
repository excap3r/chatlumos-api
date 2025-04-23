import pytest
from flask import Flask
from unittest.mock import patch, MagicMock
import json
from services.utils.error_utils import APIError
import uuid

# Assuming fixtures like 'client', 'auth_headers' are defined in conftest.py

def test_ask_non_streaming_success(client, auth_headers, mocker):
    """Test successful /ask request (always async)."""
    # Mock the Celery task delay method
    mock_delay = mocker.patch('services.tasks.question_processing.process_question_task.delay')
    
    test_question = "What is the meaning of life?"
    request_data = {"question": test_question}

    # Mock uuid.uuid4 to control the task_id
    test_task_id = str(uuid.uuid4())
    mocker.patch('uuid.uuid4', return_value=test_task_id)

    response = client.post(
        '/api/v1/ask',
        headers=auth_headers,
        json=request_data
    )

    assert response.status_code == 202
    response_data = response.get_json()
    assert response_data['task_id'] == test_task_id
    assert response_data['status'] == "Processing started"

    # Verify that the Celery task was called correctly
    # The task expects task_id and a data dict
    expected_task_data = {
        'question': test_question,
        'pdf_id': None, # Assuming pdf_id wasn't sent
        'user_id': mocker.ANY # We can check g.user was set if needed, but ANY is simpler here
    }
    # Need to check the user_id passed matches the one from auth_headers context
    # This requires a bit more work, potentially accessing the mock_auth context 
    # or inspecting g within the test. For now, checking it was called is a good start.
    
    # Get the user_id from the auth_headers fixture if possible
    # (This requires the fixture to provide it, or decode the token)
    # For simplicity, we'll assert it was called with *some* data containing the question
    # A more robust test would verify the exact user_id.
    
    # Assert delay was called once
    mock_delay.assert_called_once()
    
    # Get the arguments passed to delay
    args, kwargs = mock_delay.call_args
    
    # Check the task_id argument
    assert args[0] == test_task_id
    
    # Check the task_data argument (second positional argument)
    passed_task_data = args[1]
    assert passed_task_data['question'] == test_question
    assert 'user_id' in passed_task_data # Check user_id is present
    # assert passed_task_data['user_id'] == expected_user_id # Add this if you extract expected_user_id


def test_ask_missing_question(client, auth_headers):
    """Test /ask request with missing question parameter."""
    response = client.post(
        '/api/v1/ask',
        headers=auth_headers,
        json={}
    )
    assert response.status_code == 400
    response_data = response.get_json()
    assert "Missing 'question' parameter" in response_data['error']

def test_ask_not_json(client, auth_headers):
    """Test /ask request with non-JSON data."""
    response = client.post(
        '/api/v1/ask',
        headers=auth_headers,
        data="this is not json"
    )
    assert response.status_code == 400
    response_data = response.get_json()
    assert "Request must be JSON" in response_data['error']

def test_ask_rate_limit(client, auth_headers, redis_client, mocker):
    """Test rate limiting on the /ask endpoint."""
    # Mock the Celery task delay method to prevent connection errors
    mock_delay = mocker.patch('services.tasks.question_processing.process_question_task.delay')

    # This test requires careful setup of rate limit config and fakeredis
    # Assuming rate limit is 10 calls per 60 seconds (from decorator)
    max_calls = 10
    question = "Rate limit test?"

    # Clear potential existing rate limit keys
    # The key format depends on the rate_limit implementation
    # For now, assume we can test by making calls rapidly

    for i in range(max_calls):
        response = client.post('/api/v1/ask', headers=auth_headers, json={"question": f"{question} {i}"})
        # First 10 calls should be 202 (accepted for async processing)
        assert response.status_code == 202, f"Call {i+1} failed, expected 202"

    # The (max_calls + 1)th call should be rate limited
    response = client.post('/api/v1/ask', headers=auth_headers, json={"question": f"{question} {max_calls}"})
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.get_json().get("error", "")

# Add test for unauthorized access if not covered elsewhere
def test_ask_unauthorized(client):
    """Test /ask without authentication."""
    response = client.post('/api/v1/ask', json={"question": "??"})
    assert response.status_code == 401 # Expect Unauthorized

# Add more tests for different scenarios, error conditions in task, etc.

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
        'stream': True, # Ensure stream flag is passed
        'index_name': None, # Default value when not provided
        'top_k': None      # Default value when not provided
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

def test_ask_with_index_and_topk(client, mock_auth, mocker, redis_client):
    """Test /ask request with optional index_name and top_k parameters."""
    # Patch the celery task delay
    mock_delay = mocker.patch('services.api.routes.ask.process_question_task.delay')
    # mock_task_id = "task-with-params" # No longer need to mock return ID from delay
    # mock_delay.return_value = MagicMock(id=mock_task_id)

    # Mock uuid.uuid4 where it's called in the route handler
    test_task_id = "task-with-params-uuid"
    mocker.patch('services.api.routes.ask.uuid.uuid4', return_value=test_task_id)

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
    assert json_data['task_id'] == test_task_id
    
    # Verify the Celery task was called with the specific parameters
    mock_delay.assert_called_once()
    call_args, call_kwargs = mock_delay.call_args
    assert call_args[0] == test_task_id # Task ID
    expected_data = {
        'question': 'What about this index?',
        'pdf_id': 'pdf-for-params',
        'user_id': user_id,
        'stream': False,
        'index_name': 'my-custom-index',
        'top_k': 5
    }
    assert call_args[1] == expected_data # Data dict

    # Check Redis state for the queued task
    redis_key = f"task:{test_task_id}"
    task_state = redis_client.hgetall(redis_key)
    assert task_state['status'] == 'Queued'
    assert task_state['user_id'] == user_id
    assert task_state['question'] == 'What about this index?'
    assert task_state['pdf_id'] == 'pdf-for-params'
    assert task_state['stream'] == 'False' # Check stream flag in Redis (likely stored as string)

# Add more tests below:
# - test_ask_with_index_and_topk 