import pytest
from flask import Flask
from unittest.mock import patch, MagicMock
import json
from services.utils.error_utils import APIError

# Assuming fixtures like 'client', 'auth_headers' are defined in conftest.py

def test_ask_non_streaming_success(client, auth_headers):
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

def test_ask_streaming_success(client, auth_headers):
    """Test successful /ask request with stream=true."""
    # Mock the backend call that ask_route makes
    # In streaming mode, it should initiate a task and return a task ID
    mock_task_response = {
        "task_id": "test-task-123",
        "status": "Processing started"
    }
    
    # Assuming the same internal function is called, but it behaves differently for stream=True
    with patch('services.api.routes.ask.ask_question_internal', return_value=mock_task_response) as mock_internal_ask:
        response = client.post('/api/v1/ask', 
                                headers=auth_headers, 
                                json={'question': 'What is the streaming test question?', 'stream': True})
        
        assert response.status_code == 202 # Accepted for async task
        json_data = response.get_json()
        assert 'task_id' in json_data
        assert json_data['task_id'] == mock_task_response['task_id']
        assert json_data['status'] == mock_task_response['status']
        
        # Verify the internal function was called correctly with stream=True
        mock_internal_ask.assert_called_once_with(
            question='What is the streaming test question?',
            index_name=None, # Assuming default
            top_k=None,      # Assuming default
            stream=True,
            user_info=pytest.approx({}) # Placeholder - adjust based on actual user info passed
        )

def test_ask_missing_question(client, auth_headers):
    """Test /ask request with missing 'question' field."""
    response = client.post('/api/v1/ask', 
                            headers=auth_headers, 
                            json={'stream': False}) # Missing 'question'
    
    assert response.status_code == 400 # Bad Request
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'message' in json_data['error']
    assert 'Missing required field: question' in json_data['error']['message']

def test_ask_unauthorized(client):
    """Test /ask request without authentication."""
    response = client.post('/api/v1/ask', 
                            json={'question': 'This should fail', 'stream': False})
    
    assert response.status_code == 401 # Unauthorized
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'message' in json_data['error']
    # The exact message might vary based on the auth middleware
    assert 'Missing or invalid credentials' in json_data['error']['message']

def test_ask_internal_service_error(client, auth_headers):
    """Test /ask handles errors raised by the internal ask function."""
    # Mock the internal function to raise an error
    error_message = "Internal LLM service exploded"
    with patch('services.api.routes.ask.ask_question_internal', side_effect=APIError(error_message, status_code=503)) as mock_internal_ask:
        response = client.post('/api/v1/ask', 
                                headers=auth_headers, 
                                json={'question': 'Trigger an error', 'stream': False})
        
        assert response.status_code == 503 # Service Unavailable (or appropriate code from raised error)
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'message' in json_data['error']
        assert error_message in json_data['error']['message']
        
        # Verify the internal function was called
        mock_internal_ask.assert_called_once()

def test_ask_with_index_and_topk(client, auth_headers):
    """Test /ask request with optional index_name and top_k parameters."""
    mock_llm_response = {
        "answer": "Specific answer from custom index.",
        "context": [{"text": "Context from my-index"}]
    }
    
    with patch('services.api.routes.ask.ask_question_internal', return_value=mock_llm_response) as mock_internal_ask:
        response = client.post('/api/v1/ask', 
                                headers=auth_headers, 
                                json={
                                    'question': 'What about this index?', 
                                    'stream': False,
                                    'index_name': 'my-custom-index',
                                    'top_k': 5
                                })
        
        assert response.status_code == 200
        json_data = response.get_json()
        assert 'answer' in json_data
        assert json_data['answer'] == mock_llm_response['answer']
        
        # Verify the internal function was called with the specific parameters
        mock_internal_ask.assert_called_once_with(
            question='What about this index?',
            index_name='my-custom-index', 
            top_k=5,
            stream=False,
            user_info=pytest.approx({}) # Placeholder - adjust based on actual user info passed
        )

# Add more tests below:
# - test_ask_with_index_and_topk 