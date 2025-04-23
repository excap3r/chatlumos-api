import pytest
import json
from unittest.mock import patch, MagicMock

# Import ServiceError if needed for testing gateway errors
from services.api_gateway import ServiceError

# Assume client, mock_auth fixtures are available from conftest

API_BASE_URL = "/api/v1"
MOCK_USER_ID = "test-question-user-1"

def test_question_success(client, mock_auth):
    """Test successful question answering via POST /question"""
    question_text = "What is the meaning of this context?"
    provided_context = [{"text": "This is the context"}]
    mock_llm_answer = {"answer": "The context means X, Y, Z."}

    # Patch the api_gateway.request method
    with patch('flask.current_app.api_gateway.request', return_value=mock_llm_answer) as mock_gw_request, \
         mock_auth(user_id=MOCK_USER_ID):

        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {
            "question": question_text,
            "context": provided_context
            # Use default model
        }

        response = client.post(
            f'{API_BASE_URL}/question',
            headers=headers,
            data=json.dumps(payload)
        )

    # Assertions
    assert response.status_code == 200
    assert response.json == mock_llm_answer

    # Verify the gateway request was called correctly
    mock_gw_request.assert_called_once_with(
        service="llm",
        path="/answer",
        method="POST",
        json={
            "question": question_text,
            "context": provided_context,
            "model": "all-MiniLM-L6-v2" # Check default model
        }
    )

def test_question_missing_question_field(client, mock_auth):
    """Test /question fails with 400 if 'question' field is missing."""
    with mock_auth(user_id=MOCK_USER_ID):
        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {
            # "question": "missing",
            "context": [{"text": "Some context"}]
        }
        response = client.post(
            f'{API_BASE_URL}/question',
            headers=headers,
            data=json.dumps(payload)
        )

    assert response.status_code == 400
    assert 'error' in response.json
    assert response.json['error'] == 'Missing required field: question'

def test_question_llm_service_error(client, mock_auth):
    """Test /question fails if the downstream LLM service returns an error."""
    question_text = "query causing LLM error"
    mock_llm_error_response = {
        "error": "LLM Capacity Error",
        "message": "The LLM service is overloaded.",
        "status_code": 503 # Example status from LLM service
    }

    with patch('flask.current_app.api_gateway.request', return_value=mock_llm_error_response) as mock_gw_request, \
         mock_auth(user_id=MOCK_USER_ID):

        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {"question": question_text}
        response = client.post(f'{API_BASE_URL}/question', headers=headers, data=json.dumps(payload))

    assert response.status_code == 503 # Should match status_code from error payload
    assert 'error' in response.json
    assert "The LLM service is overloaded." in response.json['error']

    mock_gw_request.assert_called_once()

def test_question_gateway_error(client, mock_auth):
    """Test /question fails if the API gateway cannot reach the LLM service."""
    question_text = "query when gateway fails for LLM"
    gateway_comm_error = ServiceError("LLM service unreachable", status_code=504) # Gateway Timeout example

    with patch('flask.current_app.api_gateway.request', side_effect=gateway_comm_error) as mock_gw_request, \
         mock_auth(user_id=MOCK_USER_ID):

        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {"question": question_text}
        response = client.post(f'{API_BASE_URL}/question', headers=headers, data=json.dumps(payload))

    assert response.status_code == 504 # Should match status_code from ServiceError
    assert 'error' in response.json
    assert "LLM service unreachable" in response.json['error'] # Check for the specific message from ServiceError

    mock_gw_request.assert_called_once()

def test_question_unauthenticated(client):
    """Test /question fails with 401 if not authenticated."""
    headers = {'Content-Type': 'application/json'}
    payload = {"question": "unauthenticated query"}
    response = client.post(
        f'{API_BASE_URL}/question',
        headers=headers, # No Authorization header
        data=json.dumps(payload)
    )

    assert response.status_code == 401
    assert 'error' in response.json
    assert response.json['error'] == 'Authentication required' 