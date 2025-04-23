import pytest
import json
from unittest.mock import patch, MagicMock, ANY

from services.utils.error_utils import ValidationError, APIError
from services.api_gateway import ServiceError # For gateway error test
from services.config import AppConfig # Import AppConfig

# Assume client, mock_auth fixtures are available from conftest

API_BASE_URL = "/api/v1"
MOCK_USER_ID = "test-search-user-1"

def test_search_success(client, mock_auth):
    """Test successful search via POST /search"""
    search_query = "find relevant documents"
    mock_vector_results = {
        "results": [
            {"id": "doc1", "score": 0.9, "metadata": {"text": "Result 1 content..."}},
            {"id": "doc2", "score": 0.8, "metadata": {"text": "Result 2 content..."}}
        ]
    }

    # Patch the api_gateway.request method on the current_app proxy
    with patch('flask.current_app.api_gateway.request', return_value=mock_vector_results) as mock_gw_request, \
         mock_auth(user_id=MOCK_USER_ID): # Use mock_auth fixture

        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {
            "query": search_query,
            "top_k": 5 # Optional param
        }

        response = client.post(
            f'{API_BASE_URL}/search',
            headers=headers,
            data=json.dumps(payload)
        )

    # Assertions
    assert response.status_code == 200
    assert response.json == mock_vector_results

    # Verify the gateway request was called correctly
    assert mock_gw_request.call_count == 1
    call_args, call_kwargs = mock_gw_request.call_args
    
    # Check positional arguments
    assert len(call_args) >= 2
    assert call_args[0] == "vector" # service
    assert call_args[1] == "/search"  # path
    
    # Check keyword arguments
    assert call_kwargs.get('method') == "POST"
    assert isinstance(call_kwargs.get('json'), dict)
    assert call_kwargs['json'].get('query') == search_query
    assert call_kwargs['json'].get('index_name') == AppConfig.ANNOY_INDEX_PATH
    assert call_kwargs['json'].get('top_k') == 5
    assert 'metadata_filter' not in call_kwargs['json']

def test_search_missing_query(client, mock_auth):
    """Test search fails with 400 if 'query' field is missing."""
    with mock_auth(user_id=MOCK_USER_ID): # Need auth context
        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {
            # "query": "missing",
            "top_k": 5
        }
        response = client.post(
            f'{API_BASE_URL}/search',
            headers=headers,
            data=json.dumps(payload)
        )

    assert response.status_code == 400
    assert 'error' in response.json
    assert response.json['error'] == "'query' field is required and must be a string"

@pytest.mark.parametrize("invalid_top_k", ["not_an_int", 0, -5, 10.5])
def test_search_invalid_top_k(client, mock_auth, invalid_top_k):
    """Test search fails with 400 for invalid 'top_k' values."""
    with mock_auth(user_id=MOCK_USER_ID):
        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {
            "query": "test query",
            "top_k": invalid_top_k
        }
        response = client.post(
            f'{API_BASE_URL}/search',
            headers=headers,
            data=json.dumps(payload)
        )

    assert response.status_code == 400
    assert 'error' in response.json
    # The specific error message can vary slightly depending on the validation failure
    # Check if the expected substring is present in the actual error message
    actual_error = response.json['error']
    assert "'top_k' must be" in actual_error or "'top_k' must be positive" in actual_error

def test_search_invalid_filter(client, mock_auth):
    """Test search fails with 400 if 'metadata_filter' is not a dictionary."""
    with mock_auth(user_id=MOCK_USER_ID):
        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {
            "query": "test query",
            "metadata_filter": "not_a_dictionary" # Invalid filter type
        }
        response = client.post(
            f'{API_BASE_URL}/search',
            headers=headers,
            data=json.dumps(payload)
        )

    assert response.status_code == 400
    assert 'error' in response.json
    assert response.json['error'] == "'metadata_filter' must be a dictionary (object) if provided"

def test_search_gateway_service_error(client, mock_auth):
    """Test search fails if the downstream vector service returns an error."""
    search_query = "query causing downstream error"
    mock_service_error = {
        "error": "Internal Vector DB Error",
        "message": "Something broke in the vector service.",
        "status_code": 503 # Example status code from service
    }

    with patch('flask.current_app.api_gateway.request', return_value=mock_service_error) as mock_gw_request, \
         mock_auth(user_id=MOCK_USER_ID):

        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {"query": search_query}
        response = client.post(f'{API_BASE_URL}/search', headers=headers, data=json.dumps(payload))

    assert response.status_code == 503 # Should match status_code from error payload
    assert 'error' in response.json
    assert "Search service failed" in response.json['error']
    assert mock_service_error['message'] in response.json['error']

    mock_gw_request.assert_called_once()

# Import ServiceError
from services.api_gateway import ServiceError

def test_search_gateway_unavailable(client, mock_auth):
    """Test search fails if the API gateway cannot reach the vector service."""
    search_query = "query when gateway fails"
    gateway_comm_error = ServiceError("Vector service unreachable", status_code=504) # Gateway Timeout example

    with patch('flask.current_app.api_gateway.request', side_effect=gateway_comm_error) as mock_gw_request, \
         mock_auth(user_id=MOCK_USER_ID):

        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {"query": search_query}
        response = client.post(f'{API_BASE_URL}/search', headers=headers, data=json.dumps(payload))

    assert response.status_code == 504 # Should match status_code from ServiceError
    assert 'error' in response.json
    assert "Failed to communicate with search service" in response.json['error']
    assert str(gateway_comm_error) in response.json['error']

    mock_gw_request.assert_called_once()

def test_search_unauthenticated(client):
    """Test search fails with 401 if not authenticated."""
    headers = {'Content-Type': 'application/json'}
    payload = {"query": "unauthenticated query"}
    response = client.post(
        f'{API_BASE_URL}/search',
        headers=headers, # No Authorization header
        data=json.dumps(payload)
    )

    assert response.status_code == 401
    assert 'error' in response.json
    assert response.json['error'] == 'Authentication required' # Or check specific message from auth decorator

def test_search_rate_limit(): pass # TODO (requires mocking rate limit decorator state) 