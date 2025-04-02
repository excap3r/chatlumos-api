import json
from flask import url_for
from unittest.mock import patch, MagicMock

# Test Root Endpoint
def test_root_endpoint(client):
    """Test that the root endpoint returns the expected welcome message."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "message" in data
    # Corrected assertion to match the actual message
    assert data["message"] == "PDF Wisdom Extractor API Server"

# Test Health Check Endpoint
@patch('services.api.routes.health.initialize_services') # Patch the check itself
@patch('services.api.routes.health._get_service_details') # Patch detail fetching
@patch('services.api.routes.health._check_redis_status') # Patch Redis check
def test_health_endpoint(mock_redis_check, mock_details_fetch, mock_init_services, client):
    """Test the /health endpoint by mocking internal helper functions."""
    # Configure mocks to return healthy states
    mock_init_services.return_value = None # Simulate successful initialization
    mock_details_fetch.return_value = ( # Simulate successful detail fetching
        {
            'llm': {"health": {"status": "healthy"}, "capabilities": {"available_providers": ["mock_llm"], "default_provider": "mock_llm"}},
            'vector': {"health": {"status": "healthy"}, "capabilities": {"pinecone_initialized": True, "embedding_model_loaded": True}},
            'db': {"health": {"status": "healthy"}, "capabilities": {"connection_pool_active": True}}
        },
        None # No error message from detail fetching
    )
    mock_redis_check.return_value = {"status": "healthy"} # Simulate healthy Redis

    # Mock the API_VERSION config on the actual app instance
    client.application.config['API_VERSION'] = 'test-v1'

    response = client.get('/api/v1/health') # Corrected path to include /api/v1

    # Assertions for a successful health check
    assert response.status_code == 200
    data = response.json
    assert data['status'] == 'ok' # Overall endpoint status
    assert data['version'] == 'test-v1' # Check mocked version
    assert data['services']['llm']['status'] == 'healthy'
    assert data['services']['vector']['status'] == 'healthy'
    assert data['services']['db']['status'] == 'healthy'
    assert data['redis']['status'] == 'healthy'

    # Verify mocks were called
    mock_init_services.assert_called_once()
    mock_details_fetch.assert_called_once_with(client.application.api_gateway)
    mock_redis_check.assert_called_once_with(client.application.redis_client)


# Test Swagger UI Endpoint
def test_swagger_ui(client):
    """Test if the Swagger UI endpoint ('/docs') returns HTML."""
    # Corrected path to include /api/v1
    response = client.get('/api/v1/docs')
    assert response.status_code == 200
    assert response.content_type == 'text/html; charset=utf-8'
    assert b'<title>Swagger UI</title>' in response.data 