import json

# Test the root endpoint
def test_root_endpoint(client):
    """Test that the root endpoint returns the expected welcome message."""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "message" in data
    assert "Welcome to the Wisdom API" in data["message"]
    assert "version" in data

# Test the health check endpoint
def test_health_endpoint(client):
    """Test the basic health check endpoint.
    Note: This test assumes services like Redis/DB might not be available 
    in the basic test environment. More comprehensive health tests 
    might require mocking these dependencies.
    """
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "status" in data
    # In a minimal test setup, we might expect 'degraded' or 'ok' 
    # depending on mock setup. Let's just check the key exists for now.
    assert data["status"] in ["ok", "degraded", "error"] 
    assert "version" in data
    assert "services" in data
    assert isinstance(data["services"], dict) 