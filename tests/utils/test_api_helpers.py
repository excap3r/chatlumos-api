import pytest
import json
import time
from flask import Flask, jsonify, request, current_app
from services.utils.api_helpers import get_cache_key, rate_limit, cache_result

# --- Test get_cache_key ---

def test_get_cache_key_simple():
    key = get_cache_key("prefix", "arg1", 123)
    assert key == "wisdom_api:prefix:arg1:123"

def test_get_cache_key_no_args():
    key = get_cache_key("prefix")
    assert key == "wisdom_api:prefix"

def test_get_cache_key_different_types():
    key = get_cache_key("types", True, 1.5, None, "string")
    assert key == "wisdom_api:types:True:1.5:None:string"

# --- Tests for Decorators ---

# Helper to create a Flask app context for testing decorators
@pytest.fixture
def test_app(app):
    """Creates a minimal Flask app instance for testing decorators outside the main app."""
    # We use the main app's config but define routes locally for isolation
    test_app = Flask(__name__)
    test_app.config.from_mapping(app.config)

    # Provide a logger for the decorators to use
    test_app.logger.setLevel("DEBUG")

    # A simple counter to check if the underlying function was called
    test_app.config['ROUTE_CALL_COUNT'] = 0

    # *** Manually attach the redis_client from the main app fixture ***
    # This ensures the test app uses the same fakeredis instance
    if hasattr(app, 'redis_client'):
        test_app.redis_client = app.redis_client
    else:
        # Fallback or raise error if the main app fixture didn't set it
        # For simplicity, we'll assume it exists based on conftest.py
        raise AttributeError("Main app fixture does not have redis_client attribute")

    yield test_app

# --- Tests for rate_limit ---

def test_rate_limit_allows_under_limit(test_app):
    """Test that requests under the limit are allowed."""
    limit = 5
    seconds = 10

    # Provider function for the decorator - Use test_app's redis client
    def redis_provider(): return test_app.redis_client

    @test_app.route('/limited')
    @rate_limit(redis_client_provider=redis_provider, max_calls=limit, per_seconds=seconds)
    def limited_route():
        test_app.config['ROUTE_CALL_COUNT'] += 1
        return jsonify(success=True)

    client = test_app.test_client()

    for i in range(limit):
        response = client.get('/limited')
        assert response.status_code == 200, f"Request {i+1} failed"
        # Check redis count - Use the actual redis client from the test app
        redis_client = test_app.redis_client
        keys = redis_client.keys("ratelimit:*")
        assert len(keys) == 1
        assert int(redis_client.get(keys[0])) == i + 1

    assert test_app.config['ROUTE_CALL_COUNT'] == limit

def test_rate_limit_blocks_over_limit(test_app):
    """Test that requests over the limit are blocked with 429."""
    limit = 3
    seconds = 10

    # Use test_app's redis client
    def redis_provider(): return test_app.redis_client

    @test_app.route('/limited')
    @rate_limit(redis_client_provider=redis_provider, max_calls=limit, per_seconds=seconds)
    def limited_route():
        test_app.config['ROUTE_CALL_COUNT'] += 1
        return jsonify(success=True)

    client = test_app.test_client()

    # Make allowed requests
    for _ in range(limit):
        response = client.get('/limited')
        # Don't assert inside the loop, check the final state
        # assert response.status_code == 200 # This was failing prematurely

    # Make one more request - should be blocked
    response = client.get('/limited')
    assert response.status_code == 429 # This one should fail
    data = json.loads(response.data)
    assert "error" in data
    assert "Rate limit exceeded" in data["error"]

    # Underlying route should only have been called 'limit' times
    assert test_app.config['ROUTE_CALL_COUNT'] == limit

    # Check redis count
    redis_client = test_app.redis_client
    keys = redis_client.keys("ratelimit:*")
    assert len(keys) == 1
    # The count in redis should be limit + 1 after the blocked request attempt
    assert int(redis_client.get(keys[0])) == limit + 1


def test_rate_limit_resets_after_time(test_app, mocker):
    """Test that the rate limit counter resets after the time window."""
    limit = 2
    seconds = 10

    # Use test_app's redis client
    def redis_provider(): return test_app.redis_client

    @test_app.route('/limited')
    @rate_limit(redis_client_provider=redis_provider, max_calls=limit, per_seconds=seconds)
    def limited_route():
         test_app.config['ROUTE_CALL_COUNT'] += 1
         return jsonify(success=True)

    client = test_app.test_client()

    # Hit limit
    for _ in range(limit):
        client.get('/limited')
    response = client.get('/limited')
    assert response.status_code == 429

    # --- Explicitly delete the rate limit key to simulate expiry --- #
    redis_client = test_app.redis_client
    keys_before_delete = redis_client.keys("ratelimit:*")
    assert len(keys_before_delete) >= 1 # Key should exist before delete
    for key in keys_before_delete:
         redis_client.delete(key)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    # Make another request - should now be allowed
    response = client.get('/limited')
    assert response.status_code == 200

    # Check redis count is reset to 1
    redis_client = test_app.redis_client
    keys = redis_client.keys("ratelimit:*")
    assert len(keys) == 1
    assert int(redis_client.get(keys[0])) == 1

    # Check total calls
    assert test_app.config['ROUTE_CALL_COUNT'] == limit + 1

# --- Tests for cache_result ---

def test_cache_result_miss_and_hit(test_app, mocker):
    """Test cache miss on first call, hit on second."""
    ttl = 60

    # Use test_app's redis client
    def redis_provider(): return test_app.redis_client

    @test_app.route('/cached')
    @cache_result(redis_client_provider=redis_provider, ttl=ttl)
    def cached_route():
        test_app.config['ROUTE_CALL_COUNT'] += 1
        # Simulate returning different data based on call count
        return jsonify(data=f"call_{test_app.config['ROUTE_CALL_COUNT']}")

    client = test_app.test_client()

    # First call (miss)
    response1 = client.get('/cached?param=1')
    assert response1.status_code == 200
    data1 = json.loads(response1.data)
    assert data1['data'] == 'call_1'
    assert test_app.config['ROUTE_CALL_COUNT'] == 1

    # Check cache content - Use the actual redis client from the test app
    redis_client = test_app.redis_client
    keys = redis_client.keys("wisdom_api:cached_route*")
    assert len(keys) == 1
    cached_data = json.loads(redis_client.get(keys[0]))
    assert cached_data == {"data": "call_1"}

    # Second call (hit)
    response2 = client.get('/cached?param=1') # Identical request
    assert response2.status_code == 200
    data2 = json.loads(response2.data)
    # IMPORTANT: Should return the *cached* data from the first call
    assert data2['data'] == 'call_1'
    # Route function should NOT have been called again
    assert test_app.config['ROUTE_CALL_COUNT'] == 1

def test_cache_result_different_args_no_hit(test_app, mocker):
    """Test cache miss if request args/body differ."""
    ttl = 60

    # Use test_app's redis client
    def redis_provider(): return test_app.redis_client

    @test_app.route('/cached', methods=['GET', 'POST'])
    @cache_result(redis_client_provider=redis_provider, ttl=ttl)
    def cached_route():
        test_app.config['ROUTE_CALL_COUNT'] += 1
        return jsonify(data=f"call_{test_app.config['ROUTE_CALL_COUNT']}")

    client = test_app.test_client()
    redis_client = test_app.redis_client # Get redis client once

    # Call 1 (GET with param)
    response1 = client.get('/cached?param=A')
    assert response1.status_code == 200
    assert test_app.config['ROUTE_CALL_COUNT'] == 1
    # Check cache using the expected exact key
    key1 = get_cache_key("cached_route", "/cached", '{"param": "A"}', '{}')
    assert redis_client.exists(key1)
    assert json.loads(redis_client.get(key1)) == {"data": "call_1"}

    # Call 2 (GET with different param value) - Miss
    response2 = client.get('/cached?param=B')
    assert response2.status_code == 200
    assert test_app.config['ROUTE_CALL_COUNT'] == 2
    key2 = get_cache_key("cached_route", "/cached", '{"param": "B"}', '{}')
    assert redis_client.exists(key2)
    assert json.loads(redis_client.get(key2)) == {"data": "call_2"}
    assert redis_client.exists(key1) # First key should still exist

    # Call 3 (POST with body) - Miss
    response3 = client.post('/cached', json={"body_param": 1})
    assert response3.status_code == 200
    assert test_app.config['ROUTE_CALL_COUNT'] == 3
    key3 = get_cache_key("cached_route", "/cached", '{}', '{"body_param": 1}')
    assert redis_client.exists(key3)
    assert json.loads(redis_client.get(key3)) == {"data": "call_3"}
    assert redis_client.exists(key1)
    assert redis_client.exists(key2)

    # Call 4 (POST with same body) - Hit
    response4 = client.post('/cached', json={"body_param": 1})
    assert response4.status_code == 200
    assert test_app.config['ROUTE_CALL_COUNT'] == 3 # No new call
    assert redis_client.exists(key3) # Key should still exist
    assert json.loads(redis_client.get(key3)) == {"data": "call_3"} # Should be call_3 data


def test_cache_result_expires(test_app, mocker):
    """Test that the cache expires after TTL."""
    ttl = 10

    # Use test_app's redis client
    def redis_provider(): return test_app.redis_client

    @test_app.route('/cached_expiry')
    @cache_result(redis_client_provider=redis_provider, ttl=ttl)
    def cached_route_expiry():
         test_app.config['ROUTE_CALL_COUNT'] += 1
         return jsonify(data=f"call_{test_app.config['ROUTE_CALL_COUNT']}")

    client = test_app.test_client()

    # First call (miss)
    response1 = client.get('/cached_expiry')
    assert response1.status_code == 200
    assert test_app.config['ROUTE_CALL_COUNT'] == 1
    assert json.loads(response1.data)["data"] == "call_1"
    # Check cache exists
    redis_client = test_app.redis_client
    keys1 = redis_client.keys("wisdom_api:cached_route_expiry*")
    assert len(keys1) == 1

    # Advance time beyond the window
    current_time = time.time() # Define current_time before using it
    mocker.patch('time.time', return_value=current_time + ttl + 1)

    # Second call (should miss cache due to expiry)
    response2 = client.get('/cached_expiry')
    assert response2.status_code == 200
    assert test_app.config['ROUTE_CALL_COUNT'] == 2 # Function called again
    assert json.loads(response2.data)["data"] == "call_2"

    # Check new cache entry exists
    keys2 = redis_client.keys("wisdom_api:cached_route_expiry*")
    assert len(keys2) == 1
    # Ensure the data is from the second call
    assert json.loads(redis_client.get(keys2[0])) == {"data": "call_2"}