import pytest
import uuid
import json
from unittest.mock import patch, MagicMock
from flask import Flask, g, jsonify
from functools import wraps # Needed for mock decorator
from io import BytesIO
import time

# Assuming client fixture, app fixture, and mock_auth fixture are available from conftest.py
# Assuming redis_client fixture is available

# --- Helper for Mocking require_auth Directly (REMOVED - Using Fixture) ---
# def mock_require_auth_factory(...): ...

# --- Tests for /api/pdf --- #

@patch('services.api.routes.pdf.process_pdf_task.delay') # Patch Celery task delay
def test_pdf_upload_success(mock_celery_delay, client, redis_client, mock_auth): # Inject mock_auth
    """Test successful PDF upload and task queuing."""
    mock_user_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 fake pdf"
    file_name = "test_upload.pdf"
    task_id_capture = None # To capture the generated task_id

    # Mock Celery delay to capture args and prevent actual execution
    def capture_task_id(*args, **kwargs):
        nonlocal task_id_capture
        task_id_capture = kwargs.get('task_id')
        return MagicMock() # Return a mock AsyncResult
    mock_celery_delay.side_effect = capture_task_id

    # Use the mock_auth fixture to simulate authentication
    auth_patcher = mock_auth(user_id=mock_user_id)
    with auth_patcher: # Apply patch context
        data = {
            'file': (BytesIO(file_content), file_name),
            'author': 'Test Author',
            'title': 'Test Title'
        }
        response = client.post('/api/v1/pdf/upload', data=data, content_type='multipart/form-data')

        # --- Assert API Response --- 
        assert response.status_code == 202
        assert 'task_id' in response.json
        assert response.json['task_id'] == task_id_capture
        assert response.json['filename'] == file_name
        assert task_id_capture is not None

        # --- Assert Celery Task Called --- 
        mock_celery_delay.assert_called_once()
        call_args, call_kwargs = mock_celery_delay.call_args
        assert call_kwargs['task_id'] == task_id_capture
        assert call_kwargs['filename'] == file_name
        assert call_kwargs['user_id'] == mock_user_id
        assert call_kwargs['author_name'] == 'Test Author'
        assert call_kwargs['title'] == 'Test Title'
        assert 'file_content_b64' in call_kwargs

        # --- Assert Initial State in Redis --- 
        redis_key = f"task:{task_id_capture}"
        task_state = redis_client.hgetall(redis_key)
        assert task_state is not None
        assert task_state.get('status') == 'Queued'
        assert int(task_state.get('progress', -1)) == 0
        assert task_state.get('filename') == file_name
        assert task_state.get('user_id') == mock_user_id

def test_pdf_upload_no_file(client, mock_auth): # Inject mock_auth
    """Test PDF upload fails when no file part is provided."""
    auth_patcher = mock_auth(user_id="test-user")
    with auth_patcher: # Apply patch context
        response = client.post('/api/v1/pdf/upload', data={}, content_type='multipart/form-data')
        assert response.status_code == 400
        assert 'No file part' in response.json['error']

def test_pdf_upload_no_filename(client, mock_auth): # Inject mock_auth
    """Test PDF upload fails when file is present but has no filename."""
    auth_patcher = mock_auth(user_id="test-user")
    with auth_patcher: # Apply patch context
        data = {'file': (BytesIO(b"content"), '')} # Empty filename
        response = client.post('/api/v1/pdf/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        assert 'No file selected' in response.json['error']

def test_pdf_upload_wrong_type(client, mock_auth): # Inject mock_auth
    """Test PDF upload fails with non-PDF file type."""
    auth_patcher = mock_auth(user_id="test-user")
    with auth_patcher: # Apply patch context
        data = {'file': (BytesIO(b"content"), 'document.txt')} # Wrong extension
        response = client.post('/api/v1/pdf/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
        assert 'Invalid file type' in response.json['error']

def test_pdf_upload_unauthenticated(client):
    """Test /api/v1/pdf/upload fails without authentication."""
    # No mock_auth needed here
    data = {'file': (BytesIO(b"test"), 'test.pdf')}
    response = client.post('/api/v1/pdf/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 401
    assert 'Authentication required' in response.json['error']


# --- Tests for /api/ask --- #

@patch('services.api.routes.ask.process_question_task.delay')
def test_ask_endpoint_success(mock_celery_delay, client, redis_client, mock_auth): # Inject mock_auth
    """Test successful question submission to /api/ask."""
    mock_user_id = "ask-user-123"
    question = "What is the main topic?"
    pdf_id = "pdf-abc-789"
    task_id_capture = None

    def capture_task_id(*args, **kwargs):
        nonlocal task_id_capture
        task_id_capture = kwargs.get('task_id')
        return MagicMock()
    mock_celery_delay.side_effect = capture_task_id

    auth_patcher = mock_auth(user_id=mock_user_id)
    with auth_patcher: # Apply patch context
        response = client.post('/api/v1/ask', json={
            'question': question,
            'pdf_id': pdf_id,
            # 'stream': False # Assuming default is non-streaming
        })

        # --- Assert API Response --- 
        assert response.status_code == 202
        assert 'task_id' in response.json
        assert response.json['task_id'] == task_id_capture
        assert task_id_capture is not None

        # --- Assert Celery Task Called --- 
        mock_celery_delay.assert_called_once()
        call_args, call_kwargs = mock_celery_delay.call_args
        assert call_kwargs['task_id'] == task_id_capture
        assert call_kwargs['question'] == question
        assert call_kwargs['pdf_id'] == pdf_id
        assert call_kwargs['user_id'] == mock_user_id
        # Assert stream is False or default value if applicable

        # --- Assert Initial State in Redis --- 
        redis_key = f"task:{task_id_capture}"
        task_state = redis_client.hgetall(redis_key)
        assert task_state is not None
        assert task_state.get('status') == 'Queued'
        # Add more checks for initial state if needed (e.g., question stored)

def test_ask_endpoint_bad_input(client, mock_auth): # Inject mock_auth
    """Test /api/ask fails with missing required fields."""
    auth_patcher = mock_auth(user_id="test-user")
    with auth_patcher: # Apply patch context
        # Missing pdf_id
        response = client.post('/api/v1/ask', json={'question': 'test?'})
        assert response.status_code == 400
        assert 'Input validation failed' in response.json['error']
        assert 'pdf_id' in response.json['details'] # Check Pydantic error detail

        # Missing question
        response = client.post('/api/v1/ask', json={'pdf_id': 'abc'})
        assert response.status_code == 400
        assert 'Input validation failed' in response.json['error']
        assert 'question' in response.json['details']

def test_ask_unauthorized(client):
    """Test /api/ask fails without authentication."""
    # No mock_auth needed
    response = client.post('/api/v1/ask', json={'question': 'test?', 'pdf_id': 'abc'})
    assert response.status_code == 401
    assert 'Authentication required' in response.json['error']

# --- Tests for /api/progress-stream/<session_id> --- #

# Note: Testing SSE is tricky with the standard client.
# Requires a client that can handle streaming responses or direct function calls.

# Example using direct call (less integration-like, more unit-like for the generator)
# Patch the pubsub method on the Redis client class used by the app
@patch('fakeredis.FakeStrictRedis.pubsub') # Patching the pubsub method on the fake client
def test_progress_stream_generator(mock_redis_pubsub, client, redis_client, app):
    """Test the progress stream generator function directly."""
    session_id_or_task_id = "task-stream-123" # Use task_id as per route logic
    pubsub_channel = f"progress:{session_id_or_task_id}"
    # Mock the pubsub object returned by redis_client.pubsub()
    mock_pubsub_instance = MagicMock()
    mock_redis_pubsub.return_value = mock_pubsub_instance
    
    # Simulate messages published to Redis that listen() would yield
    mock_messages = [
        {'type': 'message', 'channel': pubsub_channel.encode(), 'data': json.dumps({'task_id': session_id_or_task_id, 'status': 'PROCESSING', 'progress': 50}).encode()},
        {'type': 'message', 'channel': pubsub_channel.encode(), 'data': json.dumps({'task_id': session_id_or_task_id, 'status': 'SUCCESS', 'progress': 100}).encode()},
        # Simulate timeout or end condition - the generator should break
    ]
    # mock_pubsub_instance.listen.return_value = iter(mock_messages)
    # Instead of listen(), the route uses get_message() in a loop.
    # We need to simulate multiple calls to get_message()
    # Add None to simulate timeout between real messages, and finally return None indefinitely or raise exception to stop
    get_message_return_values = [
        mock_messages[0],
        None, # Simulate a timeout
        mock_messages[1],
        None, # Simulate timeout after last message before generator stops
    ]
    mock_pubsub_instance.get_message.side_effect = get_message_return_values + [None]*10 # Return None after messages

    # Import the function we want to test directly
    from services.api.routes.progress import stream_progress # Function name changed
    
    # Simulate initial task state in Redis hash (required by the route)
    initial_state = {
        'user_id': 'test-user', # Assuming auth adds this to g
        'status': 'PROCESSING',
        'progress': 10,
        'filename': 'initial_file.pdf'
    }
    redis_client.hset(f"task:{session_id_or_task_id}", mapping=initial_state)

    # Need app context for current_app.logger and request context for g
    with app.test_request_context(f'/api/v1/progress/{session_id_or_task_id}'):
        g.user = {'id': 'test-user'} # Manually set g.user for the direct call test
        generator = stream_progress(session_id_or_task_id) # Pass task_id
        
        results = list(generator)

    # --- Assertions --- #
    # Expecting initial state + 2 updates
    assert len(results) == 3 
    # Check initial state event (might have different event name)
    assert results[0].startswith("event: processing\ndata: {") # Based on initial state status
    assert '"progress": 10' in results[0] 
    # Check first update
    assert results[1].startswith("event: processing\ndata: {")
    assert '"progress": 50' in results[1]
    # Check second update
    assert results[2].startswith("event: success\ndata: {")
    assert '"progress": 100' in results[2]
    
    # Verify pubsub interaction
    mock_redis_pubsub.assert_called_once_with(ignore_subscribe_messages=True)
    mock_pubsub_instance.subscribe.assert_called_once_with(pubsub_channel)
    assert mock_pubsub_instance.get_message.call_count >= 4 # Called until side_effect list exhausted
    mock_pubsub_instance.unsubscribe.assert_called_once_with(pubsub_channel)
    mock_pubsub_instance.close.assert_called_once()

# Actual endpoint test is harder - might need a specialized test client
@pytest.mark.skip(reason="Standard Flask test client cannot easily test SSE endpoints")
def test_progress_endpoint_sse_success(client):
    """Test the SSE endpoint /api/progress-stream/<session_id> (Skipped)."""
    session_id = "live-session-123"
    pass 