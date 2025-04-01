import pytest
import uuid
import json
from unittest.mock import patch, MagicMock
from flask import Flask, g, jsonify
from functools import wraps # Needed for mock decorator

# Assuming client fixture and app fixture are available from conftest.py

# --- Mock Celery Task ---
# Create a mock object that mimics the Celery task object
# We need to be able to mock the .delay() method
@patch('celery_app.celery_app.send_task') # Assuming celery app instance is accessible this way
def test_ask_endpoint_success(mock_send_task, client, app):
    """Test successful submission to /api/ask endpoint."""
    mock_user_id = str(uuid.uuid4())
    question_text = "What is the summary?"
    pdf_id = "test_pdf_123"
    expected_task_id = str(uuid.uuid4())
    
    # Mock the Celery send_task to simulate task enqueueing
    mock_send_task.return_value = MagicMock(id=expected_task_id)

    # Mock the Redis client (assuming it's available via current_app)
    mock_redis = MagicMock()
    app.redis_client = mock_redis # Inject mock redis into app

    # Mock the authentication decorator (assuming it sets g.user)
    with patch('services.api.middleware.auth_middleware.require_auth') as mock_require_auth:
        # Simulate the decorator setting g.user
        def mock_decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                g.user = {'id': mock_user_id}
                return f(*args, **kwargs)
            return decorated_function
        mock_require_auth.side_effect = mock_decorator

        # Make the request within the app context if needed for g
        with app.test_request_context():
             response = client.post('/api/ask', json={
                 'question': question_text,
                 'pdf_id': pdf_id
             })

    assert response.status_code == 202
    assert response.json['task_id'] == expected_task_id

    # Verify Celery task was called
    # We check send_task args. Adjust 'services.tasks.question_processing.process_question_task' if path differs.
    mock_send_task.assert_called_once_with(
        'services.tasks.question_processing.process_question_task', # Name of the task
        args=(mock_user_id, question_text, pdf_id, expected_task_id),
        # kwargs={} # Add if task uses kwargs
    )

    # Verify initial status was stored in Redis
    expected_redis_key = f"task:{expected_task_id}"
    expected_initial_data = json.dumps({
        "status": "PENDING",
        "progress": 0,
        "message": "Task received, pending execution."
    })
    mock_redis.set.assert_called_once_with(expected_redis_key, expected_initial_data, ex=3600) # Check key, value, and expiry


def test_ask_endpoint_unauthenticated(client):
    """Test /api/ask fails without authentication."""
    response = client.post('/api/ask', json={
        'question': 'test',
        'pdf_id': 'test'
    })
    assert response.status_code == 401

@patch('services.api.middleware.auth_middleware.require_auth')
def test_ask_endpoint_bad_input(mock_require_auth, client, app):
    """Test /api/ask fails with missing input fields."""
    mock_user_id = str(uuid.uuid4())
    
    # Mock auth decorator
    def mock_decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            g.user = {'id': mock_user_id}
            return f(*args, **kwargs)
        return decorated_function
    mock_require_auth.side_effect = mock_decorator
    
    with app.test_request_context():
        # Missing pdf_id
        response_missing_pdf = client.post('/api/ask', json={
            'question': 'test'
        })
        assert response_missing_pdf.status_code == 400
        assert 'Missing required field: pdf_id' in response_missing_pdf.json['error']
        
        # Missing question
        response_missing_q = client.post('/api/ask', json={
            'pdf_id': 'test'
        })
        assert response_missing_q.status_code == 400
        assert 'Missing required field: question' in response_missing_q.json['error']


# --- Tests for /api/pdf --- 

# Assuming process_pdf_task is in services.tasks.pdf_processing
@patch('celery_app.celery_app.send_task') 
def test_pdf_upload_success(mock_send_task, client, app):
    """Test successful PDF upload and task creation."""
    mock_user_id = str(uuid.uuid4())
    expected_task_id = str(uuid.uuid4())
    file_content = b"%PDF-1.4 fake pdf content"
    file_name = "test.pdf"

    # Mock Celery task
    mock_send_task.return_value = MagicMock(id=expected_task_id)

    # Mock Redis
    mock_redis = MagicMock()
    app.redis_client = mock_redis

    # Mock auth decorator
    with patch('services.api.middleware.auth_middleware.require_auth') as mock_require_auth:
        def mock_decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                g.user = {'id': mock_user_id}
                return f(*args, **kwargs)
            return decorated_function
        mock_require_auth.side_effect = mock_decorator

        # Simulate file upload using test_client
        # Need BytesIO to simulate file object
        from io import BytesIO
        data = {
            'file': (BytesIO(file_content), file_name)
        }
        
        with app.test_request_context():
            response = client.post('/api/pdf', data=data, content_type='multipart/form-data')

    assert response.status_code == 202
    assert response.json['task_id'] == expected_task_id

    # Verify Celery task call
    # The task likely receives user_id, file content, filename, task_id
    # Need to confirm exact signature of process_pdf_task
    mock_send_task.assert_called_once()
    args, kwargs = mock_send_task.call_args
    assert args[0] == 'services.tasks.pdf_processing.process_pdf_task' # Task name
    # Check positional arguments passed to the task
    task_args = args[1] 
    assert task_args[0] == mock_user_id
    # Task might receive bytes or a path to a saved temp file. Assume bytes for now.
    # If it saves temporarily, mocking file save/read might be needed.
    assert task_args[1] == file_content # Or path to saved file
    assert task_args[2] == file_name
    assert task_args[3] == expected_task_id

    # Verify Redis call
    expected_redis_key = f"task:{expected_task_id}"
    expected_initial_data = json.dumps({
        "status": "PENDING",
        "progress": 0,
        "message": "PDF received, pending processing.",
        "filename": file_name # Include filename in initial state
    })
    mock_redis.set.assert_called_once_with(expected_redis_key, expected_initial_data, ex=3600)

def test_pdf_upload_unauthenticated(client):
    """Test /api/pdf fails without authentication."""
    from io import BytesIO
    data = {'file': (BytesIO(b"test"), 'test.pdf')}
    response = client.post('/api/pdf', data=data, content_type='multipart/form-data')
    assert response.status_code == 401

@patch('services.api.middleware.auth_middleware.require_auth')
def test_pdf_upload_no_file(mock_require_auth, client, app):
    """Test /api/pdf fails when no file is provided."""
    mock_user_id = str(uuid.uuid4())
    # Mock auth decorator
    def mock_decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            g.user = {'id': mock_user_id}
            return f(*args, **kwargs)
        return decorated_function
    mock_require_auth.side_effect = mock_decorator

    with app.test_request_context():
        response = client.post('/api/pdf', data={}, content_type='multipart/form-data')
    
    assert response.status_code == 400
    assert 'No file part' in response.json['error'] # Adjust expected message

@patch('services.api.middleware.auth_middleware.require_auth')
def test_pdf_upload_no_filename(mock_require_auth, client, app):
    """Test /api/pdf fails when file has no filename."""
    mock_user_id = str(uuid.uuid4())
    # Mock auth decorator
    def mock_decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            g.user = {'id': mock_user_id}
            return f(*args, **kwargs)
        return decorated_function
    mock_require_auth.side_effect = mock_decorator

    from io import BytesIO
    data = {'file': (BytesIO(b"test"), '')} # Empty filename
    
    with app.test_request_context():
        response = client.post('/api/pdf', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 400
    assert 'No selected file' in response.json['error'] # Adjust expected message


# --- Tests for /api/progress/<task_id> --- 

# Note: Testing SSE requires careful handling of the streaming response

@patch('services.api.middleware.auth_middleware.require_auth')
@patch('redis.Redis.pubsub') # Mock the pubsub method of the Redis client
def test_progress_endpoint_sse_success(mock_redis_pubsub, mock_require_auth, client, app):
    """Test the progress SSE endpoint successfully streams updates."""
    mock_user_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    redis_channel = f"progress:{task_id}"
    
    # Mock Redis client for initial state check (if any)
    mock_redis = MagicMock()
    # Simulate initial state exists (or not, depending on route logic)
    # mock_redis.exists.return_value = True 
    app.redis_client = mock_redis

    # Mock the PubSub object and its listen() method
    mock_ps = MagicMock()
    # Simulate messages received from Redis Pub/Sub
    # Format: {'type': 'message', 'channel': b'...', 'data': b'{"status": ...}'}
    # Add a final message to break the loop or simulate connection close
    mock_messages = [
        {'type': 'pmessage', 'channel': redis_channel.encode(), 'data': json.dumps({"status": "PROCESSING", "progress": 10, "message": "Step 1"}).encode()},
        {'type': 'pmessage', 'channel': redis_channel.encode(), 'data': json.dumps({"status": "PROCESSING", "progress": 50, "message": "Step 2"}).encode()},
        {'type': 'pmessage', 'channel': redis_channel.encode(), 'data': json.dumps({"status": "SUCCESS", "progress": 100, "result": {"answer": "final"}}).encode()},
        {'type': 'pmessage', 'channel': redis_channel.encode(), 'data': b'CLOSE'} # Special message to signal end
    ]
    mock_ps.listen.return_value = iter(mock_messages)
    mock_ps.psubscribe = MagicMock()
    mock_ps.unsubscribe = MagicMock()
    mock_ps.close = MagicMock()
    mock_redis_pubsub.return_value = mock_ps

    # Mock auth decorator
    def mock_decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            g.user = {'id': mock_user_id} # Assuming progress route checks user auth
            return f(*args, **kwargs)
        return decorated_function
    mock_require_auth.side_effect = mock_decorator

    # Make the request - use stream=True
    response = client.get(f'/api/progress/{task_id}', headers={'Accept': 'text/event-stream'})
    
    assert response.status_code == 200
    assert response.mimetype == 'text/event-stream'

    # Consume the stream and check data
    # Need to decode bytes and split events properly
    stream_content = response.get_data(as_text=True)
    # Simple check for expected data payloads
    assert 'data: {"status": "PROCESSING", "progress": 10, "message": "Step 1"}\n\n' in stream_content
    assert 'data: {"status": "PROCESSING", "progress": 50, "message": "Step 2"}\n\n' in stream_content
    assert 'data: {"status": "SUCCESS", "progress": 100, "result": {"answer": "final"}}\n\n' in stream_content
    # assert 'data: CLOSE\n\n' not in stream_content # CLOSE message should terminate, not be sent
    
    # Verify pubsub interaction
    mock_redis_pubsub.assert_called_once()
    mock_ps.psubscribe.assert_called_once_with(redis_channel)
    # Check if unsubscribe/close are called depends on the route's exit logic
    # mock_ps.unsubscribe.assert_called_once_with(redis_channel)
    # mock_ps.close.assert_called_once()


def test_progress_endpoint_unauthenticated(client):
    """Test /api/progress fails without authentication."""
    task_id = str(uuid.uuid4())
    response = client.get(f'/api/progress/{task_id}', headers={'Accept': 'text/event-stream'})
    assert response.status_code == 401

# TODO: Add test for case where task_id doesn't exist initially (if route checks this)
# TODO: Add test for handling errors reported via SSE 