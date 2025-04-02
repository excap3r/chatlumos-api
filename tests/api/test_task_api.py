import pytest
import uuid
import json
from unittest.mock import patch, MagicMock
from flask import Flask, g, jsonify, current_app
from functools import wraps # Needed for mock decorator
from io import BytesIO
import time
from services.api.middleware.auth_middleware import require_auth # For direct patching in specific tests if needed

# Assuming client fixture, app fixture, and mock_auth fixture are available from conftest.py
# Assuming redis_client fixture is available

# --- Helper for Mocking require_auth Directly (REMOVED - Using Fixture) ---
# def mock_require_auth_factory(...): ...

# --- Tests for /api/pdf --- #

@patch('services.api.task_routes.process_pdf_task.delay')
def test_pdf_upload_success(mock_celery_delay, client, redis_client, mock_auth, mocker):
    """Test successful PDF upload to /api/v1/upload."""
    mock_user_id = "upload-user-123"
    file_content = b"%PDF-1.4 fake pdf content"
    file_name = "test.pdf"
    task_id_capture = None

    def capture_task_id(*args, **kwargs):
        nonlocal task_id_capture
        # Assuming the task_id is passed as a keyword argument in the actual call
        task_id_capture = kwargs.get('task_id') # Get task_id from delay call
        mock_task_result = MagicMock()
        mock_task_result.id = task_id_capture or str(uuid.uuid4()) # Use captured or generate one
        task_id_capture = mock_task_result.id # Ensure task_id_capture is set
        return mock_task_result
    mock_celery_delay.side_effect = capture_task_id

    # Use the mock_auth factory to create the context manager instance
    with mock_auth(user_id=mock_user_id):
        response = client.post(
            '/api/v1/upload',
            data={'file': (BytesIO(file_content), file_name)},
            content_type='multipart/form-data'
        )

    assert response.status_code == 202
    assert 'task_id' in response.json
    assert task_id_capture == response.json['task_id'] # Verify captured ID matches response

    # Verify Celery task was called (implicitly checked by side_effect)
    mock_celery_delay.assert_called_once()

    # Verify Redis state update (optional, but good practice)
    redis_key = f"task:{task_id_capture}"
    task_state = redis_client.hgetall(redis_key)
    # Decode keys and values if necessary (fakeredis might store bytes)
    task_state = {k.decode(): v.decode() for k, v in task_state.items()}
    assert task_state['status'] == 'PENDING'
    assert task_state['filename'] == file_name
    assert task_state['user_id'] == mock_user_id

def test_pdf_upload_no_file(client, mock_auth, mocker):
    """Test PDF upload fails when no file part is provided."""
    with mock_auth(): # Use default user from factory
        response = client.post('/api/v1/upload', data={})
    assert response.status_code == 400
    assert 'error' in response.json
    assert 'No file part' in response.json['error']

def test_pdf_upload_no_filename(client, mock_auth, mocker):
    """Test PDF upload fails when file is present but has no filename."""
    with mock_auth():
        response = client.post(
            '/api/v1/upload',
            data={'file': (BytesIO(b"content"), '')},
            content_type='multipart/form-data'
        )
    assert response.status_code == 400
    assert 'error' in response.json
    assert 'No selected file' in response.json['error']

def test_pdf_upload_wrong_type(client, mock_auth, mocker):
    """Test PDF upload fails with non-PDF file type."""
    with mock_auth():
        response = client.post(
            '/api/v1/upload',
            data={'file': (BytesIO(b"this is text"), 'test.txt')},
            content_type='multipart/form-data'
        )
    assert response.status_code == 400
    assert 'error' in response.json
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
def test_ask_endpoint_success(mock_celery_delay, client, redis_client, mock_auth, mocker):
    """Test successful question submission to /api/ask."""
    mock_user_id = "ask-user-123"
    question = "What is the main topic?"
    pdf_id = "pdf-abc-789"
    task_id_capture = None

    def capture_task_id(*args, **kwargs):
        nonlocal task_id_capture
        task_id_capture = kwargs.get('task_id')
        mock_task_result = MagicMock()
        mock_task_result.id = task_id_capture or str(uuid.uuid4())
        task_id_capture = mock_task_result.id
        return mock_task_result
    mock_celery_delay.side_effect = capture_task_id

    with mock_auth(user_id=mock_user_id):
        response = client.post(
            '/api/v1/ask',
            json={'question': question, 'pdf_id': pdf_id}
        )

    assert response.status_code == 202
    assert 'task_id' in response.json
    assert task_id_capture == response.json['task_id']

    mock_celery_delay.assert_called_once()
    # Verify args passed to delay
    called_args, called_kwargs = mock_celery_delay.call_args
    assert called_args[0] == question
    assert called_args[1] == pdf_id
    assert called_kwargs['user_id'] == mock_user_id
    assert called_kwargs['task_id'] == task_id_capture # Verify task_id passed to Celery

    # Verify Redis state
    redis_key = f"task:{task_id_capture}"
    task_state = {k.decode(): v.decode() for k, v in redis_client.hgetall(redis_key).items()}
    assert task_state['status'] == 'PENDING'
    assert task_state['question'] == question
    assert task_state['pdf_id'] == pdf_id
    assert task_state['user_id'] == mock_user_id

def test_ask_endpoint_bad_input(client, mock_auth, mocker):
    """Test /api/ask fails with missing required fields."""
    with mock_auth():
        # Missing 'question'
        response_no_q = client.post('/api/v1/ask', json={'pdf_id': 'pdf-123'})
        assert response_no_q.status_code == 400
        assert 'error' in response_no_q.json

        # Missing 'pdf_id'
        response_no_pdf = client.post('/api/v1/ask', json={'question': 'A question?'})
        assert response_no_pdf.status_code == 400
        assert 'error' in response_no_pdf.json

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
def test_progress_stream_generator(mock_redis_pubsub, client, redis_client, app, mocker):
    """Test the progress stream generator function directly."""
    session_id_or_task_id = "task-stream-123" # Use task_id as per route logic
    pubsub_channel = f"progress:{session_id_or_task_id}"
    # Mock the pubsub object returned by redis_client.pubsub()
    mock_pubsub_instance = MagicMock()
    mock_redis_pubsub.return_value = mock_pubsub_instance

    # Simulate messages published to Redis that listen() would yield
    mock_messages = [
        {'type': 'message', 'channel': pubsub_channel.encode(), 'data': json.dumps({'task_id': session_id_or_task_id, 'status': 'PROCESSING', 'progress': 50}).encode()},
        {'type': 'message', 'channel': pubsub_channel.encode(), 'data': json.dumps({'task_id': session_id_or_task_id, 'status': 'SUCCESS', 'progress': 100, 'result': {'answer': 'Final Answer'}}).encode()},
        # Add a termination message or rely on get_message timeout
        {'type': 'message', 'channel': pubsub_channel.encode(), 'data': json.dumps({'task_id': session_id_or_task_id, 'status': 'TERMINATE'}).encode()}
    ]
    get_message_return_values = [
        mock_messages[0],
        None, # Simulate a timeout
        mock_messages[1],
        mock_messages[2], # Termination message
        None, # Simulate timeout after termination
    ]
    mock_pubsub_instance.get_message.side_effect = get_message_return_values + [None]*10 # Return None after messages

    # Import the function we want to test directly
    from services.api.routes.progress import stream_progress

    # Simulate initial task state in Redis hash (required by the route)
    initial_state = {
        'user_id': 'test-user', # Assuming auth adds this to g
        'status': 'PROCESSING',
        'progress': 10,
        'filename': 'initial_file.pdf'
    }
    redis_client.hset(f"task:{session_id_or_task_id}", mapping=initial_state)

    # Need app context for current_app.logger and request context for g
    # Patch require_auth specifically for this direct call context
    mock_user_data = {'id': 'test-user', 'username': 'direct-user'}

    @wraps(require_auth)
    def mock_decorator_factory(*factory_args, **factory_kwargs):
        def mock_decorator_inner(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                g.user = mock_user_data
                g.auth_method = 'mocked_direct'
                return f(*args, **kwargs)
            return decorated_function
        return mock_decorator_inner

    with app.test_request_context(f'/api/v1/progress/{session_id_or_task_id}'):
        # Patch the decorator where it's likely used by stream_progress or its calling context
        # (Assuming stream_progress itself isn't decorated, but called by a decorated route)
        # If stream_progress IS decorated, target should be 'services.api.routes.progress.require_auth'
        # For now, assume it's used in the middleware layer which is active during request context
        patcher = mocker.patch('services.api.middleware.auth_middleware.require_auth', new=mock_decorator_factory)
        patcher.start()
        try:
            # g.user = mock_user_data # No longer needed if patch works
            generator = stream_progress(session_id_or_task_id) # Pass task_id
            results = list(generator)
        finally:
            patcher.stop()

    # --- Assertions --- #
    # Expecting initial state + 2 updates (PROCESSING, SUCCESS)
    # Termination message shouldn't yield data
    # Initial state (dict), processing update (str), success update (str)
    assert len(results) == 3
    # Check types and content
    assert isinstance(results[0], dict) # Initial state
    assert results[0]['status'] == 'PROCESSING'
    assert results[0]['progress'] == 10

    assert isinstance(results[1], str) # First update (JSON string)
    update1 = json.loads(results[1].split('data: ')[1])
    assert update1['status'] == 'PROCESSING'
    assert update1['progress'] == 50

    assert isinstance(results[2], str) # Second update (JSON string)
    update2 = json.loads(results[2].split('data: ')[1])
    assert update2['status'] == 'SUCCESS'
    assert update2['progress'] == 100
    assert 'result' in update2
    assert update2['result']['answer'] == 'Final Answer'

    # Verify Redis calls
    mock_redis_pubsub.assert_called_once_with(ignore_subscribe_messages=True)
    mock_pubsub_instance.subscribe.assert_called_once_with(pubsub_channel)
    # Check get_message calls (should match side_effect list length + extras)
    assert mock_pubsub_instance.get_message.call_count >= len(get_message_return_values)
    mock_pubsub_instance.unsubscribe.assert_called_once_with(pubsub_channel)
    mock_pubsub_instance.close.assert_called_once()

# Actual endpoint test is harder - might need a specialized test client
@pytest.mark.skip(reason="Standard Flask test client cannot easily test SSE endpoints")
def test_progress_endpoint_sse_success(client):
    """Test the SSE endpoint /api/progress-stream/<session_id> (Skipped)."""
    session_id = "live-session-123"
    pass 