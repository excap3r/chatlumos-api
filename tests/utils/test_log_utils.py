import pytest
import logging
import os
import shutil
from unittest.mock import patch, MagicMock
import json
import structlog

from services.utils.log_utils import setup_logger, log_request, log_response

# TODO: Add tests for setup_logger
# TODO: Add tests for log_request (requires Flask app context)
# TODO: Add tests for log_response (requires Flask app context)

# --- Tests for setup_logger ---

def test_setup_logger_console_only():
    """Test creating a logger with only console output."""
    logger_name = "test_console_logger"
    logger = setup_logger(logger_name, level=logging.DEBUG)

    # Check that we got a logger-like object (handles proxy)
    assert hasattr(logger, 'info') and callable(logger.info), "Logger does not have an info method"

    # Test logging
    logger.info("Test message", test_key="test_value")

    # Since structlog is configured differently than standard logging,
    # we mainly want to verify that the logger was created successfully
    # and can be used to log messages without errors


def test_setup_logger_with_file(tmp_path):
    """Test creating a logger with file output."""
    logger_name = "test_file_logger"
    log_dir = tmp_path / "logs"
    log_file = log_dir / "test.log"

    # Ensure directory doesn't exist initially
    assert not log_dir.exists()

    logger = setup_logger(logger_name, log_file=str(log_file), level=logging.INFO)

    # Check that we got a logger-like object (handles proxy)
    assert hasattr(logger, 'info') and callable(logger.info), "Logger does not have an info method"

    # Verify directory was created
    assert log_dir.exists()
    assert log_dir.is_dir()

    # Test logging
    test_message = "Test log message"
    test_data = {"key": "value"}
    logger.info(test_message, **test_data)

    # Verify log file was created and contains the message
    assert log_file.exists()
    assert log_file.is_file()

    # Read the log file and verify content
    log_content = log_file.read_text()
    assert test_message in log_content
    assert "key" in log_content
    assert "value" in log_content


def test_setup_logger_existing_dir(tmp_path):
    """Test creating a logger when the log directory already exists."""
    logger_name = "test_existing_dir_logger"
    log_dir = tmp_path / "existing_logs"
    log_file = log_dir / "test.log"

    # Create directory beforehand
    log_dir.mkdir()
    assert log_dir.exists()

    logger = setup_logger(logger_name, log_file=str(log_file))

    # Check that we got a logger-like object (handles proxy)
    assert hasattr(logger, 'info') and callable(logger.info), "Logger does not have an info method"

    # Test logging
    test_message = "Test log message"
    test_data = {"key": "value"}
    logger.info(test_message, **test_data)

    # Verify log file was created and contains the message
    assert log_file.exists()
    assert log_file.is_file()

    # Read the log file and verify content
    log_content = log_file.read_text()
    assert test_message in log_content
    assert "key" in log_content
    assert "value" in log_content

# TODO: Add tests for log_request (requires Flask app context)
# TODO: Add tests for log_response (requires Flask app context)

# --- Tests for log_request ---

from flask import Flask, request, g, jsonify

# Remove the local app fixture to use the one from conftest.py
# @pytest.fixture
# def app():
#     """Create a minimal Flask app for context."""
#     app = Flask(__name__)
#     app.config['TESTING'] = True
#     return app

@patch('services.utils.log_utils.time.time')
@patch('structlog.get_logger')
def test_log_request_get(mock_logger_factory, mock_time, app):
    """Test logging a basic GET request."""
    mock_time.return_value = 12345.6789
    mock_logger = MagicMock()
    mock_logger_factory.return_value = mock_logger

    with app.test_request_context('/test?param1=val1', method='GET', base_url="http://localhost"):
        # Simulate request_id being set by middleware
        g.request_id = "test-req-id-123"

        log_request(request)

        assert hasattr(g, 'request_start_time')
        assert g.request_start_time == 12345.6789

        # Verify the logger was called with the correct structured data
        mock_logger.info.assert_called_once()
        call_args, call_kwargs = mock_logger.info.call_args
        assert call_args[0] == 'Incoming request'
        log_data = call_kwargs

        assert log_data['method'] == 'GET'
        assert log_data['url'] == 'http://localhost/test?param1=val1'
        assert log_data['path'] == '/test'
        assert log_data['request_id'] == 'test-req-id-123'
        assert 'headers' in log_data
        assert 'args' in log_data
        assert log_data['args'] == {'param1': 'val1'}

@patch('services.utils.log_utils.time.time')
@patch('structlog.get_logger')
def test_log_request_post_json(mock_logger_factory, mock_time, app):
    """Test logging a POST request with JSON data and Auth header."""
    mock_time.return_value = 12345.0
    mock_logger = MagicMock()
    mock_logger_factory.return_value = mock_logger

    request_data = {"key": "value", "nested": {"num": 1}}
    auth_header = "Bearer some_secret_token"

    with app.test_request_context('/submit', method='POST', json=request_data,
                                headers={'Authorization': auth_header, 'Content-Type': 'application/json'},
                                base_url="http://localhost"):
        g.request_id = "test-req-id-456"
        log_request(request)

        assert hasattr(g, 'request_start_time')
        assert g.request_start_time == 12345.0

        # Verify the logger was called with the correct structured data
        mock_logger.info.assert_called_once()
        call_args, call_kwargs = mock_logger.info.call_args
        assert call_args[0] == 'Incoming request'
        log_data = call_kwargs

        assert log_data['method'] == 'POST'
        assert log_data['url'] == 'http://localhost/submit'
        assert log_data['path'] == '/submit'
        assert log_data['request_id'] == 'test-req-id-456'
        assert 'headers' in log_data
        assert 'Authorization' not in log_data['headers']  # Sensitive header should be excluded
        assert log_data['headers']['Content-Type'] == 'application/json'
        assert log_data['body'] == request_data

@patch('services.utils.log_utils.time.time')
@patch('structlog.get_logger')
def test_log_request_invalid_json(mock_logger_factory, mock_time, app):
    """Test logging a request with invalid JSON data."""
    mock_time.return_value = 12345.0
    mock_logger = MagicMock()
    mock_logger_factory.return_value = mock_logger

    invalid_json_bytes = b'{ "key": "value", '  # Invalid JSON

    with app.test_request_context('/invalid', method='POST',
                                data=invalid_json_bytes,
                                content_type='application/json',
                                base_url="http://localhost"):
        g.request_id = "test-req-id-789"
        log_request(request)

        # Verify the logger was called with the correct structured data
        mock_logger.info.assert_called_once()
        call_args, call_kwargs = mock_logger.info.call_args
        assert call_args[0] == 'Incoming request'
        log_data = call_kwargs

        assert log_data['method'] == 'POST'
        assert log_data['url'] == 'http://localhost/invalid'
        assert log_data['path'] == '/invalid'
        assert log_data['request_id'] == 'test-req-id-789'
        assert 'headers' in log_data
        assert log_data['headers']['Content-Type'] == 'application/json'
        assert log_data['body'] == '<Invalid JSON>'  # Invalid JSON should be marked as such


# --- Tests for log_response ---

@patch('services.utils.log_utils.time.time')
@patch('structlog.get_logger')
def test_log_response_json(mock_logger_factory, mock_time, app):
    """Test logging a JSON response."""
    test_logger = logging.getLogger("test_res_log_json")
    test_logger.propagate = False
    start_time = 12345.0
    end_time = 12345.5
    response_data = {"result": "success", "count": 42}

    mock_logger = MagicMock()
    mock_logger_factory.return_value = mock_logger
    mock_time.return_value = end_time

    with app.test_request_context('/test'):  # Context needed for g
        g.request_start_time = start_time
        g.request_id = "res-req-id-111"

        # Create a Flask response object
        response = jsonify(response_data)
        response.status_code = 200
        response.headers['X-Custom-Header'] = 'TestValue'

        log_response(response)

        # Verify the logger was called with the correct structured data
        mock_logger.info.assert_called_once()
        call_args, call_kwargs = mock_logger.info.call_args
        assert call_args[0] == 'Outgoing response'
        log_data = call_kwargs

        assert log_data['status_code'] == 200
        assert log_data['duration_ms'] == pytest.approx(500.0)
        assert log_data['request_id'] == 'res-req-id-111'
        assert 'headers' in log_data
        assert log_data['headers']['Content-Type'] == 'application/json'
        assert log_data['headers']['X-Custom-Header'] == 'TestValue'
        assert log_data['body'] == response_data

@patch('services.utils.log_utils.time.time')
@patch('structlog.get_logger')
def test_log_response_non_json(mock_logger_factory, mock_time, app):
    """Test logging a non-JSON response."""
    start_time = 12300.0
    end_time = 12300.2
    response_body = "<h1>Hello</h1>"

    mock_logger = MagicMock()
    mock_logger_factory.return_value = mock_logger
    mock_time.return_value = end_time

    with app.test_request_context('/test_html'):
        g.request_start_time = start_time
        g.request_id = "res-req-id-222"

        response = app.response_class(
            response=response_body,
            status=200,
            mimetype='text/html'
        )

        log_response(response)

        # Verify the logger was called with the correct structured data
        mock_logger.info.assert_called_once()
        call_args, call_kwargs = mock_logger.info.call_args
        assert call_args[0] == 'Outgoing response'
        log_data = call_kwargs

        assert log_data['status_code'] == 200
        assert log_data['duration_ms'] == pytest.approx(200.0)
        assert log_data['request_id'] == 'res-req-id-222'
        assert 'headers' in log_data
        assert log_data['headers']['Content-Type'] == 'text/html; charset=utf-8'
        assert log_data['body'] == response_body  # Include non-JSON body

@patch('services.utils.log_utils.time.time')
@patch('structlog.get_logger')
def test_log_response_no_start_time(mock_logger_factory, mock_time, app):
    """Test logging a response when start time is missing from g."""
    end_time = 12399.9
    response_data = {"error": "something failed early"}

    mock_logger = MagicMock()
    mock_logger_factory.return_value = mock_logger
    mock_time.return_value = end_time

    with app.test_request_context('/test_no_g'):
        # g.request_start_time is NOT set
        g.request_id = "res-req-id-333"

        response = jsonify(response_data)
        response.status_code = 500

        log_response(response)

        # Verify the logger was called with the correct structured data
        mock_logger.info.assert_called_once()
        call_args, call_kwargs = mock_logger.info.call_args
        assert call_args[0] == 'Outgoing response'
        log_data = call_kwargs

        assert log_data['status_code'] == 500
        assert log_data['duration_ms'] is None  # Duration should be None
        assert log_data['request_id'] == 'res-req-id-333'
        assert 'headers' in log_data
        assert log_data['body'] == response_data