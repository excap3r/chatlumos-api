import pytest
import logging
import os
import shutil
from unittest.mock import patch, MagicMock
import json

from services.utils.log_utils import setup_logger, log_request, log_response

# TODO: Add tests for setup_logger
# TODO: Add tests for log_request (requires Flask app context)
# TODO: Add tests for log_response (requires Flask app context)

# --- Tests for setup_logger --- 

def test_setup_logger_console_only():
    """Test creating a logger with only console output."""
    logger_name = "test_console_logger"
    logger = setup_logger(logger_name, level=logging.DEBUG)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    # Clean up logger to avoid interference with other tests
    logging.Logger.manager.loggerDict.pop(logger_name, None)


def test_setup_logger_with_file(tmp_path):
    """Test creating a logger with file output."""
    logger_name = "test_file_logger"
    log_dir = tmp_path / "logs"
    log_file = log_dir / "test.log"
    
    # Ensure directory doesn't exist initially
    assert not log_dir.exists()
    
    logger = setup_logger(logger_name, log_file=str(log_file), level=logging.INFO)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 2 # Console + File
    
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].baseFilename == str(log_file)
    
    # Check if directory and file were created
    assert log_dir.exists()
    assert log_dir.is_dir()
    # The file might not be created until first log message, check dir is enough

    # Test logging creates the file
    logger.info("Test message")
    assert log_file.exists()
    assert log_file.is_file()

    # Clean up logger
    logging.Logger.manager.loggerDict.pop(logger_name, None)


def test_setup_logger_existing_dir(tmp_path):
    """Test creating a logger when the log directory already exists."""
    logger_name = "test_existing_dir_logger"
    log_dir = tmp_path / "existing_logs"
    log_file = log_dir / "test.log"
    
    # Create directory beforehand
    log_dir.mkdir()
    assert log_dir.exists()
    
    logger = setup_logger(logger_name, log_file=str(log_file))
    
    assert len(logger.handlers) == 2
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].baseFilename == str(log_file)
    logger.warning("Another test")
    assert log_file.exists()

    # Clean up logger
    logging.Logger.manager.loggerDict.pop(logger_name, None)

# TODO: Add tests for log_request (requires Flask app context)
# TODO: Add tests for log_response (requires Flask app context) 

# --- Tests for log_request --- 

from flask import Flask, request, g, jsonify

@pytest.fixture
def app():
    """Create a minimal Flask app for context."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app


@patch('services.utils.log_utils.time.time')
@patch('logging.Logger.info') # Patch the logger's info method directly
def test_log_request_get(mock_logger_info, mock_time, app):
    """Test logging a basic GET request."""
    mock_time.return_value = 12345.6789
    test_logger = logging.getLogger("test_req_log")
    # Prevent logs from propagating to root logger during test
    test_logger.propagate = False 
    
    with app.test_request_context('/test?param1=val1', method='GET', base_url="http://localhost"):
        # Simulate request_id being set by middleware
        g.request_id = "test-req-id-123"
        
        log_request(request, logger=test_logger)
        
        assert hasattr(g, 'request_start_time')
        assert g.request_start_time == 12345.6789
        
        mock_logger_info.assert_called_once()
        log_message = mock_logger_info.call_args[0][0]
        assert log_message.startswith("Request: ")
        log_data = json.loads(log_message.replace("Request: ", ""))
        
        assert log_data['request_id'] == "test-req-id-123"
        assert log_data['method'] == 'GET'
        assert log_data['url'] == 'http://localhost/test?param1=val1'
        assert log_data['path'] == '/test'
        assert log_data['args'] == {'param1': 'val1'}
        assert log_data['data'] is None
        assert "Authorization" not in log_data['headers']

@patch('services.utils.log_utils.time.time')
@patch('logging.Logger.info')
def test_log_request_post_json(mock_logger_info, mock_time, app):
    """Test logging a POST request with JSON data and Auth header."""
    mock_time.return_value = 12345.0
    test_logger = logging.getLogger("test_req_log_post")
    test_logger.propagate = False
    request_data = {"key": "value", "nested": {"num": 1}}
    auth_header = "Bearer some_secret_token"
    
    with app.test_request_context('/submit', method='POST', json=request_data, 
                                headers={'Authorization': auth_header, 'Content-Type': 'application/json'}, 
                                base_url="http://localhost"):
        g.request_id = "test-req-id-456"
        log_request(request, logger=test_logger)
        
        assert hasattr(g, 'request_start_time')
        assert g.request_start_time == 12345.0

        mock_logger_info.assert_called_once()
        log_message = mock_logger_info.call_args[0][0]
        log_data = json.loads(log_message.replace("Request: ", ""))
        
        assert log_data['request_id'] == "test-req-id-456"
        assert log_data['method'] == 'POST'
        assert log_data['url'] == 'http://localhost/submit'
        assert log_data['data'] == request_data
        assert log_data['headers']['Authorization'] == "<redacted>"
        assert log_data['headers']['Content-Type'] == 'application/json'

@patch('services.utils.log_utils.time.time')
@patch('logging.Logger.info')
def test_log_request_invalid_json(mock_logger_info, mock_time, app):
    """Test logging a request with invalid JSON data."""
    mock_time.return_value = 12345.0
    test_logger = logging.getLogger("test_req_log_invalid")
    test_logger.propagate = False
    invalid_json_bytes = b'{ "key": "value", '

    with app.test_request_context('/invalid', method='POST', 
                                data=invalid_json_bytes, 
                                content_type='application/json', 
                                base_url="http://localhost"):
        g.request_id = "test-req-id-789"
        log_request(request, logger=test_logger)
        
        mock_logger_info.assert_called_once()
        log_message = mock_logger_info.call_args[0][0]
        log_data = json.loads(log_message.replace("Request: ", ""))
        
        assert log_data['data'] == "<Invalid JSON>"


# --- Tests for log_response --- 

@patch('services.utils.log_utils.time.time')
@patch('logging.Logger.info')
def test_log_response_json(mock_logger_info, mock_time, app):
    """Test logging a JSON response."""
    test_logger = logging.getLogger("test_res_log_json")
    test_logger.propagate = False
    start_time = 12345.0
    end_time = 12345.5
    response_data = {"result": "success", "count": 42}
    
    with app.test_request_context('/test'): # Context needed for g
        g.request_start_time = start_time
        g.request_id = "res-req-id-111"
        # Mock current time for duration calculation
        mock_time.return_value = end_time 
        
        # Create a Flask response object
        response = jsonify(response_data)
        response.status_code = 200
        response.headers['X-Custom-Header'] = 'TestValue'
        
        log_response(response, logger=test_logger)
        
        mock_logger_info.assert_called_once()
        log_message = mock_logger_info.call_args[0][0]
        assert log_message.startswith("Response: ")
        log_data = json.loads(log_message.replace("Response: ", ""))
        
        assert log_data['request_id'] == "res-req-id-111"
        assert log_data['status_code'] == 200
        assert log_data['duration'] == pytest.approx(end_time - start_time) 
        assert log_data['data'] == response_data
        assert log_data['headers']['Content-Type'] == 'application/json'
        assert log_data['headers']['X-Custom-Header'] == 'TestValue'

@patch('services.utils.log_utils.time.time')
@patch('logging.Logger.info')
def test_log_response_non_json(mock_logger_info, mock_time, app):
    """Test logging a non-JSON response."""
    test_logger = logging.getLogger("test_res_log_nonjson")
    test_logger.propagate = False
    start_time = 12300.0
    end_time = 12300.1
    
    with app.test_request_context('/test_html'):
        g.request_start_time = start_time
        g.request_id = "res-req-id-222"
        mock_time.return_value = end_time
        
        response = app.response_class(
            response="<h1>Hello</h1>",
            status=200,
            mimetype='text/html'
        )
        
        log_response(response, logger=test_logger)
        
        mock_logger_info.assert_called_once()
        log_message = mock_logger_info.call_args[0][0]
        log_data = json.loads(log_message.replace("Response: ", ""))
        
        assert log_data['request_id'] == "res-req-id-222"
        assert log_data['status_code'] == 200
        assert log_data['duration'] == pytest.approx(end_time - start_time)
        assert log_data['data'] == "<Non-JSON response>"
        assert log_data['headers']['Content-Type'] == 'text/html; charset=utf-8'

@patch('services.utils.log_utils.time.time')
@patch('logging.Logger.info')
def test_log_response_no_start_time(mock_logger_info, mock_time, app):
    """Test logging a response when start time is missing from g."""
    test_logger = logging.getLogger("test_res_log_no_g")
    test_logger.propagate = False
    end_time = 12399.9
    
    with app.test_request_context('/test_no_g'):
        # g.request_start_time is NOT set
        g.request_id = "res-req-id-333"
        mock_time.return_value = end_time
        
        response = jsonify({"message": "ok"})
        response.status_code = 200
        
        log_response(response, logger=test_logger)
        
        mock_logger_info.assert_called_once()
        log_message = mock_logger_info.call_args[0][0]
        log_data = json.loads(log_message.replace("Response: ", ""))
        
        assert log_data['request_id'] == "res-req-id-333"
        assert log_data['status_code'] == 200
        assert log_data['duration'] is None # Duration should be None
        assert log_data['data'] == {"message": "ok"} 