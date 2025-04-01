import pytest
import traceback
from services.utils.error_utils import (
    APIError, 
    ValidationError, 
    DatabaseError, 
    NotFoundError, 
    format_error_response
    # handle_error - Decorator, test indirectly or via integration
)

# --- Tests for Custom Exception Classes ---

def test_api_error():
    """Test the base APIError class."""
    message = "Something went wrong"
    status_code = 503
    details = {"info": "extra data"}
    
    err = APIError(message, status_code, details)
    
    assert err.message == message
    assert err.status_code == status_code
    assert err.details == details
    assert str(err) == message
    
    expected_dict = {
        "error": message,
        "status_code": status_code,
        "details": details
    }
    assert err.to_dict() == expected_dict

def test_api_error_defaults():
    """Test APIError with default values."""
    message = "Default error"
    err = APIError(message)
    
    assert err.message == message
    assert err.status_code == 500 # Default status code
    assert err.details is None
    
    expected_dict = {
        "error": message,
        "status_code": 500
    }
    assert err.to_dict() == expected_dict

def test_validation_error():
    """Test the ValidationError class."""
    message = "Invalid input"
    field = "username"
    details = ["Too short", "Must be alphanumeric"]
    
    err = ValidationError(message, field, details)
    
    assert err.message == message
    assert err.status_code == 400 # Specific status for validation
    assert err.field == field
    assert err.details == details
    assert str(err) == message
    
    expected_dict = {
        "error": message,
        "status_code": 400,
        "field": field,
        "details": details
    }
    assert err.to_dict() == expected_dict

def test_validation_error_no_field():
    """Test ValidationError without a specific field."""
    message = "Missing required fields"
    err = ValidationError(message)
    
    assert err.message == message
    assert err.status_code == 400
    assert err.field is None
    assert err.details is None
    
    expected_dict = {
        "error": message,
        "status_code": 400
    }
    assert err.to_dict() == expected_dict

def test_database_error():
    """Test the DatabaseError class."""
    details = "Connection refused"
    err = DatabaseError(details=details)
    
    assert err.message == "Database operation failed" # Default message
    assert err.status_code == 500
    assert err.details == details
    
    expected_dict = {
        "error": "Database operation failed",
        "status_code": 500,
        "details": details
    }
    assert err.to_dict() == expected_dict

def test_not_found_error():
    """Test the NotFoundError class."""
    message = "User not found"
    err = NotFoundError(message=message)
    
    assert err.message == message
    assert err.status_code == 404
    assert err.details is None
    
    expected_dict = {
        "error": message,
        "status_code": 404
    }
    assert err.to_dict() == expected_dict

# --- Tests for format_error_response --- 

def test_format_error_response_api_error():
    """Test formatting an APIError instance."""
    api_err = APIError("Specific API failure", 403, {"code": "AUTH_FAIL"})
    expected_dict = api_err.to_dict()
    expected_status = 403
    
    response_dict, status_code = format_error_response(api_err)
    
    assert status_code == expected_status
    assert response_dict == expected_dict

def test_format_error_response_validation_error():
    """Test formatting a ValidationError instance."""
    val_err = ValidationError("Bad email", field="email")
    expected_dict = val_err.to_dict()
    expected_status = 400
    
    response_dict, status_code = format_error_response(val_err)
    
    assert status_code == expected_status
    assert response_dict == expected_dict

def test_format_error_response_standard_exception():
    """Test formatting a standard Python exception."""
    std_err = ValueError("Some standard error occurred")
    expected_dict = {
        "error": str(std_err),
        "status_code": 500
    }
    expected_status = 500
    
    response_dict, status_code = format_error_response(std_err)
    
    assert status_code == expected_status
    assert response_dict == expected_dict

def test_format_error_response_standard_exception_with_traceback():
    """Test formatting a standard exception with traceback included."""
    try:
        # Intentionally raise an error to get a traceback
        _ = 1 / 0 
    except Exception as std_err:
        expected_dict = {
            "error": str(std_err),
            "status_code": 500,
            "traceback": traceback.format_exc() # Expected traceback string
        }
        expected_status = 500
        
        response_dict, status_code = format_error_response(std_err, include_traceback=True)
        
        assert status_code == expected_status
        # Compare keys first, then traceback content contains key info
        assert response_dict.keys() == expected_dict.keys()
        assert response_dict["error"] == expected_dict["error"]
        assert response_dict["status_code"] == expected_dict["status_code"]
        assert "Traceback (most recent call last)" in response_dict["traceback"]
        assert "ZeroDivisionError" in response_dict["traceback"] 