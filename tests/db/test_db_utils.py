import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import structlog

# Import the module and decorator to test
import services.db.db_utils as db_utils_module
from services.db.db_utils import handle_db_session
from services.db.exceptions import (
    QueryError,
    DatabaseError,
    ConnectionError,
    DuplicateEntryError,
    NotFoundError,
    InvalidCredentialsError, # Add other passthrough errors if needed
)

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_db_session_util():
    """Mocks the _get_session utility within db_utils."""
    with patch('services.db.db_utils._get_session') as mock_get:
        mock_sess = MagicMock(spec=Session)
        mock_get.return_value = mock_sess
        yield mock_sess # Return the mock session for assertions

@pytest.fixture
def mock_db_logger():
    """Mocks the logger within db_utils."""
    with patch('services.db.db_utils.structlog.get_logger') as mock_get_logger:
        mock_log_instance = MagicMock(spec=structlog.stdlib.BoundLogger)
        # Configure logger methods if needed, e.g., mock_log_instance.error = MagicMock()
        mock_get_logger.return_value = mock_log_instance
        yield mock_log_instance # Return the mock logger for assertions

# --- Test Cases for handle_db_session ---

def test_handle_db_session_success(mock_db_session_util, mock_db_logger):
    """Test the decorator when the wrapped function succeeds."""
    @handle_db_session
    def successful_operation():
        # In a real scenario, might interact with mock_db_session_util
        return "Success"

    result = successful_operation()

    assert result == "Success"
    mock_db_session_util.rollback.assert_not_called()
    mock_db_logger.warning.assert_not_called()
    mock_db_logger.error.assert_not_called()

def test_handle_db_session_integrity_error_duplicate(mock_db_session_util, mock_db_logger):
    """Test handling IntegrityError indicating a duplicate entry."""
    error_message = "(sqlite3.IntegrityError) UNIQUE constraint failed: users.email"
    original_exception = IntegrityError(error_message, params={}, orig=Exception(error_message))

    @handle_db_session
    def raises_duplicate_entry():
        raise original_exception

    # Patch the module-level logger used by the decorator for this test
    with patch('services.db.db_utils.logger', mock_db_logger):
        with pytest.raises(DuplicateEntryError) as exc_info:
            raises_duplicate_entry()

    assert "Duplicate entry detected" in str(exc_info.value)
    mock_db_session_util.rollback.assert_called_once()
    mock_db_logger.warning.assert_called_once() # Now asserting on the correct mock
    mock_db_logger.error.assert_not_called()

def test_handle_db_session_integrity_error_foreign_key(mock_db_session_util, mock_db_logger):
    """Test handling IntegrityError indicating a foreign key violation."""
    error_message = "FOREIGN KEY constraint failed"
    original_exception = IntegrityError(error_message, params={}, orig=Exception(error_message))

    @handle_db_session
    def raises_foreign_key_violation():
        raise original_exception

    # Patch the module-level logger used by the decorator for this test
    with patch('services.db.db_utils.logger', mock_db_logger):
        with pytest.raises(NotFoundError) as exc_info:
            raises_foreign_key_violation()

    assert "Related entity not found" in str(exc_info.value)
    mock_db_session_util.rollback.assert_called_once()
    mock_db_logger.warning.assert_called_once() # Now asserting on the correct mock
    mock_db_logger.error.assert_not_called()

def test_handle_db_session_integrity_error_other(mock_db_session_util, mock_db_logger):
    """Test handling other IntegrityErrors."""
    error_message = "Some other integrity issue"
    original_exception = IntegrityError(error_message, params={}, orig=Exception(error_message))

    @handle_db_session
    def raises_other_integrity_error():
        raise original_exception

    # Patch the module-level logger used by the decorator for this test
    with patch('services.db.db_utils.logger', mock_db_logger):
        with pytest.raises(QueryError) as exc_info:
            raises_other_integrity_error()

    assert "Database integrity error" in str(exc_info.value)
    mock_db_session_util.rollback.assert_called_once()
    mock_db_logger.warning.assert_called_once() # Now asserting on the correct mock
    mock_db_logger.error.assert_not_called()

def test_handle_db_session_sqlalchemy_error(mock_db_session_util, mock_db_logger):
    """Test handling generic SQLAlchemyError."""
    original_exception = SQLAlchemyError("Generic DB communication error")

    @handle_db_session
    def raises_sqlalchemy_error():
        raise original_exception

    # Patch the module-level logger used by the decorator for this test
    with patch('services.db.db_utils.logger', mock_db_logger):
        with pytest.raises(QueryError) as exc_info:
            raises_sqlalchemy_error()

    assert "Database query failed" in str(exc_info.value)
    mock_db_session_util.rollback.assert_called_once()
    mock_db_logger.error.assert_called_once() # Now asserting on the correct mock
    mock_db_logger.warning.assert_not_called()

def test_handle_db_session_custom_passthrough_error(mock_db_session_util, mock_db_logger):
    """Test that specific custom errors (like NotFoundError) pass through."""
    # Test with NotFoundError
    original_not_found = NotFoundError("Specific item not found")
    @handle_db_session
    def raises_not_found():
        raise original_not_found
    with pytest.raises(NotFoundError) as exc_info_nf:
        raises_not_found()
    assert exc_info_nf.value is original_not_found

    # Test with InvalidCredentialsError
    original_invalid_cred = InvalidCredentialsError("Bad password")
    @handle_db_session
    def raises_invalid_cred():
        raise original_invalid_cred
    with pytest.raises(InvalidCredentialsError) as exc_info_ic:
        raises_invalid_cred()
    assert exc_info_ic.value is original_invalid_cred
    
    # Test with DuplicateEntryError
    original_duplicate = DuplicateEntryError("Entry exists")
    @handle_db_session
    def raises_duplicate():
         raise original_duplicate
    with pytest.raises(DuplicateEntryError) as exc_info_dup:
         raises_duplicate()
    assert exc_info_dup.value is original_duplicate
    
    # Common assertions for passthrough errors
    mock_db_session_util.rollback.assert_not_called()
    mock_db_logger.warning.assert_not_called()
    mock_db_logger.error.assert_not_called()

def test_handle_db_session_unexpected_exception(mock_db_session_util, mock_db_logger):
    """Test handling unexpected generic Exceptions."""
    original_exception = Exception("Something totally unexpected happened")

    @handle_db_session
    def raises_unexpected_error():
        raise original_exception

    # Patch the module-level logger used by the decorator for this test
    with patch('services.db.db_utils.logger', mock_db_logger):
        with pytest.raises(DatabaseError) as exc_info:
            raises_unexpected_error()

    assert "An unexpected error occurred" in str(exc_info.value)
    mock_db_session_util.rollback.assert_called_once()
    mock_db_logger.error.assert_called_once() # Now asserting on the correct mock
    mock_db_logger.warning.assert_not_called()

# TODO: Add test cases here 