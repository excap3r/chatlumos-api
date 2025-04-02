"""
Database Utilities

Provides common helper functions and decorators for database operations.
"""

from functools import wraps
from flask import current_app
import structlog
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Import custom exceptions from the parent directory
from .exceptions import QueryError, DatabaseError, ConnectionError, DuplicateEntryError, NotFoundError, InvalidCredentialsError

logger = structlog.get_logger(__name__)

def _get_session() -> Session:
    """Get the SQLAlchemy session from Flask's current_app."""
    # Duplicated from db_service/user_db - consider moving to a single location like here
    if not hasattr(current_app, 'db_session') or current_app.db_session is None:
        logger.error("SQLAlchemy session not initialized in current_app.")
        raise ConnectionError("Database session not available.")
    return current_app.db_session()

def handle_db_session(func):
    """
    Decorator to manage SQLAlchemy session exceptions, rollback, and logging.
    
    It assumes the decorated function performs its own commit on success.
    It catches SQLAlchemyError, rolls back, logs, and raises QueryError.
    It catches IntegrityError, rolls back, logs, and raises specific errors
    (DuplicateEntryError, NotFoundError) based on error message heuristics,
    otherwise raises QueryError.
    It catches other Exceptions, rolls back, logs, and raises DatabaseError.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        session = _get_session() # Get session at the start
        try:
            # Execute the wrapped function (which should handle its own commit)
            result = func(*args, **kwargs)
            return result
        except IntegrityError as e:
            session.rollback()
            func_name = func.__name__
            logger.warning(f"Integrity error in {func_name}", error=str(e.orig), exc_info=True)
            # Attempt to raise more specific errors based on common patterns
            error_str = str(e.orig).upper()
            if "FOREIGN KEY" in error_str:
                 # Should ideally parse constraint name to provide better context
                 raise NotFoundError(f"Related entity not found: {e}") from e
            elif "DUPLICATE ENTRY" in error_str or "UNIQUE CONSTRAINT" in error_str:
                 # Should ideally parse constraint name/values
                 raise DuplicateEntryError(f"Duplicate entry detected: {e}") from e
            else:
                 raise QueryError(f"Database integrity error in {func_name}: {e}") from e
        except SQLAlchemyError as e:
            session.rollback()
            func_name = func.__name__
            logger.error(f"SQLAlchemyError in {func_name}", error=str(e), exc_info=True)
            raise QueryError(f"Database query failed in {func_name}: {e}") from e
        except (NotFoundError, DuplicateEntryError, InvalidCredentialsError) as e: # Let specific app errors pass through
             # No rollback needed usually, as these are raised after checks/before commit
             # or are handled specifically within the function before this decorator catches them.
             # If they can occur *after* a commit attempt fails, rollback might be needed.
             # For now, assume they are caught appropriately or happen before commit issues.
             raise
        except Exception as e:
            session.rollback()
            func_name = func.__name__
            logger.error(f"Unexpected Exception in {func_name}", error=str(e), exc_info=True)
            # Wrap unexpected errors in a generic DatabaseError
            raise DatabaseError(f"An unexpected error occurred in {func_name}: {e}") from e
        # Note: Session closing/removal is typically handled by the scoped_session
        # mechanism tied to the Flask request context, so no explicit close needed here.
    return wrapper 