# Placeholder for analytics task tests

from unittest.mock import MagicMock
import pytest
from celery.exceptions import Retry
from services.tasks import analytics_tasks

def test_log_analytics_event_unexpected_error(mock_dependencies, sample_event, mock_logging):
    """Test task failure on unexpected error during processing."""
    # Configure pipeline execute to raise a non-RedisError
    unexpected_error = TypeError("Something went wrong")
    pipeline = mock_dependencies["redis_pipeline"]
    pipeline.execute.side_effect = unexpected_error

    mock_self = MagicMock()
    mock_self.request.id = "task-analytics-unexp-err"
    mock_self.request.retries = 0
    mock_self.retry = MagicMock(side_effect=Exception("Should not retry"))
    mock_self.update_state = MagicMock()

    # --- Call the task, expecting the unexpected error to be re-raised ---
    with pytest.raises(TypeError) as excinfo:
        analytics_tasks.log_analytics_event_task(mock_self, sample_event)

    # --- Assertions ---
    # 1. Pipeline execute was called
    pipeline.execute.assert_called_once()

    # 2. Logging indicates the unexpected error
    mock_logging.error.assert_any_call("Unexpected error during analytics task execution. Task might fail permanently.", error=str(unexpected_error), exc_info=True)

    # 3. Celery retry method was NOT called
    mock_self.retry.assert_not_called()

    # 4. Task state WAS updated to FAILURE
    mock_self.update_state.assert_called_once_with(
        state='FAILURE',
        meta={'exc_type': 'TypeError', 'exc_message': str(unexpected_error)}
    )

    # 5. The original exception was re-raised
    assert excinfo.value is unexpected_error