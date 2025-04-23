# Placeholder for analytics task tests

from unittest.mock import MagicMock, patch
import pytest
from celery.exceptions import Retry
from services.tasks import analytics_tasks
import redis.exceptions

def test_log_analytics_event_unexpected_error(mocker, mock_dependencies, sample_event, mock_logging):
    """Test task failure on unexpected error during processing."""
    # --- Setup Mocks ---
    # fake_redis = MagicMock() # Create a fresh mock Redis client
    # Use the one from mock_dependencies to ensure consistency if needed elsewhere
    fake_redis = mock_dependencies['redis_client'] 
    pipeline_mock = mock_dependencies['redis_pipeline'] # Use pipeline from deps
    unexpected_error = TypeError("Something went wrong")

    # Configure mocks
    pipeline_mock.execute.side_effect = unexpected_error
    # No need to configure fake_redis.pipeline if using mock_dependencies['redis_pipeline']
    # fake_redis.pipeline.return_value = pipeline_mock 

    # Patch dependencies
    mocker.patch('services.tasks.analytics_tasks.get_redis_client', return_value=fake_redis)
    # mock_config = MagicMock()
    # mock_config.ANALYTICS_TTL_SECONDS = 3600
    # mocker.patch('services.tasks.analytics_tasks.AppConfig', mock_config)
    # AppConfig should be available via the main app fixture context implicitly

    # --- Mock Task Methods directly on the task object ---
    # Patch 'retry' and 'update_state' on the task class itself
    mock_retry = mocker.patch.object(
        analytics_tasks.log_analytics_event_task,
        'retry',
        side_effect=Exception("Should not retry on unexpected error")
    )
    mock_update_state = mocker.patch.object(
        analytics_tasks.log_analytics_event_task,
        'update_state'
    )

    # --- Call the task using .apply() and expect the error ---
    with pytest.raises(TypeError) as excinfo:
        analytics_tasks.log_analytics_event_task.apply(args=[sample_event])
    
    # Optionally assert the exception message
    assert "Something went wrong" in str(excinfo.value)

    # --- Assertions on Task Behavior ---
    pipeline_mock.execute.assert_called_once()
    # Verify state was updated to FAILURE
    mock_update_state.assert_called_once()
    call_args, call_kwargs = mock_update_state.call_args
    assert call_kwargs.get('state') == 'FAILURE'
    assert 'meta' in call_kwargs
    assert call_kwargs['meta'].get('exc_type') == 'TypeError'
    assert call_kwargs['meta'].get('exc_message') == "Something went wrong"
    # Verify retry was not called
    mock_retry.assert_not_called()

@pytest.mark.parametrize("exception_type, should_retry", [
    (redis.exceptions.ConnectionError("Connection failed"), True),
    (redis.exceptions.TimeoutError("Timeout"), True),
    (redis.exceptions.ResponseError("Redis error"), True),
])
def test_log_analytics_event_redis_error_retry(mocker, mock_dependencies, sample_event, mock_logging, exception_type, should_retry):
    """Test task retry logic on various Redis errors.
       Assumes autoretry_for allows original exception to propagate after scheduling retry.
    """
    # --- Setup Mocks ---
    fake_redis = mock_dependencies["redis_client"]
    pipeline = mock_dependencies["redis_pipeline"]
    pipeline.execute.side_effect = exception_type

    # Patch get_redis_client
    mocker.patch('services.tasks.analytics_tasks.get_redis_client', return_value=fake_redis)

    # DO NOT patch self.retry

    # --- Call the task using apply() for correct context and bind=True ---
    # Expect Celery's Retry exception because the task catches RedisError and calls self.retry
    with pytest.raises(Retry) as excinfo:
        analytics_tasks.log_analytics_event_task.apply(args=[sample_event])

    # --- Assertions ---
    # Verify the underlying operation that caused the error was called
    pipeline.execute.assert_called_once()
    # Verify the Retry exception contains the original exception
    assert isinstance(excinfo.value.exc, type(exception_type))