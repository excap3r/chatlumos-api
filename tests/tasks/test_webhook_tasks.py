from unittest.mock import MagicMock
from datetime import datetime
from unittest.mock import ANY
import json
import requests
import pytest
from services.tasks import webhook_tasks
from services.analytics.webhooks.schemas import WebhookDataError

def test_send_webhook_invalid_subscription(mocker, mock_dependencies, sample_event_data, mock_logging):
    """Test task abortion when subscription data is invalid."""
    invalid_subscription_dict = {"url": "incomplete data"} # Missing required fields

    # Configure the patch for from_dict directly
    validation_error = WebhookDataError("Invalid webhook data: Missing key 'event_types'")
    mock_from_dict = mocker.patch(
        'services.tasks.webhook_tasks.WebhookSubscriptionSchema.from_dict',
        side_effect=validation_error
    )

    # Get mocks from dependencies
    mock_requests_post = mock_dependencies['requests_post']
    fake_redis = mock_dependencies['redis_client'] # Need the fake client
    mock_update_stats = mock_dependencies['update_stats']

    # --- Patch get_redis_client used by helper ---\
    mocker.patch('services.tasks.webhook_tasks.get_redis_client', return_value=fake_redis) # Patch redis helper
    # Configure the mock logger's bind method to return itself
    mock_logging.bind.return_value = mock_logging

    # --- Call the helper function directly, expecting WebhookDataError ---\
    with pytest.raises(WebhookDataError) as excinfo:
         webhook_tasks._execute_send_webhook(
            mock_logging, # Pass the mock logger instance directly
            invalid_subscription_dict,
            sample_event_data
        )

    # --- Assertions ---\
    # 1. The patched from_dict was called
    mock_from_dict.assert_called_once_with(invalid_subscription_dict)

    # 2. requests.post was NOT called
    mock_requests_post.assert_not_called()

    # 3. Logging indicates the validation failure
    found_log = False
    # Match the actual log format in _execute_send_webhook's except block
    expected_log_message = "Webhook task failed: Invalid subscription data"
    expected_error_str = str(validation_error) # Get the expected error string
    for call_args_list, call_kwargs in mock_logging.error.call_args_list:
        log_message = call_args_list[0] if call_args_list else ""
        # Check message directly, exc_info=True, and error kwarg
        if expected_log_message == log_message and \
           call_kwargs.get('exc_info') is True and \
           call_kwargs.get('error') == expected_error_str:
            found_log = True
            break
    assert found_log, f"Expected log message '{expected_log_message}' with error='{expected_error_str}' and exc_info=True not found in error logs: {mock_logging.error.call_args_list}"

    # 4. Stats update should NOT be called for this early failure
    mock_update_stats.assert_not_called()

def test_send_webhook_payload_serialization_error(mocker, mock_dependencies, sample_subscription_dict, mock_logging):
    """Test task failure when event data cannot be serialized to JSON."""
    # Create unserializable event data
    unserializable_event_data = {
        "id": "evt-bad-data",
        "event_type": "error.occurred",
        "timestamp": datetime.utcnow().isoformat(),
        "data": object() # Objects are not JSON serializable by default
    }

    # Get mocks from dependencies
    mock_requests_post = mock_dependencies['requests_post']
    fake_redis = mock_dependencies['redis_client']

    # --- Patch get_redis_client used by helper ---
    mocker.patch('services.tasks.webhook_tasks.get_redis_client', return_value=fake_redis)
    
    # Configure the mock logger's bind method to return itself
    mock_logging.bind.return_value = mock_logging

    # --- Call the helper function directly, expecting WebhookDataError ---
    with pytest.raises(WebhookDataError) as excinfo:
        webhook_tasks._execute_send_webhook(
            mock_logging, # Pass the mock logger instance directly
            sample_subscription_dict, 
            unserializable_event_data
        )

    # --- Assertions ---
    # 1. Exception was raised with correct message
    assert isinstance(excinfo.value, WebhookDataError)
    assert "Payload serialization error" in str(excinfo.value)
    assert "not JSON serializable" in str(excinfo.value)

    # 2. requests.post was NOT called
    mock_requests_post.assert_not_called()

    # 3. Logging indicates the serialization failure
    found_log = False
    expected_msg_part = "Webhook task failed: Could not serialize event data"
    # Check calls to the error method of the mock_logger instance
    for call_args_list, call_kwargs in mock_logging.error.call_args_list:
        log_message = call_args_list[0] if call_args_list else ""
        if expected_msg_part in log_message and call_kwargs.get('exc_info') is True:
            assert "not JSON serializable" in call_kwargs.get('error', '')
            found_log = True
            break
    assert found_log, f"Expected log message containing '{expected_msg_part}' with matching error not found"

    # 4. Stats update WAS called with failure due to data error
    mock_update_stats = mock_dependencies['update_stats']
    mock_update_stats.assert_called_once()
    # Get call args to verify details
    call_args, call_kwargs = mock_update_stats.call_args
    # Check positional args for redis_client (index 0) and webhook_id (index 1)
    assert call_args[1] == sample_subscription_dict['id']
    # Check keyword args for success and error_message
    assert call_kwargs.get('success') is False
    assert "Data processing error: Payload serialization error" in call_kwargs.get('error_message', '')

    # Remove assertions for Celery-specific methods when testing helper directly
    # # 5. Retry was not called
    # mock_retry.assert_not_called()
    # 
    # # 6. State was not updated (task re-raises before this)
    # mock_update_state.assert_not_called()

def test_send_webhook_signature_generation_error(mocker, mock_dependencies, sample_subscription_dict, sample_event_data, mock_logging):
    """Test webhook sending failure when signature generation fails."""
    # Configure _generate_signature to raise TypeError
    sig_gen_error = TypeError("Bad secret type")
    # Patch the correct target
    mock_generate_signature = mocker.patch(
        'services.tasks.webhook_tasks._generate_signature',
        side_effect=sig_gen_error
    )

    # Get mocks from dependencies
    mock_requests_post = mock_dependencies['requests_post']
    fake_redis = mock_dependencies['redis_client']
    mock_update_stats = mock_dependencies['update_stats']

    # --- Patch get_redis_client and configure logger ---
    mocker.patch('services.tasks.webhook_tasks.get_redis_client', return_value=fake_redis)
    mock_logging.bind.return_value = mock_logging # Handle logger binding

    # --- Call the helper function directly, expecting WebhookDataError ---
    with pytest.raises(WebhookDataError) as excinfo:
        webhook_tasks._execute_send_webhook(
            mock_logging, 
            sample_subscription_dict, 
            sample_event_data
        )

    # --- Assertions ---
    # 1. Exception was raised with correct message
    assert isinstance(excinfo.value, WebhookDataError)
    assert "Signature generation error" in str(excinfo.value)
    assert "Bad secret type" in str(excinfo.value)

    # 2. _generate_signature was called
    # Need the expected payload_json - let's dump it here for the check
    expected_payload_json = json.dumps(sample_event_data)
    mock_generate_signature.assert_called_once_with(
        sample_subscription_dict['secret'], 
        expected_payload_json
    )

    # 3. requests.post was NOT called
    mock_requests_post.assert_not_called()

    # 4. Logging indicates the signature failure
    found_log = False
    expected_msg_part = "Webhook signature generation failed"
    for call_args_list, call_kwargs in mock_logging.error.call_args_list:
        log_message = call_args_list[0] if call_args_list else ""
        if expected_msg_part in log_message and call_kwargs.get('exc_info') is True:
            assert "Bad secret type" in call_kwargs.get('error', '')
            found_log = True
            break
    assert found_log, f"Expected log message containing '{expected_msg_part}' with matching error not found"

    # 5. Stats update WAS called once with failure due to data processing error
    mock_update_stats.assert_called_once()
    call_args, call_kwargs = mock_update_stats.call_args
    assert call_args[1] == sample_subscription_dict['id']
    assert call_kwargs.get('success') is False
    # Check that the error message reflects the underlying signature error
    # The error message comes from the catch block for WebhookDataError in _execute_send_webhook
    expected_error_msg = f"Data processing error: Signature generation error: {sig_gen_error}"
    assert call_kwargs.get('error_message') == expected_error_msg

def test_send_webhook_no_secret(mocker, mock_dependencies, sample_subscription_dict, sample_event_data, mock_logging):
    """Test successful webhook delivery when no secret is configured."""
    # Create subscription dict with no secret
    no_secret_subscription_dict = sample_subscription_dict.copy()
    no_secret_subscription_dict["secret"] = None
    webhook_id = no_secret_subscription_dict["id"] # Extract ID for convenience

    # --- Mock Schema Loading ---
    # Mock the subscription object that from_dict will return
    mock_sub_instance = MagicMock()
    mock_sub_instance.id = webhook_id
    mock_sub_instance.url = no_secret_subscription_dict['url']
    mock_sub_instance.secret = None # Explicitly set secret to None
    mock_sub_instance.enabled = True
    # Patch the Schema's from_dict method used within the helper
    mock_from_dict = mocker.patch(
        'services.tasks.webhook_tasks.WebhookSubscriptionSchema.from_dict',
        return_value=mock_sub_instance
    )

    # --- Mock Network Request ---
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests_post = mock_dependencies["requests_post"]
    mock_requests_post.return_value = mock_response

    # --- Patch Redis Client Get ---
    fake_redis = mock_dependencies['redis_client']
    mocker.patch('services.tasks.webhook_tasks.get_redis_client', return_value=fake_redis)

    # --- Configure Logger ---
    mock_logging.bind.return_value = mock_logging # Handle logger binding

    # --- Call the helper function directly ---
    result = webhook_tasks._execute_send_webhook(
        mock_logging, # Pass mock logger instance
        no_secret_subscription_dict, 
        sample_event_data
    )

    # --- Assertions ---
    # 0. Check result status
    assert result['status'] == 'success'
    assert result['status_code'] == 200

    # 1. Subscription loaded via Schema
    mock_from_dict.assert_called_once_with(no_secret_subscription_dict)

    # 2. Signature generation NOT attempted
    mock_generate_signature = mock_dependencies["generate_signature"]
    mock_generate_signature.assert_not_called()

    # 3. requests.post called correctly WITHOUT signature header
    # The helper constructs the payload differently
    expected_payload_json = json.dumps(sample_event_data) # Payload is just event data
    expected_headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'ChatlumosWebhook/1.0',
        # 'X-Chatlumos-Signature-256' header should be ABSENT
    }
    mock_requests_post.assert_called_once_with(
        no_secret_subscription_dict['url'], # Use URL from dict
        headers=expected_headers,
        data=expected_payload_json.encode('utf-8'), # Use JSON string bytes
        timeout=10 # Use correct config attribute
    )

    # 4. Stats updated for success using the helper's redis client
    mock_update_stats = mock_dependencies["update_stats"]
    mock_update_stats.assert_called_once_with(
        fake_redis, # Ensure the correct redis client mock is checked
        webhook_id, # Pass the webhook ID
        success=True,
        # error_message=None # Do not assert explicit error_message=None
    )

    # 5. Logging (check logs from helper)
    mock_logging.debug.assert_any_call("Attempting to send webhook") # Changed from info
    mock_logging.info.assert_any_call(
        "Sending webhook request", 
        event_type=sample_event_data.get('event_type'), 
        event_id=sample_event_data.get('id')
    )
    mock_logging.info.assert_any_call(
        "Webhook sent successfully", 
        status_code=200
    )
    # Ensure error/warning logs related to signature were not called
    signature_error_logs = [c for c in mock_logging.error.call_args_list if "signature" in c[0][0]]
    assert not signature_error_logs, "Signature error logs were unexpectedly found"

def test_send_webhook_max_retries_exceeded(mocker, mock_dependencies, sample_subscription_dict, sample_event_data, mock_logging):
    """Test behavior when max retries are exceeded (simulated).

    Note: This tests the helper's behavior when a retryable exception occurs.
    It doesn't fully test Celery's autoretry logic, but ensures the helper
    raises appropriately and updates stats on the final failure.
    """
    webhook_id = sample_subscription_dict['id']

    # --- Mock Schema Loading ---
    mock_sub_instance = MagicMock()
    mock_sub_instance.id = webhook_id
    mock_sub_instance.url = sample_subscription_dict['url']
    mock_sub_instance.secret = sample_subscription_dict['secret'] # Assume secret exists
    mock_sub_instance.enabled = True
    mock_from_dict = mocker.patch(
        'services.tasks.webhook_tasks.WebhookSubscriptionSchema.from_dict',
        return_value=mock_sub_instance
    )

    # --- Mock Signature Generation (assume success) ---
    mock_generate_signature = mock_dependencies["generate_signature"]
    mock_generate_signature.return_value = "dummy-signature"

    # --- Mock Network Request to fail consistently ---
    final_error_message = "Connection refused"
    final_exception = requests.exceptions.ConnectionError(final_error_message)
    mock_requests_post = mock_dependencies["requests_post"]
    mock_requests_post.side_effect = final_exception

    # --- Patch Redis Client Get ---
    fake_redis = mock_dependencies['redis_client']
    mocker.patch('services.tasks.webhook_tasks.get_redis_client', return_value=fake_redis)

    # --- Configure Logger ---
    mock_logging.bind.return_value = mock_logging

    # --- Call the helper function directly, expecting the final exception ---
    with pytest.raises(requests.exceptions.ConnectionError) as excinfo:
        webhook_tasks._execute_send_webhook(
            mock_logging,
            sample_subscription_dict,
            sample_event_data
        )

    # --- Assertions ---
    # 1. Schema loaded
    mock_from_dict.assert_called_once_with(sample_subscription_dict)

    # 2. Signature was generated (as secret exists)
    expected_payload_json = json.dumps(sample_event_data)
    mock_generate_signature.assert_called_once_with(
        sample_subscription_dict['secret'],
        expected_payload_json
    )

    # 3. requests.post was called (and failed)
    expected_headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'ChatlumosWebhook/1.0',
        'X-Chatlumos-Signature-256': 'sha256=dummy-signature' # Check signature header
    }
    mock_requests_post.assert_called_once_with(
        sample_subscription_dict['url'],
        headers=expected_headers,
        data=expected_payload_json.encode('utf-8'),
        timeout=10 # Use correct config attribute
    )

    # 4. Stats updated for FAILURE
    mock_update_stats = mock_dependencies["update_stats"]
    mock_update_stats.assert_called_once()
    call_args, call_kwargs = mock_update_stats.call_args
    assert call_args[1] == sample_subscription_dict['id']
    assert call_kwargs.get('success') is False
    # Check that the error message reflects the actual ConnectionError
    expected_error_msg = f"Request failed (None): {final_error_message}"
    assert call_kwargs.get('error_message') == expected_error_msg

    # 5. Logging indicates the failure
    mock_logging.error.assert_any_call(
        "Webhook request failed",
        status_code=None, # No status code for ConnectionError
        error=final_error_message,
        exc_info=True
    )

    # 6. Final exception was raised
    assert excinfo.value is final_exception

# --- Tests for Helper Functions (Optional) ---
def test_generate_signature():
    pass # TODO