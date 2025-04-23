import requests
import hmac
import hashlib
import json
import time
import structlog
import redis
from datetime import datetime
from celery import Task
from typing import Dict, Any, Optional
from celery.exceptions import MaxRetriesExceededError

# Import the Celery app instance
from celery_app import celery_app
from services.config import AppConfig

# Added for type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import only for type checking to avoid circular dependency
    from services.analytics.webhooks.webhook_service import WebhookSubscription

from services.analytics.webhooks.webhook_service import WebhookSubscription # Corrected import path
from services.analytics.webhooks.schemas import WebhookSubscription as WebhookSubscriptionSchema, WebhookDataError # Import the schema
from .analytics_tasks import get_redis_client # Import the redis client helper

logger = structlog.get_logger(__name__)

# Constants from WebhookService (or get from config passed to task)
DEFAULT_MAX_RETRY_COUNT = 3
DEFAULT_WEBHOOK_TIMEOUT = 5
DEFAULT_WEBHOOK_USER_AGENT = "PDFWisdomExtractor-Webhook/1.0"

def _generate_signature(secret: str, message: str) -> str:
    """Generate HMAC-SHA256 signature for the payload."""
    if not isinstance(secret, str):
         logger.error("Webhook secret is not a string, cannot generate signature.")
         raise TypeError("Webhook secret must be a string")

    hashed = hmac.new(secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
    return hashed.hexdigest()

def _update_webhook_stats_in_redis(redis_client, webhook_id: str, success: bool, error_message: Optional[str] = None):
    """Update webhook delivery statistics in Redis (adapted from WebhookService)."""
    if not redis_client:
        logger.error("Cannot update webhook stats: Redis client not available.", webhook_id=webhook_id)
        return

    key = f"webhook:subscription:{webhook_id}"
    now_iso = datetime.utcnow().isoformat()
    try:
        # Use WATCH/MULTI/EXEC to update the subscription JSON string atomically
        with redis_client.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    current_data_bytes = pipe.get(key)
                    if not current_data_bytes:
                        logger.warning("Cannot update stats: Webhook subscription data missing", webhook_id=webhook_id)
                        pipe.unwatch()
                        return # Exit if key gone

                    # Decode bytes to string before loading JSON
                    sub_data = json.loads(current_data_bytes) 

                    # Update stats in the dictionary
                    sub_data["last_triggered"] = now_iso
                    if success:
                        sub_data["success_count"] = sub_data.get("success_count", 0) + 1
                        sub_data["last_success"] = now_iso
                        sub_data.pop("last_error", None) # Remove last error on success
                    else:
                        sub_data["failure_count"] = sub_data.get("failure_count", 0) + 1
                        sub_data["last_failure"] = now_iso
                        if error_message:
                            sub_data["last_error"] = error_message[:500] # Limit length

                    # Start transaction
                    pipe.multi()
                    pipe.set(key, json.dumps(sub_data)) # Write the updated JSON back
                    pipe.execute() # Attempt transaction
                    logger.debug("Webhook stats updated via task", webhook_id=webhook_id, success=success)
                    break # Exit loop on successful transaction

                except redis.exceptions.WatchError:
                    logger.warning("WatchError during webhook stat update, retrying...", webhook_id=webhook_id)
                    continue
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error("Error decoding webhook data during stat update attempt", webhook_id=webhook_id, error=str(e))
                    pipe.unwatch()
                    return
                except Exception as e:
                    logger.error("Unexpected error during webhook stat update attempt", webhook_id=webhook_id, error=str(e), exc_info=True)
                    pipe.unwatch()
                    return
    except redis.exceptions.RedisError as e:
        logger.error("Redis error initiating pipeline/watch for stat update", webhook_id=webhook_id, error=str(e), exc_info=True)
    except Exception as e:
        logger.error("Unexpected error setting up stat update", webhook_id=webhook_id, error=str(e), exc_info=True)

# --- Helper function for core logic ---
def _execute_send_webhook(task_logger, subscription_dict: Dict, event_data: Dict):
    """Core logic for sending a webhook, handling errors."""
    try:
        # Reconstruct subscription object
        subscription = WebhookSubscriptionSchema.from_dict(subscription_dict)
    except WebhookDataError as e:
        task_logger.error(
            "Webhook task failed: Invalid subscription data",
            error=str(e),
            exc_info=True
        )
        raise # Re-raise the specific data error

    webhook_id = subscription.id
    task_logger = task_logger.bind(webhook_id=webhook_id, url=subscription.url)

    if not subscription.enabled:
        task_logger.info("Webhook subscription is disabled, skipping task.")
        return {"status": "skipped", "reason": "disabled"} # Return status

    task_logger.debug("Attempting to send webhook")

    try:
        # Serialize payload
        try:
            payload_json = json.dumps(event_data)
        except TypeError as e:
            task_logger.error(
                "Webhook task failed: Could not serialize event data",
                event_id=event_data.get('id'),
                error=str(e),
                exc_info=True
            )
            raise WebhookDataError(f"Payload serialization error: {e}") from e

        # Generate signature if secret exists
        signature = None
        if subscription.secret:
            try:
                 signature = _generate_signature(subscription.secret, payload_json)
            except Exception as e:
                 task_logger.error("Webhook signature generation failed", error=str(e), exc_info=True)
                 raise WebhookDataError(f"Signature generation error: {e}") from e

        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': AppConfig.WEBHOOK_USER_AGENT or DEFAULT_WEBHOOK_USER_AGENT
        }
        if signature:
            headers['X-Chatlumos-Signature-256'] = f"sha256={signature}"

        # Make the request
        task_logger.info("Sending webhook request", event_type=event_data.get('event_type'), event_id=event_data.get('id'))
        response = requests.post(
            subscription.url,
            headers=headers,
            data=payload_json.encode('utf-8'),
            timeout=AppConfig.WEBHOOK_TIMEOUT or DEFAULT_WEBHOOK_TIMEOUT
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        task_logger.info("Webhook sent successfully", status_code=response.status_code)
        _update_webhook_stats_in_redis(get_redis_client(), webhook_id, success=True)
        return {"status": "success", "status_code": response.status_code}

    except requests.exceptions.Timeout as e:
        task_logger.warning("Webhook request timed out", error=str(e))
        error_msg = f"Timeout error: {e}"
        _update_webhook_stats_in_redis(get_redis_client(), webhook_id, success=False, error_message=error_msg)
        raise # Re-raise to allow Celery retry
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else None
        task_logger.error("Webhook request failed", status_code=status_code, error=str(e), exc_info=True)
        error_msg = f"Request failed ({status_code}): {e}"
        _update_webhook_stats_in_redis(get_redis_client(), webhook_id, success=False, error_message=error_msg)
        raise # Re-raise to allow Celery retry
    except WebhookDataError as e:
        # Catch errors raised earlier (serialization, signature)
        task_logger.warning("Webhook task failed due to data processing error before sending", error=str(e))
        _update_webhook_stats_in_redis(get_redis_client(), webhook_id, success=False, error_message=f"Data processing error: {e}") # Use specific error
        raise # Re-raise the specific data error (don't retry)
    except Exception as e:
        task_logger.error("Unexpected error during webhook sending", error=str(e), exc_info=True)
        error_msg = f"Unexpected error: {e}"
        _update_webhook_stats_in_redis(get_redis_client(), webhook_id, success=False, error_message=error_msg)
        raise # Re-raise to allow Celery retry or mark as failure

@celery_app.task(
    bind=True,
    # Define exceptions for automatic retry
    autoretry_for=(requests.exceptions.RequestException, WebhookDataError),
    retry_backoff=True,
    # Remove retry_kwargs - access config inside task if needed for explicit retry calls
    # retry_kwargs={'max_retries': AppConfig.WEBHOOK_MAX_RETRIES if AppConfig else DEFAULT_MAX_RETRY_COUNT}
)
def send_webhook_task(self: Task, subscription_dict: Dict, event_data: Dict):
    """Celery task wrapper to send a single webhook event."""
    task_logger = structlog.get_logger(f"task.{self.request.id or 'webhook'}")
    try:
        return _execute_send_webhook(task_logger, subscription_dict, event_data)
    except WebhookDataError as e:
        # Log data errors specifically, Celery will mark as failed due to unhandled exc
        task_logger.error("Webhook task failed permanently due to data error", error=str(e), exc_info=True)
        # No need to update state here, let Celery handle it
        raise # Re-raise so Celery marks as FAILURE
    except Exception as e:
        task_logger.error("Unhandled exception reached task wrapper", error=str(e), exc_info=True)
        # For unexpected errors that might not be caught by autoretry_for,
        # ensure state is updated if needed, though Celery usually does.
        try:
            self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        except Exception as state_exc:
            task_logger.error("Failed to update task state after unhandled exception", state_update_error=str(state_exc))
        raise # Re-raise so Celery marks as FAILURE 