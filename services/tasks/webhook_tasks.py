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

# Import the Celery app instance
from celery_app import celery_app
from services.config import AppConfig

# Added for type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import only for type checking to avoid circular dependency
    from services.analytics.webhooks.webhook_service import WebhookSubscription

logger = structlog.get_logger(__name__)

# Constants from WebhookService (or get from config passed to task)
DEFAULT_MAX_RETRY_COUNT = 3
DEFAULT_WEBHOOK_TIMEOUT = 5
DEFAULT_WEBHOOK_USER_AGENT = "PDFWisdomExtractor-Webhook/1.0"

def _generate_signature(payload_body: bytes, secret: str) -> str:
    """Generate HMAC-SHA256 signature for the payload."""
    if not isinstance(payload_body, bytes):
         payload_body = payload_body.encode('utf-8')
    if not isinstance(secret, str):
         logger.error("Webhook secret is not a string, cannot generate signature.")
         raise TypeError("Webhook secret must be a string")

    hashed = hmac.new(secret.encode('utf-8'), payload_body, hashlib.sha256)
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


# Define the Celery task
# Use bind=True to access self for retry logic
@celery_app.task(bind=True, 
                 autoretry_for=(requests.exceptions.RequestException,), 
                 retry_backoff=True, 
                 retry_backoff_max=60, # Max backoff seconds
                 retry_jitter=True)
def send_webhook_task(self: Task, subscription_dict: dict, event_data: dict):
    """
    Celery task to send a webhook payload asynchronously with retries.
    Handles signature generation and updates stats in Redis.
    """
    task_logger = structlog.get_logger(f"task.{self.request.id or 'webhook'}")
    redis_client = get_redis_client() # Get Redis client for stats update

    try:
        # Reconstruct subscription object (optional, could just use dict)
        subscription = WebhookSubscription.from_dict(subscription_dict)
    except (ValueError, KeyError) as e:
         task_logger.error("Failed to reconstruct WebhookSubscription from dict. Aborting task.", error=str(e), data_keys=subscription_dict.keys())
         # Cannot update stats if ID is missing or invalid
         return # Do not retry if data is invalid

    webhook_id = subscription.id
    task_logger = task_logger.bind(webhook_id=webhook_id, url=subscription.url) # Add context

    if not subscription.enabled:
        task_logger.info("Skipping disabled webhook.")
        return # Don't send if disabled

    # Prepare payload (same as in original service)
    payload = {
        "event_id": event_data.get('id'),
        "event_type": event_data.get('event_type'),
        "timestamp": event_data.get('timestamp'),
        "data": event_data
    }
    try:
        payload_json = json.dumps(payload)
        payload_bytes = payload_json.encode('utf-8')
    except TypeError as e:
         task_logger.error("Failed to serialize webhook payload to JSON. Aborting.", error=str(e), event_id=payload.get('event_id'))
         _update_webhook_stats_in_redis(redis_client, webhook_id, success=False, error_message=f"Payload serialization error: {e}")
         return # Do not retry if payload is invalid

    headers = {
        'Content-Type': 'application/json',
        'User-Agent': AppConfig.WEBHOOK_USER_AGENT or DEFAULT_WEBHOOK_USER_AGENT # Get from AppConfig
    }

    if subscription.secret:
        try:
            signature = _generate_signature(payload_bytes, subscription.secret)
            headers['X-Webhook-Signature-256'] = f"sha256={signature}"
            task_logger.debug("Generated webhook signature.")
        except TypeError as e:
            task_logger.error("Failed to generate signature (TypeError). Skipping signature.", error=str(e))
            # Continue without signature? Or fail? Let's continue for now.

    success = False
    last_error_message = None
    response_status = None
    response_preview = None

    task_logger.info("Attempting to send webhook payload", event_type=payload['event_type'], event_id=payload['event_id'])
    start_time = time.time()
    
    try:
        response = requests.post(
            subscription.url,
            headers=headers,
            data=payload_bytes,
            timeout=AppConfig.WEBHOOK_TIMEOUT_SECONDS or DEFAULT_WEBHOOK_TIMEOUT # Get from AppConfig
        )
        duration = (time.time() - start_time) * 1000
        response_status = response.status_code
        response_preview = response.text[:200]

        if 200 <= response.status_code < 300:
            success = True
            task_logger.info("Webhook delivered successfully.", 
                         status_code=response_status, duration_ms=duration, attempt=self.request.retries + 1)
        else:
            last_error_message = f"Request failed with status {response_status}. Response: {response_preview}"
            task_logger.warning("Webhook delivery failed (non-2xx status).", 
                         status_code=response_status, attempt=self.request.retries + 1)
            # Raise an exception to trigger Celery's autoretry based on status code?
            # For now, rely on RequestException autoretry. If status is 4xx, maybe don't retry?
            if 400 <= response.status_code < 500:
                 # Client error - unlikely to succeed on retry. Stop retrying.
                 task_logger.error("Webhook delivery failed with client error (4xx). Aborting retries.")
                 # Fall through to update stats as failure without raising/retrying
                 pass
            else:
                 # Server error (5xx) or other non-2xx - raise to retry
                 response.raise_for_status() # Raise HTTPError for non-2xx status

    except requests.exceptions.Timeout as e:
        # Caught by autoretry_for=(requests.exceptions.RequestException,)
        duration = (time.time() - start_time) * 1000
        last_error_message = f"Request timed out after {AppConfig.WEBHOOK_TIMEOUT_SECONDS or DEFAULT_WEBHOOK_TIMEOUT} seconds."
        task_logger.warning("Webhook delivery timeout.", duration_ms=duration, attempt=self.request.retries + 1)
        # Let Celery handle retry via raise e
        raise e 
    except requests.exceptions.RequestException as e:
         # Caught by autoretry_for - let Celery handle retry
        duration = (time.time() - start_time) * 1000
        last_error_message = f"Request failed: {str(e)}"
        task_logger.warning("Webhook delivery failed (request exception).", 
                             error=str(e), duration_ms=duration, attempt=self.request.retries + 1)
        raise e # Re-raise for Celery retry mechanism
    except Exception as e:
         # Catch unexpected errors not covered by autoretry
         duration = (time.time() - start_time) * 1000
         last_error_message = f"Unexpected error during webhook send: {str(e)}"
         task_logger.error("Unexpected error sending webhook payload. Aborting retries.", 
                            error=str(e), duration_ms=duration, attempt=self.request.retries + 1, exc_info=True)
         # Update stats directly as failed and do not retry
         _update_webhook_stats_in_redis(redis_client, webhook_id, success=False, error_message=last_error_message)
         # Explicitly update Celery state? For now, just return.
         return # Stop processing this task

    # Update stats only after final attempt (success or max retries exceeded)
    # Celery's retry mechanism handles the looping/waiting
    # This block runs *after* the try/except finishes, either successfully or after exhausting retries.
    if not success and self.request.retries >= (AppConfig.WEBHOOK_MAX_RETRIES or DEFAULT_MAX_RETRY_COUNT):
         task_logger.error("Webhook delivery failed after max retries.", max_retries=self.request.retries +1)
         # Use the last error message recorded during the attempts
         _update_webhook_stats_in_redis(redis_client, webhook_id, success=False, error_message=last_error_message)
    elif success:
         # Update stats on success
         _update_webhook_stats_in_redis(redis_client, webhook_id, success=True, error_message=None)
    # else: it was a failure but will be retried by Celery, stats updated on final failure. 