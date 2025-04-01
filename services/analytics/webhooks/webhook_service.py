#!/usr/bin/env python3
"""
Webhook Service Module

Provides webhook subscription management and triggering functionality.
Relies on a Redis client provided through the Flask application context.
"""

import os
import json
import uuid
import requests
import threading
import hmac
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from flask import current_app # Keep current_app
import redis
import structlog

# Import the new Celery task
from services.tasks.webhook_tasks import send_webhook_task

# Configure logger
logger = structlog.get_logger(__name__)

# Constants
DEFAULT_WEBHOOK_TTL = 60 * 60 * 24 * 30  # 30 days
DEFAULT_MAX_RETRY_COUNT = 3
DEFAULT_WEBHOOK_TIMEOUT = 5  # seconds
DEFAULT_WEBHOOK_USER_AGENT = "PDFWisdomExtractor-Webhook/1.0"

# --- Custom Exceptions ---
class WebhookServiceError(Exception):
    """Base class for webhook service errors."""
    pass

class WebhookNotFoundError(WebhookServiceError):
    """Webhook subscription was not found."""
    pass

class WebhookDatabaseError(WebhookServiceError):
    """Error interacting with the underlying data store (Redis)."""
    pass

class WebhookDataError(WebhookServiceError):
    """Error decoding or validating webhook data from storage."""
    pass

class WebhookAuthorizationError(WebhookServiceError):
    """Authorization failure related to webhook access."""
    pass

# --- End Custom Exceptions ---

class WebhookSubscription:
    """Webhook subscription data structure"""

    def __init__(
        self,
        url: str,
        event_types: List[str],
        owner_id: str,
        secret: Optional[str] = None,
        description: Optional[str] = None,
        enabled: bool = True
    ):
        if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
             raise ValueError("Invalid webhook URL provided.")
        if not event_types or not isinstance(event_types, list):
             raise ValueError("Event types must be a non-empty list.")
        if not owner_id or not isinstance(owner_id, str):
             raise ValueError("Invalid owner ID provided.")

        self.id = str(uuid.uuid4())
        self.url = url
        self.event_types = list(set(event_types))
        self.owner_id = owner_id
        self.secret = secret
        self.description = description
        self.enabled = enabled
        self.created_at = datetime.utcnow().isoformat()
        self.last_triggered = None
        self.last_success = None
        self.last_failure = None
        self.success_count = 0
        self.failure_count = 0
        self.last_error = None

    def to_dict(self, include_secret: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "url": self.url,
            "event_types": self.event_types,
            "owner_id": self.owner_id,
            "description": self.description,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_triggered": self.last_triggered,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_error": self.last_error
        }
        if include_secret:
             data['secret'] = self.secret
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebhookSubscription':
        try:
            subscription = cls(
                url=data["url"],
                event_types=data["event_types"],
                owner_id=data["owner_id"],
                secret=data.get("secret"),
                description=data.get("description"),
                enabled=data.get("enabled", True)
            )
            subscription.id = data["id"]
            subscription.created_at = data["created_at"]
            subscription.last_triggered = data.get("last_triggered")
            subscription.last_success = data.get("last_success")
            subscription.last_failure = data.get("last_failure")
            subscription.success_count = data.get("success_count", 0)
            subscription.failure_count = data.get("failure_count", 0)
            subscription.last_error = data.get("last_error")
            return subscription
        except KeyError as e:
            logger.error("Missing required field in webhook data from Redis", missing_key=str(e), data_keys=list(data.keys()))
            raise WebhookDataError(f"Invalid webhook data from storage: Missing key '{e}'") from e
        except Exception as e:
             logger.error("Error creating WebhookSubscription from dict", error=str(e), data_preview=str(data)[:200])
             raise WebhookDataError(f"Error processing webhook data: {e}") from e

class WebhookService:
    """Service class for managing and triggering webhooks."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the WebhookService."""
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.webhook_ttl = self.config.get('WEBHOOK_TTL_SECONDS', DEFAULT_WEBHOOK_TTL)
        self.max_retries = self.config.get('WEBHOOK_MAX_RETRIES', DEFAULT_MAX_RETRY_COUNT)
        self.timeout = self.config.get('WEBHOOK_TIMEOUT_SECONDS', DEFAULT_WEBHOOK_TIMEOUT)
        self.user_agent = self.config.get('WEBHOOK_USER_AGENT', DEFAULT_WEBHOOK_USER_AGENT)

        # Get Redis client instance from current_app
        try:
            if hasattr(current_app, 'redis_client') and current_app.redis_client:
                self.redis_client = current_app.redis_client
                if self.redis_client.ping():
                    logger.info("WebhookService connected to Redis successfully.")
                else:
                    logger.warning("WebhookService: Redis client found but ping failed. Webhooks disabled.")
                    self.redis_client = None
            else:
                logger.warning("WebhookService: Redis client not found on current_app. Webhooks disabled.")
        except RuntimeError:
            logger.error("WebhookService init error: Not in a Flask application context.")
        except redis.exceptions.ConnectionError as e:
            logger.error("WebhookService init error: Redis connection failed.", error=str(e))
        except Exception as e:
            logger.error("WebhookService init error: Unexpected error getting Redis client.", error=str(e), exc_info=True)

    def create_webhook(self, subscription: WebhookSubscription) -> Optional[str]:
        """
        Create a new webhook subscription and store it in Redis.

        Returns:
            The webhook ID if successful.

        Raises:
            WebhookDatabaseError: If a Redis error occurs.
            WebhookServiceError: For other unexpected errors.
        """
        if not self.redis_client:
            logger.error("Cannot create webhook: Redis client not available.")
            raise WebhookDatabaseError("Redis client not available.")

        try:
            key = f"webhook:subscription:{subscription.id}"
            owner_key = f"webhook:owner:{subscription.owner_id}"
            pipe = self.redis_client.pipeline()
            pipe.set(key, json.dumps(subscription.to_dict(include_secret=True)))
            pipe.sadd(owner_key, subscription.id)
            for event_type in subscription.event_types:
                event_key = f"webhook:event:{event_type}"
                pipe.sadd(event_key, subscription.id)
            pipe.execute()
            logger.info("Webhook subscription created", webhook_id=subscription.id, owner_id=subscription.owner_id)
            return subscription.id
        except redis.exceptions.RedisError as e:
            logger.error("Redis error creating webhook", owner_id=subscription.owner_id, error=str(e), exc_info=True)
            raise WebhookDatabaseError(f"Redis error creating webhook: {e}") from e
        except Exception as e:
            logger.error("Unexpected error creating webhook", owner_id=subscription.owner_id, error=str(e), exc_info=True)
            raise WebhookServiceError(f"Unexpected error creating webhook: {e}") from e

    def get_webhook(self, webhook_id: str) -> Optional[WebhookSubscription]:
        """
        Get a webhook subscription by ID from Redis.

        Returns:
            The WebhookSubscription object if found.

        Raises:
            WebhookNotFoundError: If the webhook_id does not exist.
            WebhookDataError: If stored data is invalid.
            WebhookDatabaseError: If a Redis error occurs.
            WebhookServiceError: For other unexpected errors.
        """
        if not self.redis_client:
            logger.error("Cannot get webhook: Redis client not available.")
            raise WebhookDatabaseError("Redis client not available.")
        try:
            key = f"webhook:subscription:{webhook_id}"
            data = self.redis_client.get(key)
            if data:
                 logger.debug("Webhook subscription retrieved", webhook_id=webhook_id)
                 return WebhookSubscription.from_dict(json.loads(data))
            else:
                 logger.info("Webhook subscription not found", webhook_id=webhook_id)
                 raise WebhookNotFoundError(f"Webhook with ID '{webhook_id}' not found.")
        except redis.exceptions.RedisError as e:
            logger.error("Redis error retrieving webhook", webhook_id=webhook_id, error=str(e), exc_info=True)
            raise WebhookDatabaseError(f"Redis error retrieving webhook: {e}") from e
        except (json.JSONDecodeError, WebhookDataError) as e: # Catch decode errors and errors from from_dict
             logger.error("Error decoding or validating webhook data from Redis", webhook_id=webhook_id, error=str(e))
             raise WebhookDataError(f"Invalid data for webhook '{webhook_id}': {e}") from e
        except WebhookNotFoundError:
            raise # Re-raise not found specifically
        except Exception as e:
            logger.error("Unexpected error retrieving webhook", webhook_id=webhook_id, error=str(e), exc_info=True)
            raise WebhookServiceError(f"Unexpected error retrieving webhook: {e}") from e

    def get_webhooks_by_owner(self, owner_id: str) -> List[WebhookSubscription]:
        """Get all webhook subscriptions for a specific owner."""
        if not self.redis_client:
            logger.error("Cannot get webhooks by owner: Redis client not available.")
            return []
        subscriptions = []
        try:
            owner_key = f"webhook:owner:{owner_id}"
            webhook_ids_bytes = self.redis_client.smembers(owner_key)
            webhook_ids = [wid.decode('utf-8') for wid in webhook_ids_bytes]

            if not webhook_ids:
                 logger.debug("No webhooks found for owner", owner_id=owner_id)
                 return []

            # Fetch subscriptions in batch
            keys = [f"webhook:subscription:{wid}" for wid in webhook_ids]
            results = self.redis_client.mget(keys)

            for i, data in enumerate(results):
                if data:
                    try:
                        subscriptions.append(WebhookSubscription.from_dict(json.loads(data)))
                    except (json.JSONDecodeError, ValueError) as e:
                         logger.error("Error decoding/validating webhook data for owner", webhook_id=webhook_ids[i], owner_id=owner_id, error=str(e))
                else:
                     logger.warning("Webhook ID found in owner set but subscription data missing", webhook_id=webhook_ids[i], owner_id=owner_id)
            
            logger.info("Retrieved webhooks for owner", owner_id=owner_id, count=len(subscriptions))
            return subscriptions
        except redis.exceptions.RedisError as e:
            logger.error("Redis error retrieving webhooks by owner", owner_id=owner_id, error=str(e), exc_info=True)
            return []
        except Exception as e:
            logger.error("Unexpected error retrieving webhooks by owner", owner_id=owner_id, error=str(e), exc_info=True)
            return []

    def get_webhooks_by_event(self, event_type: str) -> List[WebhookSubscription]:
        """Get all enabled webhook subscriptions for a specific event type."""
        if not self.redis_client:
            logger.error("Cannot get webhooks by event: Redis client not available.")
            return []
        enabled_subscriptions = []
        try:
            event_key = f"webhook:event:{event_type}"
            webhook_ids_bytes = self.redis_client.smembers(event_key)
            webhook_ids = [wid.decode('utf-8') for wid in webhook_ids_bytes]

            if not webhook_ids:
                 logger.debug("No webhooks registered for event type", event_type=event_type)
                 return []

            keys = [f"webhook:subscription:{wid}" for wid in webhook_ids]
            results = self.redis_client.mget(keys)

            for i, data in enumerate(results):
                if data:
                    try:
                        subscription = WebhookSubscription.from_dict(json.loads(data))
                        if subscription.enabled:
                            enabled_subscriptions.append(subscription)
                    except (json.JSONDecodeError, ValueError) as e:
                         logger.error("Error decoding/validating webhook data for event", webhook_id=webhook_ids[i], event_type=event_type, error=str(e))
                else:
                     logger.warning("Webhook ID found in event set but subscription data missing", webhook_id=webhook_ids[i], event_type=event_type)
            
            logger.debug("Retrieved enabled webhooks for event type", event_type=event_type, count=len(enabled_subscriptions))
            return enabled_subscriptions
        except redis.exceptions.RedisError as e:
            logger.error("Redis error retrieving webhooks by event", event_type=event_type, error=str(e), exc_info=True)
            return []
        except Exception as e:
            logger.error("Unexpected error retrieving webhooks by event", event_type=event_type, error=str(e), exc_info=True)
            return []

    def update_webhook(self, webhook_id: str, updates: Dict[str, Any]) -> WebhookSubscription:
        """
        Update an existing webhook subscription.
        Supports updating url, event_types, description, secret, enabled status.

        Returns:
            The updated WebhookSubscription object.

        Raises:
            WebhookNotFoundError: If webhook_id not found.
            WebhookDataError: If validation/decoding fails.
            WebhookDatabaseError: On Redis errors.
            WebhookServiceError: For other unexpected errors.
        """
        if not self.redis_client:
            logger.error("Cannot update webhook: Redis client not available.")
            raise WebhookDatabaseError("Redis client not available.")

        key = f"webhook:subscription:{webhook_id}"
        subscription_to_return = None # Variable to hold the final subscription state
        watch_error_retries = 3 # Limit retries for WatchError

        try:
            # Use WATCH/MULTI/EXEC for transactional update
            with self.redis_client.pipeline() as pipe:
                for attempt in range(watch_error_retries + 1):
                    try:
                        pipe.watch(key) # Watch for changes
                        current_data_bytes = pipe.get(key)
                        if not current_data_bytes:
                            logger.warning("Webhook not found during update attempt", webhook_id=webhook_id)
                            pipe.unwatch()
                            raise WebhookNotFoundError(f"Webhook with ID '{webhook_id}' not found.")

                        current_data = json.loads(current_data_bytes)
                        subscription = WebhookSubscription.from_dict(current_data)
                        original_event_types = set(subscription.event_types)
                        
                        # Apply updates
                        updated = False
                        types_to_add = set()
                        types_to_remove = set()

                        if 'url' in updates and updates['url'] != subscription.url:
                            if not isinstance(updates['url'], str) or not updates['url'].startswith(('http://', 'https://')):
                                 raise ValueError("Invalid new webhook URL provided.")
                            subscription.url = updates['url']
                            updated = True
                        if 'description' in updates and updates['description'] != subscription.description:
                            subscription.description = updates['description']
                            updated = True
                        if 'secret' in updates:
                            subscription.secret = updates['secret']
                            updated = True
                        if 'enabled' in updates and updates['enabled'] != subscription.enabled:
                            subscription.enabled = bool(updates['enabled'])
                            updated = True
                        if 'event_types' in updates:
                             new_event_types = set(updates['event_types'])
                             if new_event_types != original_event_types:
                                  subscription.event_types = list(new_event_types)
                                  updated = True
                                  types_to_add = new_event_types - original_event_types
                                  types_to_remove = original_event_types - new_event_types

                        if not updated and not types_to_add and not types_to_remove:
                             logger.info("No changes detected for webhook update", webhook_id=webhook_id)
                             pipe.unwatch()
                             subscription_to_return = subscription # Store current state
                             return subscription # Return current state
                             
                        # Start transaction
                        pipe.multi()
                        pipe.set(key, json.dumps(subscription.to_dict(include_secret=True)))
                        # Update event type indexes
                        for etype in types_to_add:
                             pipe.sadd(f"webhook:event:{etype}", webhook_id)
                        for etype in types_to_remove:
                             pipe.srem(f"webhook:event:{etype}", webhook_id)

                        # Execute transaction
                        results = pipe.execute()
                        logger.info("Webhook updated successfully", webhook_id=webhook_id)
                        subscription_to_return = subscription # Store updated state
                        return subscription # Return updated object
                    
                    except redis.exceptions.WatchError:
                        # Key changed during WATCH, loop will retry
                        logger.warning("WatchError during webhook update, retrying...", webhook_id=webhook_id)
                        if attempt < watch_error_retries:
                             continue # Retry
                        else:
                             logger.error("Webhook update failed after multiple WatchError retries", webhook_id=webhook_id)
                             raise WebhookDatabaseError("Webhook update failed due to concurrent modifications.")
                    # Need to handle exceptions *within* the loop try block
                    # OR break out of the loop to handle them outside.
                    except (json.JSONDecodeError, ValueError) as e:
                         logger.error("Error decoding/validating webhook data during update", webhook_id=webhook_id, error=str(e))
                         # Need to unwatch before raising/returning
                         pipe.unwatch()
                         raise WebhookDataError(f"Invalid data during update for webhook '{webhook_id}': {e}") from e
                    except Exception as e: # Catch other potential errors inside loop
                         logger.error("Unexpected error within webhook update transaction attempt", webhook_id=webhook_id, error=str(e), exc_info=True)
                         pipe.unwatch()
                         raise WebhookServiceError(f"Unexpected error during update attempt for webhook '{webhook_id}': {e}") from e

        except redis.exceptions.RedisError as e:
            # Catches errors initiating pipeline or connection errors outside the loop
            logger.error("Redis error during webhook update", webhook_id=webhook_id, error=str(e), exc_info=True)
            raise WebhookDatabaseError(f"Redis error during webhook update: {e}") from e
        except (WebhookNotFoundError, WebhookDataError, WebhookServiceError) as e:
            # Re-raise specific errors caught and re-raised from inner try/except
            raise e
        except Exception as e:
            # Catch any unexpected errors not handled above
            logger.error("Unexpected error during webhook update process", webhook_id=webhook_id, error=str(e), exc_info=True)
            raise WebhookServiceError(f"Unexpected error updating webhook '{webhook_id}': {e}") from e

    def delete_webhook(self, webhook_id: str, owner_id: str) -> None:
        """
        Delete a webhook subscription.
        Requires owner_id for verification.

        Raises:
            WebhookNotFoundError: If webhook_id not found.
            WebhookAuthorizationError: If owner_id does not match.
            WebhookDataError: If stored data is invalid.
            WebhookDatabaseError: If a Redis error occurs.
            WebhookServiceError: For other unexpected errors.
        """
        if not self.redis_client:
            logger.error("Cannot delete webhook: Redis client not available.")
            raise WebhookDatabaseError("Redis client not available.")

        key = f"webhook:subscription:{webhook_id}"
        try:
            # Get current data first to verify owner and get event types
            current_data_bytes = self.redis_client.get(key)
            if not current_data_bytes:
                logger.warning("Webhook not found for deletion", webhook_id=webhook_id)
                # Treat as not found, even if already deleted
                raise WebhookNotFoundError(f"Webhook with ID '{webhook_id}' not found for deletion.")

            subscription_data = json.loads(current_data_bytes)
            actual_owner_id = subscription_data.get('owner_id')
            if actual_owner_id != owner_id:
                logger.error("Permission denied: Attempt to delete webhook owned by another user",
                             webhook_id=webhook_id, request_owner_id=owner_id, actual_owner_id=actual_owner_id)
                raise WebhookAuthorizationError(f"User '{owner_id}' not authorized to delete webhook '{webhook_id}'.")

            # Proceed with deletion using pipeline
            event_types = subscription_data.get('event_types', [])
            owner_key = f"webhook:owner:{owner_id}"

            pipe = self.redis_client.pipeline()
            pipe.delete(key)
            pipe.srem(owner_key, webhook_id)
            for event_type in event_types:
                event_key = f"webhook:event:{event_type}"
                pipe.srem(event_key, webhook_id)

            results = pipe.execute()
            logger.info("Webhook deleted successfully", webhook_id=webhook_id, owner_id=owner_id)
            return # Return None on success

        except redis.exceptions.RedisError as e:
            logger.error("Redis error deleting webhook", webhook_id=webhook_id, owner_id=owner_id, error=str(e), exc_info=True)
            raise WebhookDatabaseError(f"Redis error deleting webhook: {e}") from e
        except (json.JSONDecodeError, ValueError) as e:
             logger.error("Error decoding webhook data during deletion check", webhook_id=webhook_id, error=str(e))
             # Consider this a data error if we can't verify owner due to bad JSON
             raise WebhookDataError(f"Invalid data for webhook '{webhook_id}' during delete check: {e}") from e
        except (WebhookNotFoundError, WebhookAuthorizationError):
             raise # Re-raise specific errors
        except Exception as e:
            logger.error("Unexpected error deleting webhook", webhook_id=webhook_id, owner_id=owner_id, error=str(e), exc_info=True)
            raise WebhookServiceError(f"Unexpected error deleting webhook: {e}") from e

    def send_test_event(self, subscription: WebhookSubscription, test_payload: Optional[Dict] = None) -> bool:
        """
        Sends a predefined test event to a specific webhook via the async task queue.

        Args:
            subscription: The WebhookSubscription object.
            test_payload: Optional custom payload for the test event.

        Returns:
            True if the task was queued successfully.

        Raises:
            WebhookServiceError: If the Celery task cannot be queued.
        """
        if not subscription.enabled:
            logger.info("Test event skipped: Webhook is disabled", webhook_id=subscription.id)
            # Return True as no error occurred, but nothing was sent.
            # Or maybe raise a specific exception/return False?
            # Let's return True for now, API layer handles the 200 response.
            return True

        default_payload = {"message": f"Test event for webhook {subscription.id} triggered successfully!"}
        payload_to_send = test_payload if test_payload is not None else default_payload

        event_data = {
            "event_id": str(uuid.uuid4()), # Unique ID for this specific event delivery
            "event_type": "test.event",
            "data": payload_to_send,
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            # Pass the full subscription data (including secret) and event data to the task
            send_webhook_task.delay(subscription.to_dict(include_secret=True), event_data)
            logger.info("Queued test event for webhook", webhook_id=subscription.id, event_id=event_data['event_id'])
            return True
        except Exception as e:
             # Catch potential errors during task queuing (e.g., broker connection issue)
             logger.error("Failed to queue webhook test task", webhook_id=subscription.id, error=str(e), exc_info=True)
             raise WebhookServiceError(f"Failed to queue test event task: {e}") from e

    def trigger_webhooks(self, event_type: str, event_data: Dict[str, Any]):
        """
        Finds relevant webhook subscriptions for an event and triggers Celery tasks to send them.
        """
        if not self.redis_client:
             logger.error("Cannot trigger webhooks: Redis client not available.")
             return
             
        subscriptions = self.get_webhooks_by_event(event_type)
        if not subscriptions:
            logger.debug("No enabled webhooks found for event type", event_type=event_type)
            return

        logger.info("Triggering webhook tasks for event", event_type=event_type, count=len(subscriptions), event_id=event_data.get('id'))

        for subscription in subscriptions:
            try:
                # Pass subscription as dict (incl. secret) and event data
                send_webhook_task.delay(
                    subscription_dict=subscription.to_dict(include_secret=True),
                    event_data=event_data
                )
                logger.debug("Enqueued webhook task", webhook_id=subscription.id, event_type=event_type)
            except Exception as e:
                # Log error if Celery enqueueing fails
                logger.error("Failed to enqueue webhook task", 
                             webhook_id=subscription.id, 
                             event_type=event_type, 
                             error=str(e), 
                             exc_info=True)
