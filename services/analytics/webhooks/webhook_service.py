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
# Import SSRF filter
# from ssrf_filter import validate as validate_url_ssrf, FilterError

# Import the schema
from .schemas import WebhookSubscription, WebhookDataError

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

class InvalidWebhookURLError(WebhookServiceError):
    """Webhook URL is invalid or disallowed."""
    pass

class WebhookNotFoundError(WebhookServiceError):
    """Webhook subscription was not found."""
    pass

class WebhookDatabaseError(WebhookServiceError):
    """Error interacting with the underlying data store (Redis)."""
    pass

class WebhookAuthorizationError(WebhookServiceError):
    """Authorization failure related to webhook access."""
    pass

# --- End Custom Exceptions ---

# --- Added imports for new URL validation ---
import socket
import ipaddress
from urllib.parse import urlparse
# --- End added imports ---

# --- Start New URL Validation Function ---
def _validate_webhook_url(url: str):
    """
    Validates a webhook URL to prevent SSRF attacks.
    Checks if the hostname resolves to a private, loopback, or unspecified IP address.
    Raises InvalidWebhookURLError if validation fails.
    """
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname

        if not hostname:
            raise InvalidWebhookURLError(f"Invalid URL: Could not parse hostname from '{url}'")

        # Resolve hostname to IP addresses (supports IPv4 and IPv6)
        # Use socket.getaddrinfo for robust resolution
        addr_info = socket.getaddrinfo(hostname, parsed_url.port or (443 if parsed_url.scheme == 'https' else 80))
        
        if not addr_info:
             raise InvalidWebhookURLError(f"Could not resolve hostname: '{hostname}'")

        for family, socktype, proto, canonname, sockaddr in addr_info:
            ip_addr_str = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip_addr_str)
                # Check if the IP address is private, loopback, link-local, or unspecified
                if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_unspecified:
                    logger.warning("Webhook URL resolved to non-public IP", url=url, hostname=hostname, resolved_ip=ip_addr_str)
                    raise InvalidWebhookURLError(
                        f"Webhook URL hostname '{hostname}' resolves to a disallowed IP address: {ip_addr_str}"
                    )
                logger.debug("Webhook URL resolved to allowed public IP", url=url, hostname=hostname, resolved_ip=ip_addr_str)
            except ValueError:
                # Should not happen if getaddrinfo returns valid IPs, but handle anyway
                logger.error("Error parsing IP address returned by getaddrinfo", ip_str=ip_addr_str)
                raise InvalidWebhookURLError(f"Error validating IP address {ip_addr_str} for hostname '{hostname}'")

        logger.info("Webhook URL validation successful", url=url, hostname=hostname)

    except socket.gaierror as e:
        logger.warning("Webhook URL hostname resolution failed", url=url, hostname=hostname, error=str(e))
        raise InvalidWebhookURLError(f"Could not resolve hostname '{hostname}': {e}") from e
    except InvalidWebhookURLError: # Re-raise specific errors
        raise
    except Exception as e:
        # Catch other potential errors (parsing, etc.)
        logger.error("Unexpected error during webhook URL validation", url=url, error=str(e), exc_info=True)
        raise InvalidWebhookURLError(f"An unexpected error occurred validating URL '{url}': {e}") from e
# --- End New URL Validation Function ---

class WebhookService:
    """Service class for managing and triggering webhooks."""

    def __init__(self, config: Dict[str, Any], redis_client: Optional[redis.Redis]):
        """Initialize the WebhookService with an injected Redis client."""
        self.config = config
        self.redis_client = redis_client
        self.webhook_ttl = self.config.get('WEBHOOK_TTL_SECONDS', DEFAULT_WEBHOOK_TTL)
        self.max_retries = self.config.get('WEBHOOK_MAX_RETRIES', DEFAULT_MAX_RETRY_COUNT)
        self.timeout = self.config.get('WEBHOOK_TIMEOUT_SECONDS', DEFAULT_WEBHOOK_TIMEOUT)
        self.user_agent = self.config.get('WEBHOOK_USER_AGENT', DEFAULT_WEBHOOK_USER_AGENT)

        # Log status based on the injected client
        if self.redis_client:
             try:
                 if self.redis_client.ping():
                      logger.info("WebhookService initialized with active Redis client.")
                 else:
                      logger.warning("WebhookService initialized, but injected Redis client failed ping. Webhooks may fail.")
             except redis.exceptions.RedisError as e:
                  logger.error("WebhookService: Error pinging injected Redis client during init.", error=str(e))
        else:
             logger.warning("WebhookService initialized without a Redis client. Webhooks disabled.")

    def create_webhook(self, subscription: WebhookSubscription) -> Optional[str]:
        """
        Create a new webhook subscription and store it in Redis.

        Returns:
            The webhook ID if successful.

        Raises:
            WebhookDatabaseError: If a Redis error occurs.
            WebhookServiceError: For other unexpected errors.
            InvalidWebhookURLError: If the webhook URL fails SSRF validation.
        """
        if not self.redis_client:
            logger.error("Cannot create webhook: Redis client not available.")
            raise WebhookDatabaseError("Redis client not available.")

        try:
            # >> Replace URL Validation here <<
            try:
                # validate_url_ssrf(subscription.url) # Old validation
                _validate_webhook_url(subscription.url) # New validation
                logger.debug("Webhook URL passed SSRF validation.", url=subscription.url)
            # except FilterError as e: # Old exception
            except InvalidWebhookURLError as e: # Catch specific validation error
                logger.warning("Webhook URL failed validation", url=subscription.url, error=str(e))
                raise e # Re-raise the specific error
            except Exception as e: # Catch other potential validation errors
                logger.error("Unexpected error during URL validation", url=subscription.url, error=str(e), exc_info=True)
                # Wrap in our specific error type
                raise InvalidWebhookURLError(f"Unexpected error validating URL '{subscription.url}': {e}") from e

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
            InvalidWebhookURLError: If the new URL fails SSRF validation.
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
                            new_url = updates['url']
                            if not isinstance(new_url, str) or not new_url.startswith(('http://', 'https://')):
                                 raise ValueError("Invalid new webhook URL provided.")
                            # >> Validate the new URL <<
                            try:
                                # validate_url_ssrf(new_url) # Old validation
                                _validate_webhook_url(new_url) # New validation
                            except InvalidWebhookURLError as e:
                                logger.warning("New webhook URL failed validation during update", url=new_url, error=str(e))
                                raise e # Re-raise
                            except Exception as e:
                                logger.error("Unexpected error during new URL validation in update", url=new_url, error=str(e), exc_info=True)
                                raise InvalidWebhookURLError(f"Unexpected error validating new URL '{new_url}': {e}") from e
                            # Validation passed
                            subscription.url = new_url
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
