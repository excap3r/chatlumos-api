#!/usr/bin/env python3
"""
Webhook Service for Analytics

This module provides webhook functionality for real-time analytics events.
Subscribers can register webhooks to receive notifications about specific events.
"""

import os
import json
import uuid
import logging
import requests
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('webhook_service')

# Initialize Redis for webhooks if available
redis_url = os.getenv("REDIS_URL")
redis_client = None
if redis_url:
    try:
        redis_client = redis.from_url(redis_url)
        logger.info(f"Connected to Redis at {redis_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")

# Constants
WEBHOOK_TTL = 60 * 60 * 24 * 30  # 30 days retention
MAX_RETRY_COUNT = 3
WEBHOOK_TIMEOUT = 5  # seconds

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
        """
        Initialize webhook subscription.
        
        Args:
            url: Webhook URL
            event_types: List of event types to subscribe to
            owner_id: User ID of the subscription owner
            secret: Optional secret for signing webhook payloads
            description: Optional description
            enabled: Whether the webhook is enabled
        """
        self.id = str(uuid.uuid4())
        self.url = url
        self.event_types = event_types
        self.owner_id = owner_id
        self.secret = secret
        self.description = description
        self.enabled = enabled
        self.created_at = datetime.utcnow().isoformat()
        self.last_triggered = None
        self.success_count = 0
        self.failure_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert subscription to dictionary"""
        return {
            "id": self.id,
            "url": self.url,
            "event_types": self.event_types,
            "owner_id": self.owner_id,
            "description": self.description,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_triggered": self.last_triggered,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebhookSubscription':
        """Create subscription from dictionary"""
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
        subscription.success_count = data.get("success_count", 0)
        subscription.failure_count = data.get("failure_count", 0)
        return subscription

def create_webhook(subscription: WebhookSubscription) -> bool:
    """
    Create a new webhook subscription.
    
    Args:
        subscription: WebhookSubscription to create
        
    Returns:
        Success status
    """
    if not redis_client:
        logger.warning("Redis not available, webhook not created")
        return False
    
    try:
        # Store subscription in Redis
        key = f"webhook:subscription:{subscription.id}"
        redis_client.set(key, json.dumps(subscription.to_dict()))
        redis_client.expire(key, WEBHOOK_TTL)
        
        # Also store in owner-specific list
        owner_key = f"webhook:owner:{subscription.owner_id}"
        redis_client.sadd(owner_key, subscription.id)
        redis_client.expire(owner_key, WEBHOOK_TTL)
        
        # Store in event type indexes for fast lookup
        for event_type in subscription.event_types:
            event_key = f"webhook:event:{event_type}"
            redis_client.sadd(event_key, subscription.id)
            redis_client.expire(event_key, WEBHOOK_TTL)
        
        return True
    except Exception as e:
        logger.error(f"Error creating webhook: {e}")
        return False

def get_webhook(webhook_id: str) -> Optional[WebhookSubscription]:
    """
    Get a webhook subscription by ID.
    
    Args:
        webhook_id: Webhook subscription ID
        
    Returns:
        WebhookSubscription if found, None otherwise
    """
    if not redis_client:
        logger.warning("Redis not available, cannot retrieve webhook")
        return None
    
    try:
        key = f"webhook:subscription:{webhook_id}"
        data = redis_client.get(key)
        if data:
            return WebhookSubscription.from_dict(json.loads(data))
        return None
    except Exception as e:
        logger.error(f"Error retrieving webhook: {e}")
        return None

def get_webhooks_by_owner(owner_id: str) -> List[WebhookSubscription]:
    """
    Get webhook subscriptions by owner ID.
    
    Args:
        owner_id: Owner user ID
        
    Returns:
        List of WebhookSubscription objects
    """
    if not redis_client:
        logger.warning("Redis not available, cannot retrieve webhooks")
        return []
    
    try:
        owner_key = f"webhook:owner:{owner_id}"
        webhook_ids = redis_client.smembers(owner_key)
        
        subscriptions = []
        for webhook_id in webhook_ids:
            subscription = get_webhook(webhook_id.decode('utf-8') if isinstance(webhook_id, bytes) else webhook_id)
            if subscription:
                subscriptions.append(subscription)
        
        return subscriptions
    except Exception as e:
        logger.error(f"Error retrieving webhooks by owner: {e}")
        return []

def get_webhooks_by_event(event_type: str) -> List[WebhookSubscription]:
    """
    Get webhook subscriptions by event type.
    
    Args:
        event_type: Event type
        
    Returns:
        List of WebhookSubscription objects
    """
    if not redis_client:
        logger.warning("Redis not available, cannot retrieve webhooks")
        return []
    
    try:
        event_key = f"webhook:event:{event_type}"
        webhook_ids = redis_client.smembers(event_key)
        
        subscriptions = []
        for webhook_id in webhook_ids:
            subscription = get_webhook(webhook_id.decode('utf-8') if isinstance(webhook_id, bytes) else webhook_id)
            if subscription and subscription.enabled:
                subscriptions.append(subscription)
        
        return subscriptions
    except Exception as e:
        logger.error(f"Error retrieving webhooks by event: {e}")
        return []

def update_webhook(subscription: WebhookSubscription) -> bool:
    """
    Update a webhook subscription.
    
    Args:
        subscription: WebhookSubscription to update
        
    Returns:
        Success status
    """
    if not redis_client:
        logger.warning("Redis not available, webhook not updated")
        return False
    
    try:
        # Get the existing subscription to check for event type changes
        existing = get_webhook(subscription.id)
        if not existing:
            return False
        
        # Update event type indexes if needed
        if set(existing.event_types) != set(subscription.event_types):
            # Remove from old event type indexes
            for event_type in existing.event_types:
                event_key = f"webhook:event:{event_type}"
                redis_client.srem(event_key, subscription.id)
            
            # Add to new event type indexes
            for event_type in subscription.event_types:
                event_key = f"webhook:event:{event_type}"
                redis_client.sadd(event_key, subscription.id)
                redis_client.expire(event_key, WEBHOOK_TTL)
        
        # Store updated subscription
        key = f"webhook:subscription:{subscription.id}"
        redis_client.set(key, json.dumps(subscription.to_dict()))
        redis_client.expire(key, WEBHOOK_TTL)
        
        return True
    except Exception as e:
        logger.error(f"Error updating webhook: {e}")
        return False

def delete_webhook(webhook_id: str, owner_id: str) -> bool:
    """
    Delete a webhook subscription.
    
    Args:
        webhook_id: Webhook subscription ID
        owner_id: Owner user ID for validation
        
    Returns:
        Success status
    """
    if not redis_client:
        logger.warning("Redis not available, webhook not deleted")
        return False
    
    try:
        # Get the subscription
        subscription = get_webhook(webhook_id)
        if not subscription:
            return False
        
        # Verify ownership
        if subscription.owner_id != owner_id:
            logger.warning(f"Unauthorized webhook deletion attempt: {webhook_id} by {owner_id}")
            return False
        
        # Remove from event type indexes
        for event_type in subscription.event_types:
            event_key = f"webhook:event:{event_type}"
            redis_client.srem(event_key, webhook_id)
        
        # Remove from owner index
        owner_key = f"webhook:owner:{owner_id}"
        redis_client.srem(owner_key, webhook_id)
        
        # Delete the subscription
        key = f"webhook:subscription:{webhook_id}"
        redis_client.delete(key)
        
        return True
    except Exception as e:
        logger.error(f"Error deleting webhook: {e}")
        return False

def trigger_webhook(subscription: WebhookSubscription, event: Dict[str, Any]) -> bool:
    """
    Trigger a webhook with an event payload.
    
    Args:
        subscription: WebhookSubscription to trigger
        event: Event data to send
        
    Returns:
        Success status
    """
    if not subscription.enabled:
        return False
    
    # Prepare payload
    payload = {
        "event": event,
        "webhook_id": subscription.id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add signature if secret is available
    if subscription.secret:
        import hmac
        import hashlib
        payload_str = json.dumps(payload)
        signature = hmac.new(
            subscription.secret.encode('utf-8'),
            payload_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature
        }
    else:
        headers = {
            "Content-Type": "application/json"
        }
    
    # Send the webhook in a separate thread
    threading.Thread(
        target=_send_webhook, 
        args=(subscription, payload, headers),
        daemon=True
    ).start()
    
    return True

def _send_webhook(subscription: WebhookSubscription, payload: Dict[str, Any], headers: Dict[str, str]) -> None:
    """
    Send webhook request with retry logic.
    
    Args:
        subscription: WebhookSubscription
        payload: Event payload
        headers: Request headers
    """
    # Update last triggered time
    subscription.last_triggered = datetime.utcnow().isoformat()
    
    # Attempt to send the webhook with retries
    success = False
    retry_count = 0
    
    while not success and retry_count < MAX_RETRY_COUNT:
        try:
            response = requests.post(
                subscription.url,
                json=payload,
                headers=headers,
                timeout=WEBHOOK_TIMEOUT
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                success = True
                subscription.success_count += 1
                logger.info(f"Webhook {subscription.id} triggered successfully")
            else:
                retry_count += 1
                subscription.failure_count += 1
                logger.warning(f"Webhook {subscription.id} failed with status {response.status_code}: {response.text}")
                time.sleep(2 ** retry_count)  # Exponential backoff
        except Exception as e:
            retry_count += 1
            subscription.failure_count += 1
            logger.error(f"Error triggering webhook {subscription.id}: {e}")
            time.sleep(2 ** retry_count)  # Exponential backoff
    
    # Update subscription stats
    update_webhook(subscription)

def trigger_event(event_type: str, event_data: Dict[str, Any]) -> int:
    """
    Trigger all webhooks subscribed to an event type.
    
    Args:
        event_type: Event type
        event_data: Event data
        
    Returns:
        Number of webhooks triggered
    """
    if not redis_client:
        logger.warning("Redis not available, webhooks not triggered")
        return 0
    
    try:
        # Get subscriptions for this event type
        subscriptions = get_webhooks_by_event(event_type)
        
        # Create the event payload
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Trigger each webhook
        trigger_count = 0
        for subscription in subscriptions:
            if trigger_webhook(subscription, event):
                trigger_count += 1
        
        return trigger_count
    except Exception as e:
        logger.error(f"Error triggering event {event_type}: {e}")
        return 0

# API Endpoints

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "webhook-service",
        "redis_connected": redis_client is not None
    })

@app.route('/trigger', methods=['POST'])
def trigger_webhook_endpoint():
    """Trigger webhooks for an event"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        event_type = data.get("event_type")
        if not event_type:
            return jsonify({"error": "Event type is required"}), 400
        
        event_data = data.get("data", {})
        
        # Trigger webhooks
        trigger_count = trigger_event(event_type, event_data)
        
        return jsonify({
            "success": True,
            "triggered_count": trigger_count,
            "event_type": event_type
        })
    except Exception as e:
        logger.error(f"Error processing trigger request: {e}")
        return jsonify({
            "error": "Failed to trigger webhooks",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5007))) 