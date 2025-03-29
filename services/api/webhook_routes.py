#!/usr/bin/env python3
"""
Webhook API Routes

Provides API endpoints for webhook subscription management.
"""

import logging
from flask import Blueprint, request, jsonify, g
from typing import Dict, Any, List, Optional

# Import webhook services
from ..analytics.webhooks.webhook_service import (
    WebhookSubscription,
    create_webhook,
    get_webhook,
    get_webhooks_by_owner,
    update_webhook,
    delete_webhook,
    trigger_event
)

# Import authentication utilities
from ..auth_middleware import auth_required, admin_required

# Set up logging
logger = logging.getLogger("webhook_routes")

# Create Blueprint
webhook_bp = Blueprint("webhooks", __name__)

@webhook_bp.route("", methods=["POST"])
@auth_required()
def create_webhook_subscription():
    """
    Create a new webhook subscription.
    
    Request Body:
        url: Webhook URL
        event_types: List of event types to subscribe to
        secret: Optional secret for signing webhook payloads
        description: Optional description
        
    Returns:
        201: Webhook created successfully
        400: Invalid request
        401: Authentication required
    """
    data = request.get_json()
    
    # Validate request data
    if not data:
        return jsonify({"error": "Invalid request", "message": "Request body is required"}), 400
    
    url = data.get("url")
    event_types = data.get("event_types", [])
    secret = data.get("secret")
    description = data.get("description")
    
    if not url:
        return jsonify({"error": "Invalid request", "message": "URL is required"}), 400
    
    if not event_types or not isinstance(event_types, list):
        return jsonify({"error": "Invalid request", "message": "Event types must be a non-empty list"}), 400
    
    # Create subscription
    subscription = WebhookSubscription(
        url=url,
        event_types=event_types,
        owner_id=g.user["id"],
        secret=secret,
        description=description
    )
    
    success = create_webhook(subscription)
    
    if success:
        return jsonify({
            "message": "Webhook subscription created successfully",
            "webhook": subscription.to_dict()
        }), 201
    else:
        return jsonify({
            "error": "Failed to create webhook",
            "message": "Could not save webhook subscription"
        }), 500

@webhook_bp.route("", methods=["GET"])
@auth_required()
def list_webhooks():
    """
    List webhook subscriptions for the authenticated user.
    
    Returns:
        200: List of webhook subscriptions
        401: Authentication required
    """
    subscriptions = get_webhooks_by_owner(g.user["id"])
    
    return jsonify({
        "webhooks": [s.to_dict() for s in subscriptions],
        "count": len(subscriptions)
    })

@webhook_bp.route("/<webhook_id>", methods=["GET"])
@auth_required()
def get_webhook_details(webhook_id):
    """
    Get a webhook subscription by ID.
    
    Path Parameters:
        webhook_id: Webhook subscription ID
        
    Returns:
        200: Webhook subscription details
        401: Authentication required
        403: Insufficient permissions
        404: Webhook not found
    """
    subscription = get_webhook(webhook_id)
    
    if not subscription:
        return jsonify({
            "error": "Webhook not found",
            "message": f"No webhook found with ID: {webhook_id}"
        }), 404
    
    # Check ownership
    if subscription.owner_id != g.user["id"] and "admin" not in g.user.get("roles", []):
        return jsonify({
            "error": "Insufficient permissions",
            "message": "You do not have permission to access this webhook"
        }), 403
    
    return jsonify(subscription.to_dict())

@webhook_bp.route("/<webhook_id>", methods=["PUT"])
@auth_required()
def update_webhook_subscription(webhook_id):
    """
    Update a webhook subscription.
    
    Path Parameters:
        webhook_id: Webhook subscription ID
        
    Request Body:
        url: Webhook URL
        event_types: List of event types to subscribe to
        secret: Optional secret for signing webhook payloads
        description: Optional description
        enabled: Whether the webhook is enabled
        
    Returns:
        200: Webhook updated successfully
        400: Invalid request
        401: Authentication required
        403: Insufficient permissions
        404: Webhook not found
    """
    data = request.get_json()
    
    # Validate request data
    if not data:
        return jsonify({"error": "Invalid request", "message": "Request body is required"}), 400
    
    # Get existing subscription
    subscription = get_webhook(webhook_id)
    
    if not subscription:
        return jsonify({
            "error": "Webhook not found",
            "message": f"No webhook found with ID: {webhook_id}"
        }), 404
    
    # Check ownership
    if subscription.owner_id != g.user["id"] and "admin" not in g.user.get("roles", []):
        return jsonify({
            "error": "Insufficient permissions",
            "message": "You do not have permission to update this webhook"
        }), 403
    
    # Update fields
    if "url" in data:
        subscription.url = data["url"]
    
    if "event_types" in data and isinstance(data["event_types"], list):
        subscription.event_types = data["event_types"]
    
    if "secret" in data:
        subscription.secret = data["secret"]
    
    if "description" in data:
        subscription.description = data["description"]
    
    if "enabled" in data:
        subscription.enabled = bool(data["enabled"])
    
    # Save changes
    success = update_webhook(subscription)
    
    if success:
        return jsonify({
            "message": "Webhook subscription updated successfully",
            "webhook": subscription.to_dict()
        })
    else:
        return jsonify({
            "error": "Failed to update webhook",
            "message": "Could not save webhook subscription changes"
        }), 500

@webhook_bp.route("/<webhook_id>", methods=["DELETE"])
@auth_required()
def delete_webhook_subscription(webhook_id):
    """
    Delete a webhook subscription.
    
    Path Parameters:
        webhook_id: Webhook subscription ID
        
    Returns:
        200: Webhook deleted successfully
        401: Authentication required
        403: Insufficient permissions
        404: Webhook not found
    """
    # Get existing subscription
    subscription = get_webhook(webhook_id)
    
    if not subscription:
        return jsonify({
            "error": "Webhook not found",
            "message": f"No webhook found with ID: {webhook_id}"
        }), 404
    
    # Check ownership or admin role
    is_owner = subscription.owner_id == g.user["id"]
    is_admin = "admin" in g.user.get("roles", [])
    
    if not is_owner and not is_admin:
        return jsonify({
            "error": "Insufficient permissions",
            "message": "You do not have permission to delete this webhook"
        }), 403
    
    # Delete webhook
    success = delete_webhook(webhook_id, g.user["id"] if is_owner else "admin")
    
    if success:
        return jsonify({
            "message": "Webhook subscription deleted successfully"
        })
    else:
        return jsonify({
            "error": "Failed to delete webhook",
            "message": "Could not delete webhook subscription"
        }), 500

@webhook_bp.route("/test/<webhook_id>", methods=["POST"])
@auth_required()
def test_webhook(webhook_id):
    """
    Send a test event to a webhook.
    
    Path Parameters:
        webhook_id: Webhook subscription ID
        
    Returns:
        200: Test event sent successfully
        401: Authentication required
        403: Insufficient permissions
        404: Webhook not found
    """
    # Get existing subscription
    subscription = get_webhook(webhook_id)
    
    if not subscription:
        return jsonify({
            "error": "Webhook not found",
            "message": f"No webhook found with ID: {webhook_id}"
        }), 404
    
    # Check ownership
    if subscription.owner_id != g.user["id"] and "admin" not in g.user.get("roles", []):
        return jsonify({
            "error": "Insufficient permissions",
            "message": "You do not have permission to test this webhook"
        }), 403
    
    # Create test event
    test_data = {
        "message": "This is a test event",
        "user_id": g.user["id"],
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }
    
    # Trigger test event
    from ..analytics.webhooks.webhook_service import trigger_webhook
    success = trigger_webhook(subscription, {
        "type": "test",
        "data": test_data,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    })
    
    if success:
        return jsonify({
            "message": "Test event sent successfully",
            "webhook_id": webhook_id
        })
    else:
        return jsonify({
            "error": "Failed to send test event",
            "message": "Could not trigger webhook"
        }), 500

@webhook_bp.route("/events", methods=["GET"])
@admin_required
def list_event_types():
    """
    List available event types for webhooks. Admin only.
    
    Returns:
        200: List of event types
        401: Authentication required
        403: Insufficient permissions
    """
    # Get event types from analytics service
    from ..analytics.analytics_service import AnalyticsEvent
    
    event_types = [
        {
            "id": AnalyticsEvent.API_CALL,
            "name": "API Call",
            "description": "Triggered when an API endpoint is called"
        },
        {
            "id": AnalyticsEvent.PDF_PROCESSING,
            "name": "PDF Processing",
            "description": "Triggered when a PDF is processed"
        },
        {
            "id": AnalyticsEvent.SEARCH,
            "name": "Search",
            "description": "Triggered when a search is performed"
        },
        {
            "id": AnalyticsEvent.QUESTION,
            "name": "Question",
            "description": "Triggered when a question is asked"
        },
        {
            "id": AnalyticsEvent.USER_AUTH,
            "name": "User Authentication",
            "description": "Triggered on user login/logout events"
        },
        {
            "id": AnalyticsEvent.ERROR,
            "name": "Error",
            "description": "Triggered when an error occurs"
        },
        {
            "id": "test",
            "name": "Test",
            "description": "Used for testing webhook functionality"
        }
    ]
    
    return jsonify({
        "event_types": event_types
    }) 