#!/usr/bin/env python3
"""
Webhook API Routes

Provides API endpoints for webhook subscription management.
"""

import logging
import structlog
from flask import Blueprint, request, jsonify, g, current_app
from typing import Dict, Any, List, Optional
from functools import wraps # Import wraps

# Import webhook services
from ..analytics.webhooks.webhook_service import (
    WebhookSubscription,
    create_webhook,
    get_webhook,
    get_webhooks_by_owner,
    update_webhook,
    delete_webhook,
    trigger_event,
    # Import exceptions
    WebhookNotFoundError,
    WebhookDatabaseError,
    WebhookDataError,
    WebhookAuthorizationError,
    WebhookServiceError,
    WebhookService
)

# Import authentication utilities
from ..auth_middleware import auth_required, admin_required

# Pydantic and Enums for validation
from pydantic import BaseModel, HttpUrl, Field, ValidationError as PydanticValidationError
from enum import Enum

# Import base error class
from ..utils.error_utils import APIError, ValidationError

# Configure logger
logger = structlog.get_logger(__name__)

# Define allowed webhook event types
class WebhookEventType(str, Enum):
    PDF_PROCESSING_COMPLETE = "pdf.processing.complete" # Example specific events
    QUESTION_ANSWERED = "question.answered"
    SEARCH_COMPLETED = "search.completed"
    # Add other relevant event types, possibly derived from AnalyticsEvent
    # ANALYTICS_API_CALL = AnalyticsEvent.API_CALL
    # ANALYTICS_ERROR = AnalyticsEvent.ERROR

# --- Pydantic Schemas ---
class WebhookBaseSchema(BaseModel):
    url: HttpUrl
    event_types: List[WebhookEventType] = Field(..., min_items=1)
    description: Optional[str] = None
    secret: Optional[str] = None # Consider validation if format is expected

class WebhookCreateSchema(WebhookBaseSchema):
    pass # Inherits all fields

class WebhookUpdateSchema(BaseModel):
    # All fields are optional for PUT/PATCH
    url: Optional[HttpUrl] = None
    event_types: Optional[List[WebhookEventType]] = Field(None, min_items=1)
    description: Optional[str] = None
    secret: Optional[str] = None
    enabled: Optional[bool] = None

# Create Blueprint
webhook_bp = Blueprint("webhooks", __name__)

# --- Decorator for Ownership/Admin Check ---

def require_webhook_owner_or_admin(webhook_id_param: str = "webhook_id"):
    """
    Decorator factory to ensure user owns the webhook or is an admin.
    Fetches the webhook and stores it in g.webhook.

    Args:
        webhook_id_param: The name of the route parameter containing the webhook ID.
    """
    def decorator(f):
        @wraps(f)
        @auth_required() # Ensure user is authenticated first
        def decorated_function(*args, **kwargs):
            webhook_id = kwargs.get(webhook_id_param)
            if not webhook_id:
                logger.error("Webhook ID parameter missing in route for ownership check", param_name=webhook_id_param)
                raise APIError("Server configuration error: Webhook ID parameter missing.", 500)

            # Fetch the webhook
            subscription = get_webhook(webhook_id)

            if not subscription:
                logger.warning("Webhook not found during ownership check", user_id=g.user["id"], webhook_id=webhook_id)
                # Return 404, consistent with direct access attempts
                return jsonify({"error": "Webhook not found"}), 404

            # Perform ownership or admin check
            is_owner = subscription.owner_id == g.user["id"]
            is_admin = "admin" in g.user.get("roles", [])

            if not (is_owner or is_admin):
                logger.warning("Authorization failed: User does not own webhook and is not admin",
                             user_id=g.user["id"], webhook_id=webhook_id, owner_id=subscription.owner_id)
                # Return 403 Forbidden as the resource exists but access is denied
                return jsonify({"error": "Forbidden: You do not have permission to access this webhook."}), 403

            # Store the fetched subscription in g for the route handler
            g.webhook = subscription
            logger.debug("Webhook ownership/admin check passed", user_id=g.user["id"], webhook_id=webhook_id)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@webhook_bp.route("", methods=["POST"])
@auth_required()
def create_webhook_subscription():
    """
    Create a new webhook subscription. Validates input using Pydantic.
    
    Request Body:
        url: Valid HTTP/S URL for the webhook endpoint.
        event_types: List of valid event types (e.g., ["pdf.processing.complete"]) (min 1 item).
        secret: Optional secret for signing webhook payloads
        description: Optional description
        
    Returns:
        201: Webhook created successfully
        400: Invalid request data (validation error)
        401: Authentication required
        500: Server error
    """
    data = request.get_json()
    
    if not data:
        logger.warning("Create webhook attempt with missing fields", user_id=g.user["id"], provided_data=data)
        raise ValidationError("Request body cannot be empty.")

    try:
        # Validate input using Pydantic schema
        webhook_data = WebhookCreateSchema.model_validate(data)

        # Create subscription object (assuming WebhookSubscription matches schema fields)
        # Ensure owner_id is set from context, not request body
        subscription = WebhookSubscription(
            url=str(webhook_data.url), # Convert Pydantic URL back to string if needed by service
            event_types=[et.value for et in webhook_data.event_types], # Get enum values
            owner_id=g.user["id"],
            secret=webhook_data.secret,
            description=webhook_data.description
            # enabled defaults to True in WebhookSubscription class?
        )

        # Call the service function to create the webhook
        # Assume create_webhook returns the created subscription ID or raises errors
        created_subscription_id = create_webhook(subscription)

        # Fetch the created subscription to return it (get_webhook raises errors if not found)
        created_subscription_obj = get_webhook(created_subscription_id)

        logger.info("Webhook created successfully", user_id=g.user["id"], webhook_id=created_subscription_obj.id, url=created_subscription_obj.url)

        # Return the created subscription object (excluding secret)
        response_data = created_subscription_obj.to_dict()
        response_data.pop('secret', None)
        return jsonify(response_data), 201

    except PydanticValidationError as e:
        logger.warning("Webhook creation validation failed", user_id=g.user['id'], errors=e.errors())
        raise ValidationError(f"Input validation failed: {e.errors()}")

    except Exception as e: # Catch errors from create_webhook or other issues
        logger.error("Failed to create webhook in service layer", user_id=g.user["id"], url=data.get('url'), error=str(e), exc_info=True)
        return jsonify({
            "error": "Failed to create webhook",
            "message": "An internal error occurred."
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
    
    response_data = [s.to_dict() for s in subscriptions]
    for item in response_data:
        item.pop('secret', None)
    
    logger.debug("Retrieved webhooks for user", user_id=g.user["id"], count=len(response_data))
    return jsonify(response_data)

@webhook_bp.route("/<webhook_id>", methods=["GET"])
@require_webhook_owner_or_admin()
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
    # Decorator handles fetching, auth, and ownership check.
    # Webhook is available in g.webhook
    subscription = g.webhook
    response_data = subscription.to_dict()
    response_data.pop('secret', None)
    logger.debug("Retrieved webhook details", user_id=g.user["id"], webhook_id=webhook_id)
    return jsonify(response_data)

@webhook_bp.route("/<webhook_id>", methods=["PATCH"])
@require_webhook_owner_or_admin()
def update_webhook_subscription(webhook_id):
    """
    Partially update a webhook subscription (PATCH semantics).

    Path Parameters:
        webhook_id: Webhook subscription ID

    Request Body: (Provide only fields to update)
        url: Optional new webhook URL
        event_types: Optional new list of event types to subscribe to
        secret: Optional new secret for signing payloads (set to null or empty string to remove)
        description: Optional new description
        enabled: Optional boolean to enable/disable the webhook

    Returns:
        200: Webhook updated successfully
        400: Invalid request data (validation error)
        401: Authentication required
        403: Insufficient permissions
        404: Webhook not found
        500: Server error
    """
    data = request.get_json()
    
    if not data:
        logger.warning("Update webhook attempt with no data", user_id=g.user["id"], webhook_id=webhook_id)
        return jsonify({"error": "No update data provided"}), 400
    
    try:
        # Validate incoming data fields (optional fields)
        update_data = WebhookUpdateSchema.model_validate(data)

        # Decorator ensures webhook exists and user is authorized.
        # Use the webhook stored in g by the decorator.
        subscription = g.webhook

        # Apply validated updates to the subscription object
        update_fields = update_data.model_dump(exclude_unset=True) # Get only provided fields
        updated = False
        for field, value in update_fields.items():
            if hasattr(subscription, field):
                # Handle specific conversions if needed
                if field == 'url' and value is not None:
                    setattr(subscription, field, str(value)) # Convert HttpUrl back to str
                elif field == 'event_types' and value is not None:
                    setattr(subscription, field, [et.value for et in value]) # Convert Enums to strings
                else:
                    setattr(subscription, field, value)
                updated = True
            else:
                logger.warning("Attempted to update non-existent field on webhook", field=field, webhook_id=webhook_id)

        if not updated:
            # No valid fields were updated, maybe return 304 Not Modified or just the current state?
            logger.info("Webhook update request contained no valid fields to update", user_id=g.user["id"], webhook_id=webhook_id)
            response_data = subscription.to_dict()
            response_data.pop('secret', None)
            return jsonify(response_data) # Return current state

        # Save changes
        # Pass the dict of updates directly to the service function
        updated_subscription = update_webhook(webhook_id=subscription.id, updates=update_fields)

        response_data = updated_subscription.to_dict()
        response_data.pop('secret', None)
        logger.info("Webhook updated successfully", user_id=g.user["id"], webhook_id=webhook_id)
        return jsonify(response_data)

    except PydanticValidationError as e:
        logger.warning("Webhook update validation failed", user_id=g.user["id"], webhook_id=webhook_id, errors=e.errors())
        raise ValidationError(f"Input validation failed: {e.errors()}")
    except WebhookNotFoundError as e: # Raised by decorator or update_webhook
        return jsonify({"error": "Webhook not found", "message": str(e)}), 404
    except WebhookDataError as e: # Raised by update_webhook if data validation fails there
        logger.warning("Webhook update data error", user_id=g.user["id"], webhook_id=webhook_id, error=str(e))
        return jsonify({"error": "Invalid update data provided", "message": str(e)}), 400
    except WebhookDatabaseError as e:
        logger.error("Webhook update database error", user_id=g.user["id"], webhook_id=webhook_id, error=str(e), exc_info=True)
        return jsonify({"error": "Failed to update webhook due to database issue"}), 500
    except WebhookServiceError as e:
        logger.error("Webhook service error during update", user_id=g.user["id"], webhook_id=webhook_id, error=str(e), exc_info=True)
        return jsonify({"error": "An unexpected service error occurred during webhook update"}), 500
    except Exception as e:
        logger.error("Error updating webhook", user_id=g.user["id"], webhook_id=webhook_id, error=str(e), exc_info=True)
        return jsonify({
            "error": "Internal server error"
        }), 500

@webhook_bp.route("/<webhook_id>", methods=["DELETE"])
@require_webhook_owner_or_admin()
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
    try:
        # Decorator ensures webhook exists and user is authorized.
        # The delete_webhook service function likely still needs owner_id for safety/logging
        # but the authorization check is already done.
        # delete_webhook now raises exceptions on failure or returns None on success
        delete_webhook(webhook_id=g.webhook.id, owner_id=g.user["id"])

        logger.info("Webhook deleted successfully", user_id=g.user["id"], webhook_id=webhook_id)
        return '', 204

    except WebhookNotFoundError as e: # Raised by decorator or delete_webhook
        return jsonify({"error": "Webhook not found", "message": str(e)}), 404
    except WebhookAuthorizationError as e: # Raised by delete_webhook if owner check fails there
        logger.error("Webhook delete authorization error", user_id=g.user["id"], webhook_id=webhook_id, error=str(e))
        return jsonify({"error": "Forbidden", "message": "You do not have permission to delete this webhook."}), 403
    except WebhookDataError as e: # Raised by delete_webhook if data is bad during check
        logger.warning("Webhook delete data error", user_id=g.user["id"], webhook_id=webhook_id, error=str(e))
        return jsonify({"error": "Cannot delete webhook due to data inconsistency"}), 500
    except WebhookDatabaseError as e:
        logger.error("Webhook delete database error", user_id=g.user["id"], webhook_id=webhook_id, error=str(e), exc_info=True)
        return jsonify({"error": "Failed to delete webhook due to database issue"}), 500
    except WebhookServiceError as e:
        logger.error("Webhook service error during delete", user_id=g.user["id"], webhook_id=webhook_id, error=str(e), exc_info=True)
        return jsonify({"error": "An unexpected service error occurred during webhook deletion"}), 500
    except Exception as e:
        # Catch-all for truly unexpected errors
        logger.error("Error deleting webhook", user_id=g.user["id"], webhook_id=webhook_id, error=str(e), exc_info=True)
        return jsonify({
            "error": "Internal server error"
        }), 500

@webhook_bp.route("/test/<webhook_id>", methods=["POST"])
@require_webhook_owner_or_admin()
def test_webhook(webhook_id):
    """
    Send a test event to a webhook subscription.
    
    Path Parameters:
        webhook_id: Webhook subscription ID
        
    Returns:
        200: Test event sent successfully
        401: Authentication required
        403: Insufficient permissions
        404: Webhook not found
    """
    data = request.get_json() or {}
    test_event_payload = data.get("payload", {"message": "Test event triggered successfully!"})

    # Decorator ensures webhook exists and user is authorized.
    # Use the webhook stored in g.
    subscription = g.webhook

    # Get service instance
    webhook_service = get_webhook_service()

    try:
        if not subscription.enabled:
            logger.info("Test skipped: Webhook is disabled", user_id=g.user["id"], webhook_id=webhook_id)

            # Return 200 OK, but indicate it wasn't sent
            return jsonify({
                "status": "skipped",
                "message": "Webhook is disabled, test event not sent."
            }), 200

        # Call the service method to queue the test event task
        webhook_service.send_test_event(subscription, test_event_payload)

        # If service method didn't raise, assume task was queued
        logger.info("Test event queued for webhook", user_id=g.user["id"], webhook_id=webhook_id)
        return jsonify({
            "status": "queued",
            "message": "Webhook test event queued successfully. Delivery depends on the endpoint responding."
        }), 202 # 202 Accepted

    except WebhookNotFoundError: # Should be caught by decorator, but handle just in case
        return jsonify({"error": "Webhook not found"}), 404
    except WebhookServiceError as e:
        # Raised by send_test_event if queueing fails
        logger.error("Failed to queue test event", user_id=g.user["id"], webhook_id=webhook_id, error=str(e), exc_info=True)
        return jsonify({"error": "Failed to queue test event", "message": str(e)}), 500
    except Exception as e:
        logger.error("Error sending test event", user_id=g.user["id"], webhook_id=webhook_id, error=str(e), exc_info=True)
        return jsonify({
            "error": "Internal server error"
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

# Helper to get service instance (assuming it's on current_app)
# TODO: Consider dependency injection framework later
def get_webhook_service() -> WebhookService:
    if not hasattr(current_app, 'webhook_service'):
        logger.error("WebhookService not initialized or attached to current_app")
        raise RuntimeError("WebhookService is not available.")
    return current_app.webhook_service 