#!/usr/bin/env python3
"""
Webhook Data Schemas
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

# --- Custom Exceptions ---
class WebhookDataError(Exception):
    """Error decoding or validating webhook data."""
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
            logger.error("Missing required field in webhook data", missing_key=str(e), data_keys=list(data.keys()))
            raise WebhookDataError(f"Invalid webhook data: Missing key '{e}'") from e
        except Exception as e:
             logger.error("Error creating WebhookSubscription from dict", error=str(e), data_preview=str(data)[:200])
             raise WebhookDataError(f"Error processing webhook data: {e}") from e 