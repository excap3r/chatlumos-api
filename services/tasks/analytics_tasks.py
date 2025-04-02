"""
Celery tasks for handling analytics event logging asynchronously.
"""

import json
import redis
import redis.exceptions
import structlog
from datetime import datetime

# Import the Celery app instance
from celery_app import celery_app # Adjust relative import if necessary
# Import config and Redis client helper (assuming one exists or create one)
from services.config import AppConfig
# Reuse Redis client helper if suitable, e.g., from question_processing
# Or define a specific one here based on AppConfig

logger = structlog.get_logger(__name__)

# Constants (consider getting from AppConfig if they vary)
DEFAULT_ANALYTICS_TTL = 60 * 60 * 24 * 30  # 30 days

def get_redis_client():
    """Helper to get Redis client connection."""
    redis_url = AppConfig.REDIS_URL
    if not redis_url:
        logger.error("REDIS_URL not found in AppConfig for analytics task.")
        return None
    try:
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis in analytics task: {e}", exc_info=True)
        return None

@celery_app.task(bind=True, autoretry_for=(redis.exceptions.RedisError,), retry_backoff=True, retry_kwargs={'max_retries': 3})
def log_analytics_event_task(self, event_dict: dict):
    """Celery task to log a single analytics event to Redis."""
    task_logger = structlog.get_logger(f"task.{self.request.id or 'analytics'}")
    redis_client = get_redis_client()
    
    if not redis_client:
        task_logger.error("Cannot log analytics event: Redis client unavailable. Aborting task (will retry).")
        # Raising an error will trigger Celery retry based on autoretry_for
        raise ConnectionError("Redis client unavailable for analytics task.")
        
    event_id = event_dict.get('id', 'unknown')
    event_type = event_dict.get('event_type', 'unknown')
    task_logger = task_logger.bind(event_id=event_id, event_type=event_type)
    
    try:
        task_logger.debug("Processing analytics event logging task")
        event_json = json.dumps(event_dict) # Already a dict, just dump to JSON string for Redis
        
        # Determine date string for keys
        event_timestamp = event_dict.get('timestamp')
        try:
            event_dt = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00'))
            date_str = event_dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError, AttributeError):
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            task_logger.warning("Could not parse event timestamp, using current date for key", event_timestamp=event_timestamp)
            
        analytics_ttl = AppConfig.ANALYTICS_TTL_SECONDS # Use configured TTL

        # Replicate the pipeline logic from AnalyticsService.track_event
        pipe = redis_client.pipeline()
        key = f"analytics:{event_type}:{date_str}"
        pipe.lpush(key, event_json)
        pipe.expire(key, analytics_ttl)

        user_id = event_dict.get('user_id')
        if user_id:
            user_key = f"analytics:user:{user_id}:{date_str}"
            pipe.lpush(user_key, event_json)
            pipe.expire(user_key, analytics_ttl)

        endpoint = event_dict.get('endpoint')
        if endpoint:
            endpoint_key = f"analytics:counter:endpoint:{endpoint}:{date_str}"
            pipe.incr(endpoint_key)
            pipe.expire(endpoint_key, analytics_ttl)
        
        task_name = event_dict.get('task_name')
        if task_name:
            task_key = f"analytics:counter:task:{task_name}:{date_str}"
            pipe.incr(task_key)
            pipe.expire(task_key, analytics_ttl)

        error = event_dict.get('error')
        if error:
            error_key = f"analytics:errors:{date_str}"
            pipe.lpush(error_key, event_json)
            pipe.expire(error_key, analytics_ttl)

        results = pipe.execute()
        task_logger.info("Analytics event successfully logged via Celery task")
        return {"status": "success", "event_id": event_id}
        
    except redis.exceptions.RedisError as e:
        task_logger.error("Redis error during analytics task execution. Task will retry.", error=str(e), exc_info=True)
        # Re-raise to trigger Celery retry
        raise self.retry(exc=e, countdown=int(self.request.retries * 5 + 5)) # Basic exponential backoff
    except Exception as e:
        task_logger.error("Unexpected error during analytics task execution. Task might fail permanently.", error=str(e), exc_info=True)
        # Update Celery state to FAILURE, don't retry unexpected errors by default
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # Re-raise if you want Celery to log it as failed, otherwise return error info
        raise # Or return {"status": "failed", "error": str(e)} 