#!/usr/bin/env python3
"""
Analytics Service Module

Provides analytics event tracking and reporting capabilities using Redis.
"""

import os
import time
import json
import uuid
import structlog
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, request, jsonify, g, current_app # Keep g for track_api_call
import redis

# Configure logger *before* potential import errors use it
logger = structlog.get_logger(__name__)

# Import the new Celery task
try:
    from services.tasks.analytics_tasks import log_analytics_event_task
except ImportError:
    # Handle case where task might not be importable (e.g., during setup)
    log_analytics_event_task = None
    logger.warning("Could not import log_analytics_event_task. Analytics tracking might be disabled or synchronous.")

# Constants
DEFAULT_ANALYTICS_TTL = 60 * 60 * 24 * 30  # 30 days

class AnalyticsEvent:
    """Analytics event data structure"""
    # Event Types
    API_CALL = "api_call"
    PDF_PROCESSING = "pdf_processing"
    SEARCH = "search"
    QUESTION = "question"
    USER_AUTH = "user_auth"
    ERROR = "error"
    CELERY_TASK = "celery_task"

    def __init__(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        task_name: Optional[str] = None,
        duration_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.event_type = event_type
        self.user_id = user_id
        self.endpoint = endpoint
        self.task_name = task_name
        self.duration_ms = duration_ms
        self.status_code = status_code
        self.status = status
        self.error = error
        self.metadata = metadata or {}

        # Conditionally add request info only if running within a request context
        if event_type == self.API_CALL and request:
             try:
                 # Check if request object is available (might not be in background tasks)
                 if request.remote_addr:
                     self.metadata.update({
                        "ip": request.remote_addr,
                        "user_agent": request.user_agent.string if request.user_agent else None,
                        "method": request.method,
                        "origin": request.headers.get("Origin"),
                        "referer": request.headers.get("Referer")
                     })
             except RuntimeError:
                  logger.debug("Cannot add request metadata outside request context.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary, filtering out None values."""
        data = {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "endpoint": self.endpoint,
            "task_name": self.task_name,
            "duration_ms": self.duration_ms,
            "status_code": self.status_code,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata
        }
        return {k: v for k, v in data.items() if v is not None}

class AnalyticsService:
    """Service class for managing analytics events using Redis."""

    def __init__(self, config: Dict[str, Any], redis_client: Optional[redis.Redis], db_pool: Optional[Any] = None):
        """Initialize the AnalyticsService with an injected Redis client."""
        self.config = config
        # db_pool is passed but not used here currently, store if needed later
        self.db_pool = db_pool
        # Directly assign the injected client
        self.redis_client = redis_client
        self.analytics_ttl = self.config.get('ANALYTICS_TTL_SECONDS', DEFAULT_ANALYTICS_TTL)

        # Remove the old logic trying to get client from current_app
        # try:
        #     # Check both potential locations for flexibility during transition
        #     if hasattr(current_app, 'redis_client') and current_app.redis_client:
        #         self.redis_client = current_app.redis_client
        #     elif 'REDIS_CLIENT' in current_app.config and current_app.config['REDIS_CLIENT']:
        #          self.redis_client = current_app.config['REDIS_CLIENT']
        #
        #     if self.redis_client and self.redis_client.ping():
        #         logger.info("AnalyticsService connected to Redis successfully.")
        #     elif self.redis_client:
        #         logger.warning("Redis client found in config, but ping failed. Analytics disabled.")
        #         self.redis_client = None
        # except RuntimeError:
        #     logger.error("AnalyticsService init error: Not in a Flask application context during init?")
        #     self.redis_client = None
        # except redis.exceptions.ConnectionError as e:
        #     logger.error("AnalyticsService init error: Redis connection failed.", error=str(e))
        #     self.redis_client = None
        # except Exception as e:
        #     logger.error("AnalyticsService init error: Unexpected error getting Redis client.", error=str(e), exc_info=True)
        #     self.redis_client = None

        # Log status based on the injected client
        if self.redis_client:
             try:
                 if self.redis_client.ping():
                      logger.info("AnalyticsService initialized with active Redis client.")
                 else:
                      # This case is less likely if app.py already pinged, but possible
                      logger.warning("AnalyticsService initialized, but injected Redis client failed ping. Analytics may fail.")
                      # Keep the client instance, let operations fail naturally
             except redis.exceptions.RedisError as e:
                  logger.error("AnalyticsService: Error pinging injected Redis client during init.", error=str(e))
                  # Keep the client instance
        else:
             logger.warning("AnalyticsService initialized without a Redis client. Analytics disabled.")

    def track_event(self, event: AnalyticsEvent) -> bool:
        """
        Track an analytics event using the initialized Redis client.
        """
        if not self.redis_client:
            logger.debug("Analytics tracking skipped: Redis client not available.", event_type=event.event_type)
            return False

        # --- Asynchronous Tracking via Celery --- #
        if log_analytics_event_task:
            try:
                event_dict = event.to_dict()
                log_analytics_event_task.delay(event_dict)
                logger.debug("Analytics event enqueued via Celery task", event_id=event.id, event_type=event.event_type)
                return True
            except Exception as e:
                # Log error and potentially fall back to synchronous, or just fail
                logger.error("Failed to enqueue analytics event task", event_id=event.id, error=str(e), exc_info=True)
                # Fallback to synchronous? Or just return False?
                # For now, let's just return False to indicate failure to track.
                return False
        else:
            logger.warning("log_analytics_event_task not available. Analytics tracking skipped.", event_id=event.id)
            return False
        # --- End Asynchronous Tracking --- #

    def track_api_call(self, response, start_time: float) -> None:
        """
        Track an API call. Intended to be called from Flask `after_request`.
        Uses Flask's `g` for user info and `request` for details.
        """
        if not self.redis_client:
            return # Cannot track if Redis is unavailable
        
        try:
            # Check if request is available and has necessary attributes
            if not request or not hasattr(request, 'path') or not hasattr(request, 'method'):
                 logger.debug("Skipping analytics tracking: Request context or attributes missing.")
                 return
                 
            excluded_paths = self.config.get('ANALYTICS_EXCLUDE_PATHS', ['/health', '/metrics'])
            if request.path in excluded_paths or request.method == 'OPTIONS':
                 logger.debug("Skipping analytics tracking for excluded path/method", path=request.path, method=request.method)
                 return
    
            duration_ms = (time.time() - start_time) * 1000
    
            user_id = None
            if hasattr(g, 'user_id'):
                user_id = g.user_id
            elif hasattr(g, 'user') and isinstance(g.user, dict):
                user_id = g.user.get('id')
    
            error_message = None
            status = 'success' if response.status_code < 400 else 'error'
            if status == 'error':
                try:
                    error_data = response.get_json()
                    if error_data and isinstance(error_data, dict):
                        error_message = error_data.get('error') or error_data.get('message') or json.dumps(error_data)
                    else:
                        error_message = response.get_data(as_text=True)[:500]
                except Exception:
                    # Handle cases where response is not JSON or reading fails
                    try:
                         error_message = response.get_data(as_text=True)[:500]
                    except Exception as data_ex:
                         logger.warning("Could not get error details from response data", exc=str(data_ex))
                         error_message = f"Status Code {response.status_code}"
    
            # Create event (metadata added automatically using request)
            event = AnalyticsEvent(
                event_type=AnalyticsEvent.API_CALL,
                user_id=user_id,
                endpoint=request.path,
                duration_ms=duration_ms,
                status_code=response.status_code,
                status=status,
                error=error_message
            )
            # Use self.track_event which is now asynchronous
            result = self.track_event(event)
            if not result:
                # Log if enqueuing failed (track_event now returns bool)
                logger.warning("Failed to track API call event asynchronously", event_id=event.id)
            
        except RuntimeError as e:
             # Catch errors accessing request/g outside of context (shouldn't happen in after_request)
             logger.error("RuntimeError during track_api_call (likely request context issue)", error=str(e), exc_info=True)
        except Exception as e:
             logger.error("Unexpected error during track_api_call", error=str(e), exc_info=True)

    def get_analytics(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Retrieve a paginated list of analytics events from Redis.

        Note: Currently fetches all matching events then slices for pagination
              due to Redis List storage limitations for efficient cursor/offset.

        Args:
            event_type: Filter by event type.
            user_id: Filter by user ID.
            start_date: Start date filter.
            end_date: End date filter.
            page: Page number (1-indexed).
            page_size: Number of items per page.

        Returns:
            A dictionary containing paginated items and metadata:
            {'items': List[Dict], 'total_count': int, 'page': int, 'page_size': int}
        """
        if not self.redis_client:
            logger.debug("Analytics query skipped: Redis client not available.")
            return {'items': [], 'total_count': 0, 'page': page, 'page_size': page_size}

        try:
            # Find relevant Redis keys based on dates
            keys = _get_redis_keys_for_range(
                self.redis_client,
                user_id=user_id,
                event_type=event_type,
                start_date=start_date,
                end_date=end_date
            )

            if not keys:
                return {'items': [], 'total_count': 0, 'page': page, 'page_size': page_size}

            # Scan and filter data from keys
            all_events = _scan_and_filter_data(
                self.redis_client,
                keys,
                user_id=user_id if not event_type else None, # Filter by user_id only if not already filtered by key
                event_type=event_type if not user_id else None # Filter by event_type only if not already filtered by key
            )

            # Apply sorting (descending by timestamp)
            all_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            # Apply pagination
            total_count = len(all_events)
            offset = (page - 1) * page_size
            paginated_events = all_events[offset : offset + page_size]

            logger.debug("Returning paginated analytics events", total_found=total_count, page=page, page_size=page_size, returned_count=len(paginated_events))
            return {
                'items': paginated_events,
                'total_count': total_count,
                'page': page,
                'page_size': page_size
            }

        except redis.exceptions.RedisError as e:
            logger.error("Redis error retrieving analytics data", error=str(e), exc_info=True)
            # Return empty paginated structure on error for now
            return {'items': [], 'total_count': 0, 'page': page, 'page_size': page_size}
        except Exception as e:
            logger.error("Unexpected error retrieving analytics data", error=str(e), exc_info=True)
            return {'items': [], 'total_count': 0, 'page': page, 'page_size': page_size}

    def get_analytics_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Retrieve summary statistics (counters) from Redis."""
        if not self.redis_client:
            logger.warning("Cannot get analytics summary: Redis client not available.")
            return {}

        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=1)

        summary = {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "endpoint_counts": {},
            "task_counts": {},
            "total_errors": 0,
            "total_events_by_type": {}
        }

        date_strs = []
        keys_to_scan_patterns = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            date_strs.append(date_str)
            keys_to_scan_patterns.append(f"analytics:counter:*:{date_str}")
            keys_to_scan_patterns.append(f"analytics:errors:{date_str}")
            all_event_types = [getattr(AnalyticsEvent, attr) for attr in dir(AnalyticsEvent) 
                               if not callable(getattr(AnalyticsEvent, attr)) and not attr.startswith("__")]
            for et in all_event_types:
                keys_to_scan_patterns.append(f"analytics:{et}:{date_str}")
            current_date += timedelta(days=1)
        
        if not keys_to_scan_patterns:
             return summary

        # Using SCAN can be less efficient than knowing keys, but safer if patterns are complex
        # Alternative: Generate all expected keys directly as before.
        keys_to_fetch = []
        try:
             for pattern in keys_to_scan_patterns:
                  # Use scan_iter for memory efficiency if key space is huge
                  # For moderate number of keys/dates, KEYS might be acceptable but blocks Redis
                  # Let's stick to generating exact keys if possible, assuming limited types/dates
                  pass # Revert to generating exact keys below

             keys_to_fetch = [] # Re-initialize
             for date_str in date_strs:
                 # Counters
                 keys_to_fetch.extend(self.redis_client.keys(f"analytics:counter:*:{date_str}")) # Still use KEYS for counters
                 # Errors list
                 keys_to_fetch.append(f"analytics:errors:{date_str}")
                 # Event type lists
                 all_event_types = [getattr(AnalyticsEvent, attr) for attr in dir(AnalyticsEvent) 
                                     if not callable(getattr(AnalyticsEvent, attr)) and not attr.startswith("__")]
                 for et in all_event_types:
                      keys_to_fetch.append(f"analytics:{et}:{date_str}")

             if not keys_to_fetch:
                  logger.debug("No keys found matching patterns for summary")
                  return summary

             # Filter for keys that actually exist? Optional optimization
             # existing_keys = [k for k in keys_to_fetch if self.redis_client.exists(k)]
             existing_keys = keys_to_fetch # Process all potential keys

             counter_keys = [k for k in existing_keys if ':counter:' in k]
             list_keys = [k for k in existing_keys if ':counter:' not in k]

             pipe = self.redis_client.pipeline(transaction=False) # Use pipeline without transaction for mixed commands
             if counter_keys:
                 pipe.mget(counter_keys)
             if list_keys:
                 for key in list_keys:
                     pipe.llen(key)
            
             results = pipe.execute()

             result_index = 0
             counter_values = results[result_index] if counter_keys else []
             if counter_keys: result_index += 1
             list_lengths = results[result_index:] if list_keys else []

             # Process counters
             if counter_keys and counter_values:
                 for key, value in zip(counter_keys, counter_values):
                    if value is not None:
                        try:
                            count = int(value)
                            parts = key.split(':')
                            if len(parts) >= 5:
                                 counter_type = parts[2]
                                 name = parts[3]
                                 if counter_type == 'endpoint':
                                      summary['endpoint_counts'][name] = summary['endpoint_counts'].get(name, 0) + count
                                 elif counter_type == 'task':
                                      summary['task_counts'][name] = summary['task_counts'].get(name, 0) + count
                        except (ValueError, IndexError) as e:
                            logger.warning("Failed to parse counter key/value", key=key, value=value, error=str(e))
            
             # Process list lengths
             if list_keys and list_lengths:
                 for i, key in enumerate(list_keys):
                      length = list_lengths[i]
                      if length is None: continue # Handle potential errors from LLEN?
                      parts = key.split(':')
                      if len(parts) >= 3:
                           category = parts[1]
                           if category == 'errors':
                                summary['total_errors'] += length
                           else:
                                summary['total_events_by_type'][category] = summary['total_events_by_type'].get(category, 0) + length
                      else:
                           logger.warning("Unexpected list key format for summary", key=key)
            
             return summary

        except redis.exceptions.RedisError as e:
            logger.error("Redis error retrieving analytics summary", error=str(e), exc_info=True)
            return summary # Return partial summary
        except Exception as e:
            logger.error("Unexpected error retrieving analytics summary", error=str(e), exc_info=True)
            return summary # Return partial summary

    def get_user_stats_summary(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculates and returns summary statistics for a specific user.

        Args:
            user_id: The ID of the user.
            start_date: Optional start date for the period.
            end_date: Optional end date for the period.

        Returns:
            A dictionary containing user statistics.

        Raises:
            RedisError: If there's an issue communicating with Redis.
            Exception: For other unexpected errors during calculation.
        """
        logger.debug("Calculating user stats summary", user_id=user_id, start_date=start_date, end_date=end_date)
        try:
            # Fetch all relevant events for the user in the period
            # Use a reasonable limit, or consider if pagination/streaming is needed for huge histories
            events = self.get_analytics(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=self.config.get('ANALYTICS_USER_STATS_LIMIT', 5000)
            )

            # Calculate summary stats from the fetched events
            pdf_processing_count = len([e for e in events if e.get("event_type") == AnalyticsEvent.PDF_PROCESSING])
            search_count = len([e for e in events if e.get("event_type") == AnalyticsEvent.SEARCH])
            question_count = len([e for e in events if e.get("event_type") == AnalyticsEvent.QUESTION])
            api_call_count = len([e for e in events if e.get("event_type") == AnalyticsEvent.API_CALL]) # Could be useful too
            total_events = len(events)

            # Calculate average response times (only for API calls with duration)
            response_times = [
                e.get("duration_ms")
                for e in events
                if e.get("event_type") == AnalyticsEvent.API_CALL and e.get("duration_ms") is not None
            ]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            # Get the 10 most recent events for recent activity display
            # Note: get_analytics currently returns events in an unspecified order (likely insertion order)
            # If chronological order is needed, sorting might be required here or in get_analytics
            recent_activity = events[:10]

            summary = {
                "user_id": user_id,
                "period": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None
                },
                "stats": {
                    "total_events": total_events,
                    "api_calls": api_call_count,
                    "pdf_processing": pdf_processing_count,
                    "searches": search_count,
                    "questions": question_count,
                    "avg_response_time_ms": avg_response_time
                },
                "recent_activity": recent_activity
            }

            logger.debug("User stats summary calculated successfully", user_id=user_id, total_events=total_events)
            return summary

        except redis.exceptions.RedisError as e:
            logger.error("Redis error calculating user stats summary", user_id=user_id, error=str(e), exc_info=True)
            # Re-raise Redis specific errors if the caller needs to handle them
            raise
        except Exception as e:
            logger.error("Unexpected error calculating user stats summary", user_id=user_id, error=str(e), exc_info=True)
            # Re-raise other exceptions
            raise

# Removed Flask routes (/health, /track, /events, /summary)
# Removed __main__ block 