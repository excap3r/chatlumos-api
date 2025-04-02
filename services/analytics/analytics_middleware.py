#!/usr/bin/env python3
"""
Analytics Middleware

This module provides Flask middleware for tracking API requests.
"""

import time
import structlog
from functools import wraps
from flask import request, g, Flask, current_app
from typing import Callable, Any

from .analytics_service import AnalyticsEvent, AnalyticsService

# Set up logging
logger = structlog.get_logger("analytics_middleware")

def setup_analytics_tracking(app: Flask) -> None:
    """
    Set up analytics tracking middleware for Flask app.
    
    Args:
        app: Flask application to configure
    """
    @app.before_request
    def before_request() -> None:
        """Record the start time of each request."""
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response: Any) -> Any:
        """Track API request after it completes."""
        if hasattr(g, 'start_time'):
            # Store status code for tracking
            g._response_status_code = response.status_code
            
            # Store error if applicable
            if response.status_code >= 400:
                try:
                    error_data = response.get_json()
                    g._response_error = error_data.get('error') if error_data else f"HTTP {response.status_code}"
                except:
                    g._response_error = f"HTTP {response.status_code}"
            
            # Track the API call
            try:
                # Get service instance from app context
                analytics_service = getattr(current_app, 'analytics_service', None)
                if analytics_service and isinstance(analytics_service, AnalyticsService):
                    analytics_service.track_api_call(response=response, start_time=g.start_time)
                else:
                    logger.warning("AnalyticsService instance not found on current_app. Cannot track API call.")
            except Exception as e:
                logger.error("Error tracking API call analytics event", 
                             error=str(e), 
                             path=request.path, 
                             exc_info=True)
                
        return response
    
    @app.errorhandler(Exception)
    def handle_error(e: Exception) -> Any:
        """Track unhandled exceptions."""
        try:
            # Create error event
            user_id = g.get('user', {}).get('id') if hasattr(g, 'user') else None
            event = AnalyticsEvent(
                event_type=AnalyticsEvent.ERROR,
                endpoint=request.path if request else None,
                user_id=user_id,
                error=str(e),
                status_code=500, # Unhandled exceptions usually result in 500
                metadata={
                    "exception_type": e.__class__.__name__,
                    "request_args": dict(request.args) if request else None,
                    "request_path": request.path if request else None
                }
            )
            
            # Get service instance from app context
            analytics_service = getattr(current_app, 'analytics_service', None)
            if analytics_service and isinstance(analytics_service, AnalyticsService):
                analytics_service.track_event(event)
            else:
                logger.warning("AnalyticsService instance not found on current_app. Cannot track unhandled exception event.")
            
            logger.warning("Unhandled exception tracked", 
                         exc_type=e.__class__.__name__, 
                         error=str(e), 
                         path=request.path if request else None, 
                         user_id=user_id)
        except Exception as tracking_error:
            logger.error("Error tracking unhandled exception", 
                         tracking_error=str(tracking_error), 
                         original_error=str(e), 
                         exc_info=True)
        
        # Re-raise the exception for normal handling
        raise e

def track_specific_event(event_type: str, include_payload: bool = False) -> Callable:
    """
    Decorator to track specific events.
    
    Args:
        event_type: Type of event to track
        include_payload: Whether to include request payload in metadata
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            # Call the original function
            try:
                result = f(*args, **kwargs)
            except Exception as func_exc:
                 # Log and track error if the decorated function fails
                 user_id = g.get('user', {}).get('id') if hasattr(g, 'user') else None
                 logger.error("Exception in tracked function", 
                              event_type=event_type, 
                              function_name=f.__name__, 
                              error=str(func_exc), 
                              exc_info=True)
                 try:
                     error_event = AnalyticsEvent(
                         event_type=AnalyticsEvent.ERROR, # Log as error type
                         endpoint=request.path,
                         user_id=user_id,
                         error=f"Error in {event_type}: {str(func_exc)}",
                         metadata={'original_event_type': event_type}
                     )
                     # Get service instance from app context
                     analytics_service_inner = getattr(current_app, 'analytics_service', None)
                     if analytics_service_inner and isinstance(analytics_service_inner, AnalyticsService):
                         analytics_service_inner.track_event(error_event)
                     else:
                         logger.warning("AnalyticsService instance not found on current_app. Cannot track exception event within decorator.")
                 except Exception as tracking_error:
                     logger.error("Error tracking exception within decorator", 
                                  tracking_error=str(tracking_error), 
                                  exc_info=True)
                 raise func_exc # Re-raise the original exception
            
            # If function succeeded, track the specific event
            try:
                metadata = {}
                if include_payload and request.is_json:
                    try:
                        safe_payload = request.get_json() if request.is_json else {}
                        if isinstance(safe_payload, dict):
                             # Remove sensitive fields
                             for field in ['password', 'token', 'secret', 'key', 'api_key', 'authorization']:
                                 if field in safe_payload:
                                     safe_payload[field] = '[REDACTED]'
                             metadata['payload'] = safe_payload
                        elif isinstance(safe_payload, list): # Handle list payloads if necessary
                             metadata['payload_preview'] = f"List payload, length: {len(safe_payload)}"
                        else:
                             metadata['payload_type'] = type(safe_payload).__name__
                    except Exception as json_err:
                        logger.warning("Failed to parse JSON payload for tracking", error=str(json_err))
                        metadata['payload_error'] = str(json_err)
                
                status_code = 200 # Default success
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                elif isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], int):
                    status_code = result[1] # Handle Flask tuple response (body, status)
                
                event = AnalyticsEvent(
                    event_type=event_type,
                    endpoint=request.path,
                    user_id=g.get('user', {}).get('id') if hasattr(g, 'user') else None,
                    status_code=status_code,
                    metadata=metadata
                )
                # Get service instance from app context
                analytics_service = getattr(current_app, 'analytics_service', None)
                if analytics_service and isinstance(analytics_service, AnalyticsService):
                    analytics_service.track_event(event)
                else:
                    logger.warning("AnalyticsService instance not found on current_app. Cannot track specific event.")
            except Exception as e:
                logger.error("Error tracking specific analytics event", 
                             event_type=event_type,
                             error=str(e), 
                             exc_info=True) 
            
            return result
        return wrapped
    return decorator 