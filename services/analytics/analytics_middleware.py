#!/usr/bin/env python3
"""
Analytics Middleware

This module provides Flask middleware for tracking API requests.
"""

import time
import logging
from functools import wraps
from flask import request, g, Flask
from typing import Callable, Any

from .analytics_service import AnalyticsEvent, track_event, track_api_call

# Set up logging
logger = logging.getLogger("analytics_middleware")

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
            request._status_code = response.status_code
            
            # Store error if applicable
            if response.status_code >= 400:
                try:
                    error_data = response.get_json()
                    request._error = error_data.get('error') if error_data else f"HTTP {response.status_code}"
                except:
                    request._error = f"HTTP {response.status_code}"
            
            # Track the API call
            try:
                track_api_call(g.start_time)
            except Exception as e:
                logger.error(f"Error tracking API call: {e}")
                
        return response
    
    @app.errorhandler(Exception)
    def handle_error(e: Exception) -> Any:
        """Track unhandled exceptions."""
        try:
            # Create error event
            event = AnalyticsEvent(
                event_type=AnalyticsEvent.ERROR,
                endpoint=request.path,
                user_id=g.get('user', {}).get('id') if hasattr(g, 'user') else None,
                error=str(e),
                metadata={
                    "exception_type": e.__class__.__name__,
                    "query_params": dict(request.args),
                    "content_type": request.content_type
                }
            )
            
            track_event(event)
        except Exception as tracking_error:
            logger.error(f"Error tracking exception: {tracking_error}")
        
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
            result = f(*args, **kwargs)
            
            try:
                # Extract metadata
                metadata = {}
                
                if include_payload and request.json:
                    # Include only non-sensitive payload data
                    safe_payload = request.json.copy() if isinstance(request.json, dict) else {}
                    
                    # Remove sensitive fields
                    for field in ['password', 'token', 'secret', 'key', 'api_key', 'authorization']:
                        if field in safe_payload:
                            safe_payload[field] = '[REDACTED]'
                    
                    metadata['payload'] = safe_payload
                
                # Add result info if available and not too large
                if hasattr(result, 'get_json'):
                    result_json = result.get_json()
                    if result_json and isinstance(result_json, dict):
                        # Include only status info, not full response data
                        if 'status' in result_json:
                            metadata['result_status'] = result_json['status']
                
                # Create event
                event = AnalyticsEvent(
                    event_type=event_type,
                    endpoint=request.path,
                    user_id=g.get('user', {}).get('id') if hasattr(g, 'user') else None,
                    status_code=result.status_code if hasattr(result, 'status_code') else 200,
                    metadata=metadata
                )
                
                track_event(event)
            except Exception as e:
                logger.error(f"Error tracking specific event: {e}")
            
            return result
        return wrapped
    return decorator 