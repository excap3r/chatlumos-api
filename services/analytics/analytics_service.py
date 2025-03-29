#!/usr/bin/env python3
"""
Analytics Service Implementation

This module implements analytics collection, storage, and reporting functionality.
"""

import os
import time
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
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
logger = logging.getLogger('analytics_service')

# Initialize Redis for analytics if available
redis_url = os.getenv("REDIS_URL")
redis_client = None
if redis_url:
    try:
        redis_client = redis.from_url(redis_url)
        logger.info(f"Connected to Redis at {redis_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")

# Constants
API_VERSION = "v1"
ANALYTICS_TTL = 60 * 60 * 24 * 30  # 30 days retention

class AnalyticsEvent:
    """Analytics event data structure"""
    
    API_CALL = "api_call"
    PDF_PROCESSING = "pdf_processing"
    SEARCH = "search"
    QUESTION = "question"
    USER_AUTH = "user_auth"
    ERROR = "error"
    
    def __init__(
        self, 
        event_type: str,
        endpoint: str = None,
        user_id: str = None,
        duration_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize analytics event.
        
        Args:
            event_type: Type of event (API_CALL, PDF_PROCESSING, etc.)
            endpoint: API endpoint
            user_id: User ID if authenticated
            duration_ms: Request duration in milliseconds
            status_code: HTTP status code
            error: Error message if applicable
            metadata: Additional event metadata
        """
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.event_type = event_type
        self.endpoint = endpoint
        self.user_id = user_id
        self.duration_ms = duration_ms
        self.status_code = status_code
        self.error = error
        self.metadata = metadata or {}
        
        # Add request information if available
        if request:
            self.metadata.update({
                "ip": request.remote_addr,
                "user_agent": request.user_agent.string if request.user_agent else None,
                "method": request.method,
                "origin": request.headers.get("Origin"),
                "referer": request.headers.get("Referer")
            })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "endpoint": self.endpoint,
            "user_id": self.user_id,
            "duration_ms": self.duration_ms,
            "status_code": self.status_code,
            "error": self.error,
            "metadata": self.metadata
        }

def track_event(event: AnalyticsEvent) -> bool:
    """
    Track an analytics event.
    
    Args:
        event: AnalyticsEvent to track
        
    Returns:
        Success status
    """
    if not redis_client:
        logger.warning("Redis not available, analytics event not tracked")
        return False
    
    try:
        # Create timestamp-based key
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"analytics:{event.event_type}:{date_str}"
        
        # Store event in Redis
        redis_client.lpush(key, json.dumps(event.to_dict()))
        redis_client.expire(key, ANALYTICS_TTL)
        
        # Also store in user-specific list if user_id is available
        if event.user_id:
            user_key = f"analytics:user:{event.user_id}:{date_str}"
            redis_client.lpush(user_key, json.dumps(event.to_dict()))
            redis_client.expire(user_key, ANALYTICS_TTL)
        
        # Update counters
        if event.endpoint:
            endpoint_key = f"analytics:counter:endpoint:{event.endpoint}:{date_str}"
            redis_client.incr(endpoint_key)
            redis_client.expire(endpoint_key, ANALYTICS_TTL)
        
        # Track errors
        if event.error:
            error_key = f"analytics:errors:{date_str}"
            redis_client.lpush(error_key, json.dumps(event.to_dict()))
            redis_client.expire(error_key, ANALYTICS_TTL)
        
        return True
    except Exception as e:
        logger.error(f"Error tracking analytics event: {e}")
        return False

def track_api_call(start_time: float) -> None:
    """
    Track an API call.
    
    Args:
        start_time: Request start time (from time.time())
    """
    duration_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Create event
    event = AnalyticsEvent(
        event_type=AnalyticsEvent.API_CALL,
        endpoint=request.path,
        user_id=g.get('user', {}).get('id') if hasattr(g, 'user') else None,
        duration_ms=duration_ms,
        status_code=getattr(request, '_status_code', 200),
        error=getattr(request, '_error', None),
        metadata={
            "query_params": dict(request.args),
            "content_type": request.content_type
        }
    )
    
    track_event(event)

def get_analytics(
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get analytics events.
    
    Args:
        event_type: Filter by event type
        user_id: Filter by user ID
        start_date: Start date for range filter
        end_date: End date for range filter
        limit: Maximum number of events to return
        
    Returns:
        List of event dictionaries
    """
    if not redis_client:
        logger.warning("Redis not available, cannot retrieve analytics")
        return []
    
    try:
        events = []
        
        # Set date range
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=7)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Generate date range
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Determine key pattern based on filters
            if user_id:
                key_pattern = f"analytics:user:{user_id}:{date_str}"
            elif event_type:
                key_pattern = f"analytics:{event_type}:{date_str}"
            else:
                # If no specific filter, we need to check multiple keys
                for evt_type in [
                    AnalyticsEvent.API_CALL,
                    AnalyticsEvent.PDF_PROCESSING,
                    AnalyticsEvent.SEARCH,
                    AnalyticsEvent.QUESTION,
                    AnalyticsEvent.USER_AUTH,
                    AnalyticsEvent.ERROR
                ]:
                    key = f"analytics:{evt_type}:{date_str}"
                    data = redis_client.lrange(key, 0, limit - len(events))
                    for item in data:
                        events.append(json.loads(item))
                        if len(events) >= limit:
                            return events
                
                current_date += timedelta(days=1)
                continue
            
            # Get data for the specific key
            data = redis_client.lrange(key_pattern, 0, limit - len(events))
            for item in data:
                events.append(json.loads(item))
                if len(events) >= limit:
                    return events
            
            current_date += timedelta(days=1)
        
        return events
    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        return []

def get_analytics_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Get analytics summary with aggregated metrics.
    
    Args:
        start_date: Start date for range filter
        end_date: End date for range filter
        
    Returns:
        Summary dictionary with metrics
    """
    if not redis_client:
        logger.warning("Redis not available, cannot retrieve analytics summary")
        return {}
    
    try:
        # Set date range
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=7)
        if not end_date:
            end_date = datetime.utcnow()
        
        summary = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_api_calls": 0,
            "total_users": set(),
            "endpoints": {},
            "errors": 0,
            "pdf_processing": 0,
            "searches": 0,
            "questions": 0,
            "performance": {
                "avg_response_time": 0,
                "p95_response_time": 0,
                "max_response_time": 0
            }
        }
        
        response_times = []
        
        # Generate date range
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Count API calls
            api_calls = redis_client.lrange(f"analytics:{AnalyticsEvent.API_CALL}:{date_str}", 0, -1)
            summary["total_api_calls"] += len(api_calls)
            
            # Process API call data
            for call_data in api_calls:
                call = json.loads(call_data)
                
                # Track endpoints
                endpoint = call.get("endpoint")
                if endpoint:
                    summary["endpoints"][endpoint] = summary["endpoints"].get(endpoint, 0) + 1
                
                # Track users
                user_id = call.get("user_id")
                if user_id:
                    summary["total_users"].add(user_id)
                
                # Track response times
                duration = call.get("duration_ms")
                if duration:
                    response_times.append(duration)
            
            # Count errors
            errors = redis_client.lrange(f"analytics:errors:{date_str}", 0, -1)
            summary["errors"] += len(errors)
            
            # Count specific event types
            summary["pdf_processing"] += len(redis_client.lrange(
                f"analytics:{AnalyticsEvent.PDF_PROCESSING}:{date_str}", 0, -1
            ))
            summary["searches"] += len(redis_client.lrange(
                f"analytics:{AnalyticsEvent.SEARCH}:{date_str}", 0, -1
            ))
            summary["questions"] += len(redis_client.lrange(
                f"analytics:{AnalyticsEvent.QUESTION}:{date_str}", 0, -1
            ))
            
            current_date += timedelta(days=1)
        
        # Calculate performance metrics
        if response_times:
            summary["performance"]["avg_response_time"] = sum(response_times) / len(response_times)
            summary["performance"]["max_response_time"] = max(response_times)
            response_times.sort()
            p95_index = int(len(response_times) * 0.95)
            summary["performance"]["p95_response_time"] = response_times[p95_index]
        
        # Convert set to count for users
        summary["total_users"] = len(summary["total_users"])
        
        return summary
    except Exception as e:
        logger.error(f"Error retrieving analytics summary: {e}")
        return {}

# API Endpoints

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "analytics-service",
        "redis_connected": redis_client is not None
    })

@app.route('/track', methods=['POST'])
def track_event_endpoint():
    """Track an analytics event"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        event = AnalyticsEvent(
            event_type=data.get("event_type", AnalyticsEvent.API_CALL),
            endpoint=data.get("endpoint"),
            user_id=data.get("user_id"),
            duration_ms=data.get("duration_ms"),
            status_code=data.get("status_code"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )
        
        success = track_event(event)
        
        return jsonify({
            "success": success,
            "event_id": event.id
        })
    except Exception as e:
        logger.error(f"Error processing track event request: {e}")
        return jsonify({
            "error": "Failed to track event",
            "message": str(e)
        }), 500

@app.route('/events', methods=['GET'])
def get_events():
    """Get analytics events"""
    try:
        event_type = request.args.get("event_type")
        user_id = request.args.get("user_id")
        
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")
        
        start_date = None
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str)
            
        end_date = None
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
            
        limit = int(request.args.get("limit", 100))
        
        events = get_analytics(
            event_type=event_type,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return jsonify({
            "events": events,
            "count": len(events),
            "limit": limit
        })
    except Exception as e:
        logger.error(f"Error retrieving events: {e}")
        return jsonify({
            "error": "Failed to retrieve events",
            "message": str(e)
        }), 500

@app.route('/summary', methods=['GET'])
def get_summary():
    """Get analytics summary"""
    try:
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")
        
        start_date = None
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str)
            
        end_date = None
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
            
        summary = get_analytics_summary(
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error retrieving summary: {e}")
        return jsonify({
            "error": "Failed to retrieve summary",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5006))) 