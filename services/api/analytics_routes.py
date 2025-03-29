#!/usr/bin/env python3
"""
Analytics API Routes

Provides API endpoints for analytics data and reporting.
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from typing import Dict, Any, List, Optional

# Import analytics services
from ..analytics.analytics_service import (
    get_analytics,
    get_analytics_summary,
    AnalyticsEvent
)

# Import authentication utilities
from ..auth_middleware import auth_required, admin_required

# Set up logging
logger = logging.getLogger("analytics_routes")

# Create Blueprint
analytics_bp = Blueprint("analytics", __name__)

@analytics_bp.route("/dashboard", methods=["GET"])
@auth_required()
def dashboard():
    """
    Get analytics dashboard data.
    
    Query Parameters:
        start_date: Optional start date (ISO format)
        end_date: Optional end date (ISO format)
        
    Returns:
        200: Analytics summary
        500: Server error
    """
    try:
        # Parse date parameters
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")
        
        start_date = None
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str)
            
        end_date = None
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
        
        # Get analytics summary
        summary = get_analytics_summary(
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error retrieving analytics dashboard: {e}")
        return jsonify({
            "error": "Failed to retrieve analytics dashboard",
            "message": str(e)
        }), 500

@analytics_bp.route("/events", methods=["GET"])
@admin_required
def events():
    """
    Get detailed analytics events. Admin only.
    
    Query Parameters:
        event_type: Optional event type filter
        user_id: Optional user ID filter
        start_date: Optional start date (ISO format)
        end_date: Optional end date (ISO format)
        limit: Maximum number of events to return (default: 100)
        
    Returns:
        200: Events list
        403: Insufficient permissions
        500: Server error
    """
    try:
        # Parse parameters
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
        
        # Get events
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
        logger.error(f"Error retrieving analytics events: {e}")
        return jsonify({
            "error": "Failed to retrieve events",
            "message": str(e)
        }), 500

@analytics_bp.route("/user-stats", methods=["GET"])
@auth_required()
def user_stats():
    """
    Get analytics for the current user.
    
    Query Parameters:
        start_date: Optional start date (ISO format)
        end_date: Optional end date (ISO format)
        
    Returns:
        200: User analytics
        401: Authentication required
        500: Server error
    """
    try:
        # Get current user ID from authentication
        from flask import g
        user_id = g.user["id"]
        
        # Parse date parameters
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")
        
        start_date = None
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str)
            
        end_date = None
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
        
        # Get user-specific events
        events = get_analytics(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            limit=1000  # Higher limit for user-specific data
        )
        
        # Prepare summary stats
        pdf_processing_count = len([e for e in events if e.get("event_type") == AnalyticsEvent.PDF_PROCESSING])
        search_count = len([e for e in events if e.get("event_type") == AnalyticsEvent.SEARCH])
        question_count = len([e for e in events if e.get("event_type") == AnalyticsEvent.QUESTION])
        
        # Calculate average response times if available
        response_times = [e.get("duration_ms") for e in events if e.get("duration_ms") is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return jsonify({
            "user_id": user_id,
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "stats": {
                "total_api_calls": len(events),
                "pdf_processing": pdf_processing_count,
                "searches": search_count,
                "questions": question_count,
                "avg_response_time": avg_response_time
            },
            "recent_activity": events[:10]  # Include recent activity
        })
    except Exception as e:
        logger.error(f"Error retrieving user stats: {e}")
        return jsonify({
            "error": "Failed to retrieve user stats",
            "message": str(e)
        }), 500

@analytics_bp.route("/export", methods=["GET"])
@admin_required
def export_analytics():
    """
    Export analytics data. Admin only.
    
    Query Parameters:
        event_type: Optional event type filter
        start_date: Optional start date (ISO format)
        end_date: Optional end date (ISO format)
        format: Export format (json or csv, default: json)
        
    Returns:
        200: Exported data
        403: Insufficient permissions
        500: Server error
    """
    try:
        # Parse parameters
        event_type = request.args.get("event_type")
        
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")
        
        start_date = None
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str)
            
        end_date = None
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
            
        export_format = request.args.get("format", "json")
        
        # Get events with higher limit for export
        events = get_analytics(
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            limit=10000  # Higher limit for export
        )
        
        # Format response based on requested format
        if export_format == "csv":
            import csv
            from io import StringIO
            
            # Create CSV file in memory
            output = StringIO()
            
            # Get all possible field names from events
            fieldnames = set()
            for event in events:
                fieldnames.update(event.keys())
                if "metadata" in event and isinstance(event["metadata"], dict):
                    fieldnames.update([f"metadata_{k}" for k in event["metadata"].keys()])
            
            # Remove metadata field since we're flattening it
            if "metadata" in fieldnames:
                fieldnames.remove("metadata")
            
            # Create CSV writer
            writer = csv.DictWriter(output, fieldnames=sorted(fieldnames))
            writer.writeheader()
            
            # Write events to CSV
            for event in events:
                # Create a copy of the event
                row = event.copy()
                
                # Flatten metadata
                if "metadata" in row and isinstance(row["metadata"], dict):
                    for k, v in row["metadata"].items():
                        row[f"metadata_{k}"] = v
                    del row["metadata"]
                
                writer.writerow(row)
            
            # Return CSV response
            from flask import Response
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={
                    "Content-Disposition": f"attachment;filename=analytics_export_{datetime.utcnow().strftime('%Y%m%d')}.csv"
                }
            )
        else:
            # Default to JSON
            return jsonify({
                "events": events,
                "count": len(events),
                "period": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None
                }
            })
    except Exception as e:
        logger.error(f"Error exporting analytics: {e}")
        return jsonify({
            "error": "Failed to export analytics",
            "message": str(e)
        }), 500 