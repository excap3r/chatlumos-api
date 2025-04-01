#!/usr/bin/env python3
"""
Analytics API Routes

Provides API endpoints for analytics data and reporting.
"""

import logging
import structlog
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, g, current_app, Response, make_response
from typing import Dict, Any, List, Optional
import csv
from io import StringIO
import redis

# Import analytics services
from ..analytics.analytics_service import (
    get_analytics,
    get_analytics_summary,
    AnalyticsEvent,
    AnalyticsService
)

# Import authentication utilities
from ..auth_middleware import auth_required, admin_required

# Import request helpers
from .utils.request_helpers import parse_date_range_args, generate_csv_string

# Import specific exceptions for error handling
from services.utils.error_utils import ValidationError

# Configure logger
logger = structlog.get_logger(__name__)

# Create Blueprint
analytics_bp = Blueprint("analytics", __name__)

@analytics_bp.route("/dashboard", methods=["GET"])
@auth_required()
def dashboard():
    """
    Get analytics dashboard summary data.
    Supports filtering by date range using ISO 8601 format query parameters.

    Query Parameters:
        start_date: Optional start date (ISO 8601 format, e.g., YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD)
        end_date: Optional end date (ISO 8601 format)

    Returns:
        200: Analytics summary
        400: Invalid date format or range
        500: Server error
    """
    try:
        # Parse date parameters using helper
        start_date, end_date = parse_date_range_args(request.args)

        # Get analytics summary
        summary = get_analytics_summary(
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify(summary)
    except ValidationError as e:
        logger.warning("Invalid date format/range in dashboard request", args=request.args, error=str(e))
        # Raised by parse_date_range_args for invalid format/range
        return jsonify({"error": "Invalid query parameters", "message": str(e)}), 400
    except Exception as e:
        logger.error("Error retrieving analytics dashboard", error=str(e), exc_info=True)
        return jsonify({
            "error": "Failed to retrieve analytics dashboard",
            "message": "An internal error occurred."
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
        page: Page number to retrieve (default: 1)
        page_size: Number of events per page (default: from config or 100)

    Returns:
        200: Paginated events list with metadata
        403: Insufficient permissions
        400: Invalid query parameter format or value
        500: Server error
    """
    try:
        # Parse parameters
        event_type = request.args.get("event_type")
        user_id = request.args.get("user_id")
        
        # Parse date params using helper (raises ValidationError)
        start_date, end_date = parse_date_range_args(request.args)
        
        # Parse pagination parameters, catching potential ValueError
        default_page_size = current_app.config.get('ANALYTICS_DEFAULT_LIMIT', 100)
        try:
            page_arg = request.args.get("page", "1")
            size_arg = request.args.get("page_size", str(default_page_size))
            page = int(page_arg)
            page_size = int(size_arg)
            if page < 1:
                page = 1 # Ensure page is at least 1
            if page_size < 1 or page_size > 1000: # Add a max page size check
                page_size = default_page_size # Ensure page_size is positive and reasonable
        except ValueError:
            logger.warning("Invalid non-integer pagination parameter provided", page_arg=page_arg, size_arg=size_arg)
            # Raise ValidationError to be caught by the handler below
            raise ValidationError(f"Invalid pagination parameters: 'page' ('{page_arg}') and 'page_size' ('{size_arg}') must be integers.")
        
        logger.debug("Retrieving analytics events via API (paginated)",
                     page=page, page_size=page_size, event_type=event_type, user_id=user_id,
                     start_date=start_date, end_date=end_date) # Fixed parenthesis
        
        # Get events
        # Get analytics service instance and call function
        analytics_service = get_analytics_service()
        paginated_result = analytics_service.get_analytics(
            event_type=event_type,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size
        )

        # Return the paginated result structure directly
        return jsonify(paginated_result)

    except ValidationError as e: # Catch validation errors from date parsing or pagination parsing
        logger.warning("Invalid query parameter format for analytics", args=request.args, error=str(e))
        return jsonify({"error": "Invalid query parameter format or value.", "details": str(e)}), 400
    except Exception as e:
        logger.error("Error retrieving analytics events via API", error=str(e), exc_info=True)
        return jsonify({"error": "Internal server error retrieving analytics"}), 500

# Use the service instance from the app context
# Assuming it's stored like current_app.analytics_service
def get_analytics_service() -> AnalyticsService:
    if not hasattr(current_app, 'analytics_service'):
        logger.error("AnalyticsService not initialized or attached to current_app")
        # This indicates a setup problem, raise a configuration error or similar
        raise RuntimeError("AnalyticsService is not available.")
    return current_app.analytics_service

@analytics_bp.route("/user-stats", methods=["GET"])
@auth_required() # Ensures g.user is set
def user_stats():
    """
    Get analytics summary statistics for the currently authenticated user.
    
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
        user_id = g.user["id"]
        
        # Parse date parameters
        start_date, end_date = parse_date_range_args(request.args)
        
        # Get analytics service instance and call summary function
        analytics_service = get_analytics_service()
        stats_summary = analytics_service.get_user_stats_summary(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # The service function now returns the structure needed for the response
        return jsonify(stats_summary)
    except ValidationError as e:
        logger.warning("Invalid date format/range in user-stats request", user_id=user_id, args=request.args, error=str(e))
        # Raised by parse_date_range_args for invalid format/range
        return jsonify({"error": "Invalid query parameters", "message": str(e)}), 400
    except redis.exceptions.RedisError as redis_err:
        logger.error("Redis error retrieving user stats summary", user_id=g.user.get('id', 'unknown'), error=str(redis_err), exc_info=True)
        return jsonify({"error": "Failed to retrieve user stats due to a temporary issue.", "message": "Please try again later."}), 503 # Service Unavailable
    except Exception as e:
        logger.error("Error retrieving user stats", user_id=g.user.get('id', 'unknown'), error=str(e), exc_info=True)
        return jsonify({
            "error": "Failed to retrieve user stats",
            "message": "An internal error occurred."
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
        
        # Parse date params using helper
        start_date, end_date = parse_date_range_args(request.args)
        
        # Validate export format parameter
        export_format = request.args.get("format", "json")
        if export_format not in ["json", "csv"]:
            logger.warning("Invalid export format requested", requested_format=export_format)
            raise ValidationError(f"Invalid format specified: '{export_format}'. Use 'json' or 'csv'.")
        
        # Get events with configured limit for export
        export_limit = current_app.config.get('ANALYTICS_EXPORT_LIMIT', 10000)
        events = get_analytics(
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            limit=export_limit
        )
        
        # Format response based on requested format
        if export_format == "csv":
            if not events:
                csv_data = "" # Return empty string if no events
            else:
                # Define desired fieldnames for consistency, or let helper infer
                # Example: fieldnames = ['timestamp', 'event_type', 'user_id', 'endpoint', 'status', 'duration_ms', 'error']
                csv_data = generate_csv_string(events) # Let helper infer fields for now

            # Create a Flask Response object for CSV download
            response = make_response(csv_data)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            response.headers["Content-Disposition"] = f"attachment; filename=analytics_export_{timestamp_str}.csv"
            response.headers["Content-Type"] = "text/csv"
            return response
        elif export_format == "json":
            # Return JSON response
            return jsonify({
                "events": events,
                "count": len(events),
                "limit": export_limit
            })
        # else: # Should not be reachable due to validation above
        #    pass

    except ValidationError as e:
        # Raised by date parsing or format validation
        logger.warning("Invalid parameter in export request", args=request.args, error=str(e))
        return jsonify({"error": "Invalid query parameters", "message": str(e)}), 400
    except Exception as e:
        logger.error("Error exporting analytics data", format=export_format, error=str(e), exc_info=True)
        return jsonify({"error": "Failed to export analytics data"}), 500