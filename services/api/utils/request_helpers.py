#!/usr/bin/env python3
"""
API Request Helper Utilities
"""

import structlog
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any
from werkzeug.datastructures import MultiDict # For type hinting request.args

from services.utils.error_utils import ValidationError # Use existing ValidationError

logger = structlog.get_logger(__name__)

def parse_date_range_args(args: MultiDict) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parses 'start_date' and 'end_date' from request arguments.

    Args:
        args: The request arguments object (e.g., request.args).

    Returns:
        A tuple containing (start_date, end_date) as datetime objects or None.

    Raises:
        ValidationError: If date strings are provided but have an invalid ISO format.
    """
    start_date_str = args.get("start_date")
    end_date_str = args.get("end_date")

    start_date = None
    end_date = None

    try:
        if start_date_str:
            # Attempt to parse assuming ISO format with potential Z suffix
            if start_date_str.endswith('Z'):
                start_date_str = start_date_str[:-1] + '+00:00'
            start_date = datetime.fromisoformat(start_date_str)

        if end_date_str:
            if end_date_str.endswith('Z'):
                end_date_str = end_date_str[:-1] + '+00:00'
            end_date = datetime.fromisoformat(end_date_str)

    except ValueError as e:
        logger.warning("Invalid date format received in request args", 
                     start_date_arg=start_date_str, end_date_arg=end_date_str, error=str(e))
        raise ValidationError(f"Invalid date format. Use ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD). Original error: {e}")

    # Optional: Add validation logic (e.g., start_date <= end_date)
    if start_date and end_date and start_date > end_date:
        logger.warning("Invalid date range: start_date is after end_date", start_date=start_date, end_date=end_date)
        raise ValidationError("Invalid date range: start_date cannot be after end_date.")

    return start_date, end_date


import csv
from io import StringIO

def generate_csv_string(data: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> str:
    """
    Generates a CSV formatted string from a list of dictionaries.

    Args:
        data: A list of dictionaries representing the rows.
        fieldnames: Optional list of strings specifying the header and columns
                    to include and their order. If None, fieldnames are inferred
                    from the keys of the first dictionary in the data.

    Returns:
        A string containing the data in CSV format.
        Returns an empty string if the input data is empty.
    """
    if not data:
        return ""

    # Infer fieldnames from the first item if not provided
    if fieldnames is None:
        fieldnames = list(data[0].keys())

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)

    writer.writeheader()
    writer.writerows(data)

    return output.getvalue() 