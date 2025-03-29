#!/usr/bin/env python3
"""
Environment Variable Utilities

This module provides functions for securely accessing environment variables,
API keys, and configuration values.
"""

import os
import json
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_env_var(var_name: str, default: Any = None, required: bool = False) -> Any:
    """
    Load an environment variable with default value and validation.
    
    Args:
        var_name: Name of the environment variable
        default: Default value to return if not found
        required: Whether the variable is required (raises ValueError if missing)
        
    Returns:
        The value of the environment variable or default
        
    Raises:
        ValueError: If required=True and the variable is not set
    """
    value = os.getenv(var_name)
    
    if value is None:
        if required:
            raise ValueError(f"Required environment variable {var_name} not set")
        return default
    
    return value

def get_api_key(service_name: str, raise_error: bool = True) -> Optional[str]:
    """
    Get an API key for a specific service.
    
    Args:
        service_name: Name of the service (e.g., 'openai', 'pinecone')
        raise_error: Whether to raise an error if not found
        
    Returns:
        API key string or None if not found and raise_error=False
        
    Raises:
        ValueError: If the API key is not found and raise_error=True
    """
    # Standard environment variable formats to try
    formats = [
        f"{service_name.upper()}_API_KEY",
        f"{service_name}_API_KEY",
        f"{service_name.upper()}_KEY",
        f"{service_name}_KEY"
    ]
    
    # Try each format
    for var_name in formats:
        key = os.getenv(var_name)
        if key:
            return key
    
    # If we get here, the key wasn't found
    if raise_error:
        raise ValueError(f"API key for {service_name} not found in environment variables")
    return None

def get_config(config_name: str, default: Any = None) -> Any:
    """
    Get a configuration value from environment variables.
    Supports nested JSON configurations.
    
    Args:
        config_name: Name of the configuration variable
        default: Default value to return if not found
        
    Returns:
        Configuration value (parsed from JSON if applicable)
    """
    value = os.getenv(config_name)
    
    if value is None:
        return default
    
    # Try to parse as JSON
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, return as is
        return value 