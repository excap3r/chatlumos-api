"""
API Gateway Package

This package provides a central API gateway for all PDF Wisdom Extractor services:
- Service discovery and routing
- Request/response transformations
- Authentication and authorization
- Rate limiting and throttling
"""

import os
import requests
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv

# Import gateway internals
from .api_gateway import (
    create_app,
    register_service,
    get_service,
    get_service_url,
    proxy_request
)

class ServiceError(Exception):
    """Exception raised for errors in service requests."""
    def __init__(self, message: str, service: str = None, status_code: int = None, details: Any = None):
        self.message = message
        self.service = service
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

class APIGateway:
    """Client for interacting with the API Gateway service."""
    
    def __init__(self, gateway_url: str = None):
        """
        Initialize the API Gateway client.
        
        Args:
            gateway_url: The URL of the API Gateway (defaults to GATEWAY_URL env var)
        """
        # Load environment variables
        load_dotenv()
        
        self.gateway_url = gateway_url or os.getenv("GATEWAY_URL", "http://localhost:5000")
        self.gateway_url = self.gateway_url.rstrip('/')  # Remove trailing slash
    
    def health(self) -> Dict[str, Any]:
        """
        Check the health of the API Gateway and registered services.
        
        Returns:
            Dictionary with service health information
        """
        try:
            response = requests.get(f"{self.gateway_url}/health", timeout=10)
            if response.status_code != 200:
                return {
                    "status": "error",
                    "error": f"Gateway responded with status code {response.status_code}"
                }
            
            return response.json()
        except requests.RequestException as e:
            return {
                "status": "error",
                "error": f"Failed to connect to API Gateway: {str(e)}"
            }
    
    def request(
        self, 
        service: str, 
        path: str, 
        method: str = "GET", 
        params: Dict[str, Any] = None, 
        json: Dict[str, Any] = None,
        data: Any = None,
        headers: Dict[str, str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Make a request to a service through the API Gateway.
        
        Args:
            service: Service name (e.g., "llm", "vector", "db")
            path: API path for the service (e.g., "/search")
            method: HTTP method (GET, POST, PUT, DELETE)
            params: Query parameters
            json: JSON body data
            data: Form data
            headers: HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            Response data as dictionary
            
        Raises:
            ServiceError: If the service request fails
        """
        # Construct the full URL
        url = f"{self.gateway_url}/{service}{path}"
        
        # Make the request
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=json,
                data=data,
                headers=headers,
                timeout=timeout
            )
            
            # Check for success
            if response.status_code >= 200 and response.status_code < 300:
                # Return JSON response or empty dict
                try:
                    return response.json()
                except ValueError:
                    return {"content": response.text}
            else:
                # Try to get error details from response
                try:
                    error_data = response.json()
                    error_message = error_data.get("error", "Unknown error")
                except ValueError:
                    error_message = response.text or f"Service request failed with status code {response.status_code}"
                
                raise ServiceError(
                    message=error_message,
                    service=service,
                    status_code=response.status_code,
                    details=error_data if 'error_data' in locals() else None
                )
        except requests.RequestException as e:
            raise ServiceError(
                message=f"Request to {service} service failed: {str(e)}",
                service=service
            )
    
    def call_service(
        self, 
        service: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        method: str = "POST",
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Simplified method to call a service endpoint with data.
        
        Args:
            service: Service name (e.g., "llm", "vector", "db")
            endpoint: Service endpoint (e.g., "search", "extract")
            data: Data to send to the service
            method: HTTP method (default: POST)
            timeout: Request timeout in seconds
            
        Returns:
            Service response data
            
        Raises:
            ServiceError: If the service request fails
        """
        # Handle special case for method override in data
        actual_method = data.pop("method", method) if isinstance(data, dict) and "method" in data else method
        
        # Check if endpoint already starts with slash
        path = f"/{endpoint}" if not endpoint.startswith('/') else endpoint
        
        # Make the request
        return self.request(
            service=service,
            path=path,
            method=actual_method,
            json=data,
            timeout=timeout
        )
    
    def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for information using vector search.
        
        Args:
            params: Search parameters
                - query: Search query text
                - index_name: Optional vector index name
                - top_k: Optional number of results to return
                
        Returns:
            Search results
            
        Raises:
            ServiceError: If the search fails
        """
        return self.call_service("vector", "search", params)
    
    def ask(self, question: str, context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: Question to ask
            context: Optional context information
                
        Returns:
            Answer data
            
        Raises:
            ServiceError: If the question answering fails
        """
        data = {
            "question": question
        }
        
        if context:
            data["context"] = context
            
        return self.call_service("llm", "generate_answer", data)

__all__ = [
    'create_app',
    'register_service',
    'get_service',
    'get_service_url',
    'proxy_request',
    'APIGateway',
    'ServiceError'
] 