"""
Base Provider Classes

This module contains the base classes for LLM providers, including:
- LLMProvider abstract base class
- LLMResponse class for standardized responses
- StreamingHandler for handling streaming responses
"""

from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Iterator, Union


@dataclass
class LLMResponse:
    """
    Standardized response from LLM providers
    """
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    
    @property
    def is_error(self) -> bool:
        """Check if the response contains an error"""
        return self.error is not None


class StreamingHandler:
    """
    Handler for streaming responses from LLM providers
    """
    def __init__(self, 
                 callback: Callable[[str], None], 
                 accumulate: bool = True,
                 final_callback: Optional[Callable[[LLMResponse], None]] = None):
        """
        Initialize the streaming handler
        
        Args:
            callback: Function to call for each chunk of content
            accumulate: Whether to accumulate the full content
            final_callback: Function to call with the final response
        """
        self.callback = callback
        self.accumulate = accumulate
        self.final_callback = final_callback
        self.full_content = "" if accumulate else None
        
    def handle_chunk(self, chunk: str) -> None:
        """
        Handle a chunk of content from the stream
        
        Args:
            chunk: The content chunk
        """
        self.callback(chunk)
        if self.accumulate:
            self.full_content += chunk
            
    def finalize(self, response_data: Dict[str, Any]) -> LLMResponse:
        """
        Finalize the streaming response
        
        Args:
            response_data: Additional data for the response
            
        Returns:
            The final LLM response
        """
        if self.accumulate:
            response_data["content"] = self.full_content
            
        response = LLMResponse(**response_data)
        
        if self.final_callback:
            self.final_callback(response)
            
        return response


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 2):
        """
        Initialize the LLM provider
        
        Args:
            api_key: API key for the provider (default: None, read from env)
            model: Model to use (default: None, use provider's default)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 2)
        """
        self.api_key = api_key
        self.model = model or self.default_model()
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Load API key from environment if not provided
        if not self.api_key:
            self.api_key = self._get_api_key_from_env()
            
        if not self.api_key:
            raise ValueError(f"No API key provided for {self.__class__.__name__}")
        
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of the provider
        
        Returns:
            Provider name
        """
        pass
    
    @abstractmethod
    def default_model(self) -> str:
        """
        Get the default model for the provider
        
        Returns:
            Default model name
        """
        pass
    
    @abstractmethod
    def _get_api_key_from_env(self) -> Optional[str]:
        """
        Get the API key from environment variables
        
        Returns:
            API key or None if not found
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this provider
        
        Returns:
            Dictionary of capabilities
        """
        pass
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                params: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Generate a response from the LLM
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            params: Additional parameters for the provider
            
        Returns:
            Standardized LLM response
        """
        pass
    
    @abstractmethod
    def stream_generate(self,
                       prompt: str,
                       streaming_handler: StreamingHandler,
                       system_prompt: Optional[str] = None,
                       max_tokens: int = 2000,
                       temperature: float = 0.7,
                       params: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Generate a streaming response from the LLM
        
        Args:
            prompt: The user prompt
            streaming_handler: Handler for streaming chunks
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            params: Additional parameters for the provider
            
        Returns:
            Standardized LLM response
        """
        pass
    
    def _retry_with_exponential_backoff(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Retry an operation with exponential backoff
        
        Args:
            operation: Function to retry
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # If it's the last attempt, raise the exception
                if attempt == self.max_retries:
                    break
                
                # Calculate backoff time 
                backoff_time = 2 ** attempt
                print(f"Attempt {attempt+1} failed, retrying in {backoff_time}s: {str(e)}")
                time.sleep(backoff_time)
        
        # If we get here, all retries failed
        raise last_exception 