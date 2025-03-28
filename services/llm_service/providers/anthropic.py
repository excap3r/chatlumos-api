"""
Anthropic Provider Implementation

This module contains the Anthropic provider implementation for using Claude models.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any, Callable, Iterator, Union

from .base import LLMProvider, LLMResponse, StreamingHandler


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider implementation for Claude models
    """
    
    API_URL = "https://api.anthropic.com/v1/messages"
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def default_model(self) -> str:
        return "claude-3-opus-20240229"
    
    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "streaming": True,
            "models": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-2.1",
                "claude-2.0",
                "claude-instant-1.2"
            ],
            "max_tokens": 200000,  # Claude 3 has very large context windows
            "supports_system_prompt": True
        }
    
    def _prepare_request_data(self, 
                             prompt: str,
                             system_prompt: Optional[str] = None,
                             max_tokens: int = 2000,
                             temperature: float = 0.7,
                             stream: bool = False,
                             params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare the request data for the Anthropic API
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            stream: Whether to stream the response
            params: Additional parameters
            
        Returns:
            Request data dictionary
        """
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        # Add system prompt if provided
        if system_prompt:
            data["system"] = system_prompt
            
        # Add any additional parameters
        if params:
            data.update(params)
            
        return data
    
    def _make_request(self, data: Dict[str, Any], stream: bool = False) -> requests.Response:
        """
        Make a request to the Anthropic API
        
        Args:
            data: Request data
            stream: Whether to stream the response
            
        Returns:
            Response object
        """
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        return requests.post(
            self.API_URL,
            headers=headers,
            json=data,
            stream=stream,
            timeout=self.timeout
        )
    
    def _process_response(self, response: requests.Response) -> LLMResponse:
        """
        Process a regular (non-streaming) response
        
        Args:
            response: Response object
            
        Returns:
            Standardized LLM response
        """
        if response.status_code != 200:
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider_name,
                error=f"API request failed with status {response.status_code}",
                details={"response_text": response.text}
            )
        
        result = response.json()
        content = result["content"][0]["text"]
        
        # Extract usage information if available
        usage = {
            "input_tokens": result.get("usage", {}).get("input_tokens", 0),
            "output_tokens": result.get("usage", {}).get("output_tokens", 0)
        }
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            usage=usage,
            finish_reason=result.get("stop_reason")
        )
    
    def _process_stream_response(self, 
                                response: requests.Response, 
                                streaming_handler: StreamingHandler) -> LLMResponse:
        """
        Process a streaming response
        
        Args:
            response: Response object
            streaming_handler: Handler for streaming chunks
            
        Returns:
            Standardized LLM response
        """
        if response.status_code != 200:
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider_name,
                error=f"API stream request failed with status {response.status_code}",
                details={"response_text": response.text}
            )
        
        finish_reason = None
        usage = None
        
        # Process each chunk of streaming data
        for line in response.iter_lines():
            if not line:
                continue
                
            # Skip empty lines
            line_text = line.decode("utf-8")
            if not line_text.startswith("data: "):
                continue
                
            # Extract the JSON data
            json_data = line_text[6:]  # Remove "data: " prefix
            
            # Check for the [DONE] message
            if json_data.strip() == "[DONE]":
                continue
                
            try:
                chunk_data = json.loads(json_data)
                
                # Handle different event types
                event_type = chunk_data.get("type")
                
                if event_type == "content_block_delta":
                    # This is a content chunk
                    delta = chunk_data.get("delta", {})
                    content_chunk = delta.get("text", "")
                    
                    if content_chunk:
                        streaming_handler.handle_chunk(content_chunk)
                        
                elif event_type == "message_stop":
                    # This contains the finish reason
                    finish_reason = chunk_data.get("stop_reason")
                    
                elif event_type == "message_delta":
                    # This can contain usage info in the last message
                    if "usage" in chunk_data.get("delta", {}):
                        usage = chunk_data["delta"]["usage"]
                        
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from stream: {json_data}")
            except Exception as e:
                print(f"Error processing stream chunk: {e}")
        
        # Return the final response
        return streaming_handler.finalize({
            "model": self.model,
            "provider": self.provider_name,
            "usage": usage,
            "finish_reason": finish_reason
            # Content will be added by streaming_handler.finalize()
        })
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                max_tokens: int = 2000,
                temperature: float = 0.7,
                params: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Generate a response from Anthropic
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            params: Additional parameters
            
        Returns:
            Standardized LLM response
        """
        try:
            data = self._prepare_request_data(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                params=params
            )
            
            # Make the request with retry logic
            response = self._retry_with_exponential_backoff(
                self._make_request,
                data=data,
                stream=False
            )
            
            return self._process_response(response)
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider_name,
                error=f"Error during API request: {str(e)}",
                details={"exception": str(e)}
            )
    
    def stream_generate(self,
                       prompt: str,
                       streaming_handler: StreamingHandler,
                       system_prompt: Optional[str] = None,
                       max_tokens: int = 2000,
                       temperature: float = 0.7,
                       params: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """
        Generate a streaming response from Anthropic
        
        Args:
            prompt: The user prompt
            streaming_handler: Handler for streaming chunks
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            params: Additional parameters
            
        Returns:
            Standardized LLM response
        """
        try:
            data = self._prepare_request_data(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                params=params
            )
            
            # Make the request with retry logic
            response = self._retry_with_exponential_backoff(
                self._make_request,
                data=data,
                stream=True
            )
            
            return self._process_stream_response(response, streaming_handler)
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model,
                provider=self.provider_name,
                error=f"Error during API request: {str(e)}",
                details={"exception": str(e)}
            ) 