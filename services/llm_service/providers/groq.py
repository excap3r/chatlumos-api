"""
Groq Provider Implementation

This module contains the Groq provider implementation for using LLMs with high performance.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any, Callable, Iterator, Union

from .base import LLMProvider, LLMResponse, StreamingHandler


class GroqProvider(LLMProvider):
    """
    Groq API provider implementation
    """
    
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    @property
    def provider_name(self) -> str:
        return "groq"
    
    def default_model(self) -> str:
        return "llama3-70b-8192"
    
    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv("GROQ_API_KEY")
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "streaming": True,
            "models": [
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768",
                "gemma-7b-it"
            ],
            "max_tokens": 32768,  # Mixtral has 32k context
            "supports_system_prompt": True,
            "supports_functions": False,
            "low_latency": True  # Groq is known for its low latency
        }
    
    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Prepare the messages for the Groq API
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def _prepare_request_data(self, 
                             prompt: str,
                             system_prompt: Optional[str] = None,
                             max_tokens: int = 2000,
                             temperature: float = 0.7,
                             stream: bool = False,
                             params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare the request data for the Groq API
        
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
        messages = self._prepare_messages(prompt, system_prompt)
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        # Add any additional parameters
        if params:
            data.update(params)
            
        return data
    
    def _make_request(self, data: Dict[str, Any], stream: bool = False) -> requests.Response:
        """
        Make a request to the Groq API
        
        Args:
            data: Request data
            stream: Whether to stream the response
            
        Returns:
            Response object
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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
        content = result["choices"][0]["message"]["content"]
        
        # Extract usage information if available
        usage = result.get("usage", {})
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            usage=usage,
            finish_reason=result["choices"][0].get("finish_reason")
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
                
                # Get the choice data
                choice = chunk_data.get("choices", [{}])[0]
                
                # Extract the finish reason if present
                if "finish_reason" in choice and choice["finish_reason"] is not None:
                    finish_reason = choice["finish_reason"]
                
                # Extract content delta
                delta = choice.get("delta", {})
                content_chunk = delta.get("content", "")
                
                if content_chunk:
                    streaming_handler.handle_chunk(content_chunk)
                    
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from stream: {json_data}")
            except Exception as e:
                print(f"Error processing stream chunk: {e}")
        
        # Return the final response
        return streaming_handler.finalize({
            "model": self.model,
            "provider": self.provider_name,
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
        Generate a response from Groq
        
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
        Generate a streaming response from Groq
        
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