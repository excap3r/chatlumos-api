#!/usr/bin/env python3
"""
LLM Service Module

Provides language model operations like question decomposition and answer generation,
using provider instances initialized via the main application configuration.

Supports multiple LLM providers:
- DeepSeek
- Anthropic (Claude)
- OpenAI (GPT models)
- Groq
- OpenRouter
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Callable, Generator
import requests
# Removed Flask imports: Flask, request, jsonify, Response, stream_with_context, current_app
# Removed CORS
import structlog
from abc import ABC, abstractmethod
import re # Keep re for parsing

# Import provider functionality
from .providers import (
    LLMProvider,
    LLMResponse,
    StreamingHandler,
    get_provider as get_llm_provider_factory, # Renamed factory function
    list_available_providers
)

# Configure logger
logger = structlog.get_logger(__name__)

class LLMService:
    """Service class for LLM operations, managing provider instances."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLMService with configuration and providers."""
        self.config = config
        self.default_provider_name = self.config.get('DEFAULT_LLM_PROVIDER', 'openai')
        self.providers: Dict[str, LLMProvider] = {}

        logger.info("Initializing LLM Service", default_provider=self.default_provider_name)

        # Initialize configured providers (or at least the default one)
        # Example: Initialize default provider immediately
        try:
            self.default_provider = self._get_provider_instance(self.default_provider_name)
            self.providers[self.default_provider_name] = self.default_provider
            logger.info("Default LLM provider initialized successfully", provider=self.default_provider_name)
        except (ValueError, RuntimeError, KeyError) as e:
            logger.error("Failed to initialize default LLM provider", 
                         provider=self.default_provider_name, error=str(e), exc_info=True)
            self.default_provider = None # Ensure it's None if init fails
        
        # Optionally pre-initialize other providers listed in config if needed
        # other_providers = config.get('ADDITIONAL_LLM_PROVIDERS', [])
        # for provider_name in other_providers:
        #     if provider_name != self.default_provider_name:
        #         try:
        #             instance = self._get_provider_instance(provider_name)
        #             self.providers[provider_name] = instance
        #             logger.info(f"Initialized additional LLM provider: {provider_name}")
        #         except Exception as e:
        #             logger.error(f"Failed to initialize additional LLM provider: {provider_name}", error=str(e))

    def _get_provider_instance(self, provider_name: str) -> LLMProvider:
        """Internal helper to get an initialized provider using stored config."""
        if provider_name in self.providers:
             return self.providers[provider_name] # Return cached instance

        # Determine API key source from config
        config_key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'groq': 'GROQ_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
        }
        api_key_config_name = config_key_map.get(provider_name)
        if not api_key_config_name:
            raise ValueError(f"Unsupported provider or missing config key mapping: {provider_name}")

        api_key = self.config.get(api_key_config_name)
        if not api_key:
            raise ValueError(f"API key '{api_key_config_name}' not found in configuration for provider '{provider_name}'.")

        # Get model override if specified in config for this provider
        # Example config structure: LLM_MODELS = {"openai": "gpt-4o", "anthropic": "claude-3-sonnet..."}
        provider_models = self.config.get('LLM_MODELS', {})
        model = provider_models.get(provider_name) # Will be None if not specified, provider uses its default

        # Use the imported factory function
        provider_instance = get_llm_provider_factory(provider_name, api_key=api_key, model=model)
        self.providers[provider_name] = provider_instance # Cache the instance
        return provider_instance

    def get_capabilities(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Get capabilities for a specific provider or all initialized providers."""
        if provider_name:
            try:
                provider = self._get_provider_instance(provider_name)
                return provider.get_capabilities()
            except (ValueError, RuntimeError) as e:
                logger.error("Error getting provider capabilities", provider=provider_name, error=str(e))
                return {"error": str(e)}
        else:
            # Return capabilities for all successfully initialized providers
            all_caps = {}
            for name, instance in self.providers.items():
                 try:
                     all_caps[name] = instance.get_capabilities()
                 except Exception as e:
                     logger.warning(f"Could not get capabilities for {name}", error=str(e))
                     all_caps[name] = {"error": f"Failed to get capabilities: {str(e)}"}
            # Include names of providers that failed to initialize?
            # Maybe list available vs initialized separately
            return all_caps

    def list_providers(self) -> List[str]:
         """List providers successfully initialized by this service instance."""
         return list(self.providers.keys())

    # --- Core LLM Operations --- #

    def decompose_question(self,
                           question: str,
                           context: str = '',
                           provider_name: Optional[str] = None,
                           model: Optional[str] = None) -> LLMResponse:
        """Decompose a complex question into sub-questions using the specified provider."""
        
        if provider_name is None:
             provider_name = self.default_provider_name
             provider = self.default_provider # Use cached default
             if not provider:
                 err_msg = "Default LLM provider was not initialized successfully."
                 logger.error(err_msg)
                 return LLMResponse(provider=provider_name, model=model, is_error=True, error=err_msg)
        else:
            try:
                provider = self._get_provider_instance(provider_name)
            except (ValueError, RuntimeError) as e:
                 logger.error("Failed to get requested provider for decomposition", provider=provider_name, error=str(e))
                 return LLMResponse(provider=provider_name, model=model, is_error=True, error=str(e))

        system_prompt = """
        You are an expert at breaking down complex questions into simpler sub-questions.
        Your task is to analyze the given question and decompose it into 2-5 sub-questions that:
        1. Are simpler and more focused than the original question
        2. When answered together, would provide a comprehensive answer to the original question
        3. Are self-contained and can be answered independently

        Return your response STRICTLY as a JSON array of strings (sub-questions).
        Example: ["Sub-question 1?", "Sub-question 2?", "Sub-question 3?"]
        DO NOT include any other text, explanations, or markdown formatting outside the JSON array.
        """

        if context:
            prompt = f"Original question: {question}\n\nRelevant context: {context}\n\nPlease decompose this question into sub-questions following the specified JSON format."
        else:
            prompt = f"Original question: {question}\n\nPlease decompose this question into sub-questions following the specified JSON format."

        logger.info("Requesting question decomposition from LLM", provider=provider_name, model=model or provider.model)
        try:
            response = provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2, # Low temp for structured output
                model=model # Allow model override for this specific call
            )

            # Attempt to parse JSON directly from response content
            if not response.is_error and isinstance(response.content, str):
                 try:
                     parsed_content = json.loads(response.content)
                     if isinstance(parsed_content, list):
                          response.parsed_content = parsed_content # Store parsed list
                          logger.info("LLM decomposition response parsed successfully", count=len(parsed_content))
                     else:
                          logger.warning("LLM decomposition JSON was not a list", parsed_type=type(parsed_content).__name__)
                          response.is_error = True
                          response.error = "LLM response was valid JSON but not a list as expected."
                 except json.JSONDecodeError:
                     # If direct parsing fails, try extracting from markdown
                     logger.warning("LLM decomposition response was not direct JSON, attempting markdown extraction.", raw_response=response.content[:200]+"...")
                     json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response.content, re.DOTALL | re.IGNORECASE)
                     if json_match:
                         try:
                             parsed_content = json.loads(json_match.group(1))
                             if isinstance(parsed_content, list):
                                 response.parsed_content = parsed_content # Store parsed list
                                 logger.info("LLM decomposition response extracted from markdown and parsed successfully", count=len(parsed_content))
                             else:
                                 logger.warning("LLM decomposition JSON extracted from markdown was not a list", parsed_type=type(parsed_content).__name__)
                                 response.is_error = True
                                 response.error = "LLM response extracted from markdown was valid JSON but not a list."
                         except json.JSONDecodeError as json_err:
                             logger.error("Failed to parse extracted JSON from LLM decomposition response", json_string=json_match.group(1)[:100]+"...", error=str(json_err))
                             response.is_error = True
                             response.error = f"Failed to parse JSON extracted from markdown: {json_err}"
                     else:
                         logger.error("Could not parse LLM decomposition response as JSON or extract from markdown.")
                         response.is_error = True
                         response.error = "LLM response was not in the expected JSON format."
            
            return response

        except Exception as e:
            logger.error("Unexpected error during LLM question decomposition", provider=provider_name, error=str(e), exc_info=True)
            return LLMResponse(provider=provider_name, model=model or (provider.model if provider else None), is_error=True, error=f"Unexpected error: {str(e)}")

    def generate_answer(self,
                        question: str,
                        search_results: List[Dict[str, Any]],
                        provider_name: Optional[str] = None,
                        model: Optional[str] = None,
                        stream: bool = False,
                        stream_handler: Optional[StreamingHandler] = None) -> Union[LLMResponse, Generator[LLMResponse, None, None]]:
        """Generate an answer based on the question and search results using the specified provider."""
        
        if provider_name is None:
             provider_name = self.default_provider_name
             provider = self.default_provider
             if not provider:
                 err_msg = "Default LLM provider was not initialized successfully."
                 logger.error(err_msg)
                 error_resp = LLMResponse(provider=provider_name, model=model, is_error=True, error=err_msg)
                 if stream:
                      # Directly return a generator yielding the error
                      return (resp for resp in [error_resp]) 
                 else:
                      return error_resp
        else:
            try:
                provider = self._get_provider_instance(provider_name)
            except (ValueError, RuntimeError) as e:
                 error_msg = str(e)
                 logger.error("Failed to get requested provider for answer generation", provider=provider_name, error=error_msg)
                 error_resp = LLMResponse(provider=provider_name, model=model, is_error=True, error=error_msg)
                 if stream:
                      # Directly return a generator yielding the error
                      return (resp for resp in [error_resp])
                 else:
                    return error_resp

        # Build context from search results
        context_str = "\n\n".join([
            f"Source: {res.get('metadata', {}).get('filename', 'Unknown')} (Page: {res.get('metadata', {}).get('page', 'N/A')}, Score: {res.get('score', 'N/A'):.4f})\nContent: {res.get('page_content', '')}"
            for res in search_results
        ])

        system_prompt = """
        You are an AI assistant designed to answer questions based SOLELY on the provided context. 
        Do not use any external knowledge or prior information. 
        Synthesize the information from the search results to provide a comprehensive and accurate answer. 
        If the context does not contain enough information to answer the question, state that clearly. 
        Cite the sources used in your answer by referring to the filename and page number if available. Format citations like [filename, page X].
        Focus on clarity and conciseness.
        """

        prompt = f"Question: {question}\n\nContext from search results:\n---\n{context_str}\n---\n\nAnswer based only on the provided context:"

        logger.info("Requesting answer generation from LLM", provider=provider_name, model=model or provider.model, stream=stream)
        try:
            if stream:
                if not provider.supports_streaming:
                     error_msg = "Provider does not support streaming"
                     logger.error(f"Provider {provider_name} does not support streaming.")
                     error_resp = LLMResponse(provider=provider_name, model=model or provider.model, is_error=True, error=error_msg)
                     # Directly return a generator yielding the error
                     return (resp for resp in [error_resp])
                     
                # Return the generator directly from the provider
                return provider.generate_stream(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.5,
                    model=model,
                    handler=stream_handler
                )
            else:
                # Perform non-streaming generation
                response = provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.5,
                    model=model
                )
                logger.info("LLM answer generated (non-streaming)", provider=response.provider, model=response.model, error=response.error)
                return response

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error("Unexpected error during LLM answer generation", provider=provider_name, error=str(e), exc_info=True)
            error_resp = LLMResponse(provider=provider_name, model=model or (provider.model if provider else None), is_error=True, error=error_msg)
            if stream:
                 # Directly return a generator yielding the error
                 return (resp for resp in [error_resp])
            else:
                 return error_resp

    def complete(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 provider_name: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 stream: bool = False,
                 stream_handler: Optional[StreamingHandler] = None
                 ) -> Union[LLMResponse, Generator[LLMResponse, None, None]]:
        """Generic text completion using the specified provider."""
        
        if provider_name is None:
             provider_name = self.default_provider_name
             provider = self.default_provider
             if not provider:
                 err_msg = "Default LLM provider was not initialized successfully."
                 logger.error(err_msg)
                 error_resp = LLMResponse(provider=provider_name, model=model, is_error=True, error=err_msg)
                 if stream: 
                      return (resp for resp in [error_resp])
                 else: 
                      return error_resp
        else:
            try:
                provider = self._get_provider_instance(provider_name)
            except (ValueError, RuntimeError) as e:
                 error_msg = str(e)
                 logger.error("Failed to get requested provider for completion", provider=provider_name, error=error_msg)
                 error_resp = LLMResponse(provider=provider_name, model=model, is_error=True, error=error_msg)
                 if stream: 
                      return (resp for resp in [error_resp])
                 else: 
                      return error_resp

        logger.info("Requesting completion from LLM", provider=provider_name, model=model or provider.model, stream=stream)
        try:
            common_args = {
                'prompt': prompt,
                'system_prompt': system_prompt,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'model': model
            }
            if stream:
                if not provider.supports_streaming:
                    error_msg = "Provider does not support streaming for completion."
                    logger.error(f"Provider {provider_name} does not support streaming for completion.")
                    error_resp = LLMResponse(provider=provider_name, model=model or provider.model, is_error=True, error=error_msg)
                    return (resp for resp in [error_resp])
                
                # Return generator from provider
                return provider.generate_stream(**common_args, handler=stream_handler)
            else:
                # Non-streaming completion
                response = provider.generate(**common_args)
                logger.info("LLM completion generated (non-streaming)", provider=response.provider, model=response.model, error=response.error)
                return response

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error("Unexpected error during LLM completion", provider=provider_name, error=str(e), exc_info=True)
            error_resp = LLMResponse(provider=provider_name, model=model or (provider.model if provider else None), is_error=True, error=error_msg)
            if stream:
                 return (resp for resp in [error_resp])
            else:
                 return error_resp

# Removed Flask app instance, routes, CORS setup, get_config_value, get_provider
# Removed argparse and __main__ block if present