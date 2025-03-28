"""
Provider Factory

This module contains functions for creating and managing LLM providers:
- get_provider: Factory function to create the appropriate provider instance
- list_available_providers: List all available providers
"""

import os
from typing import Dict, List, Optional, Any, Type

from .base import LLMProvider
from .deepseek import DeepSeekProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .groq import GroqProvider
from .openrouter import OpenRouterProvider

# Registry of available providers
_PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "deepseek": DeepSeekProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider
}

# Default provider (can be changed via DEFAULT_LLM_PROVIDER env var)
_DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()


def list_available_providers() -> List[str]:
    """
    List all available LLM providers
    
    Returns:
        List of provider names
    """
    return list(_PROVIDERS.keys())


def get_default_provider() -> str:
    """
    Get the name of the default provider
    
    Returns:
        Default provider name
    """
    return _DEFAULT_PROVIDER


def get_provider(provider_name: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function to create the appropriate LLM provider
    
    Args:
        provider_name: Name of the provider (default: None, use default provider)
        **kwargs: Additional arguments for the provider constructor
    
    Returns:
        LLM provider instance
        
    Raises:
        ValueError: If the provider is not available
    """
    # Use default provider if none specified
    if provider_name is None:
        provider_name = _DEFAULT_PROVIDER
    
    provider_name = provider_name.lower()
    
    # Check if provider is available
    if provider_name not in _PROVIDERS:
        raise ValueError(
            f"Provider '{provider_name}' not available. Available providers: {', '.join(list_available_providers())}"
        )
    
    # Create provider instance
    provider_class = _PROVIDERS[provider_name]
    return provider_class(**kwargs) 