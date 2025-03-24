#!/usr/bin/env python3
"""
Helper module for using alternative LLM providers with PDF Wisdom Extractor.
"""

import os
import json
import requests
from typing import Dict, Any
import re

def extract_with_deepseek(chunk: str, api_key: str) -> Dict[str, Any]:
    """
    Extract concepts using DeepSeek API.
    """
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    system_prompt = """
    Jsi asistent pro extrakci informací z přepsaného textu přednášky. Tvým úkolem je:
    1. Identifikovat klíčové koncepty a myšlenky v textu
    2. Extrahovat otázky (implicitní i explicitní) a jejich odpovědi
    3. Shrnout hlavní body
    
    Vrať odpověď jako strukturovaný JSON s následujícími klíči:
    - key_concepts: Seznam klíčových konceptů a vysvětlení (každý koncept jako objekt s "concept" a "explanation")
    - qa_pairs: Seznam dvojic otázka-odpověď extrahovaných z textu (každý pár jako objekt s "question" a "answer")
    - summary: Stručné shrnutí hlavních bodů (max 3 věty)
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyzuj následující část přepisu přednášky a extrahuj klíčové informace:\n\n{chunk}"}
        ],
        "temperature": 0.2,
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            try:
                json_content = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_content:
                    return json.loads(json_content.group(1))
                else:
                    return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "error": "Could not parse JSON",
                    "raw_content": content
                }
        else:
            return {
                "error": f"API request failed with status {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {
            "error": f"Exception during API request: {str(e)}",
            "details": str(e)
        }

def extract_with_openai(chunk: str, api_key: str) -> Dict[str, Any]:
    """
    Extract concepts using OpenAI API.
    """
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
    
    system_prompt = """
    Jsi asistent pro extrakci informací z přepsaného textu přednášky. Tvým úkolem je:
    1. Identifikovat klíčové koncepty a myšlenky v textu
    2. Extrahovat otázky (implicitní i explicitní) a jejich odpovědi
    3. Shrnout hlavní body
    
    Vrať odpověď jako strukturovaný JSON s následujícími klíči:
    - key_concepts: Seznam klíčových konceptů a vysvětlení (každý koncept jako objekt s "concept" a "explanation")
    - qa_pairs: Seznam dvojic otázka-odpověď extrahovaných z textu (každý pár jako objekt s "question" a "answer")
    - summary: Stručné shrnutí hlavních bodů (max 3 věty)
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-3.5-turbo",  # or "gpt-4" for better results
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyzuj následující část přepisu přednášky a extrahuj klíčové informace:\n\n{chunk}"}
        ],
        "temperature": 0.2,
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            try:
                json_content = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_content:
                    return json.loads(json_content.group(1))
                else:
                    return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "error": "Could not parse JSON",
                    "raw_content": content
                }
        else:
            return {
                "error": f"API request failed with status {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {
            "error": f"Exception during API request: {str(e)}",
            "details": str(e)
        }

def extract_with_anthropic(chunk: str, api_key: str) -> Dict[str, Any]:
    """
    Extract concepts using Anthropic Claude API.
    """
    ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
    
    system_prompt = """
    Jsi asistent pro extrakci informací z přepsaného textu přednášky. Tvým úkolem je:
    1. Identifikovat klíčové koncepty a myšlenky v textu
    2. Extrahovat otázky (implicitní i explicitní) a jejich odpovědi
    3. Shrnout hlavní body
    
    Vrať odpověď jako strukturovaný JSON s následujícími klíči:
    - key_concepts: Seznam klíčových konceptů a vysvětlení (každý koncept jako objekt s "concept" a "explanation")
    - qa_pairs: Seznam dvojic otázka-odpověď extrahovaných z textu (každý pár jako objekt s "question" a "answer")
    - summary: Stručné shrnutí hlavních bodů (max 3 věty)
    """
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": "claude-3-haiku-20240307",  # or another Claude model
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": f"Analyzuj následující část přepisu přednášky a extrahuj klíčové informace:\n\n{chunk}"}
        ],
        "temperature": 0.2,
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            content = result["content"][0]["text"]
            
            # Extract JSON from response
            try:
                json_content = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_content:
                    return json.loads(json_content.group(1))
                else:
                    return json.loads(content)
            except json.JSONDecodeError:
                return {
                    "error": "Could not parse JSON",
                    "raw_content": content
                }
        else:
            return {
                "error": f"API request failed with status {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {
            "error": f"Exception during API request: {str(e)}",
            "details": str(e)
        }

def get_llm_extractor(provider: str = "deepseek"):
    """
    Get the appropriate LLM extractor function based on the provider.
    
    Args:
        provider: The LLM provider to use ('deepseek', 'openai', or 'anthropic')
        
    Returns:
        Function that can be used to extract concepts using the specified LLM
    """
    providers = {
        "deepseek": extract_with_deepseek,
        "openai": extract_with_openai,
        "anthropic": extract_with_anthropic
    }
    
    if provider not in providers:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: {', '.join(providers.keys())}")
    
    return providers[provider] 