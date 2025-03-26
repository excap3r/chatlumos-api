#!/usr/bin/env python3
"""
Wisdom QA - Console Interface for Semantic Search with Vector Database

This script provides a command-line interface for asking questions,
decomposing them into sub-questions and concepts, and retrieving
relevant information from the Pinecone vector database to generate answers.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
import requests
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import textwrap
import random

# Import Pinecone
try:
    import pinecone
except ImportError:
    print("Error: Pinecone package not installed. Run 'pip install pinecone'")
    sys.exit(1)

# Import embedding model
try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required ML packages not installed. Run 'pip install torch sentence-transformers'")
    sys.exit(1)

# Global Pinecone instance
pc = None
embedding_model = None

# ----------------- Configuration -----------------

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Default embedding model
DEFAULT_INDEX = "wisdom-embeddings"   # Default Pinecone index
DEFAULT_PROVIDER = "deepseek"        # Default LLM provider
DEFAULT_TOP_K = 10                   # Default number of results to retrieve

# ----------------- DeepSeek Integration -----------------

def query_deepseek(prompt: str, system_prompt: str = None, api_key: str = None, 
                  max_tokens: int = 2000, timeout: int = 30, 
                  retries: int = 2, task_type: str = "general", 
                  stream: bool = False, stream_callback = None) -> Dict[str, Any]:
    """
    Query the DeepSeek API with a prompt.
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt to set context
        api_key: DeepSeek API key (defaults to env var)
        max_tokens: Maximum tokens to generate (default 2000)
        timeout: Request timeout in seconds (default 30)
        retries: Number of retry attempts (default 2)
        task_type: Task type for optimal parameter settings ("general", "translation", etc.)
        stream: Whether to stream the response (default False)
        stream_callback: Callback function to receive streamed chunks (required if stream=True)
    
    Returns:
        Dictionary with response content or error
    """
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return {"error": "No DeepSeek API key provided"}
    
    # Validate stream parameters
    if stream and stream_callback is None:
        return {"error": "stream_callback must be provided when stream=True"}
    
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # Set task-specific parameters
    if task_type == "translation":
        # Use lower values for translation tasks
        actual_max_tokens = min(max_tokens, 800)  # Cap at 800 for translations
        temperature = 0.2  # Lower temperature for more accurate translations
    else:
        actual_max_tokens = max_tokens
        temperature = 0.3  # Default temperature
    
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": actual_max_tokens,
        "stream": stream  # Add stream parameter to the API request
    }
    
    # Initialize retry backoff
    backoff = 1
    attempts = 0
    max_attempts = retries + 1  # Initial attempt + retries
    
    while attempts < max_attempts:
        attempts += 1
        try:
            print(f"DeepSeek API call attempt {attempts}/{max_attempts} for {task_type} task...")
            
            # Handle streaming or non-streaming responses differently
            if stream:
                # Streaming response
                with requests.post(
                    DEEPSEEK_API_URL,
                    headers=headers,
                    json=data,
                    stream=True,
                    timeout=timeout
                ) as response:
                    # Check for error in initial response
                    if response.status_code != 200:
                        error_msg = f"API stream request failed with status {response.status_code}"
                        try:
                            error_details = response.json()
                            error_msg += f": {error_details}"
                        except:
                            error_msg += f": {response.text}"
                        
                        print(f"DeepSeek API streaming error: {error_msg}")
                        
                        # Only retry server errors (5xx)
                        if response.status_code >= 500 and attempts < max_attempts:
                            print(f"Retrying stream in {backoff} seconds...")
                            time.sleep(backoff)
                            backoff *= 2  # Exponential backoff
                            continue
                        
                        return {
                            "error": error_msg,
                            "details": response.text
                        }
                    
                    # Process the stream
                    full_content = ""
                    
                    # Process each chunk of streaming data
                    for line in response.iter_lines():
                        if line:
                            # Skip empty lines and SSE comments
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                # Extract the JSON data
                                json_data = line_text[6:]  # Remove 'data: ' prefix
                                
                                # Skip the '[DONE]' message
                                if json_data.strip() == "[DONE]":
                                    continue
                                
                                try:
                                    chunk_data = json.loads(json_data)
                                    # Extract content delta
                                    delta = chunk_data.get('choices', [{}])[0].get('delta', {})
                                    content_chunk = delta.get('content', '')
                                    
                                    if content_chunk:
                                        # Accumulate the full content
                                        full_content += content_chunk
                                        
                                        # Call the callback with the chunk
                                        stream_callback(content_chunk)
                                except json.JSONDecodeError:
                                    print(f"Failed to decode JSON from stream: {json_data}")
                                except Exception as e:
                                    print(f"Error processing stream chunk: {e}")
                    
                    # Return the full content after stream is complete
                    return {"content": full_content}
            else:
                # Non-streaming response (original behavior)
                response = requests.post(
                    DEEPSEEK_API_URL, 
                    headers=headers, 
                    json=data, 
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    return {"content": content}
                elif response.status_code == 429:
                    # Rate limit error - always retry with backoff
                    print(f"DeepSeek API rate limit error (429). Backing off for {backoff} seconds...")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                    continue
                else:
                    error_msg = f"API request failed with status {response.status_code}"
                    try:
                        error_details = response.json()
                        error_msg += f": {error_details}"
                    except:
                        error_msg += f": {response.text}"
                    
                    print(f"DeepSeek API error: {error_msg}")
                    
                    # Only retry server errors (5xx)
                    if response.status_code >= 500 and attempts < max_attempts:
                        print(f"Retrying in {backoff} seconds...")
                        time.sleep(backoff)
                        backoff *= 2  # Exponential backoff
                        continue
                    
                    return {
                        "error": error_msg,
                        "details": response.text
                    }
        except requests.exceptions.Timeout:
            if attempts < max_attempts:
                print(f"DeepSeek API timeout after {timeout} seconds. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
                continue
            return {
                "error": f"API request timed out after {timeout} seconds",
                "details": f"Request timed out after {attempts} attempts"
            }
        except Exception as e:
            if attempts < max_attempts:
                print(f"DeepSeek API exception: {str(e)}. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
                continue
            return {
                "error": f"Exception during API request: {str(e)}",
                "details": str(e)
            }
    
    # If we get here, all retries failed
    return {
        "error": f"Failed after {max_attempts} attempts",
        "details": "Exceeded maximum retry attempts"
    }

def decompose_question(question: str, api_key: str = None, language: str = "") -> Dict[str, Any]:
    """
    Decompose a complex question into sub-questions and concepts using DeepSeek.
    
    Args:
        question: The main question to decompose
        api_key: DeepSeek API key (optional)
        language: The language code of the question (default: "", which means auto-detect)
    
    Returns:
        Dictionary with sub-questions, concepts, analysis, and language information
    """
    # Add translation instructions if language is not specified (auto-detect)
    translation_instruction = ""
    if not language:
        translation_instruction = """
        IMPORTANT: First identify the language of the question and include it in your response using the key "detected_language".
        Use the ISO 639-1 two-letter language code (e.g., "en" for English, "cs" for Czech).
        
        If the question is not in English, translate it to English and include the translation using the key "translated_question".
        Then decompose the TRANSLATED version of the question.
        
        If the question is in English, set "detected_language" to "en" and "translated_question" to the original question.
        """
    elif language != "en":
        # Language is specified and not English
        translation_instruction = """
        IMPORTANT: The question is not in English. Your first task is to translate it to English. 
        Include the translated question in your response using the key "translated_question". 
        Then decompose the TRANSLATED version of the question.
        """
    
    system_prompt = f"""
    You are an expert spiritual knowledge question decomposer. Your task is to break down complex spiritual questions into:
    1. A set of simpler sub-questions that together would help answer the main spiritual question
    2. Key spiritual concepts mentioned in the question that would be important to understand
    
    {translation_instruction}
    
    Return your analysis as a structured JSON with:
    - sub_questions: Array of simpler questions (3-5 items) focused on spiritual aspects of the query
    - concepts: Array of SIMPLE CONCEPT STRINGS with no descriptions (3-5 items). JUST PLAIN TERMS like "karma", "meditation", "energy healing" - NOT objects or dictionaries with explanations.
    - search_queries: Array of effective search queries for a vector database of spiritual teachings (3-7 items)
    - body_parts: Array of any specific body parts mentioned in the question (e.g., "legs", "heart", "left leg", "right knee")
    - analysis: Brief analysis of what spiritual knowledge is needed to answer this question
    - translated_question: The English translation of the question (if not already in English)
    - detected_language: The detected ISO language code for this question (e.g., "en", "cs", "de")
    
    When generating your response consider the following concepts:
    - 5D
    - Consciousness
    - Hermeticism
    - Oneness
    
    For spiritual concepts, FOCUS ON EXTRACTING SHORT KEYWORD PHRASES that would be good for searching, NOT detailed explanations.
    Each concept should be a single string of 1-4 words maximum, focused on core spiritual principles, practices, or ideas.
    
    IMPORTANT: If the question mentions specific body parts or physical symptoms, ALWAYS include those exact terms in both the concepts list AND as specific search queries. For example, if someone asks about "left leg pain", make sure "left leg" is included in concepts and create search queries like "left leg spiritual meaning" and "left leg energy".
    
    IMPORTANT FOR NON-ENGLISH QUESTIONS: If translating from another language, ALSO include any body parts, symptoms or key concepts in English. For example, if someone asks about "bolest v levé noze" in Czech, include "left leg", "leg pain", etc. in English in your concepts and search queries.
    
    Be precise and focused on extracting information that would help retrieve relevant spiritual knowledge from a vector database containing wisdom teachings.
    """
    
    prompt = f"Decompose this spiritual question for semantic search:\n\n{question}"
    
    # Use a longer timeout (45s) for this more complex cognitive task
    response = query_deepseek(
        prompt, 
        system_prompt, 
        api_key, 
        max_tokens=2000,
        timeout=45,
        retries=2,
        task_type="decomposition"
    )
    
    if "error" in response:
        print(f"Error with DeepSeek: {response['error']}")
        return {
            "sub_questions": [],
            "concepts": [],
            "search_queries": [],
            "body_parts": [],
            "analysis": f"Failed to decompose question: {response.get('error', 'Unknown error')}",
            "translated_question": question,  # Default to original
            "detected_language": language if language else "en"  # Default to specified language or English
        }
    
    content = response["content"]
    
    # Try to extract JSON from the response
    try:
        # First attempt: look for JSON block
        import re
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            # Second attempt: try to parse the entire content as JSON
            data = json.loads(content)
        
        # Ensure all expected keys exist
        for key in ["sub_questions", "concepts", "search_queries", "body_parts", "analysis"]:
            if key not in data:
                data[key] = []
        
        # Ensure language fields exist
        if "detected_language" not in data:
            # If specified, use that, otherwise default to English
            data["detected_language"] = language if language else "en"
            
        # If no translation but needed, use original
        if "translated_question" not in data:
            # For English or if language matches detected language, use original question
            if data["detected_language"] == "en" or data["detected_language"] == language:
                data["translated_question"] = question
            else:
                # Try to extract translation from content if it exists
                translation_match = re.search(r'translated_question["\s:]+([^"]+)', content)
                if translation_match:
                    data["translated_question"] = translation_match.group(1).strip()
                else:
                    # If all else fails, use original question
                    data["translated_question"] = question
        
        # Convert any complex concept objects to simple strings if needed
        if "concepts" in data:
            simple_concepts = []
            for concept in data["concepts"]:
                if isinstance(concept, dict) and "name" in concept:
                    simple_concepts.append(concept["name"])
                elif isinstance(concept, str):
                    simple_concepts.append(concept)
            data["concepts"] = simple_concepts
            
        # If body parts were identified, add them to concepts and search queries
        if "body_parts" in data and data["body_parts"]:
            # Add body parts to concepts if not already there
            for part in data["body_parts"]:
                if part.lower() not in [c.lower() for c in data["concepts"]]:
                    data["concepts"].append(part)
                    
            # Add specific body part search queries
            for part in data["body_parts"]:
                spiritual_queries = [
                    f"{part} spiritual meaning",
                    f"{part} energy",
                    f"{part} symbolism"
                ]
                
                # Add these to search queries if not similar to existing ones
                for query in spiritual_queries:
                    if not any(q.lower() in query.lower() or query.lower() in q.lower() for q in data["search_queries"]):
                        data["search_queries"].append(query)
                
        return data
    except Exception as e:
        print(f"Failed to parse DeepSeek response: {e}")
        print("Raw response:", content)
        return {
            "sub_questions": [],
            "concepts": [],
            "search_queries": [],
            "body_parts": [],
            "analysis": "Failed to parse question decomposition",
            "translated_question": question,  # Default to original
            "detected_language": language if language else "en"  # Default to specified language or English
        }

# ----------------- Vector Database Functions -----------------

def init_pinecone(api_key: str) -> bool:
    """Initialize Pinecone client."""
    global pc
    try:
        # Initialize with the current API (v6.0.2)
        pc = pinecone.Pinecone(api_key=api_key)
        
        # Verify connection by trying to list indexes
        try:
            indexes = pc.list_indexes()
            return True
        except Exception as e:
            print(f"Error verifying Pinecone connection: {e}")
            return False
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return False

def load_embedding_model(model_name: str = DEFAULT_MODEL) -> bool:
    """Load the embedding model."""
    global embedding_model
    try:
        embedding_model = SentenceTransformer(model_name)
        return True
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return False

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text."""
    global embedding_model
    if not embedding_model:
        raise ValueError("Embedding model not initialized")
    
    try:
        # Handle potentially long texts by truncating if necessary
        # Most models have context limits
        max_chars = 8000  # Adjust based on model
        if len(text) > max_chars:
            text = text[:max_chars]
            
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def search_pinecone(query: str, index_name: str = DEFAULT_INDEX, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
    """
    Search Pinecone for vectors similar to the query.
    
    Args:
        query: Text query to search for
        index_name: Name of the Pinecone index
        top_k: Number of results to return
    
    Returns:
        List of matches with metadata
    """
    global pc
    if not pc:
        raise ValueError("Pinecone not initialized")
    
    try:
        # Generate embedding for query
        query_embedding = generate_embedding(query)
        if not query_embedding:
            return []
        
        # Connect to index
        index = pc.Index(index_name)
        
        # Query the index
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Process and return matches
        matches = []
        for match in results.matches:
            matches.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
        
        return matches
    except Exception as e:
        print(f"Error searching Pinecone: {e}")
        return []

def batch_search_pinecone(queries: List[str], index_name: str = DEFAULT_INDEX, top_k: int = DEFAULT_TOP_K) -> Dict[str, List[Dict]]:
    """
    Search Pinecone for multiple queries in batch.
    
    Args:
        queries: List of text queries
        index_name: Name of the Pinecone index
        top_k: Number of results per query
        
    Returns:
        Dictionary mapping queries to their search results
    """
    results = {}
    for query in queries:
        results[query] = search_pinecone(query, index_name, top_k)
    return results

# ----------------- Answer Generation -----------------

def generate_answer(question: str, retrieved_data: Dict[str, List[Dict]], api_key: str = None, 
                   ensure_relevance: bool = False, target_language: str = "en", 
                   stream: bool = False, stream_callback = None) -> str:
    """
    Generate an answer based on retrieved data.
    
    Args:
        question: The original question
        retrieved_data: Dictionary of search queries and their results
        api_key: DeepSeek API key (optional)
        ensure_relevance: If True, emphasize strict relevance to the user's specific question
        target_language: Language code to generate the answer in (default: "en")
        stream: Whether to stream the response (default False)
        stream_callback: Callback function to receive streamed chunks
        
    Returns:
        Generated answer
    """
    # Prepare context from retrieved data
    context_pieces = []
    
    for query, results in retrieved_data.items():
        if not results:
            continue
            
        # Add query and results
        context_pieces.append(f"Search Query: {query}")
        
        for i, result in enumerate(results[:5]):  # Limit to top 5 per query
            metadata = result["metadata"]
            # Format based on the type of result
            if metadata.get("type") == "concept":
                context = (
                    f"CONCEPT ({result['score']:.2f}): {metadata.get('concept', 'Unknown')}\n"
                    f"EXPLANATION: {metadata.get('explanation', 'No explanation')}\n"
                    f"SOURCE: {metadata.get('document_title', 'Unknown')} by {metadata.get('document_author', 'Unknown')}"
                )
            elif metadata.get("type") == "qa_pair":
                context = (
                    f"QA PAIR ({result['score']:.2f}):\n"
                    f"Q: {metadata.get('question', 'Unknown')}\n"
                    f"A: {metadata.get('answer', 'No answer')}\n"
                    f"SOURCE: {metadata.get('document_title', 'Unknown')} by {metadata.get('document_author', 'Unknown')}"
                )
            elif metadata.get("type") == "question":
                context = (
                    f"QUESTION ({result['score']:.2f}): {metadata.get('question', 'Unknown')}\n"
                    f"A: {metadata.get('answer', 'No answer')}\n"
                    f"SOURCE: {metadata.get('document_title', 'Unknown')} by {metadata.get('document_author', 'Unknown')}"
                )
            else:
                context = f"RESULT ({result['score']:.2f}): {metadata.get('text', 'No text')}"
            
            context_pieces.append(context)
    
    # Create the full context
    full_context = "\n\n".join(context_pieces)
    
    # If there's no context, handle gracefully
    if not full_context:
        no_info_message = "I don't have enough information to answer this question based on the available knowledge."
        if target_language != "en":
            # Translate the message to target language
            system_prompt = f"You are a professional translator. Translate the text to {target_language}. Return only the translated text."
            response = query_deepseek(no_info_message, system_prompt, api_key, max_tokens=200, timeout=10)
            if "content" in response:
                return response["content"]
        return no_info_message
    
    # Create the system prompt
    relevance_instructions = ""
    if ensure_relevance:
        relevance_instructions = """
    8. IMPORTANT: Only include information that is directly relevant to answering the specific question asked
    9. Do not include tangential information or context that doesn't directly address the user's specific question
    10. Filter out any information that seems only loosely related to the question's core intent
    11. Focus on providing a precise answer to exactly what was asked, nothing more
    """
    
    # Add translation instruction if needed
    translation_instruction = ""
    if target_language != "en":
        translation_instruction = f"\n12. IMPORTANT: Your final answer must be in {target_language}. First understand the question and formulate your answer in English, then translate the final answer to {target_language}."
    
    # Original user question to help with personalization
    original_question_instruction = f"\nThe original user question is: '{question}'. Use this to make your response more personalized."
    
    system_prompt = f"""
    You are a knowledgeable and compassionate advisor who helps answer people's questions based on spiritual teachings.
    
    Follow these guidelines:
    1. Use the information in the provided context to inform your answer, but don't mention or reference the context directly
    2. If the context doesn't contain enough information, make a reasonable guess based on spiritual principles
    3. DO NOT cite sources or mention where the information comes from
    4. Create a personalized, direct answer as if you're speaking directly to the person
    5. Keep responses personal, compassionate and focused on helping the individual
    6. DO NOT include any meta-instructions or notes about how to structure your response
    7. NEVER say phrases like "Based on the information provided" or "According to the context"{relevance_instructions}{translation_instruction}
    
    Your answer should feel like it comes from your own wisdom and compassion, not from external sources.{original_question_instruction}
    """
    
    # Create the user prompt
    user_prompt = f"""
    Question: {question}
    
    Please answer using ONLY the following reference information (but don't mention this information directly):
    
    {full_context}
    """
    
    # Prepare streaming parameters if needed
    actual_stream = stream and stream_callback is not None
    
    # Query the LLM for an answer with appropriate parameters for a complete answer
    response = query_deepseek(
        user_prompt, 
        system_prompt, 
        api_key,
        max_tokens=2000,  # Keep max_tokens high for comprehensive answers
        timeout=60,       # Use longer timeout for this complex task
        retries=2,        # Two retries are reasonable here
        task_type="answer_generation",
        stream=actual_stream,
        stream_callback=stream_callback
    )
    
    if "error" in response:
        error_msg = f"Error generating answer: {response['error']}"
        if target_language != "en":
            # Translate the error message
            system_prompt = f"You are a professional translator. Translate the text to {target_language}. Return only the translated text."
            error_response = query_deepseek(error_msg, system_prompt, api_key, max_tokens=200, timeout=10)
            if "content" in error_response:
                return error_response["content"]
        return error_msg
    
    # Clean up the response to remove any meta-instructions
    content = response["content"]
    
    # Remove common patterns of meta-instructions that might appear at the end
    instruction_patterns = [
        "Make response more personal and friendly",
        "you don't need to talk about concepts",
        "use that information as knowledge base",
        "Based on the provided context",
        "According to the context provided",
        "According to the information",
        "Based on the information given",
        "From the information provided"
    ]
    
    # Check if any instruction patterns are at the end of the response
    for pattern in instruction_patterns:
        if pattern.lower() in content.lower():
            # Find where the instructions start and truncate
            index = content.lower().find(pattern.lower())
            if index > 0:
                content = content[:index].strip()
    
    return content

# ----------------- Command-Line Interface -----------------

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Wisdom QA - Ask questions using vector search")
    
    parser.add_argument("--question", type=str, help="The question to ask")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Embedding model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX,
                        help=f"Pinecone index name (default: {DEFAULT_INDEX})")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of results to retrieve per query (default: {DEFAULT_TOP_K})")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive mode")
    
    return parser.parse_args()

def interactive_mode():
    """Run the QA system in interactive console mode."""
    print("\nWisdom QA - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        try:
            question = input("\nYour question: ")
            if question.lower() in ["exit", "quit"]:
                print("Exiting session. Goodbye!")
                break
                
            if not question.strip():
                continue
                
            print("\nDecomposing question...")
            decomposition = decompose_question(question)
            
            # Print decomposition
            print("\n--- Question Analysis ---")
            print(decomposition["analysis"])
            
            print("\n--- Searching for Information ---")
            
            # Combine all search queries
            all_queries = []
            all_queries.append(question)  # The original question
            all_queries.extend(decomposition["sub_questions"])
            all_queries.extend(decomposition["concepts"])
            all_queries.extend(decomposition["search_queries"])
            
            # Remove duplicates while preserving order
            unique_queries = []
            seen = set()
            for q in all_queries:
                if isinstance(q, str) and q and q not in seen:
                    unique_queries.append(q)
                    seen.add(q)
            
            # Search for all queries
            print(f"Executing {len(unique_queries)} search queries...")
            search_results = batch_search_pinecone(unique_queries)
            
            # Generate answer
            print("\nGenerating answer...\n")
            answer = generate_answer(question, search_results)
            
            # Print answer with pretty formatting
            print("\n--- Answer ---")
            for line in answer.split('\n'):
                wrapped = textwrap.fill(line, width=100)
                print(wrapped)
            
            print("\n" + "-" * 100)
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Main function."""
    # Load environment variables
    load_dotenv()
    
    # Get Pinecone credentials
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not pinecone_api_key:
        print("Error: PINECONE_API_KEY must be set in .env file")
        sys.exit(1)
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize Pinecone
    print("Initializing Pinecone...")
    if not init_pinecone(pinecone_api_key):
        print("Failed to initialize Pinecone")
        sys.exit(1)
    
    # Load embedding model
    print(f"Loading embedding model: {args.model}...")
    if not load_embedding_model(args.model):
        print("Failed to load embedding model")
        sys.exit(1)
    
    # Check DeepSeek API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Warning: DEEPSEEK_API_KEY not set in .env file")
        print("This will affect question decomposition and answer generation")
    
    # Run in interactive mode or process single question
    if args.interactive:
        interactive_mode()
    elif args.question:
        print(f"\nQuestion: {args.question}")
        
        # Decompose the question
        print("\nDecomposing question...")
        decomposition = decompose_question(args.question)
        
        # Print decomposition
        print("\n--- Question Analysis ---")
        print(decomposition["analysis"])
        
        if decomposition["sub_questions"]:
            print("\nSub-questions:")
            for i, q in enumerate(decomposition["sub_questions"], 1):
                print(f"  {i}. {q}")
        
        if decomposition["concepts"]:
            print("\nKey concepts:")
            for concept in decomposition["concepts"]:
                print(f"  - {concept}")
        
        # Search for information
        print("\n--- Searching for Information ---")
        
        # Combine all search queries
        all_queries = []
        all_queries.append(args.question)  # The original question
        all_queries.extend(decomposition["sub_questions"])
        all_queries.extend(decomposition["concepts"])
        all_queries.extend(decomposition["search_queries"])
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in all_queries:
            if isinstance(q, str) and q and q not in seen:
                unique_queries.append(q)
                seen.add(q)
        
        # Search for all queries
        print(f"Executing {len(unique_queries)} search queries...")
        search_results = batch_search_pinecone(unique_queries, args.index, args.top_k)
        
        # Generate answer
        print("\nGenerating answer...\n")
        answer = generate_answer(args.question, search_results)
        
        # Print answer with pretty formatting
        print("\n--- Answer ---")
        for line in answer.split('\n'):
            wrapped = textwrap.fill(line, width=100)
            print(wrapped)
    else:
        print("Error: Please provide a question or use interactive mode")
        print("Run with --help for more information")

if __name__ == "__main__":
    main() 