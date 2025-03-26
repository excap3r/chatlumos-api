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

def query_deepseek(prompt: str, system_prompt: str = None, api_key: str = None) -> Dict[str, Any]:
    """
    Query the DeepSeek API with a prompt.
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt to set context
        api_key: DeepSeek API key (defaults to env var)
    
    Returns:
        Dictionary with response content or error
    """
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return {"error": "No DeepSeek API key provided"}
    
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return {"content": content}
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

def decompose_question(question: str, api_key: str = None) -> Dict[str, Any]:
    """
    Decompose a complex question into sub-questions and concepts using DeepSeek.
    
    Args:
        question: The main question to decompose
        api_key: DeepSeek API key (optional)
    
    Returns:
        Dictionary with sub-questions, concepts, and analysis
    """
    system_prompt = """
    You are an expert question decomposer. Your task is to break down complex questions into:
    1. A set of simpler sub-questions that together would help answer the main question
    2. Key concepts mentioned in the question that would be important to understand
    
    Return your analysis as a structured JSON with:
    - sub_questions: Array of simpler questions (3-5 items)
    - concepts: Array of key concepts with brief descriptions (2-4 items)
    - search_queries: Array of effective search queries for a vector database (3-5 items)
    - analysis: Brief analysis of what knowledge is needed to answer this question
    
    Be precise and focused on extracting information that would help retrieve relevant knowledge from a vector database.
    """
    
    prompt = f"Decompose this question for semantic search:\n\n{question}"
    
    response = query_deepseek(prompt, system_prompt, api_key)
    
    if "error" in response:
        print(f"Error with DeepSeek: {response['error']}")
        return {
            "sub_questions": [],
            "concepts": [],
            "search_queries": [],
            "analysis": f"Failed to decompose question: {response.get('error', 'Unknown error')}"
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
        for key in ["sub_questions", "concepts", "search_queries", "analysis"]:
            if key not in data:
                data[key] = []
                
        return data
    except Exception as e:
        print(f"Failed to parse DeepSeek response: {e}")
        print("Raw response:", content)
        return {
            "sub_questions": [],
            "concepts": [],
            "search_queries": [],
            "analysis": "Failed to parse question decomposition"
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

def generate_answer(question: str, retrieved_data: Dict[str, List[Dict]], api_key: str = None, ensure_relevance: bool = False) -> str:
    """
    Generate an answer based on retrieved data.
    
    Args:
        question: The original question
        retrieved_data: Dictionary of search queries and their results
        api_key: DeepSeek API key (optional)
        ensure_relevance: If True, emphasize strict relevance to the user's specific question
        
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
        return "I don't have enough information to answer this question based on the available knowledge."
    
    # Create the system prompt
    relevance_instructions = ""
    if ensure_relevance:
        relevance_instructions = """
    8. IMPORTANT: Only include information that is directly relevant to answering the specific question asked
    9. Do not include tangential information or context that doesn't directly address the user's specific question
    10. Filter out any information that seems only loosely related to the question's core intent
    11. Focus on providing a precise answer to exactly what was asked, nothing more
    """
    
    system_prompt = f"""
    You are a knowledgeable assistant that answers questions based ONLY on the provided context information.
    Follow these guidelines:
    1. Only use information explicitly stated in the context
    2. If the context doesn't contain enough information to answer the question completely, state that clearly
    3. Cite sources of information from the context where appropriate
    4. Do not introduce information beyond what's provided in the context
    5. Answer in a clear, concise, and well-structured format
    6. Make responses personal and friendly, avoiding technical language when possible
    7. DO NOT include any meta-instructions or notes about how to structure your response in your final answer{relevance_instructions}
    """
    
    # Create the user prompt
    user_prompt = f"""
    Question: {question}
    
    Please answer using ONLY the following context information:
    
    {full_context}
    """
    
    # Query the LLM for an answer
    response = query_deepseek(user_prompt, system_prompt, api_key)
    
    if "error" in response:
        return f"Error generating answer: {response['error']}"
    
    # Clean up the response to remove any meta-instructions
    content = response["content"]
    
    # Remove common patterns of meta-instructions that might appear at the end
    instruction_patterns = [
        "Make response more personal and friendly",
        "you don't need to talk about concepts",
        "use that information as knowledge base",
        "Based on the provided context",
        "According to the context provided",
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
            all_queries.extend([c for c in decomposition["concepts"] if isinstance(c, str)])
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
                if isinstance(concept, dict) and "name" in concept and "description" in concept:
                    print(f"  - {concept['name']}: {concept['description']}")
                elif isinstance(concept, str):
                    print(f"  - {concept}")
        
        # Search for information
        print("\n--- Searching for Information ---")
        
        # Combine all search queries
        all_queries = []
        all_queries.append(args.question)  # The original question
        all_queries.extend(decomposition["sub_questions"])
        all_queries.extend([c for c in decomposition["concepts"] if isinstance(c, str)])
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