#!/usr/bin/env python3
import os
import re
import json
import argparse
import requests
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import time
import datetime
import pytz
import sys
from tqdm import tqdm
import random
from collections import defaultdict
import threading

# Import database utilities
import db_utils

# Global rate limiting
api_rate_limits = defaultdict(lambda: {"last_call": 0, "backoff": 0})
api_lock = threading.Lock()  # Lock for thread-safe API call scheduling

# Try to import tiktoken, but use a fallback if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available, using simple character-based token estimation")

# Load environment variables from .env file
load_dotenv()

# 1. Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.strip()
        
        print(f"Successfully extracted {len(text)} characters from {pdf_path}")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

# Simple token counter that approximates token count for when tiktoken is not available
def estimate_tokens(text: str) -> int:
    """Estimate token count based on simple rules."""
    # Rough approximation: one token is about 4 characters on average
    return len(text) // 4

# 2. Split text into smaller, processable chunks
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split the text into overlapping chunks of specified size."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    i = 0
    while i < len(words):
        # Get chunk of words with specified size
        chunk = words[i:i + chunk_size]
        # Join words back into text
        chunks.append(" ".join(chunk))
        # Move to next chunk, considering overlap
        i += (chunk_size - overlap)
    
    # Ensure minimum meaningful size
    return [chunk for chunk in chunks if len(chunk.split()) > 10]

def is_off_peak_time():
    """Check if current time is during DeepSeek's off-peak hours (UTC 16:30-00:30)."""
    now = datetime.datetime.now(pytz.UTC)
    hour = now.hour
    minute = now.minute
    
    # Convert to minutes since midnight for easier comparison
    current_time_mins = hour * 60 + minute
    start_off_peak_mins = 16 * 60 + 30  # 16:30 UTC
    end_off_peak_mins = 24 * 60 + 30    # 00:30 UTC next day
    
    # Check if current time is between 16:30 and 00:30 UTC
    if start_off_peak_mins <= current_time_mins or current_time_mins <= (end_off_peak_mins % (24 * 60)):
        return True
    return False

# 3. Extract key concepts using LLM
def extract_concepts_with_llm(chunk: str, api_key: str, author_name: str, translate_to_english: bool = True) -> Dict[str, Any]:
    """Use LLM to extract key concepts, Q&A pairs, and summary from a text chunk."""
    global api_rate_limits, api_lock
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    # Apply rate limiting if needed for this key
    with api_lock:
        key_limits = api_rate_limits[api_key]
        current_time = time.time()
        time_since_last_call = current_time - key_limits["last_call"]
        if time_since_last_call < key_limits["backoff"]:
            sleep_time = key_limits["backoff"] - time_since_last_call
            print(f"  - Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        # Update last call time
        api_rate_limits[api_key]["last_call"] = time.time()
    
    # Check if we're in off-peak hours (50-75% discount)
    off_peak = is_off_peak_time()
    if off_peak:
        print("  - Using off-peak pricing (discount applied)")
    
    # Check if chunk is too large for the model
    if TIKTOKEN_AVAILABLE:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(chunk))
        # Optimize for token count to reduce costs
        if token_count > 8000:  # DeepSeek has 16K context window but we keep some room for response
            # Take the first part of the chunk (more important content usually starts here)
            max_safe_tokens = 7500  # Leave room for system prompt and response
            chunk = encoding.decode(encoding.encode(chunk)[:max_safe_tokens])
            # Add a note about the truncation
            chunk += "\n\n[... text truncated due to length limitations ...]"
    
    # Adjust system prompt based on translation setting
    if translate_to_english:
        system_prompt = f"""
        You are an assistant for extracting information from the transcribed lecture text by {author_name} (spiritual teacher). Extract only key information in a structured format without unnecessary text.
        
        When extracting, use the name "{author_name}" directly instead of generic terms like "speaker", "lecturer", "author", etc.
        
        IMPORTANT: Translate ALL outputs to ENGLISH, including concepts, explanations, questions, and answers.
        
        Return your response ONLY as JSON in the following format (without markdown code or explanations):
        {{
          "key_concepts": [
            {{"concept": "Concept name in English", "explanation": "Explanation in English (max 100 words)"}}
          ],
          "qa_pairs": [
            {{"question": "Question from text in English", "answer": "Answer to the question in English (max 80 words)"}}
          ]
        }}
        """
    else:
        system_prompt = f"""
        You are an assistant for extracting information from the transcribed lecture text by {author_name} (spiritual teacher). Extract only key information in a structured format without unnecessary text.
        
        When extracting, use the name "{author_name}" directly instead of generic terms like "speaker", "lecturer", "author", etc.
        
        IMPORTANT: Respond in the SAME LANGUAGE as the input text. Do not translate the content.
        
        Return your response ONLY as JSON in the following format (without markdown code or explanations):
        {{
          "key_concepts": [
            {{"concept": "Concept name", "explanation": "Explanation of the concept (max 100 words)"}}
          ],
          "qa_pairs": [
            {{"question": "Question from text", "answer": "Answer to the question (max 80 words)"}}
          ]
        }}
        """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # If in off-peak hours, we can afford to be more detailed
    max_output_tokens = 1200 if off_peak else 800
    
    data = {
        "model": "deepseek-chat",  # Using deepseek-chat which has lower pricing
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract key information from this part of the lecture{' and translate it to English' if translate_to_english else ''}:\n\n{chunk}"}
        ],
        "temperature": 0.1,  # Lower temperature for more consistent results and potentially fewer tokens
        "max_tokens": max_output_tokens,
        # Enable request caching to reduce costs (cached input tokens cost less)
        "use_cache": True
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            # Reset backoff for successful calls but maintain a minimal spacing
            with api_lock:
                api_rate_limits[api_key]["backoff"] = max(0.5, api_rate_limits[api_key]["backoff"] * 0.5)
            
            content = result["choices"][0]["message"]["content"]
            
            # Check for token usage if available
            if "usage" in result:
                print(f"  - Token usage: {result['usage']['total_tokens']} " +
                      f"(prompt: {result['usage']['prompt_tokens']}, " +
                      f"completion: {result['usage']['completion_tokens']})")
            
            # Extract JSON from response (handle different possible formats)
            try:
                # Try direct JSON parsing first
                try:
                    extracted_data = json.loads(content)
                    # Validate structure
                    if not all(k in extracted_data for k in ["key_concepts", "qa_pairs"]):
                        raise ValueError("Missing required keys in JSON")
                    return extracted_data
                except (json.JSONDecodeError, ValueError):
                    # If direct parsing fails, look for JSON in code blocks
                    json_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                        try:
                            extracted_data = json.loads(json_content)
                            if not all(k in extracted_data for k in ["key_concepts", "qa_pairs"]):
                                raise ValueError("Missing required keys in JSON")
                            print("  - Successfully extracted JSON from code block")
                            return extracted_data
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"  - Error parsing JSON from code block: {str(e)}")
                    
                    # If that fails, try to find anything that looks like JSON
                    json_pattern = r'({[\s\S]*?})'
                    json_candidates = re.findall(json_pattern, content)
                    for candidate in json_candidates:
                        try:
                            extracted_data = json.loads(candidate)
                            if all(k in extracted_data for k in ["key_concepts", "qa_pairs"]):
                                print("  - Successfully extracted JSON from content")
                                return extracted_data
                        except:
                            continue
                    
                    # If no valid JSON found, create a basic structure with any extractable content
                    print("  - Warning: Could not parse response as valid JSON, creating basic structure")
                    
                    # Try to extract concepts using regex patterns
                    concepts = []
                    qa_pairs = []
                    
                    # Try to detect JSON-like content even when parsing failed
                    if '"key_concepts"' in content and '"concept"' in content and '"explanation"' in content:
                        print("  - Detected JSON-like content, trying to extract manually")
                        # Try to extract concepts
                        concept_json_matches = re.findall(r'"concept"\s*:\s*"([^"]+)"[^}]*"explanation"\s*:\s*"([^"]+)"', content)
                        for c_match in concept_json_matches:
                            concepts.append({"concept": c_match[0], "explanation": c_match[1]})
                        
                        # Try to extract QA pairs
                        qa_json_matches = re.findall(r'"question"\s*:\s*"([^"]+)"[^}]*"answer"\s*:\s*"([^"]+)"', content)
                        for qa_match in qa_json_matches:
                            qa_pairs.append({"question": qa_match[0], "answer": qa_match[1]})
                    
                    # Look for concept sections in plain text format
                    if not concepts:
                        concept_matches = re.findall(r'(?:Koncept|Klíčový koncept)[^\n]*:\s*([^\n]+)(?:\n+(?:Vysvětlení|Explanation)[^\n]*:\s*([^\n]+))?', content, re.MULTILINE) 
                        for concept_match in concept_matches:
                            concept = concept_match[0].strip()
                            explanation = concept_match[1].strip() if len(concept_match) > 1 else ""
                            concepts.append({"concept": concept, "explanation": explanation})
                    
                    # Look for Q&A sections in plain text format
                    if not qa_pairs:
                        qa_matches = re.findall(r'(?:Otázka|Q)[^\n]*:\s*([^\n]+)\n+(?:Odpověď|A)[^\n]*:\s*([^\n]+)', content, re.MULTILINE)
                        for qa_match in qa_matches:
                            question = qa_match[0].strip()
                            answer = qa_match[1].strip()
                            qa_pairs.append({"question": question, "answer": answer})
                    
                    # If nothing was found with regex, return minimal structure with warning
                    if not concepts and not qa_pairs:
                        return {
                            "key_concepts": [{"concept": "Extrakce selhala", "explanation": "Model nevrátil validní JSON strukturu."}],
                            "qa_pairs": [],
                            "raw_content": content
                        }
                    
                    return {
                        "key_concepts": concepts,
                        "qa_pairs": qa_pairs
                    }
            except Exception as parse_error:
                print(f"  - Error parsing JSON response: {str(parse_error)}")
                return {
                    "key_concepts": [{"concept": "Chyba zpracování JSON", "explanation": f"Error: {str(parse_error)}"}],
                    "qa_pairs": [],
                    "raw_content": content[:500] + ("..." if len(content) > 500 else "")
                }
        else:
            error_message = f"API request failed with status {response.status_code}"
            try:
                error_details = response.json()
                
                # Check for rate limit errors
                if response.status_code == 429 or "rate limit" in str(error_details).lower():
                    # Exponential backoff with jitter
                    with api_lock:
                        current_backoff = api_rate_limits[api_key]["backoff"]
                        new_backoff = max(2.0, current_backoff * 2) + random.uniform(0, 1)
                        api_rate_limits[api_key]["backoff"] = min(30, new_backoff)  # Cap at 30 seconds
                        
                        print(f"  - Rate limit detected! Adding backoff of {api_rate_limits[api_key]['backoff']:.2f} seconds")
                    
                    time.sleep(api_rate_limits[api_key]["backoff"])
                    
                    # Retry immediately with the same chunks
                    print("  - Retrying request after rate limit backoff...")
                    return extract_concepts_with_llm(chunk, api_key, author_name, translate_to_english)
                
                # Check for token limit errors
                if "error" in error_details and "maximum context length" in error_details.get("error", {}).get("message", ""):
                    # Retry with a shorter chunk
                    if TIKTOKEN_AVAILABLE:
                        encoding = tiktoken.get_encoding("cl100k_base")
                        token_count = len(encoding.encode(chunk))
                        # Take the first part only (more important usually)
                        max_safe_tokens = 5000
                        shortened_chunk = encoding.decode(encoding.encode(chunk)[:max_safe_tokens])
                        # Add a note about the truncation
                        shortened_chunk += "\n\n[... text truncated due to length limitations ...]"
                        
                        # Try again with the shortened chunk
                        data["messages"][1]["content"] = f"Extract key information from this part of the lecture (text was shortened):\n\n{shortened_chunk}"
                        retry_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
                        
                        if retry_response.status_code == 200:
                            retry_result = retry_response.json()
                            retry_content = retry_result["choices"][0]["message"]["content"]
                            
                            try:
                                # Try direct JSON parsing first
                                try:
                                    extracted_data = json.loads(retry_content)
                                    # Validate structure
                                    if not all(k in extracted_data for k in ["key_concepts", "qa_pairs"]):
                                        raise ValueError("Missing required keys in JSON")
                                    return extracted_data
                                except (json.JSONDecodeError, ValueError):
                                    # If direct parsing fails, look for JSON in code blocks
                                    json_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', retry_content, re.DOTALL)
                                    if json_match:
                                        json_content = json_match.group(1).strip()
                                        try:
                                            extracted_data = json.loads(json_content)
                                            if not all(k in extracted_data for k in ["key_concepts", "qa_pairs"]):
                                                raise ValueError("Missing required keys in JSON")
                                            print("  - Successfully extracted JSON from code block (retry)")
                                            return extracted_data
                                        except (json.JSONDecodeError, ValueError) as e:
                                            print(f"  - Error parsing JSON from code block in retry: {str(e)}")
                                    
                                    # If that fails too, try to find anything that looks like JSON
                                    json_pattern = r'({[\s\S]*?})'
                                    json_candidates = re.findall(json_pattern, retry_content)
                                    for candidate in json_candidates:
                                        try:
                                            extracted_data = json.loads(candidate)
                                            if all(k in extracted_data for k in ["key_concepts", "qa_pairs"]):
                                                print("  - Successfully extracted JSON from retry content")
                                                return extracted_data
                                        except:
                                            continue
                                            
                                    # Return error if all extraction attempts failed
                                    return {
                                        "key_concepts": [{"concept": "Extrakce selhala (retry)", "explanation": "Model nevrátil validní JSON strukturu ani po opakovaném pokusu."}],
                                        "qa_pairs": [],
                                        "raw_content": retry_content[:500] + ("..." if len(retry_content) > 500 else "")
                                    }
                            except Exception as parse_error:
                                print(f"  - Error parsing JSON from retry: {str(parse_error)}")
                                return {
                                    "key_concepts": [{"concept": "Chyba zpracování JSON (retry)", "explanation": f"Error: {str(parse_error)}"}],
                                    "qa_pairs": [],
                                    "raw_content": retry_content[:500] + ("..." if len(retry_content) > 500 else "")
                                }
            except Exception as e:
                print(f"  - Error handling API failure: {str(e)}")
                
            return {
                "key_concepts": [{"concept": "Chyba API", "explanation": error_message}],
                "qa_pairs": [],
                "error": error_message,
                "details": response.text[:500] + ("..." if len(response.text) > 500 else "")
            }
    except Exception as e:
        print(f"  - Exception during API request: {str(e)}")
        return {
            "key_concepts": [{"concept": "Výjimka", "explanation": f"Exception during API request: {str(e)}"}],
            "qa_pairs": [],
            "error": f"Exception during API request: {str(e)}",
            "details": str(e)
        }

# 4. Deduplicate concepts
def deduplicate_concepts(concepts: List[Dict], author_name: str) -> List[Dict]:
    """Remove duplicate concepts and merge similar ones."""
    # Create a dictionary to track concepts by name
    concept_dict = {}
    
    # Regular expressions to find generic speaker references
    speaker_patterns = [
        (r'\břečnice\b', author_name),
        (r'\bpřednášející\b', author_name),
        (r'\bautorka\b', author_name),
        (r'\bmluvčí\b', author_name)
    ]
    
    for concept in concepts:
        # Skip if the concept doesn't have the expected structure
        if not isinstance(concept, dict) or 'concept' not in concept or 'explanation' not in concept:
            continue
        
        # Replace generic speaker references with the author's name
        for pattern, replacement in speaker_patterns:
            if 'explanation' in concept:
                concept['explanation'] = re.sub(pattern, replacement, concept['explanation'], flags=re.IGNORECASE)
            
        name = concept['concept'].lower()
        if name in concept_dict:
            # Merge explanations if it brings new information
            current_exp = concept_dict[name]['explanation']
            new_exp = concept['explanation']
            if len(new_exp) > len(current_exp) * 1.2:  # Add only if significantly longer
                concept_dict[name]['explanation'] = new_exp
        else:
            concept_dict[name] = concept
    
    return list(concept_dict.values())

# Process Q&A pairs
def process_qa_pairs(qa_pairs: List[Dict], author_name: str) -> List[Dict]:
    """Replace generic speaker references with the author's name in Q&A pairs."""
    processed_pairs = []
    
    # Regular expressions to find generic speaker references
    speaker_patterns = [
        (r'\břečnice\b', author_name),
        (r'\bpřednášející\b', author_name),
        (r'\bautorka\b', author_name),
        (r'\bmluvčí\b', author_name)
    ]
    
    for qa in qa_pairs:
        if not isinstance(qa, dict) or 'question' not in qa or 'answer' not in qa:
            processed_pairs.append(qa)
            continue
            
        # Create a new processed entry
        processed_qa = {
            'question': qa['question'],
            'answer': qa['answer']
        }
        
        # Replace generic speaker references in both question and answer
        for pattern, replacement in speaker_patterns:
            processed_qa['question'] = re.sub(pattern, replacement, processed_qa['question'], flags=re.IGNORECASE)
            processed_qa['answer'] = re.sub(pattern, replacement, processed_qa['answer'], flags=re.IGNORECASE)
            
        processed_pairs.append(processed_qa)
    
    return processed_pairs

# 5. Functions for processing and summarizing extraction results
def deduplicate_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
    """Remove duplicate Q&A pairs and keep the most detailed ones."""
    # Similar to deduplicating concepts
    qa_dict = {}
    
    for qa in qa_pairs:
        if not isinstance(qa, dict) or 'question' not in qa or 'answer' not in qa:
            continue
            
        question_key = qa['question'].lower()
        if question_key in qa_dict:
            # Keep the longer, more detailed answer
            if len(qa['answer']) > len(qa_dict[question_key]['answer']):
                qa_dict[question_key] = qa
        else:
            qa_dict[question_key] = qa
            
    return list(qa_dict.values())

def generate_combined_summary(concepts, qa_pairs, chunk_count):
    """Generate a brief summary of the entire document from key concepts and Q&A pairs."""
    top_concepts = concepts[:5] if len(concepts) > 5 else concepts
    
    # Get author name from first concept if available, fallback to a default
    author_name = "Author"
    if concepts and len(concepts) > 0 and 'explanation' in concepts[0]:
        # Try to extract author name from any explanation that contains it
        for concept in concepts:
            if 'Adamcová' in concept.get('explanation', ''):
                author_name = "Iva Adamcová"
                break
            # Look for other common author references that might be in the explanation
            for name_pattern in ['učitel', 'přednášející', 'autor']:
                if name_pattern in concept.get('explanation', '').lower():
                    # Try to get nearby words that might be names
                    text = concept['explanation']
                    words = text.split()
                    for i, word in enumerate(words):
                        if name_pattern in word.lower() and i+1 < len(words):
                            # Next word might be author name
                            potential_name = words[i+1]
                            if potential_name[0].isupper():
                                author_name = potential_name
                                break
    
    summary = f"# Shrnutí extrakce učení {author_name}\n\n"
    
    # Add statistics
    summary += f"## Statistiky extrakce\n"
    summary += f"- Celkem zpracováno částí textu: {chunk_count}\n"
    summary += f"- Celkem unikátních konceptů: {len(concepts)}\n"
    summary += f"- Celkem otázek a odpovědí: {len(qa_pairs)}\n\n"
    
    # Add top concepts
    summary += "## Klíčové koncepty\n\n"
    for i, concept in enumerate(top_concepts, 1):
        summary += f"{i}. **{concept['concept']}**: {concept['explanation']}\n\n"
    
    # Add notice about all concepts and Q&A pairs
    summary += "Všechny koncepty a otázky s odpověďmi jsou k dispozici v databázi.\n"
    
    return summary

# 6. Create summary document from database
def create_summary_document(document_id: int, output_dir: str):
    """Create a more detailed summary document with all key concepts and Q&A pairs from database."""
    # Get document info
    document_info = db_utils.get_document_info(document_id)
    if not document_info:
        print(f"Error: Could not retrieve document information for ID {document_id}")
        return
    
    # Get concepts and QA pairs from database
    concepts = db_utils.get_all_concepts(document_id)
    qa_pairs = db_utils.get_all_qa_pairs(document_id)
    
    # Create summary content
    content = f"# Shrnutí přednášky {document_info['author']} - {document_info['title']}\n\n"
    content += f"Datum zpracování: {document_info['processed_date']}\n\n"
    
    # Add key concepts section
    content += "## Klíčové koncepty\n\n"
    for i, concept in enumerate(concepts, 1):
        content += f"{i}. **{concept['concept']}**\n"
        content += f"   {concept['explanation']}\n\n"
    
    # Add Q&A section
    content += "## Otázky a odpovědi\n\n"
    for i, qa in enumerate(qa_pairs, 1):
        content += f"{i}. **Otázka**: {qa['question']}\n"
        content += f"   **Odpověď**: {qa['answer']}\n\n"
    
    # Save the summary document
    summary_path = os.path.join(output_dir, f"{document_info['filename']}_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Summary document saved to {summary_path}")
    
    return summary_path

def process_single_chunk(chunk_idx: int, chunk: str, api_key: str, document_id: int, translate_to_english: bool = False) -> Tuple[int, List[Dict], List[Dict]]:
    """Process a single chunk and store results in the database."""
    print(f"\nProcessing chunk {chunk_idx+1}...")
    start_time = time.time()
    
    try:
        # Get the chunk_id from the database - use a new connection for thread safety
        conn = db_utils.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, status) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE status = %s, chunk_text = %s",
                (document_id, chunk_idx, chunk, "processing", "processing", chunk)
            )
            conn.commit()
            
            # Get the chunk_id
            cursor.execute(
                "SELECT chunk_id FROM document_chunks WHERE document_id = %s AND chunk_index = %s",
                (document_id, chunk_idx)
            )
            result = cursor.fetchone()
            chunk_id = result[0] if result else None
        finally:
            cursor.close()
            conn.close()
            
        if not chunk_id:
            print(f"  - Error: Failed to create chunk record in database")
            return chunk_idx, [], []
        
        # Get author name from document info
        document_info = db_utils.get_document_info(document_id)
        author_name = document_info['author'] if document_info else "Unknown Author"
        
        # Extract key concepts and Q&A pairs
        result = extract_concepts_with_llm(chunk, api_key, author_name, translate_to_english)
        
        # Extract concepts and Q&A pairs
        concepts = result.get("key_concepts", [])
        qa_pairs = result.get("qa_pairs", [])
        
        # Store concepts and QA pairs in the database - use new connections for thread safety
        if concepts:
            conn = db_utils.get_connection()
            cursor = conn.cursor()
            try:
                # Insert concepts
                for concept in concepts:
                    cursor.execute(
                        """
                        INSERT INTO concepts (chunk_id, document_id, concept_name, explanation) 
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            explanation = IF(LENGTH(VALUES(explanation)) > LENGTH(explanation), VALUES(explanation), explanation),
                            chunk_id = IF(LENGTH(VALUES(explanation)) > LENGTH(explanation), VALUES(chunk_id), chunk_id)
                        """,
                        (chunk_id, document_id, concept.get('concept', ''), concept.get('explanation', ''))
                    )
                conn.commit()
            except Exception as e:
                print(f"  - Warning: Failed to store concepts in database: {str(e)}")
                conn.rollback()
            finally:
                cursor.close()
                conn.close()
        
        if qa_pairs:
            conn = db_utils.get_connection()
            cursor = conn.cursor()
            try:
                # Insert QA pairs
                for qa in qa_pairs:
                    cursor.execute(
                        """
                        INSERT INTO qa_pairs (chunk_id, document_id, question, answer) 
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            answer = IF(LENGTH(VALUES(answer)) > LENGTH(answer), VALUES(answer), answer),
                            chunk_id = IF(LENGTH(VALUES(answer)) > LENGTH(answer), VALUES(chunk_id), chunk_id)
                        """,
                        (chunk_id, document_id, qa.get('question', ''), qa.get('answer', ''))
                    )
                conn.commit()
            except Exception as e:
                print(f"  - Warning: Failed to store QA pairs in database: {str(e)}")
                conn.rollback()
            finally:
                cursor.close()
                conn.close()
        
        # Update chunk status to completed
        conn = db_utils.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE document_chunks SET status = %s WHERE chunk_id = %s",
                ("completed", chunk_id)
            )
            conn.commit()
        except Exception as e:
            print(f"  - Warning: Failed to update chunk status: {str(e)}")
        finally:
            cursor.close()
            conn.close()
        
        # Print statistics
        elapsed_time = time.time() - start_time
        print(f"  - Found {len(concepts)} concepts and {len(qa_pairs)} Q&A pairs")
        print(f"  - Processing time: {elapsed_time:.2f} seconds")
        
        # Return results for this chunk
        return chunk_idx, concepts, qa_pairs
    
    except Exception as e:
        print(f"  - Error in process_single_chunk: {str(e)}")
        # Try to mark the chunk as failed if possible
        try:
            conn = db_utils.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE document_chunks SET status = %s WHERE document_id = %s AND chunk_index = %s",
                ("failed", document_id, chunk_idx)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except:
            pass  # If this fails too, just continue
            
        return chunk_idx, [], []

# Process PDF text with LLM
def process_pdf_with_llm(chunks: List[str], api_keys: List[str], document_id: int, batch_size: int = 1, resume: bool = True, translate_to_english: bool = True):
    """Process PDF text chunks with LLM to extract key concepts, Q&A pairs, and summary."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import math
    
    # Initialize collections for storing results (for temporary use)
    all_concepts = []
    all_qa_pairs = []
    
    # Get document info
    document_info = db_utils.get_document_info(document_id)
    if not document_info:
        print(f"Error: Could not retrieve document information for ID {document_id}")
        return
    
    # Get author name from document info
    author_name = document_info['author']
    
    print(f"Processing document: {document_info['filename']}")
    if translate_to_english:
        print("Translation to English is enabled - output will be in English")
    else:
        print("Translation is disabled - output will be in the original language")
        
    # Check for existing chunks and their status
    conn = db_utils.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT chunk_index, status FROM document_chunks WHERE document_id = %s ORDER BY chunk_index",
        (document_id,)
    )
    existing_chunks = {row['chunk_index']: row['status'] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    
    # Filter chunks that need processing (skip completed ones if resume=True)
    chunks_to_process = []
    chunk_indices = []
    
    for i, chunk in enumerate(chunks):
        if i in existing_chunks:
            # If resume is enabled and chunk is already completed, skip it
            if resume and existing_chunks[i] == 'completed':
                print(f"Skipping already completed chunk {i+1}")
                continue
        chunks_to_process.append(chunk)
        chunk_indices.append(i)
    
    if not chunks_to_process:
        print("All chunks are already processed. Nothing to do.")
        return
        
    print(f"Processing {len(chunks_to_process)} out of {len(chunks)} chunks")
    
    # Create progress bar
    progress_bar = tqdm(total=len(chunks_to_process), desc="Processing chunks", unit="chunk")
    
    # Process chunks in appropriate batches
    if batch_size > 1:
        # Calculate effective batch size (no longer requiring multiple API keys)
        effective_batch_size = min(batch_size, 10)  # Cap at 10 parallel threads to avoid overwhelming
        print(f"Using batch processing with {effective_batch_size} parallel requests")
        
        # Define a function for processing a single chunk (for ThreadPoolExecutor)
        def process_chunk(chunk_data):
            chunk_idx, chunk_text, key_idx = chunk_data
            api_key = api_keys[key_idx % len(api_keys)]
            
            # Use thread-local variable for retries
            retries = 0
            max_retries = 3
            
            while retries < max_retries:
                try:
                    result = process_single_chunk(chunk_idx, chunk_text, api_key, document_id, translate_to_english)
                    return result
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        print(f"\n  - Failed to process chunk {chunk_idx} after {max_retries} attempts: {str(e)}")
                        # Return empty results
                        return chunk_idx, [], []
                    
                    # Exponential backoff
                    sleep_time = 2 ** retries + random.uniform(0, 1)
                    print(f"\n  - Retrying chunk {chunk_idx} in {sleep_time:.2f}s ({retries}/{max_retries})")
                    time.sleep(sleep_time)
        
        # Prepare chunks with API key indices
        chunk_data = [(chunk_indices[i], chunks_to_process[i], i % len(api_keys)) 
                     for i in range(len(chunks_to_process))]
        
        # Process chunks in parallel using ThreadPoolExecutor with as_completed
        futures = []
        chunk_results = []
        
        with ThreadPoolExecutor(max_workers=effective_batch_size) as executor:
            # Submit all tasks
            for data in chunk_data:
                futures.append(executor.submit(process_chunk, data))
            
            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                chunk_results.append(result)
                progress_bar.update(1)  # Update progress bar as tasks complete
                
                # Aggregate results for summary generation
                chunk_idx, chunk_concepts, chunk_qa_pairs = result
                all_concepts.extend(chunk_concepts)
                all_qa_pairs.extend(chunk_qa_pairs)
    else:
        # Sequential processing
        print("Using sequential processing")
        for i, chunk in enumerate(chunks_to_process):
            # Rotate through available API keys
            api_key = api_keys[i % len(api_keys)]
            
            # Process each chunk
            _, chunk_concepts, chunk_qa_pairs = process_single_chunk(chunk_indices[i], chunk, api_key, document_id, translate_to_english)
            
            # Add chunk results to the collections for summary generation
            all_concepts.extend(chunk_concepts)
            all_qa_pairs.extend(chunk_qa_pairs)
            
            # Update progress bar
            progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    # Deduplicate concepts and Q&A pairs
    unique_concepts = deduplicate_concepts(all_concepts, author_name)
    unique_qa_pairs = deduplicate_qa_pairs(all_qa_pairs)
    
    # Process QA pairs to replace generic speaker references
    unique_qa_pairs = process_qa_pairs(unique_qa_pairs, author_name)
    
    # Generate combined summary
    combined_summary = generate_combined_summary(unique_concepts, unique_qa_pairs, len(chunks_to_process))
    
    # Store the summary in the database
    if db_utils.store_summary(document_id, combined_summary):
        print(f"Summary stored in database")
    else:
        print(f"Error: Failed to store summary in database")
    
    # Update document status to completed
    db_utils.update_document_status(document_id, "completed")
    
    # Get statistics from database
    stats = db_utils.get_document_statistics(document_id)
    
    print("\nExtraction completed successfully!")
    print(f"Found {stats['total_concepts']} unique concepts and {stats['total_qa_pairs']} unique Q&A pairs")
    print(f"All data stored in database for document ID: {document_id}")

def export_document_data(document_id: int):
    """Export concepts and QA pairs from database to JSON files."""
    document_info = db_utils.get_document_info(document_id)
    if not document_info:
        print(f"Error: Document ID {document_id} not found")
        return False
        
    concepts = db_utils.get_all_concepts(document_id)
    qa_pairs = db_utils.get_all_qa_pairs(document_id)
    
    # Export to JSON files
    filename_base = document_info['filename'].split('.')[0]
    
    with open(f"{filename_base}_concepts.json", "w", encoding="utf-8") as f:
        json.dump(concepts, f, ensure_ascii=False, indent=2)
        
    with open(f"{filename_base}_qa_pairs.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
    print(f"Exported concepts to {filename_base}_concepts.json")
    print(f"Exported QA pairs to {filename_base}_qa_pairs.json")
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract wisdom from PDF lecture transcripts")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of words per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Number of overlapping words between chunks")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of chunks to process in parallel (default: 3, set to 1 for sequential processing)")
    parser.add_argument("--api_keys_file", type=str, help="Path to file containing multiple API keys (one per line, recommended for higher parallelism)")
    parser.add_argument("--init_db", action="store_true", help="Force recreate database tables")
    parser.add_argument("--title", type=str, help="Document title (default: filename)")
    parser.add_argument("--author", type=str, default="Iva Adamcová", help="Document author")
    parser.add_argument("--no_resume", action="store_true", help="Don't resume interrupted processing, start from beginning")
    parser.add_argument("--no_translation", action="store_true", help="Do not translate concepts and QA pairs to English, keep them in the original language")
    
    # Database operations
    parser.add_argument("--document_id", type=int, help="Existing document ID to work with")
    parser.add_argument("--list_documents", action="store_true", help="List all documents in the database")
    parser.add_argument("--export_data", action="store_true", help="Export concepts and QA pairs to JSON files")
    parser.add_argument("--create_summary", action="store_true", help="Create summary document for existing document")
    parser.add_argument("--create_visualization", action="store_true", help="Create visualization for existing document")
    
    return parser.parse_args()

# 7. Create a concept visualization
def create_concept_visualization(concepts: List[Dict], output_dir: str, author_name: str):
    """Create a simple HTML visualization of concepts using D3.js."""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Koncept Mapa - {author_name}</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f7f9fc; }}
            h1 {{ color: #444; text-align: center; margin-bottom: 30px; }}
            .node {{ cursor: pointer; }}
            .node circle {{ 
                stroke-width: 2px; 
                transition: all 0.3s ease;
            }}
            .node text {{ 
                font: 12px sans-serif; 
                fill: #333;
                transition: all 0.3s ease;
            }}
            .node:hover circle {{ 
                stroke-width: 4px; 
            }}
            .node:hover text {{ 
                font-weight: bold;
                font-size: 14px;
            }}
            .link {{ 
                fill: none; 
                stroke: #ddd; 
                stroke-width: 1.5px; 
            }}
            .tooltip {{ 
                position: absolute; 
                background: white; 
                border: 1px solid #ddd; 
                padding: 15px; 
                border-radius: 8px; 
                max-width: 320px; 
                z-index: 10; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                line-height: 1.5;
                color: #333;
                font-size: 14px;
            }}
            #concept-list {{
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-top: 40px;
            }}
            #concept-list h2 {{
                color: #555;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            .concept-item {{
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #f0f0f0;
            }}
            .concept-item h3 {{
                color: #3366cc;
                margin-bottom: 5px;
            }}
            .concept-item p {{
                margin-top: 5px;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <h1>Mapa konceptů z přednášky {author_name}</h1>
        <div id="visualization"></div>
        
        <script>
        // Data from extracted concepts
        const conceptsData = CONCEPTS_JSON_PLACEHOLDER;
        
        // Create hierarchical structure
        const root = {{
            name: "Hlavní koncepty",
            children: conceptsData.map(c => ({{ 
                name: c.concept, 
                explanation: c.explanation,
                size: c.explanation.length / 10 // Size based on explanation length
            }}))
        }};
        
        // Set up dimensions and radius
        const width = 960;
        const height = 700;
        const radius = Math.min(width, height) / 2 - 90;
        
        // Create tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);
            
        // Create SVG
        const svg = d3.select("#visualization").append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", `translate(${{width / 2}},${{height / 2}})`);
            
        // Create cluster layout
        const cluster = d3.cluster()
            .size([360, radius]);
            
        // Process data
        const hierarchy = d3.hierarchy(root);
        cluster(hierarchy);
        
        // Create category colors
        const colors = d3.scaleOrdinal(d3.schemeCategory10);
        
        // Create links
        svg.selectAll("path")
            .data(hierarchy.links())
            .enter()
            .append("path")
            .attr("class", "link")
            .attr("d", d3.linkRadial()
                .angle(d => d.x * Math.PI / 180)
                .radius(d => d.y));
                
        // Create nodes
        const node = svg.selectAll(".node")
            .data(hierarchy.descendants())
            .enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", d => `rotate(${{d.x - 90}}) translate(${{d.y}}, 0)`)
            .on("mouseover", function(event, d) {{
                if (d.data.explanation) {{
                    tooltip.transition().duration(200).style("opacity", .9);
                    tooltip.html(`<strong>${{d.data.name}}</strong><br>${{d.data.explanation}}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px");
                }}
            }})
            .on("mouseout", function() {{
                tooltip.transition().duration(500).style("opacity", 0);
            }});
                
        // Add circles to nodes
        node.append("circle")
            .attr("r", d => d.data.size ? Math.min(Math.max(d.data.size, 5), 15) : 5)
            .style("fill", d => d.depth === 0 ? "#fff" : colors(d.data.name.length % 10))
            .style("stroke", d => d.depth === 0 ? "#999" : colors(d.data.name.length % 10));
            
        // Add text to nodes
        node.append("text")
            .attr("dy", ".31em")
            .attr("x", d => d.x < 180 ? 10 : -10)
            .attr("text-anchor", d => d.x < 180 ? "start" : "end")
            .attr("transform", d => d.x < 180 ? null : "rotate(180)")
            .text(d => d.data.name);
            
        </script>
        
        <!-- Add a list of concepts below the visualization -->
        <div id="concept-list">
            <h2>Seznam klíčových konceptů</h2>
            <div id="concepts">
                <!-- Concepts will be inserted here -->
            </div>
        </div>
        
        <script>
            // Add concepts to the list
            const conceptsList = document.getElementById('concepts');
            conceptsData.forEach((concept, index) => {{
                const item = document.createElement('div');
                item.className = 'concept-item';
                item.innerHTML = `
                    <h3>${{index + 1}}. ${{concept.concept}}</h3>
                    <p>${{concept.explanation}}</p>
                `;
                conceptsList.appendChild(item);
            }});
        </script>
    </body>
    </html>
    """
    
    # Replace placeholder with actual concepts data
    html_content = html_template.replace('CONCEPTS_JSON_PLACEHOLDER', json.dumps(concepts, ensure_ascii=False))
    
    # Write HTML file
    visualization_path = os.path.join(output_dir, "concept_visualization.html")
    with open(visualization_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Concept visualization saved to {visualization_path}")

# Main function
if __name__ == "__main__":
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key (prioritize .env file)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_keys = []
    
    # If an API keys file is provided, load all keys from it
    if args.api_keys_file and os.path.exists(args.api_keys_file):
        with open(args.api_keys_file, 'r') as key_file:
            api_keys = [line.strip() for line in key_file if line.strip()]
        print(f"Loaded {len(api_keys)} API keys from {args.api_keys_file}")
    
    # If no API keys file or no keys in file, use the single API key from .env
    if not api_keys and api_key:
        api_keys = [api_key]
    
    # Initialize database
    print("Initializing database...")
    if not db_utils.initialize_database(force_recreate=args.init_db):
        print("Error: Could not initialize database")
        sys.exit(1)
    
    # Export data if requested
    if args.export_data:
        if not args.document_id:
            print("Error: --document_id is required with --export_data")
            sys.exit(1)
        
        if export_document_data(args.document_id):
            print("Data export completed successfully")
        sys.exit(0)
    
    # List all documents if requested
    if args.list_documents:
        documents = db_utils.get_all_documents()
        print("\nDocuments in the database:")
        print("-" * 80)
        print(f"{'ID':5} | {'Filename':<30} | {'Title':<30} | {'Concepts':<8} | {'QA Pairs':<8} | {'Status':<10}")
        print("-" * 80)
        for doc in documents:
            print(f"{doc['document_id']:<5} | {doc['filename'][:30]:<30} | {doc['title'][:30]:<30} | {doc['concept_count']:<8} | {doc['qa_count']:<8} | {doc['status']:<10}")
        sys.exit(0)
    
    # Create summary for existing document if requested
    if args.create_summary:
        if not args.document_id:
            print("Error: --document_id is required with --create_summary")
            sys.exit(1)
        
        output_dir = "."  # Current directory
        summary_path = create_summary_document(args.document_id, output_dir)
        if summary_path:
            print(f"Summary document created: {summary_path}")
        sys.exit(0)
    
    # Create visualization for existing document if requested
    if args.create_visualization:
        if not args.document_id:
            print("Error: --document_id is required with --create_visualization")
            sys.exit(1)
        
        # Get document info for author name
        document_info = db_utils.get_document_info(args.document_id)
        if not document_info:
            print(f"Error: Document ID {args.document_id} not found")
            sys.exit(1)
            
        concepts = db_utils.get_all_concepts(args.document_id)
        if concepts:
            output_dir = "."  # Current directory
            create_concept_visualization(concepts, output_dir, document_info['author'])
            print(f"Visualization created: concept_visualization.html")
        else:
            print(f"No concepts found for document ID {args.document_id}")
        sys.exit(0)
    
    # Process a new document if pdf_path is provided
    if args.pdf_path:
        pdf_path = args.pdf_path
        pdf_filename = os.path.basename(pdf_path)
        
        # Extract document title from filename if not provided
        document_title = args.title if args.title else os.path.splitext(pdf_filename)[0]
        
        # Create document record in database
        print(f"Creating document record for {pdf_filename}...")
        document_id = db_utils.create_document(
            filename=pdf_filename,
            title=document_title,
            author=args.author,
            file_path=os.path.abspath(pdf_path)
        )
        
        if not document_id:
            print("Error: Could not create document record in database")
            sys.exit(1)
        
        print(f"Document ID: {document_id}")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        if not pdf_text:
            print("Error: Could not extract text from PDF")
            db_utils.update_document_status(document_id, "failed")
            sys.exit(1)
        
        # Update document with full text
        db_utils.update_document_fulltext(document_id, pdf_text, 0)  # Will update total_chunks later
        
        # Split text into chunks
        chunks = chunk_text(pdf_text, args.chunk_size, args.chunk_overlap)
        print(f"Text split into {len(chunks)} chunks")
        
        # Update document with total chunks
        db_utils.update_document_fulltext(document_id, pdf_text, len(chunks))
        
        # Check if API keys are available for processing
        if not api_keys:
            print("Error: No API key provided. Please set DEEPSEEK_API_KEY in .env file or provide --api_keys_file")
            db_utils.update_document_status(document_id, "failed")
            sys.exit(1)
        
        # Process with LLM - invert the no_translation flag for translate_to_english
        process_pdf_with_llm(chunks, api_keys, document_id, batch_size=args.batch_size, resume=not args.no_resume, translate_to_english=not args.no_translation)
        
        # No automatic JSON export - data is stored in database
        print("\nExtraction completed successfully!")
        print(f"All data stored in database for document ID: {document_id}")
        print(f"Use --list_documents to see all processed documents")
    else:
        # If no pdf_path and no other action specified, display help
        if not (args.list_documents or args.create_summary or args.create_visualization or args.export_data):
            print("Error: Please provide either a PDF path to process or use --list_documents, --export_data, --create_summary, or --create_visualization")
            print("\nFor help, use: python pdf_wisdom_extractor.py -h")
            sys.exit(1) 