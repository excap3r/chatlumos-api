import json
import time
import logging

# Assuming ServiceError is defined elsewhere
from services.api_gateway import ServiceError 
# Import helper
from services.utils.api_helpers import get_cache_key

# Default value - consider making this configurable via app context or env var
DEFAULT_TOP_K = 10

def process_question_async(question, session_id, logger, progress_events, api_gateway, redis_client):
    """
    Process a question asynchronously, sending progress updates.
    
    Args:
        question: The question to process
        session_id: The session ID for tracking progress
        logger: Logger instance
        progress_events: Dictionary for progress queues
        api_gateway: API Gateway client instance
        redis_client: Redis client instance or None
    """
    event_queue = progress_events.get(session_id)
    if not event_queue:
        logger.error(f"Error: No event queue found for session {session_id}")
        return
    
    try:
        # Initialize process steps
        process_steps = [
            {"id": "decompose", "name": "Analyzing question", "status": "pending", "details": None},
            {"id": "search", "name": "Searching knowledge base", "status": "pending", "details": None},
            {"id": "answer", "name": "Generating answer", "status": "pending", "details": None}
        ]
        
        # Send initial process steps
        logger.info(f"[Session {session_id}] Initializing process steps")
        event_queue.put({
            "event": "process_init",
            "process_steps": process_steps,
            "progress": 0
        })
        
        # Calculate progress increment per step
        progress_increment = 100.0 / len(process_steps)
        current_progress = 0
        
        # Track result data
        result_data = {
            "question": question,
            "sub_questions": [],
            "search_results": {},
            "answer": "",
            "error": None
        }
        
        # Step 1: Decompose the question
        logger.info(f"[Session {session_id}] Decomposing question: {question}")
        event_queue.put({
            "event": "process_update",
            "step_id": "decompose",
            "status": "running"
        })
        
        sub_questions = [question] # Default
        try:
            # Check cache first if available
            cache_hit = False
            if redis_client:
                cache_key = get_cache_key("decompose", question) 
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    decompose_result = json.loads(cached_result)
                    cache_hit = True
                    logger.info(f"[Session {session_id}] Cache hit for decomposition")
            
            if not cache_hit:
                # Use LLM service to decompose question
                decompose_result = api_gateway.request(
                    "llm", 
                    "/decompose_question", 
                    method="POST",
                    json={"question": question}
                )
                
                # Cache the result
                if redis_client and "error" not in decompose_result:
                    redis_client.setex(
                        cache_key, 
                        3600,  # 1 hour cache
                        json.dumps(decompose_result)
                    )
            
            if "error" in decompose_result:
                raise Exception(f"Error decomposing question: {decompose_result['error']}")
                
            sub_questions = decompose_result.get("sub_questions", [question])
            result_data["sub_questions"] = sub_questions
            
            # Mark step as complete
            current_progress += progress_increment
            event_queue.put({
                "event": "process_update",
                "step_id": "decompose",
                "status": "complete",
                "details": {
                    "sub_questions": sub_questions
                },
                "progress": min(int(current_progress), 99)
            })
        except Exception as e:
            logger.error(f"[Session {session_id}] Error in decomposition: {str(e)}", exc_info=True)
            event_queue.put({
                "event": "process_update",
                "step_id": "decompose",
                "status": "error",
                "details": {"error": str(e)}
            })
            # Fall back to using the original question
            sub_questions = [question]
            result_data["sub_questions"] = sub_questions
            result_data["error"] = f"Error during question analysis: {str(e)}"
            # Do not stop processing, just use original question
            # Update progress anyway
            current_progress += progress_increment 
            event_queue.put({
                "event": "process_update",
                "step_id": "decompose",
                "status": "complete_with_fallback",
                "details": {"fallback_reason": str(e), "sub_questions": sub_questions},
                "progress": min(int(current_progress), 99)
            })

        
        # Step 2: Search for each sub-question
        logger.info(f"[Session {session_id}] Searching for {len(sub_questions)} sub-questions")
        event_queue.put({
            "event": "process_update",
            "step_id": "search",
            "status": "running"
        })
        
        all_results = []
        search_errors = []
        try:
            for sub_q in sub_questions:
                try:
                    # Check cache first if available
                    cache_hit = False
                    if redis_client:
                        cache_key = get_cache_key("search", sub_q, DEFAULT_TOP_K)
                        cached_result = redis_client.get(cache_key)
                        if cached_result:
                            search_result = json.loads(cached_result)
                            cache_hit = True
                            logger.info(f"[Session {session_id}] Cache hit for search: {sub_q[:30]}...")
                    
                    if not cache_hit:
                        # Generate embedding and search using vector service
                        search_result = api_gateway.request(
                            "vector", 
                            "/search", 
                            method="POST",
                            json={
                                "query": sub_q,
                                "top_k": DEFAULT_TOP_K
                            }
                        )
                        
                        # Cache the result
                        if redis_client and "error" not in search_result:
                            redis_client.setex(
                                cache_key, 
                                3600 * 24,  # 24 hour cache
                                json.dumps(search_result)
                            )

                    if "error" in search_result:
                        raise Exception(f"Error searching for '{sub_q}': {search_result['error']}")

                    # Collect results
                    found_results = search_result.get("results", [])
                    result_data["search_results"][sub_q] = found_results
                    all_results.extend(found_results)
                    logger.info(f"[Session {session_id}] Found {len(found_results)} results for: {sub_q[:30]}...")

                except Exception as sub_e:
                    logger.error(f"[Session {session_id}] Error searching for sub-question '{sub_q}': {str(sub_e)}")
                    search_errors.append(str(sub_e))
                    result_data["search_results"][sub_q] = [] # Ensure key exists
            
            # Mark step as complete (even if some searches failed)
            current_progress += progress_increment
            event_queue.put({
                "event": "process_update",
                "step_id": "search",
                "status": "complete" if not search_errors else "complete_with_errors",
                "details": {
                    "total_results": len(all_results),
                    "errors": search_errors
                },
                "progress": min(int(current_progress), 99)
            })

        except Exception as e:
            logger.error(f"[Session {session_id}] Critical error during search phase: {str(e)}", exc_info=True)
            event_queue.put({
                "event": "process_update",
                "step_id": "search",
                "status": "error",
                "details": {"error": str(e)}
            })
            # Cannot proceed without search results
            result_data["error"] = f"Critical error during knowledge search: {str(e)}"
            raise # Re-raise to be caught by outer try/except
            
        # Step 3: Generate answer
        logger.info(f"[Session {session_id}] Generating answer based on {len(all_results)} results")
        event_queue.put({
            "event": "process_update",
            "step_id": "answer",
            "status": "running"
        })
        
        try:
            if not all_results:
                 raise Exception("No search results found to generate an answer.")

            # Prepare context from search results
            context = "\n".join([r.get("text", "") for r in all_results])
            
            # Check cache first if available
            cache_hit = False
            if redis_client:
                # Use a hash of the question and search result IDs as cache key
                result_ids = sorted([r.get("id", "") for r in all_results])
                cache_key = get_cache_key("answer", question, "-".join(result_ids[:20]))
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    answer_result = json.loads(cached_result)
                    cache_hit = True
                    logger.info(f"[Session {session_id}] Cache hit for answer generation")
            
            if not cache_hit:
                # Use LLM service to generate answer
                answer_result = api_gateway.request(
                    "llm", 
                    "/generate_answer", 
                    method="POST",
                    json={
                        "question": question,
                        "context": context
                    }
                )
                
                # Cache the result
                if redis_client and "error" not in answer_result:
                    redis_client.setex(
                        cache_key, 
                        3600 * 12,  # 12 hour cache
                        json.dumps(answer_result)
                    )

            if "error" in answer_result:
                raise Exception(f"Error generating answer: {answer_result['error']}")

            answer = answer_result.get("answer", "Could not generate an answer.")
            result_data["answer"] = answer

            # Mark step as complete
            current_progress = 100 # Final step completes progress
            event_queue.put({
                "event": "process_update",
                "step_id": "answer",
                "status": "complete",
                "details": {
                    "answer_length": len(answer)
                },
                "progress": current_progress
            })

        except Exception as e:
            logger.error(f"[Session {session_id}] Error in answer generation: {str(e)}", exc_info=True)
            event_queue.put({
                "event": "process_update",
                "step_id": "answer",
                "status": "error",
                "details": {"error": str(e)}
            })
            result_data["answer"] = "Error generating answer."
            result_data["error"] = f"Error during answer generation: {str(e)}"
            # Proceed to final result despite error

        # Send final result event
        event_queue.put({
            "event": "final_result",
            "data": result_data
        })
        logger.info(f"[Session {session_id}] Processing complete.")

    except Exception as e:
        # Catch errors from Step 2 re-raise or other unexpected errors
        logger.error(f"[Session {session_id}] Unhandled error in process_question_async: {str(e)}", exc_info=True)
        if event_queue: # Check if queue exists before putting error
            event_queue.put({
                "event": "process_error",
                "error": f"Unhandled processing error: {str(e)}"
            })
    finally:
        if event_queue: # Check if queue exists before signaling end
            try:
                event_queue.put(None)  # Signal end of processing
            except Exception as q_e:
                 logger.error(f"[Session {session_id}] Error signaling end of processing: {q_e}") 