import json
import time
import logging
from datetime import datetime
import os
import redis
from dotenv import load_dotenv
import traceback
import structlog
from typing import Any

# Import the Celery app instance
from celery_app import celery_app # Corrected: Import from root
from services.config import AppConfig # Added
from services.llm_service.llm_service import LLMService # Corrected import path
from services.vector_search.vector_search import VectorSearchService # Corrected import path

# Load env vars at module level for worker
load_dotenv()

# Configure logger
logger = structlog.get_logger(__name__)
# Basic config if running standalone (adjust as needed)
# if not logger.hasHandlers(): # Removing this, assuming standard logging setup
#     logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# --- Removed Mock/Unused API Gateway --- 
# class APIGateway:
#     ...
# class ServiceError(Exception):
#     pass

# --- Helper to get Redis Client (using AppConfig) ---
def get_redis_client():
    redis_url = AppConfig.REDIS_URL
    if not redis_url:
        logger.error("REDIS_URL not found in AppConfig.")
        return None
    try:
        # Use decode_responses=True for easier handling
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping() # Verify connection
        logger.info("Redis client connected successfully for task.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
        return None

# --- Removed get_api_gateway() --- 
# def get_api_gateway():
#     ...

# Initialize services (consider dependency injection) - Services are now initialized within the task
# llm_service = LLMService()
# vector_service = VectorSearchService()

# Decorate the function as a Celery task
@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 3}) # Added basic retry
def process_question_task(self, task_id: str, data: dict):
    """Processes a question asynchronously, updating progress in Redis."""
    logger = structlog.get_logger(f"task.{self.request.id}") # Task-specific logger instance
    logger.info(f"Starting question processing task", passed_task_id=task_id, data=data)
    
    # Initialize clients/services within the task using AppConfig
    redis_client = get_redis_client()
    redis_key = f"task:{task_id}"
    pubsub_channel = f"progress:{task_id}"

    llm_service: Optional[LLMService] = None
    vector_service: Optional[VectorSearchService] = None

    try:
        # Create a temporary config dict from AppConfig attributes for service init
        # This avoids passing the whole class if attributes are simple types
        # Adjust if AppConfig becomes more complex
        config_dict = {
            attr: getattr(AppConfig, attr)
            for attr in dir(AppConfig)
            if not callable(getattr(AppConfig, attr)) and not attr.startswith("__")
        }
        llm_service = LLMService(config=config_dict)
        vector_service = VectorSearchService(config=config_dict)
        logger.info("LLM and Vector Search services initialized for task.")
    except Exception as service_init_err:
         logger.error("Failed to initialize services within task", error=str(service_init_err), exc_info=True)
         # Update state and raise if services are critical
         if redis_client:
              # Use internal helper to prevent code duplication
              _update_task_progress(redis_client, redis_key, pubsub_channel, task_id, logger,
                                    status="Failed", progress=0,
                                    details="Service initialization failed.",
                                    error=f"Service init error: {service_init_err}")
         self.update_state(state='FAILURE', meta={'exc_type': type(service_init_err).__name__, 'exc_message': f"Service init failed: {service_init_err}"})
         raise # Re-raise to trigger Celery failure/retry logic

    # Check if essential clients are available
    if not redis_client:
        logger.critical("Cannot proceed without Redis connection.")
        self.update_state(state='FAILURE', meta={'exc_type': 'ConnectionError', 'exc_message': 'Redis client unavailable'})
        # No need to raise again if state is set
        return {"error": "Redis client unavailable"} # Return error info

    if not llm_service or not vector_service:
        logger.critical("Cannot proceed without LLM or Vector Search service.")
        _update_task_progress(redis_client, redis_key, pubsub_channel, task_id, logger,
                              status="Failed", progress=0,
                              details="Core service initialization failed.",
                              error="LLM or Vector Search service unavailable")
        self.update_state(state='FAILURE', meta={'exc_type': 'RuntimeError', 'exc_message': 'LLM/Vector Search service unavailable'})
        return {"error": "LLM/Vector Search service unavailable"}

    try:
        logger.info("Starting question processing logic.")
        _update_task_progress(redis_client, redis_key, pubsub_channel, task_id, logger,
                              status="Processing", progress=10, details="Starting analysis")

        question = data.get('question')
        user_id = data.get('user_id') # Get user_id if passed
        
        # --- Get config from data dict, falling back to AppConfig --- 
        index_name = data.get('index_name', AppConfig.PINECONE_INDEX_NAME) # Use request value or default
        try:
            # Attempt to get top_k from data, ensuring it's an int
            top_k_raw = data.get('top_k')
            if top_k_raw is not None:
                 top_k = int(top_k_raw)
            else:
                 top_k = AppConfig.DEFAULT_TOP_K
        except (ValueError, TypeError):
            logger.warning(f"Invalid top_k value '{top_k_raw}' received, using default {AppConfig.DEFAULT_TOP_K}")
            top_k = AppConfig.DEFAULT_TOP_K
        # --- End Config Extraction --- 

        if not question:
            raise ValueError("Question is required in task data")
        if not user_id:
             logger.warning("user_id not provided in task data, search may not be user-specific.")
             # Decide how to handle missing user_id (e.g., raise error, proceed without user filter)
             # For now, proceed but log warning. The vector_service.query should handle user_id internally.

        # Step 1: Search using VectorSearchService
        _update_task_progress(redis_client, redis_key, pubsub_channel, task_id, logger,
                              status="Processing", progress=30, details=f"Searching index '{index_name}' using top_k={top_k}...") # Updated log

        try:
            # Assuming user_id is needed for filtering in the query method
            search_results_raw = vector_service.query(user_id=user_id, query_text=question, top_k=top_k, index_name=index_name)
            # Extract context (assuming 'metadata' contains 'text' or similar)
            context_list = []
            for res in search_results_raw:
                if 'metadata' in res and 'text' in res['metadata']:
                    context_list.append(res['metadata']['text'])
                elif 'text' in res: # Fallback if text is top-level
                     context_list.append(res['text'])
            context = "\n".join(context_list) # Join context pieces
            logger.info("Search completed.", num_results=len(search_results_raw), context_length=len(context))

        except Exception as search_err:
            logger.error("Error during vector search", error=str(search_err), exc_info=True)
            raise ServiceError(f"Search service error: {search_err}") from search_err

        if not context:
            result_data = {"answer": "I couldn't find relevant information to answer your question."}
            _update_task_progress(redis_client, redis_key, pubsub_channel, task_id, logger,
                                  status="Completed", progress=100, details="No relevant context found.", result=result_data)
            return result_data # Return result for Celery backend

        # Step 2: Generate answer using LLMService
        _update_task_progress(redis_client, redis_key, pubsub_channel, task_id, logger,
                              status="Processing", progress=70, details="Generating answer...")
        try:
            # Combine search results into a context string for the LLM
            context_for_llm = "\n\n".join([res['metadata'].get('text', '') for res in search_results_raw if res.get('metadata', {}).get('text')])

            # Use the generate_answer method of the initialized llm_service
            llm_response = llm_service.generate_answer(
                question=question,
                search_results=search_results_raw, # Pass raw results if needed by method, or just context
                # context=context_for_llm, # If method expects context string
                # provider_name=None, # Use default provider
                # model=None # Use default model
            )

            if llm_response.is_error:
                 raise ServiceError(f"LLM service error: {llm_response.error}")

            final_answer = llm_response.content if isinstance(llm_response.content, str) else str(llm_response.content)
            final_result_data = {"answer": final_answer}
            logger.info("Answer generation completed.")

        except Exception as generate_err:
            logger.error("Error during answer generation", error=str(generate_err), exc_info=True)
            raise ServiceError(f"LLM service error: {generate_err}") from generate_err


        # Step 3: Finalize
        _update_task_progress(redis_client, redis_key, pubsub_channel, task_id, logger,
                              status="Completed", progress=100, details="Answer generated successfully.", result=final_result_data)
        logger.info("Question processing task completed successfully.")
        return final_result_data # Return final result for Celery backend

    except Exception as e:
        logger.error("Unhandled error during question processing task", error=str(e), exc_info=True)
        error_message = f"An error occurred: {str(e)}"
        if redis_client:
            _update_task_progress(redis_client, redis_key, pubsub_channel, task_id, logger,
                                  status="Failed", progress=100, details=error_message, error=str(e))
        # Update Celery state before raising
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': traceback.format_exc()}) # Include traceback
        raise # Re-raise the exception to ensure Celery knows it failed

# --- Remove unused async function and helpers --- 
# def process_question_async(question, session_id, logger, progress_events, api_gateway, redis_client):
#    ...

# --- Internal Helper Function for Progress Updates ---
def _update_task_progress(redis_client, redis_key, pubsub_channel, task_id, task_logger,
                          status: str, progress: int, details: str = "", result: Any = None, error: str = None):
    """Helper function to update task state in Redis hash and publish to Pub/Sub."""
    if not redis_client:
        task_logger.error("Redis client unavailable during progress update.")
        return
    try:
        update_data = {
            'status': status,
            'progress': progress,
            'details': details,
            'updated_at': datetime.utcnow().isoformat(),
            'task_id': task_id
        }
        result_json = None
        if result is not None:
            try:
                # Attempt to serialize result to JSON
                result_json = json.dumps(result)
                update_data['result'] = result_json
            except TypeError as e:
                task_logger.warning("Result data is not JSON serializable", error=str(e))
                # Store a representation or error message instead
                update_data['result'] = json.dumps({"error": "Result data not serializable", "type": type(result).__name__})
        if error is not None:
            update_data['error'] = str(error) # Ensure error is string

        # Update the Redis Hash (use hmset for atomic update of multiple fields)
        redis_client.hmset(redis_key, update_data)
        task_logger.info("Progress Hash updated", status=status, progress=progress)

        # Publish the update to the Pub/Sub channel
        # Prepare message for pub/sub (use the same data written to hash)
        publish_message = json.dumps(update_data)
        redis_client.publish(pubsub_channel, publish_message)
        task_logger.info("Published update to Pub/Sub channel", channel=pubsub_channel)

    except redis.exceptions.RedisError as redis_err:
        task_logger.error("Redis error during progress update", error=str(redis_err), exc_info=True)
    except Exception as e:
        task_logger.error("Unexpected error during progress update", error=str(e), exc_info=True) 