import time
import logging
import os
import json
import io
import traceback
from datetime import datetime
import redis
from dotenv import load_dotenv
from typing import Optional # Added for type hints

# Import the Celery app instance
# Use absolute import from project root
from celery_app import celery_app

# Load env vars at module level for worker
load_dotenv()

# Initialize logger for the module
logger = logging.getLogger(__name__)
# Basic config if running standalone (adjust as needed)
if not logger.hasHandlers():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Import Services and Config
from services.config import AppConfig # Added
from services.llm_service.llm_service import LLMService
from services.vector_search.vector_search import VectorSearchService
from services.pdf_processor.pdf_processor import PDFProcessor # Added for text extraction/chunking
from services.utils.error_utils import APIError # Use shared error type
import structlog # Added for consistency

# Placeholder if translate_text isn't easily importable - requires refactor
def translate_text_placeholder(text, target_lang, source_lang, logger):
    logger.warning("translate_text_placeholder used. Need to implement proper call.")
    # Simulate basic pass-through or error
    if target_lang == source_lang:
        return {"translated_text": text}
    return {"error": "Translation function not implemented in this context"}

def process_pdf_async(file_path, author_name, title, language, translate_to_english, session_id, user_id, logger, progress_events, api_gateway):
    """
    Processes a PDF file asynchronously, sending progress updates.
    Args:
        file_path: Path to the temporary PDF file
        author_name: Document author
        title: Document title
        language: Original language of the document
        translate_to_english: Boolean, whether to translate
        session_id: Session ID for progress tracking
        user_id: ID of the user who uploaded the document
        logger: Logger instance
        progress_events: Dictionary for progress queues
        api_gateway: API Gateway client instance
    """
    event_queue = progress_events.get(session_id)
    if not event_queue:
        logger.error(f"No event queue found for session {session_id}")
        return

    current_progress = 0
    total_steps = 6 # Estimate: Extract, Translate(opt), Chunk, Concepts, DB Store, Vector Store
    if not translate_to_english: total_steps -= 1
    progress_increment = 100.0 / total_steps
    
    def update_progress(step_id, status, details=None, error=None):
        nonlocal current_progress
        if status == "complete":
             current_progress += progress_increment
        payload = {
            "event": "process_update",
            "step_id": step_id,
            "status": status,
            "progress": min(int(current_progress), 100) # Cap progress at 100
        }
        if details:
             payload["details"] = details
        if error:
             payload["error"] = error
        try:
             event_queue.put(payload)
        except Exception as q_e:
             logger.error(f"[Session {session_id}] Error putting event on queue: {q_e}")

    try:
        logger.info(f"[Session {session_id}] Starting PDF processing for: {file_path}")
        update_progress("init", "running", details={"message": "Starting PDF processing..."})

        # Step 1: Extract Text using PDF Processor Service
        update_progress("extract_text", "running", details={"message": "Extracting text from PDF..."})
        try:
            extract_result = api_gateway.request(
                 "pdf_processor", 
                 "/extract", 
                 method="POST",
                 json={"file_path": file_path} # Service needs access to this path
            )
            if "error" in extract_result:
                 raise APIError(f"Text extraction failed: {extract_result['error']}", 500)
            text = extract_result.get("text", "")
            page_count = extract_result.get("page_count", 0)
            if not text:
                 raise ValueError("Extracted text is empty.")
            update_progress("extract_text", "complete", details={"page_count": page_count, "text_length": len(text)})
            logger.info(f"[Session {session_id}] Text extracted ({len(text)} chars, {page_count} pages)")
        except Exception as e:
             logger.error(f"[Session {session_id}] Error during text extraction: {e}", exc_info=True)
             update_progress("extract_text", "error", error=str(e))
             raise APIError(f"Text extraction failed: {str(e)}", 500) # Stop processing

        # Step 2: Translate Text (Optional)
        original_text = text # Keep original for potential use
        if translate_to_english and language != 'en':
            update_progress("translate_text", "running", details={"message": f"Translating text from {language} to English..."})
            try:
                 # Use placeholder or actual internal function
                 # translated_text_dict = translate_text_internal(text, "en", language)
                 translated_text_dict = translate_text_placeholder(text, "en", language, logger)
                 if "error" in translated_text_dict:
                      raise APIError(f"Translation failed: {translated_text_dict['error']}", 500)
                 text = translated_text_dict.get("translated_text", "")
                 if not text:
                      raise ValueError("Translated text is empty.")
                 update_progress("translate_text", "complete", details={"original_lang": language, "translated_length": len(text)})
                 logger.info(f"[Session {session_id}] Text translated to English ({len(text)} chars)")
            except Exception as e:
                 logger.error(f"[Session {session_id}] Error during translation: {e}", exc_info=True)
                 update_progress("translate_text", "error", error=str(e))
                 # Decide whether to continue with original text or stop
                 logger.warning(f"[Session {session_id}] Proceeding with original text due to translation error.")
                 text = original_text # Fallback to original
                 # Adjust total steps and progress if translation was skipped/failed?
                 # For simplicity, let's assume we continue but mark the error.
        
        # Step 3: Chunk Text using PDF Processor Service
        update_progress("chunk_text", "running", details={"message": "Chunking text..."})
        try:
            # Define chunk size (consider making configurable)
            chunk_size = 1000
            chunk_overlap = 100
            chunk_result = api_gateway.request(
                 "pdf_processor", 
                 "/chunk", 
                 method="POST",
                 json={
                      "text": text,
                      "chunk_size": chunk_size,
                      "chunk_overlap": chunk_overlap
                 }
            )
            if "error" in chunk_result:
                 raise APIError(f"Text chunking failed: {chunk_result['error']}", 500)
            chunks = chunk_result.get("chunks", [])
            if not chunks:
                 raise ValueError("No chunks created from text.")
            update_progress("chunk_text", "complete", details={"chunk_count": len(chunks), "chunk_size": chunk_size})
            logger.info(f"[Session {session_id}] Text chunked into {len(chunks)} chunks.")
        except Exception as e:
             logger.error(f"[Session {session_id}] Error during text chunking: {e}", exc_info=True)
             update_progress("chunk_text", "error", error=str(e))
             raise APIError(f"Text chunking failed: {str(e)}", 500) # Stop processing

        # Step 4: Extract Concepts/Keywords (Optional but useful) using LLM Service
        update_progress("extract_concepts", "running", details={"message": "Extracting key concepts..."})
        concepts = []
        try:
             # Use a sample or summary of the text for concept extraction?
             sample_text = text[:2000] # Use first 2k chars as sample
             concept_prompt = f"Extract the main keywords and concepts from the following text, separated by commas: {sample_text}"
             concept_result = api_gateway.request(
                 "llm", 
                 "/complete", 
                 method="POST",
                 json={
                      "prompt": concept_prompt, 
                      "max_tokens": 100, 
                      "temperature": 0.3
                 }
             )
             if "error" in concept_result:
                  logger.warning(f"[Session {session_id}] Failed to extract concepts: {concept_result['error']}")
             else:
                  concepts = [c.strip() for c in concept_result.get("content", "").split(',') if c.strip()]
             update_progress("extract_concepts", "complete", details={"concept_count": len(concepts)})
             logger.info(f"[Session {session_id}] Extracted concepts: {concepts}")
        except Exception as e:
             logger.warning(f"[Session {session_id}] Error during concept extraction: {e}")
             update_progress("extract_concepts", "complete_with_errors", error=str(e)) # Continue without concepts

        # Step 5: Store Document Metadata in Database Service
        update_progress("store_metadata", "running", details={"message": "Storing document metadata..."})
        doc_id = None
        try:
            doc_data = {
                "title": title,
                "author": author_name,
                "language": language if not translate_to_english else 'en', # Store final language
                "page_count": page_count,
                "text_length": len(text),
                "chunk_count": len(chunks),
                "concepts": concepts,
                "uploader_user_id": user_id,
                # Add source file info if needed
            }
            db_result = api_gateway.request(
                 "db", 
                 "/documents", 
                 method="POST",
                 json=doc_data
            )
            if "error" in db_result:
                 raise APIError(f"Failed to store document metadata: {db_result['error']}", 500)
            doc_id = db_result.get("id")
            if not doc_id:
                 raise ValueError("Database did not return a document ID.")
            update_progress("store_metadata", "complete", details={"document_id": doc_id})
            logger.info(f"[Session {session_id}] Document metadata stored with ID: {doc_id}")
        except Exception as e:
             logger.error(f"[Session {session_id}] Error storing document metadata: {e}", exc_info=True)
             update_progress("store_metadata", "error", error=str(e))
             raise APIError(f"Database error during document creation: {str(e)}", 500) # Stop processing if metadata fails
             
        # Step 6: Create Embeddings and Store in Vector Service
        update_progress("store_vectors", "running", details={"message": f"Generating and storing {len(chunks)} vector embeddings..."})
        try:
            metadata_list = [
                 {
                      "document_id": doc_id,
                      "author": author_name,
                      "title": title,
                      "language": language if not translate_to_english else 'en',
                      "chunk_index": i
                      # Add any other relevant metadata per chunk
                 } for i in range(len(chunks))
            ]
            
            # Batch embedding generation might be needed for large number of chunks
            # Assuming vector service handles batching or we call it in batches
            embed_result = api_gateway.request(
                 "vector", 
                 "/embed_and_store", # Assuming an endpoint like this
                 method="POST",
                 json={
                      "texts": chunks,
                      "metadata": metadata_list
                      # Specify index name if needed by service
                 }
            )
            if "error" in embed_result:
                 raise APIError(f"Failed to store vector embeddings: {embed_result['error']}", 500)
            
            stored_count = embed_result.get("stored_count", 0)
            if stored_count != len(chunks):
                 logger.warning(f"[Session {session_id}] Mismatch in stored vectors: expected {len(chunks)}, got {stored_count}")
                 
            update_progress("store_vectors", "complete", details={"stored_count": stored_count})
            logger.info(f"[Session {session_id}] Stored {stored_count} vector embeddings.")

        except Exception as e:
             logger.error(f"[Session {session_id}] Error storing vector embeddings: {e}", exc_info=True)
             update_progress("store_vectors", "error", error=str(e))
             # Consider cleanup or marking document as partially indexed?
             raise APIError(f"Vector service error: {str(e)}", 500) # Stop processing

        # Processing Successful
        logger.info(f"[Session {session_id}] PDF processing completed successfully for doc ID {doc_id}")
        event_queue.put({
            "event": "final_result",
            "data": {
                "message": "PDF processed successfully",
                "document_id": doc_id,
                "chunk_count": len(chunks),
                "vector_count": stored_count
            }
        })

    except Exception as e:
        # Catch errors from any step that re-raised
        logger.error(f"[Session {session_id}] PDF processing failed: {str(e)}", exc_info=True)
        if event_queue:
            event_queue.put({
                "event": "process_error",
                "error": f"PDF processing failed: {str(e)}"
            })
    finally:
        # Cleanup temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"[Session {session_id}] Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.error(f"[Session {session_id}] Error cleaning up temp file {file_path}: {e}")
            
        # Signal end of processing to SSE stream
        if event_queue:
            try:
                event_queue.put(None)
            except Exception as q_e:
                 logger.error(f"[Session {session_id}] Error signaling end of processing: {q_e}")
                 
        # Note: The cleanup thread for progress_events queue item is started in the route handler 

# --- Helper to get Redis Client (using AppConfig) ---
def get_redis_client():
    redis_url = AppConfig.REDIS_URL
    if not redis_url:
        logger.error("REDIS_URL not found in AppConfig.")
        return None
    try:
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping() # Verify connection
        logger.info("Redis client connected successfully for PDF processing task.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
        return None

# --- Helper Function for Progress Updates (Adapted from question_processing) ---
def _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, task_id, task_logger,
                              status: str, progress: int, details: str = "", result: dict = None, error: str = None):
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
                result_json = json.dumps(result)
                update_data['result'] = result_json
            except TypeError as e:
                task_logger.warning("Result data is not JSON serializable", error=str(e), result_type=type(result).__name__)
                update_data['result'] = json.dumps({"error": "Result data not serializable", "type": type(result).__name__})
        if error is not None:
            update_data['error'] = str(error)

        # Update the Redis Hash (use hmset for atomic update of multiple fields)
        redis_client.hmset(redis_key, update_data)
        task_logger.info("Progress Hash updated", status=status, progress=progress)

        # Publish the update to the Pub/Sub channel
        publish_message = json.dumps(update_data)
        redis_client.publish(pubsub_channel, publish_message)
        task_logger.info("Published update to Pub/Sub channel", channel=pubsub_channel)

    except redis.exceptions.RedisError as redis_err:
        task_logger.error("Redis error during progress update", error=str(redis_err), exc_info=True)
    except Exception as e:
        task_logger.error("Unexpected error during progress update", error=str(e), exc_info=True)

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 3}) # Added retry
def process_pdf_task(self, task_id: str, file_content_b64: str, filename: str, user_id: str,
                     author_name: str = None, title: str = None,
                     language: str = 'en', translate_to_english: bool = False):
    """
    Processes a PDF asynchronously: extracts text, chunks, stores metadata/chunks,
    generates embeddings, and stores vectors. Updates progress in Redis.

    Args:
        task_id (str): Unique ID for tracking this task.
        file_content_b64 (str): Base64 encoded content of the PDF file.
        filename (str): Original filename of the PDF.
        user_id (str): ID of the user initiating the processing.
        author_name (str, optional): Author of the document. Defaults to None.
        title (str, optional): Title of the document. Defaults to filename if None.
        language (str, optional): Original language code (e.g., 'en', 'es'). Defaults to 'en'.
        translate_to_english (bool, optional): Whether to translate to English if language is not 'en'. Defaults to False.
    """
    # Use task ID from Celery's request object if available, otherwise use passed task_id
    effective_task_id = self.request.id or task_id
    logger = structlog.get_logger(f"task.{effective_task_id}")
    logger.info("Starting PDF processing task", filename=filename, user_id=user_id, task_id=effective_task_id)

    # Initialize clients/services
    redis_client = get_redis_client()
    redis_key = f"task:{effective_task_id}"
    pubsub_channel = f"progress:{effective_task_id}"

    pdf_processor: Optional[PDFProcessor] = None
    llm_service: Optional[LLMService] = None
    vector_service: Optional[VectorSearchService] = None

    try:
        # Create config dict for service init
        config_dict = {
            attr: getattr(AppConfig, attr)
            for attr in dir(AppConfig)
            if not callable(getattr(AppConfig, attr)) and not attr.startswith("__")
        }

        pdf_processor = PDFProcessor() # Doesn't need config currently
        llm_service = LLMService(config=config_dict)
        vector_service = VectorSearchService(config=config_dict)

        logger.info("Services initialized for PDF task.")

    except Exception as service_init_err:
        logger.error("Failed to initialize services within PDF task", error=str(service_init_err), exc_info=True)
        if redis_client:
            _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                      status="Failed", progress=0, details="Service initialization failed.",
                                      error=f"Service init error: {service_init_err}")
        self.update_state(state='FAILURE', meta={'exc_type': type(service_init_err).__name__, 'exc_message': f"Service init failed: {service_init_err}"})
        raise APIError(f"Service initialization failed: {service_init_err}", 500)

    # Check essential clients
    if not redis_client:
        logger.critical("Cannot proceed without Redis connection.")
        self.update_state(state='FAILURE', meta={'exc_type': 'ConnectionError', 'exc_message': 'Redis client unavailable'})
        return {"error": "Redis client unavailable"}

    if not all([pdf_processor, llm_service, vector_service]): # Check needed services
        error_msg = "Core service initialization failed (PDFProcessor, LLMService, or VectorSearchService)."
        logger.critical(error_msg)
        _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                  status="Failed", progress=0, details=error_msg, error=error_msg)
        self.update_state(state='FAILURE', meta={'exc_type': 'RuntimeError', 'exc_message': error_msg})
        return {"error": error_msg}

    # Decode file content
    try:
        import base64
        file_content = base64.b64decode(file_content_b64)
        if not file_content:
            raise ValueError("Decoded file content is empty.")
        logger.info("File content decoded successfully.")
    except Exception as decode_err:
        logger.error("Failed to decode base64 file content", error=str(decode_err))
        _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                  status="Failed", progress=0, details="Invalid file content encoding.", error=str(decode_err))
        self.update_state(state='FAILURE', meta={'exc_type': type(decode_err).__name__, 'exc_message': f"File decode error: {decode_err}"})
        return {"error": "Invalid file content encoding"}

    # Main processing logic
    document_id = None # Initialize document ID
    try:
        _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                  status="Processing", progress=5, details="Starting PDF processing...")

        # --- Step 1: Store Initial Document Metadata ---
        # Store basic info first to get a document_id
        _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                  status="Processing", progress=10, details="Registering document...")
        try:
            effective_title = title or filename # Use filename if title is missing
            # --- Refactored: Replace direct DB call with API call placeholder ---
            # TODO: Make internal API call: POST /api/v1/internal/documents
            # Body: {'filename': filename, 'title': effective_title, 'author': author_name, 'initial_status': 'processing'}
            # Example response: {'document_id': 123}
            # Simulating success for now:
            logger.info("[Placeholder] Making API call to create document record...")
            # Replace with actual API call result handling
            # --- Start Simulation ---
            temp_api_response = {'document_id': int(time.time()) % 10000} # Simulate getting an ID
            if 'document_id' in temp_api_response:
                 document_id = temp_api_response['document_id']
            else:
                 # Handle API error appropriately (log, update status, potentially raise)
                 raise APIError("API call to create document record failed.", 500)
            # --- End Simulation ---
            # document_id = DBService.create_document( # Original call removed
            #     filename=filename,
            #     title=effective_title,
            #     author=author_name,
            # )
            if document_id is None:
                raise APIError("Failed to create initial document record (via API call).", 500) # Adjusted error message
            logger.info("Document registered via API call (placeholder)", document_id=document_id)
            _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                      status="Processing", progress=15, details="Document registered.")
        except Exception as db_err:
            logger.error("Error registering document in DB", error=str(db_err), exc_info=True)
            raise APIError(f"Database error during document creation: {db_err}", 500) from db_err

        # --- Step 2: Extract Text ---
        _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                  status="Processing", progress=20, details="Extracting text from PDF...")
        try:
            pdf_file_like = io.BytesIO(file_content)
            extracted_text = pdf_processor.extract_text(pdf_file_like.getvalue()) # Pass bytes
            if not extracted_text:
                # Handle empty extraction - maybe update status and stop?
                raise APIError("Extracted text is empty. PDF might be image-based or corrupted.", 500)
            logger.info("Text extracted successfully", length=len(extracted_text))
            _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                      status="Processing", progress=35, details=f"Text extracted ({len(extracted_text)} chars).")
        except Exception as extract_err:
            logger.error("Error during text extraction", error=str(extract_err), exc_info=True)
            raise APIError(f"Text extraction failed: {extract_err}", 500) from extract_err

        # --- Step 3: Translate Text (Optional) ---
        final_text = extracted_text
        processing_language = language
        if translate_to_english and language != 'en':
            _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                      status="Processing", progress=40, details=f"Translating text from {language} to English...")
            try:
                translation_prompt = f"Translate the following text from {language} to English:\n\n{extracted_text[:3000]}" # Limit length for translation
                # Use LLM for translation
                llm_response = llm_service.complete(
                    prompt=translation_prompt,
                    temperature=0.2,
                    max_tokens=int(len(extracted_text) * 1.5 / 3) # Rough estimate
                )
                if llm_response.is_error:
                     raise APIError(f"Translation failed: {llm_response.error}", 500)
                final_text = llm_response.content
                processing_language = 'en'
                logger.info("Text translated successfully", length=len(final_text))
                _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                          status="Processing", progress=50, details=f"Text translated to English ({len(final_text)} chars).")
            except Exception as translate_err:
                 logger.warning("Error during translation, proceeding with original text", error=str(translate_err), exc_info=True)
                 _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                           status="Processing", progress=50, details=f"Translation failed, using original text. Error: {translate_err}")
                 # Keep final_text as extracted_text

        # --- Step 4: Chunk Text ---
        _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                  status="Processing", progress=55, details="Chunking text...")
        try:
            chunk_size = AppConfig.CHUNK_SIZE # Get from config
            chunk_overlap = AppConfig.CHUNK_OVERLAP # Get from config
            # Use the chunk_text utility function (assuming it's part of PDFProcessor or utils)
            # If it's a standalone function in pdf_processor module:
            from services.pdf_processor.pdf_processor import chunk_text
            chunks = chunk_text(final_text, chunk_size=chunk_size, overlap=chunk_overlap)

            if not chunks:
                raise APIError("No chunks created from text.", 500)
            logger.info(f"Text chunked successfully", chunk_count=len(chunks))
            _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                      status="Processing", progress=70, details=f"Text chunked into {len(chunks)} pieces.")
        except Exception as chunk_err:
            logger.error("Error during text chunking", error=str(chunk_err), exc_info=True)
            raise APIError(f"Text chunking failed: {chunk_err}", 500) from chunk_err

        # --- Step 5: Store Chunks and Update Document ---
        _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                  status="Processing", progress=75, details="Storing text chunks...")
        chunk_ids = []
        try:
            for i, chunk_text_content in enumerate(chunks):
                # --- Refactored: Replace direct DB call with API call placeholder ---
                # TODO: Make internal API call: POST /api/v1/internal/documents/{document_id}/chunks
                # Body: {'chunk_index': i, 'chunk_text': chunk_text_content}
                # Example response: {'chunk_id': 456}
                # Simulating success for now:
                logger.debug(f"[Placeholder] Making API call to create chunk {i}...")
                # Replace with actual API call result handling
                # --- Start Simulation ---
                temp_chunk_api_response = {'chunk_id': document_id * 1000 + i} # Simulate getting an ID
                chunk_id = None
                if 'chunk_id' in temp_chunk_api_response:
                     chunk_id = temp_chunk_api_response['chunk_id']
                else:
                     # Handle API error appropriately (log, skip chunk, or collect errors)
                     logger.warning(f"API call to create chunk {i} failed.")
                # --- End Simulation ---
                # chunk_id = DBService.create_document_chunk( # Original call removed
                #     document_id=document_id,
                #     chunk_index=i,
                #     chunk_text=chunk_text_content
                # )
                if chunk_id is None:
                    logger.warning(f"Failed to store chunk {i} for document {document_id} (via API call)")
                    # Decide how to handle partial failure - continue or raise?
                else:
                    chunk_ids.append(chunk_id)

            if len(chunk_ids) != len(chunks):
                 logger.warning("Mismatch between successful chunk API calls and expected chunks", expected=len(chunks), stored=len(chunk_ids))
                 # Proceeding, but this indicates potential data loss

            # Update document status and full text now that processing is mostly done
            # --- Refactored: Replace direct DB call with API call placeholder ---
            # TODO: Make internal API call: PUT /api/v1/internal/documents/{document_id}/status
            # Body: {'status': 'processing_embeddings', 'total_chunks': len(chunks), 'full_text': final_text}
            logger.info("[Placeholder] Making API call to update document status to processing_embeddings...")
            # Replace with actual API call result handling (check for success/failure)
            # DBService.update_document_status( # Original call removed
            #     document_id=document_id,
            #     status='processing_embeddings', # Indicate next step
            #     total_chunks=len(chunks),
            #     full_text=final_text # Store the final processed text
            # )
            logger.info("Document status update API call sent (placeholder). Text chunks processing step complete.")
            _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                      status="Processing", progress=85, details=f"{len(chunk_ids)} chunks stored (via API calls)." ) # Adjusted message
        except Exception as db_chunk_err:
            logger.error("Error during chunk storage step (API calls)", error=str(db_chunk_err), exc_info=True)
            # -- Removed direct DB status update on error --
            # DBService.update_document_status(document_id=document_id, status='error_db') # Removed
            raise APIError(f"Error during chunk storage API calls: {db_chunk_err}", 500) from db_chunk_err # Adjusted error message

        # --- Step 6: Generate Embeddings and Store Vectors ---
        _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                  status="Processing", progress=90, details="Generating and storing vector embeddings...")
        try:
            docs_to_embed = []
            for i, chunk_text_content in enumerate(chunks):
                 # Find corresponding chunk_id if available, otherwise handle error/skip?
                 # This assumes chunk_ids and chunks lists correspond index-wise
                 current_chunk_id = chunk_ids[i] if i < len(chunk_ids) else None
                 if current_chunk_id is None: continue # Skip if chunk wasn't stored

                 metadata = {
                     "document_id": str(document_id), # Ensure string
                     "user_id": str(user_id), # Ensure string
                     "filename": filename,
                     "author": author_name or "Unknown",
                     "title": title or filename,
                     "language": processing_language,
                     "chunk_index": i,
                     "chunk_id": str(current_chunk_id), # Link vector to DB chunk ID
                     "text": chunk_text_content # Include text in metadata for easier retrieval later
                 }
                 docs_to_embed.append((chunk_text_content, metadata)) # Tuple format for add_documents

            if not docs_to_embed:
                raise APIError("No valid chunks available for embedding after storage.", 500)

            # Use VectorSearchService to add documents (handles embedding)
            success_count, failure_count = vector_service.add_documents(user_id=user_id, documents=docs_to_embed)

            if failure_count > 0:
                logger.warning(f"Failed to embed/store {failure_count} vectors out of {len(docs_to_embed)}. ")
                # Consider more specific error handling or partial success reporting

            logger.info("Vector embeddings generated and stored.", success_count=success_count, failure_count=failure_count)
            final_status = 'completed' if failure_count == 0 else 'completed_with_errors'
            final_details = f"Processing complete. {success_count}/{len(docs_to_embed)} vectors stored."
            if failure_count > 0:
                final_details += f" ({failure_count} failed)."

            # Update final document status in DB via API
            # --- Refactored: Replace direct DB call with API call placeholder ---
            # TODO: Make internal API call: PUT /api/v1/internal/documents/{document_id}/status
            # Body: {'status': final_status}
            logger.info(f"[Placeholder] Making API call to update final document status to {final_status}...")
            # Replace with actual API call result handling
            # DBService.update_document_status(document_id=document_id, status=final_status) # Original call removed
            logger.info("Final document status update API call sent (placeholder).", status=final_status)

            _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                      status=final_status, progress=100, details=final_details,
                                      result={"document_id": document_id, "chunks_processed": success_count}) # Add result info

            return {"document_id": document_id, "status": final_status, "chunks_processed": success_count}

        except Exception as vector_err:
            logger.error("Error during vector embedding/storage", error=str(vector_err), exc_info=True)
            # -- Removed direct DB status update on error --
            # DBService.update_document_status(document_id=document_id, status='error_vector_storage') # Removed
            raise APIError(f"Vector service error: {vector_err}", 500) from vector_err

    except Exception as e:
        logger.error("Unhandled error during PDF processing task", error=str(e), exc_info=True)
        error_message = f"An error occurred: {str(e)}"
        # Update Redis progress to Failed
        if redis_client: 
            _update_pdf_task_progress(redis_client, redis_key, pubsub_channel, effective_task_id, logger,
                                      status="Failed", progress=100, details=error_message, error=traceback.format_exc())
        # Update DB status if possible - REMOVED direct call
        if document_id:
             # -- Removed direct DB status update on error --
             # try:
             #     DBService.update_document_status(document_id=document_id, status='error_task_failed') # Removed
             # except Exception as db_update_err:
             #      logger.error("Failed to update final document error status in DB", error=db_update_err)
             logger.warning("Final error occurred, DB status not updated directly from task. Relying on Redis/Celery state.", document_id=document_id)
        # Update Celery state
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': traceback.format_exc()})
        # Ensure we raise APIError for consistency if it's a service-related failure, otherwise let original exception propagate for retry
        if isinstance(e, APIError): # Check if it's already an APIError we raised
            self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': e.message, 'status_code': e.status_code})
            # Don't re-raise if state is set? Or re-raise to ensure Celery retry logic if appropriate?
            # Let's re-raise for now to be safe with retry logic.
            raise e 
        else: # For other unexpected errors, wrap in APIError before raising?
             self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': traceback.format_exc()}) 
             # Wrap unexpected errors in APIError for consistent task failure reporting?
             raise APIError(f"Unhandled error during PDF processing: {str(e)}", 500) from e
             # Or just re-raise original for Celery's default retry based on Exception
             # raise # Let's stick with raising the original for now to leverage default retry