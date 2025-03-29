import os
import time
import json
import logging

# Assuming ServiceError and translate_text_internal are available
# Adjust imports based on where translate_text_internal ends up.
# If translate_text is moved to its own service/module, import from there.
# For now, let's assume it might be in translate.py
from services.api_gateway import ServiceError
# from services.api.routes.translate import translate_text_internal

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
                 raise ServiceError(f"Text extraction failed: {extract_result['error']}")
            text = extract_result.get("text", "")
            page_count = extract_result.get("page_count", 0)
            if not text:
                 raise ValueError("Extracted text is empty.")
            update_progress("extract_text", "complete", details={"page_count": page_count, "text_length": len(text)})
            logger.info(f"[Session {session_id}] Text extracted ({len(text)} chars, {page_count} pages)")
        except Exception as e:
             logger.error(f"[Session {session_id}] Error during text extraction: {e}", exc_info=True)
             update_progress("extract_text", "error", error=str(e))
             raise # Stop processing

        # Step 2: Translate Text (Optional)
        original_text = text # Keep original for potential use
        if translate_to_english and language != 'en':
            update_progress("translate_text", "running", details={"message": f"Translating text from {language} to English..."})
            try:
                 # Use placeholder or actual internal function
                 # translated_text_dict = translate_text_internal(text, "en", language)
                 translated_text_dict = translate_text_placeholder(text, "en", language, logger)
                 if "error" in translated_text_dict:
                      raise ServiceError(f"Translation failed: {translated_text_dict['error']}")
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
                 raise ServiceError(f"Text chunking failed: {chunk_result['error']}")
            chunks = chunk_result.get("chunks", [])
            if not chunks:
                 raise ValueError("No chunks created from text.")
            update_progress("chunk_text", "complete", details={"chunk_count": len(chunks), "chunk_size": chunk_size})
            logger.info(f"[Session {session_id}] Text chunked into {len(chunks)} chunks.")
        except Exception as e:
             logger.error(f"[Session {session_id}] Error during text chunking: {e}", exc_info=True)
             update_progress("chunk_text", "error", error=str(e))
             raise # Stop processing

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
                 raise ServiceError(f"Failed to store document metadata: {db_result['error']}")
            doc_id = db_result.get("id")
            if not doc_id:
                 raise ValueError("Database did not return a document ID.")
            update_progress("store_metadata", "complete", details={"document_id": doc_id})
            logger.info(f"[Session {session_id}] Document metadata stored with ID: {doc_id}")
        except Exception as e:
             logger.error(f"[Session {session_id}] Error storing document metadata: {e}", exc_info=True)
             update_progress("store_metadata", "error", error=str(e))
             raise # Stop processing if metadata fails
             
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
                 raise ServiceError(f"Failed to store vector embeddings: {embed_result['error']}")
            
            stored_count = embed_result.get("stored_count", 0)
            if stored_count != len(chunks):
                 logger.warning(f"[Session {session_id}] Mismatch in stored vectors: expected {len(chunks)}, got {stored_count}")
                 
            update_progress("store_vectors", "complete", details={"stored_count": stored_count})
            logger.info(f"[Session {session_id}] Stored {stored_count} vector embeddings.")

        except Exception as e:
             logger.error(f"[Session {session_id}] Error storing vector embeddings: {e}", exc_info=True)
             update_progress("store_vectors", "error", error=str(e))
             # Consider cleanup or marking document as partially indexed?
             raise # Stop processing

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