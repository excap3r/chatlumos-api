import json
import time
from flask import Blueprint, jsonify, Response, stream_with_context, current_app

progress_bp = Blueprint('progress', __name__)

@progress_bp.route('/progress-stream/<session_id>')
def progress_stream(session_id):
    """
    Server-Sent Events endpoint to stream question processing progress.
    
    This endpoint streams real-time updates about the processing stages
    of a question being answered or a PDF being processed.
    """
    logger = current_app.logger
    progress_events = current_app.progress_events
    
    if session_id not in progress_events:
        logger.warning(f"Attempt to access invalid progress stream session: {session_id}")
        return jsonify({"error": "Invalid or expired session ID"}), 404
    
    def generate_events():
        event_queue = progress_events.get(session_id)
        if not event_queue: # Check again, might have been deleted between check and here
            logger.warning(f"Progress queue for session {session_id} disappeared.")
            yield 'data: {"event": "error", "message": "Session expired or invalid"}\n\n'
            return

        try:
            logger.info(f"SSE stream opened for session: {session_id}")
            # Send initial event
            yield 'data: {"event": "connected", "message": "Connected to progress stream"}\n\n'
            
            # Listen for events until completion or timeout
            timeout = time.time() + 600  # 10 minute timeout (increased from 5)
            while time.time() < timeout:
                try:
                    # Non-blocking get with timeout
                    event = event_queue.get(timeout=1.0)
                    if event is None:  # None signals end of stream
                        logger.info(f"End of stream signal received for session: {session_id}")
                        yield 'data: {"event": "complete", "message": "Processing complete"}\n\n'
                        break
                    
                    if isinstance(event, dict):
                        event_json = json.dumps(event)
                        yield f'data: {event_json}\n\n'
                    else:
                        logger.warning(f"Received non-dict event in queue for session {session_id}: {type(event)}")
                except Exception:
                    # Queue.Empty exception signifies no new message, just continue polling
                    # logger.debug(f"No event in queue for session {session_id}, continuing...")
                    continue # Continue loop to check timeout or wait for next event
            else:
                 # Loop finished due to timeout
                 logger.warning(f"SSE stream timed out for session: {session_id}")
                 yield 'data: {"event": "timeout", "message": "Stream timed out"}\n\n'
                    
            # Send end event regardless of how loop ended (complete or timeout)
            yield 'data: {"event": "end", "message": "Stream ended"}\n\n'
            
        except GeneratorExit:
            # Client disconnected
            logger.info(f"Client disconnected from SSE stream for session: {session_id}")
        except Exception as e:
             logger.error(f"Error during SSE generation for session {session_id}: {e}", exc_info=True)
             try:
                 yield 'data: {"event": "error", "message": "An internal error occurred"}\n\n'
             except Exception:
                 pass # Cannot send if generator is broken
        finally:
            # Clean up the queue for this session ID
            if session_id in progress_events:
                try:
                    # Attempt to clear the queue before deleting
                    while not event_queue.empty():
                        try: 
                            event_queue.get_nowait()
                        except Exception:
                            break
                    del progress_events[session_id]
                    logger.info(f"Cleaned up progress queue for session: {session_id}")
                except KeyError:
                     logger.warning(f"Session {session_id} already cleaned up.")
                except Exception as cleanup_e:
                     logger.error(f"Error cleaning up progress queue for session {session_id}: {cleanup_e}")

    
    # Return Response with SSE mimetype
    return Response(stream_with_context(generate_events()),
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'X-Accel-Buffering': 'no',  # For Nginx proxy buffering
                       'Connection': 'keep-alive'
                   }) 