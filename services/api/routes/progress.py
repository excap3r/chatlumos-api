import json
import time
from flask import Blueprint, jsonify, Response, stream_with_context, current_app, g
from datetime import datetime, timedelta
from werkzeug.exceptions import NotFound, Forbidden # Import Forbidden
from redis.exceptions import LockError

# Import utility
from services.utils.error_utils import APIError, handle_error, ValidationError
from services.config import AppConfig
# Import auth decorator
from services.api.middleware.auth_middleware import require_auth

progress_bp = Blueprint('progress_bp', __name__)

# Configuration - Now read from AppConfig
# PROGRESS_TIMEOUT = 300 # seconds (Overall timeout for the connection)
# PUBSUB_LISTEN_TIMEOUT = 1.0 # Timeout for pubsub.listen() in seconds

# Constants
REDIS_TASK_TTL = 86400 # 24 hours in seconds
DEFAULT_PROGRESS_TIMEOUT = 300 # 5 minutes
PUBSUB_LISTEN_TIMEOUT = 1.0 # Check connection/timeout every 1 second

@progress_bp.route('/progress/<task_id>', methods=['GET'])
@require_auth() # Add authentication check
def stream_progress(task_id):
    """Stream task progress updates using Server-Sent Events (SSE)."""
    logger = current_app.logger
    redis_client = current_app.redis_client
    progress_timeout = current_app.config.get('PROGRESS_STREAM_TIMEOUT', DEFAULT_PROGRESS_TIMEOUT)
    pubsub_channel = f"progress:{task_id}"
    redis_key = f"task:{task_id}"
    current_user_id = g.user.get('id')

    if not redis_client:
        logger.error("Redis client not configured for progress streaming.")
        raise APIError("Server configuration error: Cannot stream progress.", 500)

    # --- Authorization Check (Moved before generator) --- #
    try:
        initial_state = redis_client.hgetall(redis_key)
        if not initial_state:
            logger.warning(f"Task key {redis_key} not found for progress stream.")
            return jsonify({
                "error": "Not Found", 
                "message": "Task not found or expired"
            }), 404

        task_user_id = initial_state.get('user_id')
        if not task_user_id or task_user_id != current_user_id:
            logger.warning(f"Authorization failed for task progress stream - TaskID: {task_id}, Owner: {task_user_id}, Requester: {current_user_id}")
            return jsonify({
                "error": "Forbidden",
                "message": "You do not have permission to view this task progress."
            }), 403

        # Initial state is valid and user is authorized, prepare it for sending
        # No decoding needed here if fakeredis returns strings
        task_data = initial_state
        if 'result' in task_data and task_data['result']:
            try: task_data['result'] = json.loads(task_data['result'])
            except: pass
        if 'progress' in task_data:
            try: task_data['progress'] = int(task_data['progress'])
            except: task_data['progress'] = 0
            
    except redis.RedisError as e:
         logger.error(f"Redis error fetching initial state for task {task_id}: {e}")
         raise APIError("Failed to fetch task state.", 500)
    except Exception as e:
        logger.error(f"Unexpected error during initial check for task {task_id}: {e}", exc_info=True)
        raise APIError("An unexpected error occurred.", 500)
    # --- End Authorization Check --- #

    # If authorization passed, define the generator
    def generate_progress():
        pubsub = None # Initialize pubsub to None for finally block
        start_time = time.time()
        try:
            # 1. Send initial state immediately
            event_type = task_data.get('status', 'update').lower()
            yield f"event: {event_type}\ndata: {json.dumps(task_data)}\n\n"
            logger.debug(f"Sent initial state for task {task_id}: Status={task_data.get('status')}")

            # If already completed/failed, end stream
            if task_data.get('status') in ["Completed", "Failed"]:
                logger.info(f"Task {task_id} already in terminal state ({task_data.get('status')}) on connect.")
                return # End the generator

            # 2. Subscribe and listen for updates
            pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(pubsub_channel)
            logger.info(f"Subscribed to Redis channel: {pubsub_channel} for task {task_id}")

            while True:
                # Check for overall timeout
                if time.time() - start_time > progress_timeout:
                    logger.warning(f"SSE stream overall timeout ({progress_timeout}s) for task {task_id}")
                    yield f"event: timeout\ndata: {json.dumps({'message': f'Stream timed out after {progress_timeout} seconds'})}\n\n"
                    break

                # Listen for messages
                message = pubsub.get_message(timeout=PUBSUB_LISTEN_TIMEOUT)
                
                if message is None:
                    # No message received within timeout, continue loop to check overall timeout
                    continue
                
                if message['type'] == 'message':
                    # Assuming messages are already strings due to decode_responses=True
                    message_data_str = message['data'] 
                    logger.debug(f"Received message on {pubsub_channel}: {message_data_str[:100]}...")
                    try:
                        # Data should be the JSON string published by the task
                        received_task_data = json.loads(message_data_str)
                        
                        # Yield the received update
                        event_type = received_task_data.get('status', 'update').lower()
                        yield f"event: {event_type}\ndata: {message_data_str}\n\n" 
                        
                        # Check for terminal states in the message
                        if received_task_data.get('status') in ["Completed", "Failed"]:
                            logger.info(f"Task {task_id} reached terminal state via PubSub: {received_task_data.get('status')}. Closing stream.")
                            break # Exit loop
                            
                    except json.JSONDecodeError as json_e:
                        logger.error(f"Failed to decode JSON from PubSub message for task {task_id}: {json_e} - Data: {message_data_str}")
                    except Exception as proc_e:
                         logger.error(f"Error processing PubSub message for task {task_id}: {proc_e}")
                         yield f"event: error\ndata: {json.dumps({'message': f'Error processing update: {str(proc_e)}'})}\n\n"

        except GeneratorExit:
             logger.info(f"Client disconnected from SSE stream for task {task_id}.")
        except Exception as e:
            logger.error(f"Unhandled error in SSE generator for task {task_id}: {e}", exc_info=True)
            try:
                 yield f"event: error\ndata: {json.dumps({'message': 'An internal server error occurred generating progress.'})}\n\n"
            except Exception:
                 pass
        finally:
            if pubsub:
                try:
                    pubsub.unsubscribe(pubsub_channel)
                    pubsub.close()
                    logger.info(f"Unsubscribed and closed PubSub for task {task_id}")
                except Exception as pubsub_close_e:
                     logger.error(f"Error closing PubSub for task {task_id}: {pubsub_close_e}")
            logger.debug(f"Finished SSE stream generator for task {task_id}")

    # Return the streaming response, wrapping the generator
    return Response(stream_with_context(generate_progress()), mimetype='text/event-stream') 