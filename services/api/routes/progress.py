import json
import time
from flask import Blueprint, jsonify, Response, stream_with_context, current_app
from datetime import datetime, timedelta

# Import utility
from services.utils.error_utils import APIError
from services.config import AppConfig

progress_bp = Blueprint('progress_bp', __name__)

# Configuration - Now read from AppConfig
# PROGRESS_TIMEOUT = 300 # seconds (Overall timeout for the connection)
# PUBSUB_LISTEN_TIMEOUT = 1.0 # Timeout for pubsub.listen() in seconds

@progress_bp.route('/progress/<task_id>', methods=['GET'])
def stream_progress(task_id):
    """Streams progress updates for a given task ID using Redis Pub/Sub and SSE."""
    redis_client = current_app.redis_client
    logger = current_app.logger
    redis_key = f"task:{task_id}" # Key for initial state fetch
    pubsub_channel = f"progress:{task_id}" # Channel to subscribe to
    
    # Read timeouts from config
    progress_timeout = AppConfig.PROGRESS_STREAM_TIMEOUT
    pubsub_listen_timeout = AppConfig.PROGRESS_PUBSUB_LISTEN_TIMEOUT

    if not redis_client:
        # Raise 503 Service Unavailable if Redis is essential
        raise APIError("Service temporarily unavailable: Cannot stream progress.", status_code=503)

    def generate_progress():
        start_time = time.time()
        pubsub = None # Initialize pubsub object
        
        try:
            # 1. Fetch and send initial state from Hash immediately
            try:
                if redis_client.exists(redis_key):
                    task_data_bytes = redis_client.hgetall(redis_key)
                    task_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in task_data_bytes.items()}
                    
                    # Prepare and send initial state
                    if 'result' in task_data and task_data['result']:
                         try: task_data['result'] = json.loads(task_data['result']) 
                         except: pass # Ignore decode error on initial fetch
                    if 'progress' in task_data:
                         try: task_data['progress'] = int(task_data['progress'])
                         except: task_data['progress'] = 0
                         
                    event_type = task_data.get('status', 'update').lower()
                    yield f"event: {event_type}\ndata: {json.dumps(task_data)}\n\n"
                    logger.debug(f"Sent initial state for task {task_id}: Status={task_data.get('status')}")
                    
                    # If already completed/failed, end stream
                    if task_data.get('status') in ["Completed", "Failed"]:
                        logger.info(f"Task {task_id} already in terminal state ({task_data.get('status')}) on connect.")
                        return # End the generator
                else:
                     logger.warning(f"Task key {redis_key} not found on initial fetch.")
                     # Optionally send an error event immediately
                     # yield f"event: error\ndata: {json.dumps({'message': 'Task not found or expired'})}\n\n"
                     # return 

            except Exception as initial_e:
                logger.error(f"Error fetching initial state for task {task_id}: {initial_e}")
                # Decide if we should send error and exit, or just continue to listen
                yield f"event: error\ndata: {json.dumps({'message': f'Error fetching initial state: {str(initial_e)}'})}\n\n"
                # return # Optionally end stream here

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
                message = pubsub.get_message(timeout=pubsub_listen_timeout)
                
                if message is None:
                    # No message received within timeout, continue loop to check overall timeout
                    continue
                
                if message['type'] == 'message':
                    message_data_str = message['data'].decode('utf-8')
                    logger.debug(f"Received message on {pubsub_channel}: {message_data_str[:100]}...")
                    try:
                        # Data should be the JSON string published by the task
                        task_data = json.loads(message_data_str)
                        
                        # Yield the received update
                        event_type = task_data.get('status', 'update').lower()
                        yield f"event: {event_type}\ndata: {message_data_str}\n\n" 
                        
                        # Check for terminal states in the message
                        if task_data.get('status') in ["Completed", "Failed"]:
                            logger.info(f"Task {task_id} reached terminal state via PubSub: {task_data.get('status')}. Closing stream.")
                            break # Exit loop
                            
                    except json.JSONDecodeError as json_e:
                        logger.error(f"Failed to decode JSON from PubSub message for task {task_id}: {json_e} - Data: {message_data_str}")
                    except Exception as proc_e:
                         logger.error(f"Error processing PubSub message for task {task_id}: {proc_e}")
                         # Optionally send an error event to the client
                         yield f"event: error\ndata: {json.dumps({'message': f'Error processing update: {str(proc_e)}'})}\n\n"
                         # Decide whether to break or continue listening
                         # break 

        except GeneratorExit:
             logger.info(f"Client disconnected from SSE stream for task {task_id}.")
        except Exception as e:
            logger.error(f"Unhandled error in SSE generator for task {task_id}: {e}", exc_info=True)
            try:
                 # Try to send a final error message if possible
                 yield f"event: error\ndata: {json.dumps({'message': 'An internal server error occurred generating progress.'})}\n\n"
            except Exception:
                 pass # Ignore if cannot send
        finally:
            # Clean up PubSub subscription
            if pubsub:
                try:
                    pubsub.unsubscribe(pubsub_channel)
                    pubsub.close()
                    logger.info(f"Unsubscribed and closed PubSub for task {task_id}")
                except Exception as pubsub_close_e:
                     logger.error(f"Error closing PubSub for task {task_id}: {pubsub_close_e}")
            logger.debug(f"Finished SSE stream generator for task {task_id}")

    # Return the streaming response
    return Response(stream_with_context(generate_progress()), mimetype='text/event-stream') 