import json
import redis
import logging
import hashlib
from functools import wraps
from flask import request, jsonify, current_app
from typing import Callable

# Note: The 'redis_client' parameter must be passed to the decorator factories
# cache_result and rate_limit when they are applied in app.py or other modules.
# Example: @cache_result(redis_client, ttl=3600)

def get_cache_key(prefix: str, *args) -> str:
    """Generate a cache key from arguments."""
    key_parts = [prefix] + [str(arg) for arg in args]
    key_str = ":".join(key_parts)
    return f"wisdom_api:{key_str}"

def cache_result(redis_client_provider: Callable[[], redis.Redis | None], ttl: int = 3600):
    """Cache decorator for API endpoints. Requires Redis client provider function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            redis_client = redis_client_provider()
            if not redis_client:
                # Log using Flask's current_app logger if available in context
                try:
                    logger = current_app.logger
                    logger.debug(f"Redis client not available for caching {func.__name__}")
                except RuntimeError: # Outside app context
                    logging.debug(f"Redis client not available for caching {func.__name__}")
                return func(*args, **kwargs)

            # Generate cache key
            try:
                cache_key = get_cache_key(
                    func.__name__,
                    request.path,
                    json.dumps(request.args.to_dict(), sort_keys=True),
                    json.dumps(request.get_json(silent=True) or {}, sort_keys=True)
                )
            except Exception as e:
                 try: current_app.logger.error(f"Error generating cache key for {func.__name__}: {e}", exc_info=True)
                 except RuntimeError: logging.error(f"Error generating cache key for {func.__name__}: {e}", exc_info=True)
                 return func(*args, **kwargs) # Proceed without caching if key generation fails

            # Try to get from cache
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    try: current_app.logger.debug(f"Cache hit for key: {cache_key}")
                    except RuntimeError: logging.debug(f"Cache hit for key: {cache_key}")
                    response_data = json.loads(cached)
                    return jsonify(response_data)
                try: current_app.logger.debug(f"Cache miss for key: {cache_key}")
                except RuntimeError: logging.debug(f"Cache miss for key: {cache_key}")
            except redis.RedisError as e:
                 try: current_app.logger.warning(f"Redis GET error for key {cache_key}: {e}")
                 except RuntimeError: logging.warning(f"Redis GET error for key {cache_key}: {e}")
            except json.JSONDecodeError as e:
                 try: current_app.logger.warning(f"Error decoding cached JSON for key {cache_key}: {e}. Re-fetching.")
                 except RuntimeError: logging.warning(f"Error decoding cached JSON for key {cache_key}: {e}. Re-fetching.")

            # Generate response
            response = func(*args, **kwargs)

            # Cache the response
            try:
                # Handle Flask response tuple (data, status, headers) or Response object
                response_data = None
                status_code = 200 # Default
                headers = {}

                if isinstance(response, tuple):
                    data = response[0]
                    status_code = response[1] if len(response) > 1 else 200
                    headers = response[2] if len(response) > 2 else {}
                    response_data = data.get_json() if hasattr(data, 'get_json') else json.loads(data.data)
                elif hasattr(response, 'get_json') and callable(response.get_json):
                     response_data = response.get_json()
                     status_code = response.status_code
                     headers = dict(response.headers)
                elif hasattr(response, 'data'): # Fallback for simple responses
                     # Only cache if data looks like JSON? Or based on content-type?
                     # Let's assume jsonify was intended if caching is used.
                     # If direct data caching is needed, this logic must be smarter.
                     logging.warning(f"Attempting to cache non-standard response type for {func.__name__}. Caching skipped.")

                # Only cache successful responses with valid JSON data
                if 200 <= status_code < 300 and response_data is not None:
                     redis_client.setex(cache_key, ttl, json.dumps(response_data))
                     try: current_app.logger.debug(f"Cached response for key: {cache_key}")
                     except RuntimeError: logging.debug(f"Cached response for key: {cache_key}")

            except redis.RedisError as e:
                 try: current_app.logger.warning(f"Redis SETEX error for key {cache_key}: {e}")
                 except RuntimeError: logging.warning(f"Redis SETEX error for key {cache_key}: {e}")
            except Exception as e:
                 try: current_app.logger.error(f"Error processing response for caching in function {func.__name__}: {e}", exc_info=True)
                 except RuntimeError: logging.error(f"Error processing response for caching in function {func.__name__}: {e}", exc_info=True)

            return response
        return wrapper
    return decorator

def rate_limit(redis_client_provider: Callable[[], redis.Redis | None], max_calls: int = 100, per_seconds: int = 60):
    """Rate limiting decorator for API endpoints. Requires Redis client provider function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            redis_client = redis_client_provider()
            if not redis_client:
                try: current_app.logger.debug(f"Redis client not available for rate limiting {func.__name__}")
                except RuntimeError: logging.debug(f"Redis client not available for rate limiting {func.__name__}")
                return func(*args, **kwargs)

            # Get client IP - Consider X-Forwarded-For header if behind a proxy
            client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)

            # Create rate limit key
            limit_key = f"ratelimit:{client_ip}:{func.__name__}"

            try:
                # Use pipeline for atomic incr and expire
                pipe = redis_client.pipeline()
                pipe.incr(limit_key)
                pipe.expire(limit_key, per_seconds) # Set expiry every time to handle clock drift

                # Handle both real Redis and mock Redis in tests
                result = pipe.execute()
                if result:
                    count = result[0] if isinstance(result, list) and len(result) > 0 else 1
                else:
                    # For tests with mocks that return empty results
                    count = int(redis_client.get(limit_key) or 1)


                # Check if over limit
                if count > max_calls:
                     try: current_app.logger.info(f"Rate limit exceeded for {client_ip} on {func.__name__}")
                     except RuntimeError: logging.info(f"Rate limit exceeded for {client_ip} on {func.__name__}")
                     # Consider adding 'Retry-After' header
                     return jsonify({
                         "error": "Rate limit exceeded",
                         "message": f"Maximum {max_calls} requests per {per_seconds} seconds"
                     }), 429

            except redis.RedisError as e:
                 try: current_app.logger.warning(f"Redis rate limiting error for key {limit_key}: {e}. Allowing request.")
                 except RuntimeError: logging.warning(f"Redis rate limiting error for key {limit_key}: {e}. Allowing request.")
                 # Fail open: Log warning and allow request if Redis fails

            # Execute function
            return func(*args, **kwargs)
        return wrapper
    return decorator