from flask import Blueprint, request, jsonify, current_app

# Import decorators and helpers
from services.utils.api_helpers import rate_limit, cache_result
from services.analytics.analytics_middleware import track_specific_event
from services.analytics.analytics_service import AnalyticsEvent
from services.api_gateway import ServiceError

translate_bp = Blueprint('translate', __name__)

def translate_text_internal(text, target_lang="en", source_lang=None):
    """
    Internal function to translate text using LLM service via API Gateway.
    Uses current_app context for gateway access.
    
    Args:
        text: Text to translate
        target_lang: Target language code
        source_lang: Source language code (if known)
    
    Returns:
        Dictionary with translation or error
    """
    api_gateway = current_app.api_gateway
    logger = current_app.logger
    
    if not api_gateway:
         logger.error("API Gateway not available for translation")
         return {"error": "Translation service not configured"}
         
    # Create translation prompt
    source_lang_prompt = f" from {source_lang}" if source_lang else ""
    system_prompt = f"You are a professional translator. Translate the user's text{source_lang_prompt} to {target_lang}. Return only the translated text without explanations or notes."
    
    try:
        logger.info(f"Requesting translation from LLM service for target {target_lang}")
        # Use LLM service for translation
        result = api_gateway.request(
            "llm", 
            "/complete", 
            method="POST",
            json={
                "prompt": text,
                "system_prompt": system_prompt,
                "temperature": 0.2, # Low temperature for more deterministic translation
                "max_tokens": int(len(text) * 1.5) + 50, # Estimate max tokens needed
                "task_type": "translation"
            }
        )
        
        if "error" in result:
            logger.error(f"LLM service translation error: {result['error']}")
            return {"error": result["error"]}
        
        logger.info(f"Translation successful for target {target_lang}")
        return {"translated_text": result["content"]}
        
    except ServiceError as e:
        logger.error(f"ServiceError during translation: {str(e)}")
        return {"error": f"Translation service error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}", exc_info=True)
        return {"error": f"Unexpected error during translation: {str(e)}"}

@translate_bp.route('/translate', methods=['POST'])
@rate_limit(lambda: current_app.redis_client, max_calls=30, per_seconds=60)
@cache_result(lambda: current_app.redis_client, ttl=86400)  # Cache for 24 hours
@track_specific_event(AnalyticsEvent.API_CALL)
def translate_route():
    """API endpoint to translate text."""
    logger = current_app.logger
    data = request.json
    
    if not data or not data.get('text'):
        return jsonify({"error": "No text provided"}), 400
    
    text = data.get('text')
    target_lang = data.get('target_lang', 'en')
    source_lang = data.get('source_lang')
    
    logger.info(f"Received translation request for target language: {target_lang}")
    result = translate_text_internal(text, target_lang, source_lang)
    
    if "error" in result:
        # Determine appropriate status code based on error type if possible
        status_code = 500 
        if "service error" in result["error"].lower():
             status_code = 503 # Service Unavailable
        return jsonify(result), status_code
    
    return jsonify(result) 