from flask import Blueprint, request, jsonify, current_app
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound

# Import decorators and helpers
from services.utils.api_helpers import rate_limit, cache_result
from services.analytics.analytics_middleware import track_specific_event
from services.analytics.analytics_service import AnalyticsEvent
from services.utils.error_utils import APIError, ValidationError
# Import auth decorator
from services.api.middleware.auth_middleware import require_auth

translate_bp = Blueprint('translate', __name__)

def translate_text_internal(text, target_lang="en", source_lang="auto"):
    """
    Internal function to translate text using the deep-translator library.
    
    Args:
        text: Text to translate
        target_lang: Target language code (e.g., 'en', 'es')
        source_lang: Source language code ('auto' detects automatically)
    
    Returns:
        Dictionary with translation or raises APIError
    """
    logger = current_app.logger
    
    try:
        logger.info(f"Requesting translation via deep-translator from '{source_lang}' to '{target_lang}'")
        
        # Use deep-translator (Google)
        translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        
        if translated_text is None:
             # Handle cases where translation might return None unexpectedly
             logger.warning("deep-translator returned None, indicating possible issue.")
             raise APIError("Translation failed or returned an empty result.", status_code=500)

        logger.info(f"Translation successful for target {target_lang}")
        return {"translated_text": translated_text}
        
    except TranslationNotFound as e:
        logger.error(f"Translation not found via deep-translator: {e}")
        raise APIError(f"Translation failed: {e}", status_code=404) # Or 400 if lang codes invalid?
    except Exception as e:
        # Catch other potential exceptions from deep-translator or network issues
        logger.error(f"Unexpected error during deep-translator execution: {str(e)}", exc_info=True)
        # Check for common error messages if possible, otherwise generic 500/503
        if "invalid source language" in str(e).lower() or "invalid target language" in str(e).lower():
             raise APIError(f"Invalid language code provided: {e}", status_code=400)
        raise APIError(f"An unexpected error occurred during translation: {str(e)}", status_code=503) # Assume service unavailable

@translate_bp.route('/translate', methods=['POST'])
@rate_limit(lambda: current_app.redis_client, max_calls=30, per_seconds=60)
@require_auth() # Add authentication check
@cache_result(lambda: current_app.redis_client, ttl=86400)  # Cache for 24 hours
@track_specific_event(AnalyticsEvent.API_CALL)
def translate_route():
    """API endpoint to translate text."""
    logger = current_app.logger
    data = request.json
    
    if not data or not data.get('text'):
        raise ValidationError("Missing required field: text")
    
    text = data.get('text')
    target_lang = data.get('target_lang', 'en')
    source_lang = data.get('source_lang', 'auto') # Default to auto-detect
    
    logger.info(f"Received translation request for target language: {target_lang}")
    
    # Call the updated internal function
    result = translate_text_internal(text, target_lang, source_lang)
    
    return jsonify(result) 