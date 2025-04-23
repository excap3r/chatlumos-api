import pytest
import json
from unittest.mock import patch, MagicMock
from deep_translator.exceptions import TranslationNotFound

# Assume client, mock_auth fixtures are available from conftest

API_BASE_URL = "/api/v1"
MOCK_USER_ID = "test-translate-user-1"

def test_translate_success(client, mock_auth):
    """Test successful translation via POST /translate"""
    original_text = "Hello, world!"
    target_lang = "es"
    expected_translation = "¡Hola Mundo!"

    # Mock the GoogleTranslator class within the target module
    with patch('services.api.routes.translate.GoogleTranslator') as MockGoogleTranslator, \
         mock_auth(user_id=MOCK_USER_ID): # Use mock_auth fixture

        # Configure the mock translator instance returned by the constructor
        mock_translator_instance = MockGoogleTranslator.return_value
        mock_translator_instance.translate.return_value = expected_translation

        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {
            "text": original_text,
            "target_lang": target_lang
            # source_lang defaults to 'auto'
        }

        response = client.post(
            f'{API_BASE_URL}/translate',
            headers=headers,
            data=json.dumps(payload)
        )

    # Assertions
    assert response.status_code == 200
    assert response.json == {"translated_text": expected_translation}

    # Verify GoogleTranslator was initialized correctly
    MockGoogleTranslator.assert_called_once_with(source='auto', target=target_lang)

    # Verify the translate method was called on the instance
    mock_translator_instance.translate.assert_called_once_with(original_text)

def test_translate_missing_text(client, mock_auth):
    """Test translation fails with 400 if 'text' field is missing."""
    with mock_auth(user_id=MOCK_USER_ID):
        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {
            # "text": "missing",
            "target_lang": "fr"
        }
        response = client.post(
            f'{API_BASE_URL}/translate',
            headers=headers,
            data=json.dumps(payload)
        )

    assert response.status_code == 400
    assert 'error' in response.json
    assert response.json['error'] == 'Missing required field: text'

def test_translate_api_error(client, mock_auth):
    """Test error handling when deep-translator raises TranslationNotFound."""
    original_text = "Text that causes error"
    target_lang = "xx" # Invalid target to potentially trigger error

    # Mock the translate method to raise the specific error
    with patch('services.api.routes.translate.GoogleTranslator') as MockGoogleTranslator, \
         mock_auth(user_id=MOCK_USER_ID):

        mock_translator_instance = MockGoogleTranslator.return_value
        mock_translator_instance.translate.side_effect = TranslationNotFound(original_text)

        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {"text": original_text, "target_lang": target_lang}

        response = client.post(
            f'{API_BASE_URL}/translate',
            headers=headers,
            data=json.dumps(payload)
        )

    # Based on current code, TranslationNotFound raises APIError 404
    assert response.status_code == 404
    assert 'error' in response.json
    assert "Translation failed" in response.json['error']

def test_translate_invalid_lang(client, mock_auth):
    """Test error handling for invalid language codes (generic exception)."""
    original_text = "Some text"
    invalid_lang = "invalid-lang-code"

    # Mock the translate method to raise a generic error containing the trigger phrase
    with patch('services.api.routes.translate.GoogleTranslator') as MockGoogleTranslator, \
         mock_auth(user_id=MOCK_USER_ID):

        mock_translator_instance = MockGoogleTranslator.return_value
        # Simulate the error message check in the except block
        mock_translator_instance.translate.side_effect = Exception(f"something wrong: invalid target language {invalid_lang}")

        headers = {'Authorization': 'Bearer dummy', 'Content-Type': 'application/json'}
        payload = {"text": original_text, "target_lang": invalid_lang}

        response = client.post(
            f'{API_BASE_URL}/translate',
            headers=headers,
            data=json.dumps(payload)
        )

    # Based on current code, this specific error raises APIError 400
    assert response.status_code == 400
    assert 'error' in response.json
    assert "Invalid language code provided" in response.json['error']

def test_translate_unauthenticated(client):
    """Test translation fails with 401 if not authenticated."""
    headers = {'Content-Type': 'application/json'}
    payload = {"text": "unauthenticated text", "target_lang": "de"}
    response = client.post(
        f'{API_BASE_URL}/translate',
        headers=headers, # No Authorization header
        data=json.dumps(payload)
    )

    assert response.status_code == 401
    assert 'error' in response.json
    assert response.json['error'] == 'Authentication required'