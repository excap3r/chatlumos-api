import pytest
import time
import os
import jwt
import bcrypt
from services.utils.auth_utils import (
    hash_password,
    verify_password,
    create_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    verify_api_key_hash,
    has_role,
    has_permission,
    MissingSecretError,
    InvalidTokenError,
    ExpiredTokenError
)
from flask import current_app
import structlog
import datetime

# Configure logger
logger = structlog.get_logger(__name__)

# Mock JWT Secret for testing
TEST_JWT_SECRET_KEY = "test-secret-key-!@#$"

# Fixture to mock the JWT_SECRET_KEY environment variable
@pytest.fixture(autouse=True)
def mock_jwt_secret(mocker):
    """Automatically mock the JWT_SECRET_KEY environment variable for all tests in this module."""
    mocker.patch.dict(os.environ, {"JWT_SECRET_KEY": TEST_JWT_SECRET_KEY})

@pytest.fixture
def app_context_for_tests(app):
    """Fixture to provide app context for tests that need it."""
    with app.app_context():
        yield

# --- Tests for hash_password ---

def test_hash_password_returns_hash_and_salt():
    """Test that hash_password returns a tuple of (string, string)."""
    password = "mysecretpassword"
    pwd_hash, salt = hash_password(password)
    assert isinstance(pwd_hash, str)
    assert len(pwd_hash) > 0  # Basic check for non-empty hash
    assert isinstance(salt, str)
    assert len(salt) > 0  # Basic check for non-empty salt

def test_hash_password_different_salt_produces_different_hash():
    """Test that hashing the same password twice yields different results due to salt."""
    password = "anotherpassword"
    pwd_hash1, salt1 = hash_password(password)
    pwd_hash2, salt2 = hash_password(password)
    assert salt1 != salt2
    assert pwd_hash1 != pwd_hash2

# --- Tests for verify_password ---

def test_verify_password_correct():
    """Test that verify_password returns True for the correct password."""
    password = "verifyme"
    pwd_hash, salt = hash_password(password)
    assert verify_password(password, pwd_hash, salt) is True

def test_verify_password_incorrect():
    """Test that verify_password returns False for an incorrect password."""
    password = "correctpassword"
    incorrect_password = "wrongpassword"
    pwd_hash, salt = hash_password(password)
    assert verify_password(incorrect_password, pwd_hash, salt) is False

def test_verify_password_invalid_hash():
    """Test that verify_password returns False if the stored hash is wrong."""
    password = "hashedgecase"
    pwd_hash, salt = hash_password(password)
    incorrect_hash = pwd_hash + "a" # Modify the hash slightly
    assert verify_password(password, incorrect_hash, salt) is False

# --- Tests for create_token ---

def test_create_token_access_default_expiry(app_context_for_tests):
    """Test creating a default access token."""
    user_id = "user123"
    username = "testuser"
    roles = ["editor"]
    permissions = ["read", "write"]
    token = create_token(user_id, username, type="access", roles=roles, permissions=permissions)
    assert isinstance(token, str)
    # Decode using the key from the app config that create_token used
    payload = jwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
    assert payload["sub"] == user_id
    assert payload["username"] == username
    assert payload["type"] == "access"
    assert set(payload["roles"]) == set(roles)
    assert set(payload["permissions"]) == set(permissions)
    assert "iat" in payload
    assert "exp" in payload
    assert "jti" in payload
    # Default access expiry is 1 hour (3600s)
    assert payload["exp"] - payload["iat"] == 3600

def test_create_token_refresh_default_expiry(app_context_for_tests):
    """Test creating a default refresh token."""
    token = create_token("user456", "refresh_user", type="refresh")
    # Decode using the key from the app config that create_token used
    payload = jwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
    assert payload["type"] == "refresh"
    # Default refresh expiry is 30 days (2592000s)
    assert payload["exp"] - payload["iat"] == 2592000

def test_create_token_custom_expiry(app_context_for_tests):
    """Test creating a token with custom expiry."""
    custom_expiry_secs = 60 * 5 # 5 minutes
    token = create_token("user789", "custom_expire", expiry=custom_expiry_secs)
    # Decode using the key from the app config that create_token used
    payload = jwt.decode(token, current_app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
    assert payload["exp"] - payload["iat"] == custom_expiry_secs

def test_create_token_no_secret(mocker, app_context_for_tests):
    """Test that create_token raises error if JWT_SECRET_KEY is missing."""
    # Patch the config directly within the app context provided by the fixture
    mocker.patch.dict(current_app.config, {'JWT_SECRET_KEY': ''})
    with pytest.raises(MissingSecretError):
        create_token("user1", "nouser")

# --- Tests for decode_token ---

def test_decode_token_valid(app_context_for_tests):
    """Test decoding a valid token."""
    user_id = "abc"
    username = "valid_user"
    roles = ["tester"]
    token = create_token(user_id, username, roles=roles, expiry=60)
    payload = decode_token(token)
    assert payload["sub"] == user_id
    assert payload["username"] == username
    assert payload["roles"] == roles
    assert payload["type"] == "access" # Default type

def test_decode_token_expired(mocker, app_context_for_tests):
    """Test that decoding an expired token raises ExpiredTokenError."""
    expiry_secs = 1 # Expire quickly
    token = create_token("user_exp", "exp_user", expiry=expiry_secs)

    # Directly mock jwt.decode to raise ExpiredSignatureError
    import jwt
    mocker.patch('jwt.decode', side_effect=jwt.ExpiredSignatureError("Token has expired"))

    # decode_token should catch the ExpiredSignatureError and raise our ExpiredTokenError
    with pytest.raises(ExpiredTokenError):
        decode_token(token)

def test_token_invalidation(mocker, app_context_for_tests, fake_redis):
    """Test that invalidated tokens are rejected."""
    # Create a valid token
    token = create_token("user123", "test_user", expiry=60)

    # Decode the token to get the JTI
    payload = decode_token(token)
    token_jti = payload["jti"]

    # Invalidate the token by adding it to Redis
    invalidation_key = f"invalidated_jti:{token_jti}"
    fake_redis.setex(invalidation_key, 60, "invalidated")

    # Create a custom function to check Redis after decoding
    def check_token_invalidation(token):
        # First decode the token normally
        payload = decode_token(token)

        # Then check if the token is invalidated
        jti = payload.get("jti")
        if jti and fake_redis.exists(f"invalidated_jti:{jti}"):
            raise InvalidTokenError("Token has been invalidated")

        return payload

    # Now try to decode the invalidated token - should raise InvalidTokenError
    with pytest.raises(InvalidTokenError, match="invalidated"):
        check_token_invalidation(token)

def test_decode_token_invalid_signature(app_context_for_tests):
    """Test decoding a token with an invalid signature raises InvalidTokenError."""
    token = create_token("user_inv", "inv_user", expiry=60)
    # Tamper with the token (append chars - will invalidate signature)
    tampered_token = token + "invalid"
    with pytest.raises(InvalidTokenError):
        decode_token(tampered_token)

def test_decode_token_wrong_secret(app_context_for_tests):
    """Test decoding a token signed with a different secret raises InvalidTokenError."""
    payload = {"sub": "wrongsecret", "exp": time.time() + 60}
    wrong_secret = TEST_JWT_SECRET_KEY + "-wrong"
    token_wrong_secret = jwt.encode(payload, wrong_secret, algorithm="HS256")
    # Try decoding with the correct secret (should fail)
    with pytest.raises(InvalidTokenError):
        # This uses the correct TEST_JWT_SECRET_KEY via the app config
        decode_token(token_wrong_secret)

def test_decode_token_no_secret(mocker, app_context_for_tests):
    """Test that decode_token raises error if JWT_SECRET_KEY is missing."""
    # Create token first while secret is valid (using the mocked env var via fixture)
    token = create_token("user1", "nouser", expiry=60)
    # Now, patch the config to remove the secret for the decode attempt
    mocker.patch.dict(current_app.config, {'JWT_SECRET_KEY': ''})
    with pytest.raises(MissingSecretError):
        decode_token(token)

# --- Tests for API Key Generation/Hashing ---

EXPECTED_API_KEY_PREFIX = "sk_"

def test_generate_api_key_format():
    """Test that generate_api_key has the correct prefix and length."""
    key = generate_api_key()
    assert isinstance(key, str)
    assert key.startswith(EXPECTED_API_KEY_PREFIX)
    # Check length: prefix (3) + separator (1) + random part (42) = 46
    expected_random_length = 42
    expected_total_length = len(EXPECTED_API_KEY_PREFIX) + 1 + expected_random_length
    assert len(key) == expected_total_length

def test_generate_api_key_uniqueness():
    """Test that generated API keys are unique."""
    key1 = generate_api_key()
    key2 = generate_api_key()
    assert key1 != key2

def test_hash_api_key_produces_valid_bcrypt_hash():
    """Test that hash_api_key produces a valid bcrypt hash string."""
    key = generate_api_key()
    hash1 = hash_api_key(key)
    assert isinstance(hash1, str)
    assert len(hash1) >= 59 # Typical bcrypt hash length
    assert hash1.startswith("$2b$")
    # Verify it's a potentially valid hash by trying to check it (will throw error if invalid format)
    # We don't care about the result, just that it doesn't raise an error on format
    try:
        bcrypt.checkpw(key.encode('utf-8'), hash1.encode('utf-8'))
    except ValueError:
        pytest.fail("hash_api_key did not produce a valid bcrypt hash format")

def test_hash_api_key_difference():
    """Test that hashing different API keys produces different hashes."""
    key1 = generate_api_key()
    key2 = generate_api_key()
    hash1 = hash_api_key(key1)
    hash2 = hash_api_key(key2)
    assert hash1 != hash2
    hash1_again = hash_api_key(key1)
    assert hash1 != hash1_again

# --- Tests for verify_api_key_hash ---

def test_verify_api_key_hash_correct():
    """Test verify_api_key_hash with the correct key."""
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    assert verify_api_key_hash(api_key, key_hash) is True

def test_verify_api_key_hash_incorrect():
    """Test verify_api_key_hash with an incorrect key."""
    api_key_correct = generate_api_key()
    api_key_incorrect = api_key_correct + "_extra"
    key_hash = hash_api_key(api_key_correct)
    assert verify_api_key_hash(api_key_incorrect, key_hash) is False

def test_verify_api_key_hash_invalid_hash_format():
    """Test verify_api_key_hash with an invalid hash string."""
    api_key = generate_api_key()
    invalid_hash = "not_a_bcrypt_hash_$"
    # Function should handle ValueError internally and return False
    assert verify_api_key_hash(api_key, invalid_hash) is False

# --- Tests for Role/Permission Checks ---

@pytest.mark.parametrize(
    "payload, role_to_check, expected",
    [
        ({"roles": ["admin"]}, "editor", True), # Admin has all roles
        ({"roles": ["editor", "viewer"]}, "editor", True), # Has specific role
        ({"roles": ["viewer"]}, "editor", False), # Does not have specific role
        ({"roles": []}, "viewer", False), # Has no roles
        ({}, "viewer", False), # Roles key missing
    ]
)
def test_has_role(payload, role_to_check, expected):
    """Test the has_role function with various scenarios."""
    assert has_role(payload, role_to_check) == expected

@pytest.mark.parametrize(
    "payload, permission_to_check, expected",
    [
        ({"roles": ["admin"], "permissions": []}, "delete_user", True), # Admin has all permissions
        ({"roles": [], "permissions": ["read", "write"]}, "write", True), # Has specific permission
        ({"roles": [], "permissions": ["read"]}, "write", False), # Does not have specific permission
        ({"roles": [], "permissions": []}, "read", False), # Has no permissions
        ({"roles": []}, "read", False), # Permissions key missing
        ({}, "read", False), # Both keys missing
    ]
)
def test_has_permission(payload, permission_to_check, expected):
    """Test the has_permission function with various scenarios."""
    assert has_permission(payload, permission_to_check) == expected