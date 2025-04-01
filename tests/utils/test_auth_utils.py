import pytest
import time
import os
import jwt
from services.utils.auth_utils import (
    hash_password, 
    verify_password,
    create_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    has_role,
    has_permission,
    MissingSecretError,
    InvalidTokenError,
    ExpiredTokenError
)

# Mock JWT Secret for testing
TEST_JWT_SECRET = "test-secret-key-!@#$"

# Fixture to mock the JWT_SECRET environment variable
@pytest.fixture(autouse=True)
def mock_jwt_secret(mocker):
    """Automatically mock the JWT_SECRET environment variable for all tests in this module."""
    mocker.patch.dict(os.environ, {"JWT_SECRET": TEST_JWT_SECRET})

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

def test_verify_password_incorrect_salt():
    """Test that verify_password returns False if salt is incorrect (edge case)."""
    password = "saltedgecase"
    pwd_hash, salt = hash_password(password)
    incorrect_salt = salt + "a" # Modify the salt slightly
    # This should result in a different hash being calculated
    assert verify_password(password, pwd_hash, incorrect_salt) is False

def test_verify_password_incorrect_hash():
    """Test that verify_password returns False if the stored hash is wrong."""
    password = "hashedgecase"
    pwd_hash, salt = hash_password(password)
    incorrect_hash = pwd_hash + "a" # Modify the hash slightly
    assert verify_password(password, incorrect_hash, salt) is False

# --- Tests for create_token --- 

def test_create_token_access_default_expiry():
    """Test creating a default access token."""
    user_id = "user123"
    username = "testuser"
    roles = ["editor"]
    permissions = ["read", "write"]
    token = create_token(user_id, username, type="access", roles=roles, permissions=permissions)
    assert isinstance(token, str)
    # Decode to check basic payload structure (expiry checked in decode tests)
    payload = jwt.decode(token, TEST_JWT_SECRET, algorithms=["HS256"])
    assert payload["sub"] == user_id
    assert payload["username"] == username
    assert payload["type"] == "access"
    assert payload["roles"] == roles
    assert payload["permissions"] == permissions
    assert "iat" in payload
    assert "exp" in payload
    # Default access expiry is 1 hour (3600s)
    assert payload["exp"] - payload["iat"] == 3600 

def test_create_token_refresh_default_expiry():
    """Test creating a default refresh token."""
    token = create_token("user456", "refresh_user", type="refresh")
    payload = jwt.decode(token, TEST_JWT_SECRET, algorithms=["HS256"])
    assert payload["type"] == "refresh"
    # Default refresh expiry is 30 days (2592000s)
    assert payload["exp"] - payload["iat"] == 2592000

def test_create_token_custom_expiry():
    """Test creating a token with custom expiry."""
    custom_expiry_secs = 60 * 5 # 5 minutes
    token = create_token("user789", "custom_expire", expiry=custom_expiry_secs)
    payload = jwt.decode(token, TEST_JWT_SECRET, algorithms=["HS256"])
    assert payload["exp"] - payload["iat"] == custom_expiry_secs

def test_create_token_no_secret(mocker):
    """Test that create_token raises error if JWT_SECRET is missing."""
    mocker.patch.dict(os.environ, {"JWT_SECRET": ""}) # Unset the secret
    with pytest.raises(MissingSecretError):
        create_token("user1", "nouser")

# --- Tests for decode_token --- 

def test_decode_token_valid():
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

def test_decode_token_expired(mocker):
    """Test that decoding an expired token raises ExpiredTokenError."""
    expiry_secs = 1 # Expire quickly
    token = create_token("user_exp", "exp_user", expiry=expiry_secs)
    # Mock time to be after the token expiry
    mocker.patch('time.time', return_value=time.time() + expiry_secs + 5)
    with pytest.raises(ExpiredTokenError):
        decode_token(token)

def test_decode_token_invalid_signature():
    """Test decoding a token with an invalid signature raises InvalidTokenError."""
    token = create_token("user_inv", "inv_user", expiry=60)
    # Tamper with the token (append chars - will invalidate signature)
    tampered_token = token + "invalid"
    with pytest.raises(InvalidTokenError):
        decode_token(tampered_token)

def test_decode_token_wrong_secret():
    """Test decoding a token signed with a different secret raises InvalidTokenError."""
    payload = {"sub": "wrongsecret", "exp": time.time() + 60}
    wrong_secret = TEST_JWT_SECRET + "-wrong"
    token_wrong_secret = jwt.encode(payload, wrong_secret, algorithm="HS256")
    # Try decoding with the correct secret (should fail)
    with pytest.raises(InvalidTokenError):
        decode_token(token_wrong_secret)

def test_decode_token_no_secret(mocker):
    """Test that decode_token raises error if JWT_SECRET is missing."""
    # Create token first while secret is mocked
    token = create_token("user1", "nouser", expiry=60)
    # Unset the secret for decoding attempt
    mocker.patch.dict(os.environ, {"JWT_SECRET": ""})
    with pytest.raises(MissingSecretError):
        decode_token(token)

# --- Tests for API Key Generation/Hashing --- 

EXPECTED_API_KEY_PREFIX = "pdf_wisdom."

def test_generate_api_key_format():
    """Test that generate_api_key has the correct prefix."""
    key = generate_api_key()
    assert isinstance(key, str)
    assert key.startswith(EXPECTED_API_KEY_PREFIX)
    assert len(key) > len(EXPECTED_API_KEY_PREFIX)

def test_generate_api_key_uniqueness():
    """Test that generated API keys are unique."""
    key1 = generate_api_key()
    key2 = generate_api_key()
    assert key1 != key2

def test_hash_api_key_consistency():
    """Test that hashing the same API key produces the same hash."""
    key = generate_api_key()
    hash1 = hash_api_key(key)
    hash2 = hash_api_key(key)
    assert isinstance(hash1, str)
    assert len(hash1) == 64 # SHA-256 hex digest length
    assert hash1 == hash2

def test_hash_api_key_difference():
    """Test that hashing different API keys produces different hashes."""
    key1 = generate_api_key()
    key2 = generate_api_key()
    hash1 = hash_api_key(key1)
    hash2 = hash_api_key(key2)
    assert hash1 != hash2

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