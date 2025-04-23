import pytest
from unittest.mock import patch, MagicMock, ANY, call
import uuid
from datetime import datetime, timedelta, timezone
import bcrypt
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import redis
import json
import hashlib
import structlog

# Import the module to test
import services.db.user_db as user_db_module
from services.db.user_db import (
    create_user,
    authenticate_user,
    get_user_by_id,
    update_user,
    update_user_roles,
    update_user_password,
    create_api_key,
    verify_api_key,
    get_user_api_keys
)
from services.db.models.user_models import User, UserRoleAssociation, APIKey
from services.db.exceptions import (
    UserAlreadyExistsError, DuplicateEntryError, QueryError, DatabaseError, InvalidCredentialsError, UserNotFoundError
)

# Import utilities
from services.db.db_utils import handle_db_session
from services.utils.auth_utils import verify_api_key_hash

# Configure logger
logger = structlog.get_logger(__name__)

# --- Define Cache Constants Locally for Test Isolation ---
API_KEY_CACHE_PREFIX = "apikey_verify_cache:"
API_KEY_CACHE_TTL = 60 # Match the TTL in user_db for realistic testing
API_KEY_CACHE_TTL_INVALID = 10 # Match the TTL used in _cache_invalid_result
API_KEY_CACHE_INVALID_MARKER = "__INVALID__" # Match the marker string
INVALID_MARKER_BYTES = API_KEY_CACHE_INVALID_MARKER.encode('utf-8')

# --- Mocks and Fixtures ---

@pytest.fixture(autouse=True)
def mock_bcrypt():
    """Mocks bcrypt functions."""
    with patch("bcrypt.gensalt") as mock_gensalt, patch(
        "bcrypt.hashpw"
    ) as mock_hashpw, patch("bcrypt.checkpw") as mock_checkpw:

        mock_salt = b"$2b$12$abcdefghijklmnopqrstuvwx"
        mock_gensalt.return_value = mock_salt

        # Make hashpw return a predictable bytes hash based on input password
        def hashpw_side_effect(password_bytes, salt):
            # Simple predictable hash for testing, includes salt prefix
            return salt + b":" + password_bytes

        mock_hashpw.side_effect = hashpw_side_effect

        # Make checkpw compare based on the predictable hash
        def checkpw_side_effect(password_bytes, hashed_password_bytes):
            salt, original_pw_bytes = hashed_password_bytes.split(b":", 1)
            return password_bytes == original_pw_bytes

        mock_checkpw.side_effect = checkpw_side_effect

        yield {"gensalt": mock_gensalt, "hashpw": mock_hashpw, "checkpw": mock_checkpw}


@pytest.fixture
def mock_redis():
    """Provides a mock Redis client via the _get_redis_client function."""
    with patch("services.db.user_db._get_redis_client") as mock_getter:
        mock_client = MagicMock(spec=redis.Redis)
        mock_client.ping.return_value = True  # Assume connected
        mock_client.get.return_value = None  # Default: cache miss
        mock_client.setex.return_value = True
        mock_client.delete.return_value = 1
        mock_getter.return_value = mock_client
        yield mock_client


# --- create_user Tests ---


# Remove parametrization as test only checks default role behavior
# @pytest.mark.parametrize("roles", [None, [UserRole.ADMIN]])
@patch("services.utils.auth_utils.hash_password")
@patch("services.db.user_db.User")
def test_create_user_success_default_role(
    MockUser, mock_hash_password, db_session, mock_bcrypt, mocker
):
    """Test successful user creation with the default 'user' role."""
    username = "testuser"
    email = "test@example.com"
    password = "password123"
    # Get mock bcrypt from the fixture
    hashed_pw_bytes = mock_bcrypt["hashpw"](password.encode("utf-8"), mock_bcrypt["gensalt"]())
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    # Mock the User instance returned after commit/refresh
    mock_user_instance = MagicMock(spec=User)
    mock_user_instance.id = uuid.uuid4()
    mock_user_instance.username = username
    mock_user_instance.email = email
    mock_user_instance.is_active = True
    mock_user_instance.created_at = datetime.utcnow()
    mock_user_instance.updated_at = datetime.utcnow()
    mock_user_instance.last_login = None
    mock_user_instance.password_hash = hashed_pw_str # Set hash on mock

    # Mock session methods using mocker.patch.object
    mock_add = mocker.patch.object(db_session, 'add')
    mock_flush = mocker.patch.object(db_session, 'flush')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_refresh = mocker.patch.object(db_session, 'refresh')
    mock_query = mocker.patch.object(db_session, 'query') # Need to mock the role query

    # Capture added objects
    added_objects = []
    def add_side_effect(obj):
        added_objects.append(obj)
        # Simulate ID assignment on flush for User object
        if isinstance(obj, User):
             obj.id = mock_user_instance.id

    mock_add.side_effect = add_side_effect

    # Simulate refresh loading defaults and roles relationship
    def refresh_side_effect(obj):
        if isinstance(obj, User):
            obj.id = mock_user_instance.id # Ensure ID is set before accessing relationships
            obj.is_active = mock_user_instance.is_active
            obj.created_at = mock_user_instance.created_at
            obj.updated_at = mock_user_instance.updated_at
            obj.last_login = mock_user_instance.last_login
            # No longer need to mock obj.roles here
            # mock_role_assoc = MagicMock(spec=UserRoleAssociation)
            # mock_role_assoc.role = "user"
            # obj.roles = [mock_role_assoc] # Set the relationship attribute

    mock_refresh.side_effect = refresh_side_effect

    # Mock the explicit role query using side_effect - return the final list directly
    def query_side_effect(*args, **kwargs):
        # Be less strict about query args, check if UserRoleAssociation is involved
        if UserRoleAssociation in args or (len(args) > 0 and getattr(args[0], 'class_', None) == UserRoleAssociation):
            print(f"MATCHED ROLE QUERY: {args}") # Debug print
            # Simulate the query chain returning the final list of tuples
            mock_chain = MagicMock()
            mock_chain.filter_by.return_value.all.return_value = [("user",)]
            return mock_chain
        # Return a default MagicMock for other unexpected queries
        print(f"UNEXPECTED QUERY ARGS: {args}") # Debug print
        return MagicMock()
    mock_query.side_effect = query_side_effect

    # Call the function under test (without roles argument for default behavior)
    result = create_user(username, email, password)

    # Assertions on mocks
    assert len(added_objects) >= 1
    added_user = next((o for o in added_objects if isinstance(o, User)), None)
    assert added_user is not None
    mock_add.assert_called() # Check add was called
    mock_flush.assert_called_once()
    mock_commit.assert_called_once()
    # Ensure refresh is called AFTER commit and with the user object
    mock_refresh.assert_called_once_with(added_user)
    # Assert the role query was called after refresh
    mock_query.assert_any_call(UserRoleAssociation.role)
    # Get the actual call arguments made to filter_by
    filter_by_call = next((c for c in mock_query.return_value.filter_by.call_args_list if c.kwargs.get('user_id') == mock_user_instance.id), None)
    assert filter_by_call is not None, "filter_by was not called with the expected user_id"

    # Assertions on logic
    mock_hash_password.assert_called_once_with(password)
    added_role_assoc = next((o for o in added_objects if isinstance(o, UserRoleAssociation)), None)
    assert added_role_assoc is not None
    assert added_role_assoc.user_id == added_user.id
    assert added_role_assoc.role == "user"

    # Check the dictionary returned by create_user
    assert result["id"] == str(mock_user_instance.id)
    assert result["username"] == username
    assert result["roles"] == ["user"]
    assert result["created_at"] is not None


@patch("services.utils.auth_utils.hash_password")
@patch("services.db.user_db.User")
def test_create_user_success_specific_roles(
    MockUser, mock_hash_password, db_session, mock_bcrypt, mocker
):
    """Test successful user creation with specific roles."""
    username = "adminuser"
    email = "admin@example.com"
    password = "complexpassword"
    roles = ["admin", "editor"]
    hashed_pw_bytes = mock_bcrypt["hashpw"](
        password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")
    mock_user_id = uuid.uuid4()

    # Mock session methods
    mock_add = mocker.patch.object(db_session, 'add')
    mock_flush = mocker.patch.object(db_session, 'flush')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_refresh = mocker.patch.object(db_session, 'refresh')
    mock_query = mocker.patch.object(db_session, 'query') # Mock query for roles

    added_objects = []
    def add_side_effect(obj):
        added_objects.append(obj)
        if isinstance(obj, User): obj.id = mock_user_id
    mock_add.side_effect = add_side_effect

    # Simulate refresh loading defaults and roles relationship
    def refresh_side_effect(obj):
        if isinstance(obj, User):
            obj.id = mock_user_id # Ensure ID is set
            obj.created_at = datetime.utcnow()
            obj.updated_at = datetime.utcnow()
            obj.is_active = True
            obj.last_login = None
            # No longer mock obj.roles here
            # mock_roles_assoc = []
            # for role in roles:
            #     mock_role_assoc = MagicMock(spec=UserRoleAssociation)
            #     mock_role_assoc.role = role
            #     mock_roles_assoc.append(mock_role_assoc)
            # obj.roles = mock_roles_assoc # Set the relationship attribute

    mock_refresh.side_effect = refresh_side_effect

    # Mock the explicit role query using side_effect - return the final list directly
    def query_side_effect_specific(*args, **kwargs):
        # Be less strict
        if UserRoleAssociation in args or (len(args) > 0 and getattr(args[0], 'class_', None) == UserRoleAssociation):
            print(f"MATCHED SPECIFIC ROLE QUERY: {args}") # Debug print
            mock_chain = MagicMock()
            mock_chain.filter_by.return_value.all.return_value = [(r,) for r in roles]
            return mock_chain
        print(f"UNEXPECTED QUERY ARGS (specific): {args}") # Debug print
        return MagicMock() # Default for other queries
    mock_query.side_effect = query_side_effect_specific

    # Call the function under test
    result = create_user(username, email, password, roles=roles)

    # Assertions on mocks
    assert len(added_objects) >= 2 # User + roles
    added_user = next((o for o in added_objects if isinstance(o, User)), None)
    assert added_user is not None
    mock_add.assert_called() # Check add was called
    mock_flush.assert_called_once()
    mock_commit.assert_called_once()
    mock_refresh.assert_called_once_with(added_user)
    # Assert the role query was called
    mock_query.assert_any_call(UserRoleAssociation.role)
    # Get the actual call arguments made to filter_by
    filter_by_call_specific = next((c for c in mock_query.return_value.filter_by.call_args_list if c.kwargs.get('user_id') == mock_user_id), None)
    assert filter_by_call_specific is not None, "filter_by was not called with the expected user_id"

    # Assertions on logic
    mock_hash_password.assert_called_once_with(password)
    role_assocs = [o for o in added_objects if isinstance(o, UserRoleAssociation)]
    assert len(role_assocs) == 2
    assert {assoc.role for assoc in role_assocs} == set(roles)
    for assoc in role_assocs: assert assoc.user_id == mock_user_id

    assert result["id"] == str(mock_user_id)
    assert result["username"] == username
    assert set(result["roles"]) == set(roles)


@patch("services.db.user_db.bcrypt") # Patch bcrypt used within the function
def test_create_user_already_exists(mock_bcrypt_inner, db_session, mock_bcrypt, mocker):
    """Test that UserAlreadyExistsError is raised for duplicate username/email."""
    username = "existinguser"
    email = "existing@example.com"
    password = "password123"

    # Mock bcrypt hashpw call within create_user
    hashed_pw_bytes = mock_bcrypt["hashpw"](password.encode("utf-8"), mock_bcrypt["gensalt"]())
    mock_bcrypt_inner.hashpw.return_value = hashed_pw_bytes
    mock_bcrypt_inner.gensalt.return_value = mock_bcrypt["gensalt"].return_value

    # Mock session.commit() to raise the IntegrityError that leads to DuplicateEntryError
    # The decorator then converts IntegrityError -> DuplicateEntryError
    # The create_user function then catches DuplicateEntryError -> UserAlreadyExistsError
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_rollback = mocker.patch.object(db_session, 'rollback')
    mock_add = mocker.patch.object(db_session, 'add')

    # Ensure the original exception in IntegrityError contains 'DUPLICATE'
    mock_commit.side_effect = IntegrityError("commit failed", {}, Exception("Some DUPLICATE key error"))

    # Expect UserAlreadyExistsError
    with pytest.raises(UserAlreadyExistsError):
        create_user(username, email, password)

    # Assertions
    mock_bcrypt_inner.hashpw.assert_called_once()
    mock_add.assert_called() # User and default role should be added
    mock_flush = mocker.patch.object(db_session, 'flush')
    mock_flush.assert_called_once()
    mock_commit.assert_called_once() # Commit was attempted
    mock_rollback.assert_called()


@patch("services.utils.auth_utils.hash_password")
def test_create_user_other_db_error(mock_hash_password, db_session, mocker):
    """Test user creation failure due to a generic database error."""
    username = "erroruser"
    email = "error@example.com"
    password = "password"

    # Mock session methods
    mock_add = mocker.patch.object(db_session, 'add')
    mock_flush = mocker.patch.object(db_session, 'flush')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_rollback = mocker.patch.object(db_session, 'rollback')

    # Simulate a generic SQLAlchemyError on commit
    mock_commit.side_effect = SQLAlchemyError("DB connection lost")

    with pytest.raises(QueryError):
        create_user(username, email, password)

    mock_add.assert_called()
    mock_commit.assert_called_once()
    mock_rollback.assert_called_once()


# --- authenticate_user Tests ---


# Remove unnecessary datetime mock and use mock_bcrypt fixture
# @patch("services.db.user_db.datetime") # Datetime is handled within the function, no need to mock here unless testing specific timestamp
def test_authenticate_user_success(db_session, mock_bcrypt, mocker): # Removed mock_datetime
    """Test successful user authentication."""
    username = "authuser"
    password = "correctpassword"
    user_id = uuid.uuid4()
    # Use mock_bcrypt fixture to generate the expected hash
    hashed_pw_bytes = mock_bcrypt["hashpw"](password.encode("utf-8"), mock_bcrypt["gensalt"]())
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    mock_user = MagicMock(spec=User, id=user_id, username=username, password_hash=hashed_pw_str, is_active=True)
    mock_user.last_login = None # Will be updated

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_rollback = mocker.patch.object(db_session, 'rollback') # For asserting it's not called

    # Mock finding the user by username/email (authenticate_user uses filter with OR)
    mock_user_query_chain = MagicMock()
    # The filter condition is complex (User.username == username) | (User.email == username)
    # We mock the final .first() call
    mock_user_query_chain.filter.return_value.first.return_value = mock_user


    # Mock fetching roles
    # Need to mock the specific query for UserRoleAssociation.role
    # session.query(UserRoleAssociation.role).filter(UserRoleAssociation.user_id == user.id).all()
    mock_role_query_chain = MagicMock()
    mock_role_filter = mock_role_query_chain.filter
    mock_role_filter.return_value.all.return_value = [("user",)] # Example role tuple

    # Configure mock_query to return the user query result first, then the role query result
    mock_query.side_effect = [
        mock_user_query_chain, # For User query
        mock_role_query_chain    # For UserRoleAssociation query
    ]

    # Call the function
    result = user_db_module.authenticate_user(username, password)

    # Verify bcrypt checkpw was called using the mock_bcrypt fixture
    mock_bcrypt["checkpw"].assert_called_once_with(
        password.encode("utf-8"), hashed_pw_bytes
    )

    # Verify user query was made (first call to query)
    mock_query.assert_any_call(User)
    mock_user_query_chain.filter.assert_called_once() # Check filter was called
    # Verify role query was made (second call to query)
    mock_query.assert_any_call(UserRoleAssociation.role)
    mock_role_filter.assert_called_once_with(UserRoleAssociation.user_id == user_id)
    mock_role_filter.return_value.all.assert_called_once()


    # Verify last_login was updated (commit called within authenticate_user)
    mock_commit.assert_called_once()
    mock_rollback.assert_not_called() # Should not rollback on success
    assert mock_user.last_login is not None
    assert isinstance(mock_user.last_login, datetime)

    # Verify the returned dictionary
    assert result["id"] == str(user_id)
    assert result["username"] == username
    assert result["is_active"] is True
    assert result["roles"] == ["user"]
    # Optionally check last_login if needed, though it's dynamic
    assert "last_login" in result and isinstance(result["last_login"], datetime)


# Keep the patch for verify_password here because this test specifically asserts
# that verify_password is *not* called when the user isn't found.
@patch("services.utils.auth_utils.verify_password")
def test_authenticate_user_not_found(mock_verify_password, db_session, mocker):
    """Test authentication failure when user is not found."""
    username = "nouser"
    password = "password"

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')

    mock_query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(user_db_module.InvalidCredentialsError):
        user_db_module.authenticate_user(username, password)

    mock_verify_password.assert_not_called()
    mock_commit.assert_not_called()


def test_authenticate_user_incorrect_password(db_session, mock_bcrypt, mocker):
    """Test authentication failure due to incorrect password."""
    username = "wrongpassuser"
    password = "wrongpassword"
    correct_password = "correctpassword"
    hashed_pw_bytes = mock_bcrypt["hashpw"](
        correct_password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    # The mock_user needs username and id if authenticate_user tries to fetch roles
    mock_user = MagicMock(spec=User, id=uuid.uuid4(), username=username, password_hash=hashed_pw_str, is_active=True)

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_rollback = mocker.patch.object(db_session, 'rollback')

    # Mock finding the user
    mock_query.return_value.filter.return_value.first.return_value = mock_user
    # No need to mock roles query as it fails before that
    # No need to set mock_verify_password.return_value = False, mock_bcrypt handles it

    with pytest.raises(user_db_module.InvalidCredentialsError):
        user_db_module.authenticate_user(username, password)

    # Assert that bcrypt.checkpw was called
    mock_bcrypt["checkpw"].assert_called_once_with(
        password.encode("utf-8"), hashed_pw_bytes
    )
    # Verify user query was made
    mock_query.assert_called_once()
    # Commit and rollback should not be called if password check fails
    mock_commit.assert_not_called()
    mock_rollback.assert_not_called()


def test_authenticate_user_inactive(db_session, mock_bcrypt, mocker):
    """Test authentication failure for an inactive user."""
    username = "inactiveuser"
    password = "correctpassword"
    hashed_pw_bytes = mock_bcrypt["hashpw"](
        password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    mock_user = MagicMock(spec=User, password_hash=hashed_pw_str, is_active=False)

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_rollback = mocker.patch.object(db_session, 'rollback')

    mock_query.return_value.filter.return_value.first.return_value = mock_user

    with pytest.raises(
        user_db_module.InvalidCredentialsError, match="User account is not active"
    ):
        user_db_module.authenticate_user(username, password)

    # Assert bcrypt checkpw was called
    mock_bcrypt["checkpw"].assert_called_once_with(
        password.encode("utf-8"), hashed_pw_bytes
    )
    # Verify user query was made
    mock_query.assert_called_once()
    # Commit and rollback should not be called
    mock_commit.assert_not_called()
    mock_rollback.assert_not_called()


# Remove unnecessary patch for verify_password and datetime
# @patch("services.utils.auth_utils.verify_password")
# @patch("services.db.user_db.datetime") # Datetime is handled within the function
def test_authenticate_user_last_login_update_fails(
    db_session, mock_bcrypt, mocker # Removed mock_datetime, mock_verify_password
):
    """Test authentication succeeds even if last_login update fails."""
    username = "loginupdatefail"
    password = "correctpassword"
    # Use mock_bcrypt fixture
    hashed_pw_bytes = mock_bcrypt["hashpw"](
        password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")
    user_id = uuid.uuid4()
    original_last_login = datetime(2023, 1, 10)

    mock_user = MagicMock(spec=User, id=user_id, username=username, password_hash=hashed_pw_str, is_active=True, last_login=original_last_login)

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_rollback = mocker.patch.object(db_session, 'rollback')

    # Mock finding the user
    mock_user_query_chain = MagicMock()
    mock_user_query_chain.filter.return_value.first.return_value = mock_user

    # Mock fetching roles (needed even if commit fails, as roles are fetched before commit)
    mock_role_query_chain = MagicMock()
    mock_role_filter = mock_role_query_chain.filter
    mock_role_filter.return_value.all.return_value = [("user",)] # Example roles

    # Configure mock_query side_effect
    mock_query.side_effect = [
        mock_user_query_chain, # For User query
        mock_role_query_chain    # For UserRoleAssociation query
    ]

    # Simulate failure during the commit for last_login update
    mock_commit.side_effect = IntegrityError("DB Error during commit", params={}, orig=None)

    # Call the function under test
    result = user_db_module.authenticate_user(username, password)

    # Verify bcrypt checkpw was called
    mock_bcrypt["checkpw"].assert_called_once_with(
        password.encode("utf-8"), hashed_pw_bytes
    )

    # Verify user query was made
    mock_query.assert_any_call(User)
    mock_user_query_chain.filter.assert_called_once()
    # Verify role query was made
    mock_query.assert_any_call(UserRoleAssociation.role)
    mock_role_filter.assert_called_once_with(UserRoleAssociation.user_id == user_id)
    mock_role_filter.return_value.all.assert_called_once()

    # Verify commit was attempted and rollback occurred (due to the commit side_effect)
    mock_commit.assert_called_once()
    mock_rollback.assert_called_once()  # Rollback should be called within the function's except block

    # Verify authentication still succeeded and returned user data
    assert result["id"] == str(user_id)
    assert result["username"] == username
    assert result["roles"] == ["user"]
    assert mock_user.last_login != original_last_login  # Modified in-place before failed commit
    # The returned last_login should reflect the *attempted* update, even though it was rolled back in DB
    assert "last_login" in result and isinstance(result["last_login"], datetime)
    assert result["last_login"] != original_last_login


# --- get_user_by_id Tests ---


def test_get_user_by_id_success(db_session, mocker):
    """Test successfully getting a user by ID."""
    user_id = uuid.uuid4()
    username = "getme"
    roles = ["viewer"]

    mock_user = MagicMock(spec=User, id=user_id, username=username, email="getme@example.com", is_active=True)
    mock_user.roles = [MagicMock(spec=UserRoleAssociation, role=r) for r in roles]

    # Mock session query
    mock_query = mocker.patch.object(db_session, 'query')
    mock_query.return_value.options.return_value.filter.return_value.first.return_value = mock_user

    result = user_db_module.get_user_by_id(str(user_id))

    # Verify query
    mock_query.assert_called_once_with(User)
    filter_arg = mock_query.return_value.options.return_value.filter.call_args[0][0]
    assert str(filter_arg) == "users.id = :id_1"
    assert filter_arg.right.value == user_id

    # Verify result
    assert result["id"] == str(user_id)
    assert result["roles"] == roles


def test_get_user_by_id_not_found(db_session, mocker):
    """Test getting a user by ID when the user does not exist."""
    user_id = uuid.uuid4()

    # Mock session query
    mock_query = mocker.patch.object(db_session, 'query')
    mock_query.return_value.options.return_value.filter.return_value.first.return_value = None

    with pytest.raises(
        user_db_module.UserNotFoundError, match=f"User with ID {str(user_id)} not found"
    ):
        user_db_module.get_user_by_id(str(user_id))

    mock_query.assert_called_once() # Verify query was attempted


def test_get_user_by_id_invalid_uuid_format(db_session, mocker):
    """Test getting user by ID with an invalid UUID string."""
    invalid_user_id = "not-a-uuid"

    # Mock session query (should not be called)
    mock_query = mocker.patch.object(db_session, 'query')

    with pytest.raises(
        user_db_module.UserNotFoundError, match="Invalid user ID format"
    ):
        user_db_module.get_user_by_id(invalid_user_id)

    mock_query.assert_not_called()


# --- update_user Tests ---


# Need to patch get_user_by_id within the user_db module
@patch("services.db.user_db.get_user_by_id")
@patch("services.db.user_db.datetime") # Keep datetime patch if needed by internal logic or assertions
def test_update_user_success(mock_get_user_by_id, mock_datetime, db_session, mocker):
    """Test successfully updating user data."""
    user_id = uuid.uuid4()
    user_id_str = str(user_id)
    original_email = "original@test.com"
    new_email = "new@test.com"
    new_is_active = False

    mock_user = MagicMock(spec=User, id=user_id, email=original_email, is_active=True)
    # No need to mock roles on mock_user, as get_user_by_id will provide them

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')

    # Mock finding the user
    mock_query.return_value.filter.return_value.first.return_value = mock_user

    # Mock the return value of the final get_user_by_id call
    mock_updated_user_data = {
        "id": user_id_str,
        "username": "original_user", # Assuming username wasn't updated
        "email": new_email,
        "is_active": new_is_active,
        "roles": ["user"], # Example roles
        "created_at": datetime(2023, 1, 1), # Example timestamp
        "updated_at": datetime.utcnow(), # Should reflect update
        "last_login": None
    }
    mock_get_user_by_id.return_value = mock_updated_user_data

    update_data = {"email": new_email, "is_active": new_is_active}
    result = user_db_module.update_user(user_id_str, update_data)

    # Verify user attributes were updated *before* commit
    assert mock_user.email == new_email
    assert mock_user.is_active == new_is_active
    # Verify commit was called
    mock_commit.assert_called_once()
    # Verify get_user_by_id was called with the correct ID *after* commit
    mock_get_user_by_id.assert_called_once_with(user_id_str)

    # Verify the result matches the mocked return value of get_user_by_id
    assert result == mock_updated_user_data


@patch("services.db.user_db.get_user_by_id")
@patch("services.db.user_db.datetime")
def test_update_user_only_one_field(mock_get_user_by_id, mock_datetime, db_session, mocker):
    """Test updating only a single user field."""
    user_id = uuid.uuid4()
    user_id_str = str(user_id)
    new_username = "new_username"
    original_email = "old@test.com"

    mock_user = MagicMock(spec=User, id=user_id, username="old", email=original_email, is_active=True)
    # No roles needed on mock_user

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')

    # Mock finding the user
    mock_query.return_value.filter.return_value.first.return_value = mock_user

    # Mock the return value of the final get_user_by_id call
    mock_updated_user_data = {
        "id": user_id_str,
        "username": new_username,
        "email": original_email, # Email remains unchanged
        "is_active": True,       # is_active remains unchanged
        "roles": [],             # Example roles
        "created_at": datetime(2023, 1, 1), # Example timestamp
        "updated_at": datetime.utcnow(), # Should reflect update
        "last_login": None
    }
    mock_get_user_by_id.return_value = mock_updated_user_data

    update_data = {"username": new_username}
    result = user_db_module.update_user(user_id_str, update_data)

    # Verify attribute was updated before commit
    assert mock_user.username == new_username
    assert mock_user.email == original_email  # Email should be unchanged
    # Verify commit was called
    mock_commit.assert_called_once()
    # Verify get_user_by_id was called
    mock_get_user_by_id.assert_called_once_with(user_id_str)
    # Verify the result matches the mocked return value
    assert result == mock_updated_user_data


@patch("services.db.user_db.get_user_by_id")
@patch("services.db.user_db.datetime")
def test_update_user_no_valid_fields(mock_get_user_by_id, mock_datetime, db_session, mocker):
    """Test updating with no valid fields in the data dictionary."""
    user_id = uuid.uuid4()
    user_id_str = str(user_id)
    original_username = "original_user"
    original_email = "original@example.com"

    mock_user = MagicMock(spec=User, id=user_id, username=original_username, email=original_email, is_active=True)
    # No roles needed on mock_user

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')

    # Mock finding the user
    mock_query.return_value.filter.return_value.first.return_value = mock_user

    # Mock the return value of the final get_user_by_id call (should be current data)
    mock_current_user_data = {
        "id": user_id_str,
        "username": original_username,
        "email": original_email,
        "is_active": True,
        "roles": [], # Example roles
        "created_at": datetime(2023, 1, 1),
        "updated_at": datetime(2023, 1, 1), # Not updated
        "last_login": None
    }
    mock_get_user_by_id.return_value = mock_current_user_data

    # Data contains fields not directly updatable by update_user
    update_data = {"password": "newpass", "roles": ["admin"], "invalid_field": "xyz"}
    result = user_db_module.update_user(user_id_str, update_data)

    # User object attributes should remain unchanged
    assert mock_user.username == original_username
    assert mock_user.email == original_email
    # Commit should NOT be called
    mock_commit.assert_not_called()
    # get_user_by_id *should* be called to return current state
    mock_get_user_by_id.assert_called_once_with(user_id_str)
    # Verify the result matches the mocked return value of get_user_by_id
    assert result == mock_current_user_data


@patch("services.db.user_db.datetime")
def test_update_user_not_found(mock_datetime, db_session, mocker):
    """Test updating a user that does not exist."""
    user_id = uuid.uuid4()

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')

    mock_query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(user_db_module.UserNotFoundError):
        user_db_module.update_user(str(user_id), {"email": "new@test.com"})

    mock_commit.assert_not_called()


# --- update_user_roles Tests ---


@patch("services.db.user_db._user_to_dict")
@patch("services.db.user_db.datetime")
def test_update_user_roles_add_only(mock_dt, mock_user_to_dict, db_session, mocker):
    """Test adding roles when none are removed."""
    user_id = uuid.uuid4()
    user_id_str = str(user_id)
    existing_roles_set = {"user"}
    new_roles_input = ["user", "editor"]
    new_roles_set = set(new_roles_input)
    role_to_add = "editor"
    final_roles = ["user", "editor"]

    mock_user = MagicMock(spec=User, id=user_id, username="addroleuser", email="add@test.com", is_active=True)

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_add = mocker.patch.object(db_session, 'add')
    mock_delete = MagicMock()

    # Mock Query Chain
    mock_find_user_query = MagicMock()
    mock_find_user_query.filter.return_value.first.return_value = mock_user

    mock_current_roles_query = MagicMock()
    mock_current_roles_assocs = [MagicMock(spec=UserRoleAssociation, role=r) for r in existing_roles_set]
    mock_current_roles_query.filter.return_value.all.return_value = mock_current_roles_assocs

    mock_delete_chain = MagicMock() # Delete query setup (delete method not expected to be called)
    mock_delete_chain.filter.return_value.filter.return_value.delete = mock_delete

    mock_final_roles_query = MagicMock()
    mock_final_roles_query.filter.return_value.all.return_value = [(r,) for r in final_roles]

    mock_query.side_effect = [
        mock_find_user_query,
        mock_current_roles_query,
        mock_delete_chain, # The query setup for delete *is* called, but delete() on it is not
        mock_final_roles_query
    ]

    # Mock _user_to_dict helper
    expected_result_dict = {
        "id": user_id_str,
        "username": mock_user.username,
        "email": mock_user.email,
        "is_active": mock_user.is_active,
        "roles": final_roles,
        "created_at": datetime(2023,1,1),
        "updated_at": datetime.utcnow(),
        "last_login": None,
        "permissions": []
    }
    mock_user_to_dict.return_value = expected_result_dict

    added_objects = []
    mock_add.side_effect = added_objects.append

    result = user_db_module.update_user_roles(user_id_str, new_roles_input)

    # Assertions
    mock_query.assert_any_call(User)
    mock_query.assert_any_call(UserRoleAssociation) # Current roles
    mock_query.assert_any_call(UserRoleAssociation) # Delete setup
    mock_query.assert_any_call(UserRoleAssociation.role) # Final roles

    mock_delete.assert_not_called() # Verify no roles were deleted
    assert len(added_objects) == 1
    assert added_objects[0].user_id == user_id
    assert added_objects[0].role == role_to_add

    mock_commit.assert_called_once()
    assert mock_user.updated_at is not None

    mock_user_to_dict.assert_called_once_with(mock_user, roles=final_roles)
    assert result == expected_result_dict


@patch("services.db.user_db._user_to_dict")
@patch("services.db.user_db.datetime")
def test_update_user_roles_remove_only(mock_dt, mock_user_to_dict, db_session, mocker):
    """Test removing roles when none are added."""
    user_id = uuid.uuid4()
    user_id_str = str(user_id)
    existing_roles_set = {"user", "editor", "reader"}
    new_roles_input = ["user", "reader"]
    new_roles_set = set(new_roles_input)
    role_to_remove = "editor"
    final_roles = ["user", "reader"]

    mock_user = MagicMock(spec=User, id=user_id, username="removeroleuser", email="remove@test.com", is_active=True)

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_add = mocker.patch.object(db_session, 'add')
    mock_delete = MagicMock()

    # Mock Query Chain
    mock_find_user_query = MagicMock()
    mock_find_user_query.filter.return_value.first.return_value = mock_user

    mock_current_roles_query = MagicMock()
    mock_current_roles_assocs = [MagicMock(spec=UserRoleAssociation, role=r) for r in existing_roles_set]
    mock_current_roles_query.filter.return_value.all.return_value = mock_current_roles_assocs

    mock_delete_chain = MagicMock()
    # Mock the delete call that happens after the filter chain
    mock_delete_chain.filter.return_value.filter.return_value.delete = mock_delete

    mock_final_roles_query = MagicMock()
    mock_final_roles_query.filter.return_value.all.return_value = [(r,) for r in final_roles]

    mock_query.side_effect = [
        mock_find_user_query,
        mock_current_roles_query,
        mock_delete_chain, # Setup for the delete query
        mock_final_roles_query
    ]

    # Mock _user_to_dict helper
    expected_result_dict = {
        "id": user_id_str,
        "username": mock_user.username,
        "email": mock_user.email,
        "is_active": mock_user.is_active,
        "roles": final_roles,
        "created_at": datetime(2023,1,1),
        "updated_at": datetime.utcnow(), # updated_at should be set
        "last_login": None,
        "permissions": []
    }
    mock_user_to_dict.return_value = expected_result_dict

    added_objects = []
    mock_add.side_effect = added_objects.append

    result = user_db_module.update_user_roles(user_id_str, new_roles_input)

    # Assertions
    mock_query.assert_any_call(User)
    mock_query.assert_any_call(UserRoleAssociation) # Current roles
    mock_query.assert_any_call(UserRoleAssociation) # Delete setup
    mock_query.assert_any_call(UserRoleAssociation.role) # Final roles

    mock_delete.assert_called_once_with(synchronize_session=False)
    # Check the filters applied before delete
    delete_filter_calls = mock_delete_chain.filter.call_args_list
    assert len(delete_filter_calls) == 2 # user_id filter and role filter
    # Check the role filter argument
    role_filter_arg = delete_filter_calls[1][0][0] # Get the second filter arg (role filter)
    # Ensure it's filtering for the roles to be removed
    assert str(role_filter_arg).strip() == "user_roles.role IN (:role_1)" # Check structure
    assert role_filter_arg.right.value == [role_to_remove] # Check value

    mock_add.assert_not_called() # Verify no roles were added

    mock_commit.assert_called_once()
    assert mock_user.updated_at is not None

    mock_user_to_dict.assert_called_once_with(mock_user, roles=final_roles)
    assert result == expected_result_dict


@patch("services.db.user_db._user_to_dict")
@patch("services.db.user_db.datetime")
def test_update_user_roles_no_change(mock_dt, mock_user_to_dict, db_session, mocker):
    """Test updating roles when the input matches existing roles."""
    user_id = uuid.uuid4()
    user_id_str = str(user_id)
    existing_roles_set = {"user", "reader"}
    new_roles_input = ["user", "reader"]
    final_roles = ["user", "reader"]

    mock_user = MagicMock(spec=User, id=user_id, username="nochangeuser", email="nochange@test.com", is_active=True)

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_add = mocker.patch.object(db_session, 'add')
    mock_delete = MagicMock()

    # Mock Query Chain
    mock_find_user_query = MagicMock()
    mock_find_user_query.filter.return_value.first.return_value = mock_user

    mock_current_roles_query = MagicMock()
    mock_current_roles_assocs = [MagicMock(spec=UserRoleAssociation, role=r) for r in existing_roles_set]
    mock_current_roles_query.filter.return_value.all.return_value = mock_current_roles_assocs

    # The final role query is called even if no change, to get the return value
    mock_final_roles_query = MagicMock()
    mock_final_roles_query.filter.return_value.all.return_value = [(r,) for r in final_roles]

    # Configure side_effect for session.query (delete setup query is NOT called)
    mock_query.side_effect = [
        mock_find_user_query,
        mock_current_roles_query,
        mock_final_roles_query
    ]

    # Mock _user_to_dict helper
    expected_result_dict = {
        "id": user_id_str,
        "username": mock_user.username,
        "email": mock_user.email,
        "is_active": mock_user.is_active,
        "roles": final_roles,
        "created_at": datetime(2023,1,1),
        "updated_at": datetime(2023,1,1), # Should NOT be updated
        "last_login": None,
        "permissions": []
    }
    mock_user_to_dict.return_value = expected_result_dict

    result = user_db_module.update_user_roles(user_id_str, new_roles_input)

    # Assertions
    mock_query.assert_any_call(User)
    mock_query.assert_any_call(UserRoleAssociation) # Current roles
    mock_query.assert_any_call(UserRoleAssociation.role) # Final roles query

    # Verify delete, add, and commit were NOT called
    mock_delete.assert_not_called()
    mock_add.assert_not_called()
    mock_commit.assert_not_called()
    # Verify updated_at was NOT updated on the mock object
    # assert mock_user.updated_at is None # Need to know initial value

    mock_user_to_dict.assert_called_once_with(mock_user, roles=final_roles)
    assert result == expected_result_dict


@patch("services.db.user_db.datetime") # Keep patch as it was there, though maybe unused
def test_update_user_roles_user_not_found(mock_datetime, db_session, mocker):
    """Test updating roles for a user that does not exist."""
    user_id = uuid.uuid4()
    user_id_str = str(user_id)
    new_roles_input = ["admin"]

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_add = mocker.patch.object(db_session, 'add')
    mock_delete = MagicMock()

    # Mock finding the user (returns None)
    mock_find_user_query = MagicMock()
    mock_find_user_query.filter.return_value.first.return_value = None

    # Configure query side effect (only first query happens)
    mock_query.side_effect = [mock_find_user_query]

    with pytest.raises(user_db_module.UserNotFoundError):
        user_db_module.update_user_roles(user_id_str, new_roles_input)

    # Assertions
    mock_query.assert_called_once_with(User) # Only user query should happen
    mock_find_user_query.filter.assert_called_once_with(User.id == user_id)
    mock_find_user_query.filter.return_value.first.assert_called_once()

    # Verify other DB operations did not occur
    mock_add.assert_not_called()
    mock_delete.assert_not_called()
    mock_commit.assert_not_called()


# --- update_user_password Tests ---


@patch("services.utils.auth_utils.hash_password")
def test_update_user_password_success(mock_hash_password, db_session, mock_bcrypt, mocker):
    """Test successfully updating a user's password."""
    user_id = uuid.uuid4()
    new_password = "newSecurePassword!"
    new_hashed_pw_bytes = mock_bcrypt["hashpw"](
        new_password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    new_hashed_pw_str = new_hashed_pw_bytes.decode("utf-8")

    mock_user = MagicMock(spec=User, id=user_id, password_hash="old_hash")

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')

    mock_query.return_value.filter.return_value.first.return_value = mock_user
    mock_hash_password.return_value = new_hashed_pw_str

    result = user_db_module.update_user_password(str(user_id), new_password)

    # Verify user was fetched
    mock_query.assert_called_once()

    # Verify password hash was updated on the mock user object
    assert mock_user.password_hash == new_hashed_pw_str
    assert mock_user.updated_at is not None  # Should be updated

    # Verify commit was called
    mock_commit.assert_called_once()

    # Verify result
    assert result is True


def test_update_user_password_user_not_found(db_session, mock_bcrypt, mocker):
    """Test updating password for a non-existent user."""
    user_id = uuid.uuid4()
    new_password = "newpassword"

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')

    mock_query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(user_db_module.UserNotFoundError):
        user_db_module.update_user_password(str(user_id), new_password)

    # Verify bcrypt hashpw was not called via the util function patch target if applicable
    # (Need to patch services.utils.auth_utils.hash_password)
    # For now, check commit
    mock_commit.assert_not_called()


@patch("services.db.user_db.bcrypt")
def test_update_user_password_db_error(mock_bcrypt_db, db_session, mock_bcrypt, mocker):
    """Test handling database error during password update commit."""
    user_id = uuid.uuid4()
    user_id_str = str(user_id)
    new_password = "newSecurePassword!"
    # Use the mock_bcrypt fixture's salt for consistency if needed, but hashpw mock overrides it
    hashed_pw_bytes = b"mocked_hashed_password_bytes_from_bcrypt_patch"
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    # Mock the return value of bcrypt.hashpw within the function
    mock_bcrypt_db.hashpw.return_value = hashed_pw_bytes

    mock_user = MagicMock(spec=User, id=user_id, password_hash="old_hash")

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_rollback = mocker.patch.object(db_session, 'rollback')

    mock_query.return_value.filter.return_value.first.return_value = mock_user

    # Simulate error on commit
    mock_commit.side_effect = IntegrityError("Commit failed", params={}, orig=None)

    # Use handle_db_session's expected wrapping (it catches SQLAlchemyError -> raises QueryError)
    with pytest.raises(QueryError):
        user_db_module.update_user_password(user_id_str, new_password)

    # Verify bcrypt.hashpw was called
    mock_bcrypt_db.hashpw.assert_called_once_with(
        new_password.encode('utf-8'), mock_bcrypt_db.gensalt.return_value
    )

    # Verify user's password_hash attribute was updated before the failed commit
    assert mock_user.password_hash == hashed_pw_str

    # Verify commit was attempted and rollback occurred (due to the commit side_effect and handle_db_session)
    mock_commit.assert_called_once()
    mock_rollback.assert_called_once()


# --- API Key Tests ---


# Patch bcrypt directly where it's used in create_api_key
@patch("services.db.user_db.bcrypt")
@patch("services.db.user_db.uuid.uuid4")
def test_create_api_key_success(mock_uuid, mock_bcrypt_db, db_session, mocker, mock_bcrypt): # Use mock_bcrypt fixture for salt
    """Test successfully creating an API key."""
    user_id = uuid.uuid4()
    user_id_str = str(user_id)
    key_name = "test_key"
    key_id = uuid.uuid4()
    api_key_val = "mysecretapikey123" # The raw key value
    api_key_prefix = "chatlumos_"
    full_api_key = f"{api_key_prefix}{api_key_val}"
    hashed_key_bytes = b"hashed_key_bytes" # Mocked hash

    # Mock the APIKey model constructor
    MockAPIKey = mocker.patch("services.db.user_db.APIKey", autospec=True)
    mock_api_key_instance = MockAPIKey.return_value
    mock_api_key_instance.id = key_id
    mock_api_key_instance.name = key_name
    mock_api_key_instance.prefix = api_key_prefix
    mock_api_key_instance.expires_at = None
    mock_api_key_instance.created_at = datetime.now(timezone.utc)
    mock_api_key_instance.last_used_at = None
    mock_api_key_instance.is_active = True
    mock_api_key_instance.user_id = user_id

    # Mock the APIKey creation utility
    mock_create_raw = mocker.patch("services.db.user_db.create_raw_api_key", return_value=(full_api_key, api_key_val))

    # Mock uuid.uuid4 to return a predictable value for the key ID
    mock_uuid.return_value = key_id

    # Mock bcrypt.hashpw used within create_api_key
    mock_bcrypt_db.hashpw.return_value = hashed_key_bytes

    # Mock session methods
    mock_add = mocker.patch.object(db_session, 'add')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_refresh = mocker.patch.object(db_session, 'refresh')

    # Side effects to capture added/refreshed objects if needed
    added_object = None
    refreshed_object = None

    def add_side_effect(obj):
        nonlocal added_object
        added_object = obj

    def refresh_side_effect(obj):
        nonlocal refreshed_object
        refreshed_object = obj
        # Simulate refresh setting the id if it wasn't set before commit
        if hasattr(obj, 'id') and not obj.id:
            obj.id = key_id

    mock_add.side_effect = add_side_effect
    mock_refresh.side_effect = refresh_side_effect

    # Mock finding the user (needed to associate the key)
    mock_user = MagicMock(spec=User, id=user_id)
    mocker.patch.object(db_session, 'get', return_value=mock_user)

    # Call the function under test
    result = user_db_module.create_api_key(user_id_str, key_name)

    # --- Assertions ---
    # Verify API key generation was called
    mock_create_raw.assert_called_once()

    # Verify bcrypt hashing was called with the raw key value
    mock_bcrypt_db.hashpw.assert_called_once()
    # Check the first argument of the hashpw call is the raw key bytes
    call_args, _ = mock_bcrypt_db.hashpw.call_args
    assert call_args[0] == api_key_val.encode('utf-8')

    # Verify session get was called for the user
    db_session.get.assert_called_once_with(User, user_id)

    # Verify APIKey was instantiated correctly
    MockAPIKey.assert_called_once_with(
        id=key_id,
        name=key_name,
        prefix=api_key_prefix,
        hashed_key=hashed_key_bytes.decode('utf-8'), # Stored as string in DB model
        user_id=user_id,
        expires_at=None, # Default expiry
        is_active=True
    )

    # Verify session add, commit, refresh
    mock_add.assert_called_once_with(added_object)
    assert added_object is mock_api_key_instance # Ensure the correct object was added
    mock_commit.assert_called_once()
    mock_refresh.assert_called_once_with(mock_api_key_instance)

    # Verify the result structure
    assert result['api_key'] == full_api_key # Raw key returned to user
    assert result['key_info']['id'] == str(key_id)
    assert result['key_info']['name'] == key_name
    assert result['key_info']['prefix'] == api_key_prefix
    assert not result['key_info']['expires_at'] # None


# Use the same mock_bcrypt fixture from conftest
@patch("services.db.user_db.bcrypt")
def test_create_api_key_user_not_found(mock_bcrypt_db, db_session, mocker):
    """Test creating an API key when the user does not exist."""
    user_id_str = str(uuid.uuid4())
    key_name = "Test Key Fail"

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_add = mocker.patch.object(db_session, 'add')
    mock_commit = mocker.patch.object(db_session, 'commit')

    mock_query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(user_db_module.UserNotFoundError):
        user_db_module.create_api_key(user_id_str, key_name)

    mock_bcrypt_db.assert_not_called()
    mock_add.assert_not_called()
    mock_commit.assert_not_called()


@patch("services.db.user_db.bcrypt")
def test_create_api_key_db_error(mock_bcrypt_db, db_session, mocker):
    """Test error handling during API key creation (e.g., commit fails)."""
    user_id_uuid = uuid.uuid4()
    user_id_str = str(user_id_uuid)
    key_name = "Test Key DB Error"

    mock_user = MagicMock(spec=User, id=user_id_uuid)

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_add = mocker.patch.object(db_session, 'add')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_rollback = mocker.patch.object(db_session, 'rollback')

    mock_query.return_value.filter.return_value.first.return_value = mock_user
    mock_bcrypt_db.return_value = "some_hash"

    # Simulate error on commit
    mock_commit.side_effect = IntegrityError("Commit failed", params={}, orig=None)

    with pytest.raises(IntegrityError): # Decorator might re-raise or wrap
        user_db_module.create_api_key(user_id_str, key_name)

    # Verify add was called, but commit failed
    mock_add.assert_called_once()
    mock_commit.assert_called_once()
    # Rollback handled by decorator


# --- verify_api_key Tests ---

# Use a realistic-looking hash for mocks
MOCK_API_KEY_HASH_BYTES = b'$2b$12$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
MOCK_API_KEY_HASH_STR = MOCK_API_KEY_HASH_BYTES.decode('utf-8')
# INVALID_MARKER_BYTES is defined above

# Define common key prefix
KEY_PREFIX = "sk"

# Helper to generate a valid test API key string
def create_test_api_key_str(key_id_uuid: uuid.UUID) -> str:
    return f"{KEY_PREFIX}_{key_id_uuid.hex}"

# Note: The patch target "services.db.user_db.verify_api_key_hash" is correct
# because verify_api_key calls verify_api_key_hash defined in the same module.
@patch("services.db.user_db.verify_api_key_hash")
def test_verify_api_key_success_cache_miss(mock_verify_hash, db_session, mock_redis, mocker):
    """Test successful API key verification when not in cache."""
    key_id_part_uuid = uuid.uuid4() # Use a real UUID for the key ID part
    key_id_part_str = key_id_part_uuid.hex
    api_key_str = create_test_api_key_str(key_id_part_uuid)
    db_key_id = uuid.uuid4() # Separate DB primary key UUID for the APIKey record
    user_id = uuid.uuid4()
    username = "api_user"
    roles = ["api_access"]

    # Configure mock verify_api_key_hash to return True for success
    mock_verify_hash.return_value = True

    # Mock Redis cache miss
    # Calculate cache key inline using locally defined constants
    key_id_hash = hashlib.sha256(key_id_part_str.encode('utf-8')).hexdigest()
    cache_key_expected = f"{API_KEY_CACHE_PREFIX}{key_id_hash}"
    mock_redis.get.return_value = None

    # Mock DB query result for ApiKey based on key_id_part_uuid
    mock_api_key = MagicMock(
        spec=APIKey,
        id=db_key_id, # DB Primary Key
        user_id=user_id,
        key=key_id_part_str, # Store the hex string part in the model's 'key' field
        key_hash=MOCK_API_KEY_HASH_STR, # Use string format hash here as per function logic
        is_active=True,
        expires_at=None,
        user=MagicMock(
            spec=User, id=user_id, username=username, is_active=True
        )
    )
    # Mock the relationship loading for roles
    mock_user_roles = [MagicMock(spec=UserRoleAssociation, role=r) for r in roles]
    mocker.patch.object(mock_api_key.user, 'roles', mock_user_roles)

    mock_query_apikey = mocker.patch.object(db_session, 'query')
    # Simulate finding the ApiKey by the key_id_part_uuid
    mock_query_apikey.return_value.options.return_value.filter.return_value.first.return_value = mock_api_key

    # Mock Redis set call
    mock_redis.setex = mocker.Mock()
    # Mock session commit for last_used update
    mock_commit = mocker.patch.object(db_session, 'commit')

    # Call the function
    result = user_db_module.verify_api_key(api_key_str)

    # Assertions
    mock_redis.get.assert_called_once_with(cache_key_expected)
    mock_query_apikey.assert_called_once_with(APIKey) # DB query happened
    # Check the filter condition used in the query
    filter_arg = mock_query_apikey.return_value.options.return_value.filter.call_args[0][0]
    # The DB query in verify_api_key filters by APIKey.id == key_uuid, where key_uuid is uuid.UUID(key_id_str)
    assert str(filter_arg) == "api_keys.id = :id_1" # Check filter column
    assert filter_arg.right.value == key_id_part_uuid # Check the value passed to the filter is the UUID

    # Ensure hash verification was called with the correct args (full key, stored hash string)
    mock_verify_hash.assert_called_once_with(api_key_str, MOCK_API_KEY_HASH_STR)

    # Assert commit was called for last_used update
    mock_commit.assert_called_once()
    assert mock_api_key.last_used is not None # Check that last_used was updated

    # Assert redis.setex was called correctly with the structure the function creates
    expected_cache_data = {
        "key_hash": MOCK_API_KEY_HASH_STR,
        "user_id": str(user_id),
        "is_active": True,
        "user_is_active": True,
        "expires_at": None # Or ISO format string if expires_at was set
    }
    mock_redis.setex.assert_called_once()
    args, _ = mock_redis.setex.call_args
    assert args[0] == cache_key_expected
    assert args[1] == API_KEY_CACHE_TTL
    # Removed assertion: assert isinstance(args[2], bytes)
    # Fakeredis might store as string, load directly:
    assert json.loads(args[2]) == expected_cache_data

    # Assert result structure (function returns user info, not cache structure)
    expected_user_info = {
        "id": str(user_id),
        "key_id": key_id_part_str # Return the string hex part of the key
    }
    assert result == expected_user_info

@patch("services.db.user_db.verify_api_key_hash")
def test_verify_api_key_success_cache_hit_valid(
    mock_verify_hash, db_session, mock_redis, mocker
):
    """Test successful API key verification when valid data is in cache."""
    key_id_part_uuid = uuid.uuid4()
    key_id_part_str = key_id_part_uuid.hex
    api_key_str = create_test_api_key_str(key_id_part_uuid)
    user_id = uuid.uuid4()

    # This is the structure expected in the cache by verify_api_key
    cached_data_dict = {
        "key_hash": MOCK_API_KEY_HASH_STR,
        "user_id": str(user_id),
        "is_active": True,
        "user_is_active": True,
        "expires_at": None
    }
    cached_data_bytes = json.dumps(cached_data_dict).encode("utf-8")
    # Calculate cache key inline
    key_id_hash = hashlib.sha256(key_id_part_str.encode('utf-8')).hexdigest()
    cache_key_expected = f"{API_KEY_CACHE_PREFIX}{key_id_hash}"

    # Mock Redis cache hit
    mock_redis.get.return_value = cached_data_bytes
    # Mock verify_api_key_hash to return True when called (as it will be by cache check logic)
    mock_verify_hash.return_value = True

    # Mock DB query (should not be called)
    mock_query_apikey = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit') # Should not be called

    # Call the function
    result = user_db_module.verify_api_key(api_key_str)

    # Assertions
    mock_redis.get.assert_called_once_with(cache_key_expected)
    # Hash verification SHOULD be called by cache logic
    mock_verify_hash.assert_called_once_with(api_key_str, MOCK_API_KEY_HASH_STR)
    mock_query_apikey.assert_not_called() # DB query should NOT happen
    mock_commit.assert_not_called() # DB commit should NOT happen on cache hit

    # Assert result matches expected user info derived from cache
    expected_user_info = {
        "id": str(user_id),
        "key_id": key_id_part_str
    }
    assert result == expected_user_info


@patch("services.db.user_db.verify_api_key_hash")
def test_verify_api_key_cache_hit_invalid_marker(
    mock_verify_hash, db_session, mock_redis, mocker
):
    """Test API key verification failure when cache has the 'invalid' marker."""
    key_id_part_uuid = uuid.uuid4()
    key_id_part_str = key_id_part_uuid.hex
    api_key_str = create_test_api_key_str(key_id_part_uuid)
    # Calculate cache key inline
    key_id_hash = hashlib.sha256(key_id_part_str.encode('utf-8')).hexdigest()
    cache_key_expected = f"{API_KEY_CACHE_PREFIX}{key_id_hash}"

    # Mock Redis cache hit with the invalid marker (bytes)
    mock_redis.get.return_value = INVALID_MARKER_BYTES

    # Mock DB query (should not be called)
    mock_query_apikey = mocker.patch.object(db_session, 'query')
    # Mock redis set (should not be called again)
    mock_redis.setex = mocker.Mock()
    mock_commit = mocker.patch.object(db_session, 'commit')

    # Call the function and expect the specific error for invalid marker cache hit
    with pytest.raises(
        user_db_module.InvalidCredentialsError, match="Invalid API Key \(cached\)" # Match error from cache invalid marker
    ):
        user_db_module.verify_api_key(api_key_str)

    # Assertions
    mock_redis.get.assert_called_once_with(cache_key_expected)
    mock_query_apikey.assert_not_called() # DB query should NOT happen
    mock_verify_hash.assert_not_called() # Hash verification should NOT happen
    mock_redis.setex.assert_not_called() # Cache should not be set again
    mock_commit.assert_not_called()


@patch("services.db.user_db.verify_api_key_hash")
def test_verify_api_key_fail_hash_mismatch_cache_miss(
    mock_verify_hash, db_session, mock_redis, mocker
):
    """Test API key verification failure due to hash mismatch (cache miss)."""
    key_id_part_uuid = uuid.uuid4()
    key_id_part_str = key_id_part_uuid.hex
    api_key_str = create_test_api_key_str(key_id_part_uuid)
    db_key_id = uuid.uuid4() # DB ID
    user_id = uuid.uuid4()
    # Calculate cache key inline
    key_id_hash = hashlib.sha256(key_id_part_str.encode('utf-8')).hexdigest()
    cache_key_expected = f"{API_KEY_CACHE_PREFIX}{key_id_hash}"

    # Configure mock verify_api_key_hash to return False for failure
    mock_verify_hash.return_value = False

    # Mock Redis cache miss
    mock_redis.get.return_value = None

    # Mock DB query result for ApiKey (needs to be found for hash check)
    mock_api_key = MagicMock(
        spec=APIKey,
        id=db_key_id,
        user_id=user_id,
        key=key_id_part_str,
        key_hash=MOCK_API_KEY_HASH_STR, # Provide a hash to compare against
        is_active=True,
        expires_at=None,
        user=MagicMock(spec=User, id=user_id, is_active=True) # Active user needed
    )
    mock_query_apikey = mocker.patch.object(db_session, 'query')
    mock_query_apikey.return_value.options.return_value.filter.return_value.first.return_value = mock_api_key

    # Mock Redis set call (should be called with invalid marker)
    mock_redis.setex = mocker.Mock()
    mock_commit = mocker.patch.object(db_session, 'commit') # Should not be called

    # Call the function and expect the specific error for hash mismatch after DB lookup
    with pytest.raises(
        user_db_module.InvalidCredentialsError, match="Invalid API Key" # General error after failed hash check
    ):
        user_db_module.verify_api_key(api_key_str)

    # Assertions
    mock_redis.get.assert_called_once_with(cache_key_expected)
    mock_query_apikey.assert_called_once_with(APIKey) # DB query happened
    # Check DB query filter value
    filter_arg = mock_query_apikey.return_value.options.return_value.filter.call_args[0][0]
    assert str(filter_arg) == "api_keys.id = :id_1"
    assert filter_arg.right.value == key_id_part_uuid

    mock_verify_hash.assert_called_once_with(api_key_str, MOCK_API_KEY_HASH_STR) # Hash verification was called

    # Assert redis.setex was called with the invalid marker
    mock_redis.setex.assert_called_once_with(
        cache_key_expected,
        API_KEY_CACHE_TTL_INVALID,
        INVALID_MARKER_BYTES # Check if "invalid" marker (bytes) was cached
    )
    mock_commit.assert_not_called() # Commit for last_used update shouldn't happen on failure


@patch("services.db.user_db.verify_api_key_hash")
def test_verify_api_key_fail_key_inactive(mock_verify_hash, db_session, mock_redis, mocker):
    """Test API key verification failure because the key is inactive."""
    key_id_part_uuid = uuid.uuid4()
    key_id_part_str = key_id_part_uuid.hex
    api_key_str = create_test_api_key_str(key_id_part_uuid)
    db_key_id = uuid.uuid4() # DB ID
    user_id = uuid.uuid4()
    # Calculate cache key inline
    key_id_hash = hashlib.sha256(key_id_part_str.encode('utf-8')).hexdigest()
    cache_key_expected = f"{API_KEY_CACHE_PREFIX}{key_id_hash}"

    # Mock Redis cache miss
    mock_redis.get.return_value = None

    # Mock DB query result for ApiKey (inactive)
    mock_api_key = MagicMock(
        spec=APIKey,
        id=db_key_id,
        user_id=user_id,
        key=key_id_part_str,
        key_hash=MOCK_API_KEY_HASH_STR,
        is_active=False, # Key is inactive
        expires_at=None,
        user=MagicMock(spec=User, id=user_id, is_active=True) # User might still be active
    )
    mock_query_apikey = mocker.patch.object(db_session, 'query')
    mock_query_apikey.return_value.options.return_value.filter.return_value.first.return_value = mock_api_key

    # Mock Redis set call (should be called with invalid marker)
    mock_redis.setex = mocker.Mock()
    mock_commit = mocker.patch.object(db_session, 'commit') # Should not be called

    # Call the function and expect the specific error for inactive key after DB lookup
    with pytest.raises(
        user_db_module.InvalidCredentialsError, match="Invalid API Key" # Error raised before hash check now
    ):
        user_db_module.verify_api_key(api_key_str)

    # Assertions
    mock_redis.get.assert_called_once_with(cache_key_expected)
    mock_query_apikey.assert_called_once_with(APIKey) # DB query happened
    # Check DB query filter value
    filter_arg = mock_query_apikey.return_value.options.return_value.filter.call_args[0][0]
    assert str(filter_arg) == "api_keys.id = :id_1"
    assert filter_arg.right.value == key_id_part_uuid

    mock_verify_hash.assert_not_called() # Hash verification should NOT happen if key is inactive

    # Assert redis.setex was called with the invalid marker
    mock_redis.setex.assert_called_once_with(
        cache_key_expected,
        API_KEY_CACHE_TTL_INVALID, # Check correct TTL
        INVALID_MARKER_BYTES
    )
    mock_commit.assert_not_called() # No last_used update on failure


# --- get_user_api_keys Tests ---


def test_get_user_api_keys_success(db_session, mocker):
    user_id_uuid = uuid.uuid4()
    user_id_str = str(user_id_uuid)

    # Create the mock key and explicitly configure its attributes
    mock_key_1 = MagicMock(spec=APIKey)
    mock_key_1.id = uuid.uuid4()
    mock_key_1.name = "Key One" # Explicitly set attribute to the string value
    mock_key_1.prefix = "sk"
    mock_key_1.created_at = datetime.utcnow()
    mock_key_1.expires_at = None
    mock_key_1.last_used_at = None
    mock_key_1.is_active = True
    mock_key_1.user_id = user_id_uuid # Need user_id for the dict creation

    mock_keys = [mock_key_1]

    mock_session_instance = MagicMock(spec=Session)
    mock_get_session = mocker.patch('services.db.user_db._get_session', return_value=mock_session_instance)

    mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_keys

    results = user_db_module.get_user_api_keys(user_id_str)

    mock_get_session.assert_called_once()
    mock_session_instance.query.assert_called_once_with(APIKey)
    mock_session_instance.query.return_value.filter.assert_called_once()
    filter_arg = mock_session_instance.query.return_value.filter.call_args[0][0]
    assert str(filter_arg) == "api_keys.user_id = :user_id_1"
    assert filter_arg.right.value == user_id_uuid

    mock_session_instance.query.return_value.filter.return_value.order_by.assert_called_once()
    mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.assert_called_once()

    assert len(results) == 1
    assert results[0]['name'] == "Key One"


def test_get_user_api_keys_none_found(db_session, mocker):
    user_id_uuid = uuid.uuid4()
    user_id_str = str(user_id_uuid)

    mock_session_instance = MagicMock(spec=Session)
    mock_get_session = mocker.patch('services.db.user_db._get_session', return_value=mock_session_instance)

    mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

    results = user_db_module.get_user_api_keys(user_id_str)

    mock_get_session.assert_called_once()
    mock_session_instance.query.assert_called_once_with(APIKey)
    mock_session_instance.query.return_value.filter.assert_called_once()
    filter_arg = mock_session_instance.query.return_value.filter.call_args[0][0]
    assert str(filter_arg) == "api_keys.user_id = :user_id_1"
    assert filter_arg.right.value == user_id_uuid

    mock_session_instance.query.return_value.filter.return_value.order_by.assert_called_once()
    mock_session_instance.query.return_value.filter.return_value.order_by.return_value.all.assert_called_once()

    assert results == []


def test_get_user_api_keys_invalid_user_id_format():
    invalid_user_id = "not-a-valid-uuid"
    results = user_db_module.get_user_api_keys(invalid_user_id)
    assert results == []


# --- revoke_api_key Tests ---

def test_revoke_api_key_success(db_session, mocker):
    """Test successfully revoking an API key."""
    key_id_uuid = uuid.uuid4()
    key_id_str = str(key_id_uuid)

    # Use autospec=True to ensure the mock behaves like the real APIKey
    mock_api_key = MagicMock(spec=APIKey, id=key_id_uuid, is_active=True)
    # Create a specific mock for the is_active setter to track calls
    mock_is_active_setter = mocker.PropertyMock()
    type(mock_api_key).is_active = mock_is_active_setter

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')

    # Mock finding the key - use filter instead of filter_by if needed based on func
    # revoke_api_key uses filter(APIKey.id == key_id)
    mock_query.return_value.filter.return_value.first.return_value = mock_api_key

    result = user_db_module.revoke_api_key(key_id_str)

    # Assertions
    mock_query.assert_called_once()
    mock_query.return_value.filter.assert_called_once()
    # Verify that is_active was set to False (check last call)
    assert mock_is_active_setter.call_args == call(False)
    mock_commit.assert_called_once()
    assert result is True


def test_revoke_api_key_not_found(db_session, mocker):
    """Test revoking an API key that does not exist raises ApiKeyNotFoundError."""
    key_id_str = str(uuid.uuid4())

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    # Mock rollback as the decorator might call it on exception
    mock_rollback = mocker.patch.object(db_session, 'rollback')

    mock_query.return_value.filter.return_value.first.return_value = None

    # Expect ApiKeyNotFoundError to be raised
    with pytest.raises(user_db_module.ApiKeyNotFoundError, match=f"API Key with ID {key_id_str} not found"):
        user_db_module.revoke_api_key(key_id_str)

    # Assertions
    mock_query.assert_called_once()
    mock_commit.assert_not_called() # Commit should not be called if key not found
    # Rollback might be called by the decorator, so don't assert not_called
    # mock_rollback.assert_called_once() # Optional: assert decorator calls rollback


def test_revoke_api_key_db_error(db_session, mocker):
    """Test handling DB error during API key revocation."""
    key_id_uuid = uuid.uuid4()
    key_id_str = str(key_id_uuid)

    # Use autospec=True
    mock_api_key = MagicMock(spec=APIKey, id=key_id_uuid, is_active=True)
    # Create a specific mock for the is_active setter
    mock_is_active_setter = mocker.PropertyMock()
    type(mock_api_key).is_active = mock_is_active_setter

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')
    mock_commit = mocker.patch.object(db_session, 'commit')
    mock_rollback = mocker.patch.object(db_session, 'rollback')

    # Mock finding the key - use filter
    mock_query.return_value.filter.return_value.first.return_value = mock_api_key
    mock_commit.side_effect = SQLAlchemyError("Commit failed")

    with pytest.raises(QueryError):
        user_db_module.revoke_api_key(key_id_str)

    # Verify bcrypt.hashpw was called
    mock_bcrypt_db.hashpw.assert_called_once_with(
        new_password.encode('utf-8'), mock_bcrypt_db.gensalt.return_value
    )

    # Verify user's password_hash attribute was updated before the failed commit
    assert mock_user.password_hash == hashed_pw_str

    # Verify commit was attempted and rollback occurred (due to the commit side_effect and handle_db_session)
    mock_commit.assert_called_once()
    mock_rollback.assert_called_once()


# --- get_all_users Tests ---

def test_get_all_users_success(db_session, mocker):
    """Test retrieving all users successfully."""
    mock_users = [MagicMock(spec=User, id=uuid.uuid4()) for _ in range(3)]
    mock_roles_assocs = [MagicMock(spec=UserRoleAssociation, user_id=mock_users[i].id, role=f"role{i}") for i in range(3)]

    # Mock session methods
    mock_query = mocker.patch.object(db_session, 'query')

    # Mock base user query
    mock_query.return_value.order_by.return_value.limit.return_value.offset.return_value.all.return_value = mock_users
    # Mock roles query
    mock_query.return_value.filter.return_value.all.return_value = mock_roles_assocs

    results = user_db_module.get_all_users()

    # Assertions (basic check)
    assert len(results) == 3
    assert results[0]['roles'] == ['role0']


def test_get_all_users_pagination(db_session, mocker):
    """Test pagination parameters for retrieving users."""
    mock_users = [MagicMock(spec=User)]
    mock_roles_assocs = []

    # Mock session query and chained calls
    mock_query = mocker.patch.object(db_session, 'query')
    mock_order_query = MagicMock()
    mock_limit_query = MagicMock()
    mock_offset_query = MagicMock()

    mock_query.return_value.order_by.return_value = mock_order_query
    mock_order_query.limit.return_value = mock_limit_query
    mock_limit_query.offset.return_value = mock_offset_query
    mock_offset_query.all.return_value = mock_users

    # Mock roles query (will be called after base query)
    mock_query.return_value.filter.return_value.all.return_value = mock_roles_assocs

    user_db_module.get_all_users(limit=5, offset=10)

    # Assertions on chain calls
    mock_query.assert_any_call(User)
    mock_order_query.limit.assert_called_once_with(5)
    mock_limit_query.offset.assert_called_once_with(10)
    mock_offset_query.all.assert_called_once()


def test_get_all_users_no_users(db_session, mocker):
    """Test retrieving all users when none exist."""
    # Mock session query
    mock_query = mocker.patch.object(db_session, 'query')
    mock_query.return_value.order_by.return_value.limit.return_value.offset.return_value.all.return_value = []

    results = user_db_module.get_all_users()

    assert results == []
    # Assert roles query was not made if no users found
    mock_query.return_value.filter.assert_not_called()
