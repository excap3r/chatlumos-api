import pytest
from unittest.mock import patch, MagicMock, ANY
import uuid
from datetime import datetime, timedelta
import bcrypt
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import redis
import json
import hashlib

# Import the module to test
import services.db.user_db as user_db_module
from services.db.user_db import (
    create_user,
    # Other functions will be imported as needed
)
from services.db.models.user_models import User, UserRoleAssociation
from services.db.exceptions import UserAlreadyExistsError, DuplicateEntryError, QueryError

# --- Mocks and Fixtures ---


@pytest.fixture
def mock_session():
    """Provides a mock SQLAlchemy Session and mocks _get_session in both user_db and db_utils."""
    # Patch _get_session where it's called by the functions under test AND the decorator
    with patch('services.db.user_db._get_session') as mock_get_user_db, \
         patch('services.db.db_utils._get_session') as mock_get_db_utils:
        
        mock_sess = MagicMock(spec=Session)
        # Make both patches return the same mock session
        mock_get_user_db.return_value = mock_sess
        mock_get_db_utils.return_value = mock_sess

        # Configure query chaining
        mock_query = MagicMock()
        mock_sess.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.filter_by.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.first.return_value = None
        mock_query.count.return_value = 0
        mock_query.delete.return_value = 0

        # Mock add to do nothing substantial by default
        mock_sess.add.return_value = None
        mock_sess.flush.return_value = None
        mock_sess.commit.return_value = None
        mock_sess.rollback.return_value = None
        mock_sess.refresh.return_value = None

        yield mock_sess


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


def test_create_user_success_default_role(mock_session, mock_bcrypt):
    """Test successful user creation with the default 'user' role."""
    username = "testuser"
    email = "test@example.com"
    password = "password123"
    hashed_pw_bytes = mock_bcrypt["hashpw"](
        password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    # Mock the state *after* flush assigns an ID and commit refreshes defaults
    mock_user_instance = MagicMock(spec=User)
    mock_user_instance.id = uuid.uuid4()
    mock_user_instance.username = username
    mock_user_instance.email = email
    mock_user_instance.is_active = True
    mock_user_instance.created_at = datetime.utcnow()
    mock_user_instance.updated_at = datetime.utcnow()
    mock_user_instance.last_login = None

    # Capture added objects
    added_objects = []

    def add_side_effect(obj):
        added_objects.append(obj)
        # Simulate ID assignment on flush for User object
        if isinstance(obj, User):
            obj.id = mock_user_instance.id

    mock_session.add.side_effect = add_side_effect

    # Simulate refresh loading defaults
    def refresh_side_effect(obj):
        if isinstance(obj, User):
            obj.is_active = mock_user_instance.is_active
            obj.created_at = mock_user_instance.created_at
            obj.updated_at = mock_user_instance.updated_at
            obj.last_login = mock_user_instance.last_login

    mock_session.refresh.side_effect = refresh_side_effect

    # Mock query for roles after commit/refresh
    mock_session.query.return_value.filter_by.return_value.all.return_value = [
        ("user",)
    ]

    result = create_user(username, email, password)

    # Assertions
    assert len(added_objects) == 2  # User + UserRoleAssociation
    added_user = next(o for o in added_objects if isinstance(o, User))
    added_role_assoc = next(
        o for o in added_objects if isinstance(o, UserRoleAssociation)
    )

    assert added_user.username == username
    assert added_user.email == email
    assert added_user.password_hash == hashed_pw_str
    assert added_role_assoc.user_id == added_user.id
    assert added_role_assoc.role == "user"  # Default role

    mock_session.flush.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once_with(added_user)

    assert result["id"] == str(mock_user_instance.id)
    assert result["username"] == username
    assert result["roles"] == ["user"]
    assert result["created_at"] is not None


def test_create_user_success_specific_roles(mock_session, mock_bcrypt):
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

    added_objects = []

    def add_side_effect(obj):
        added_objects.append(obj)
        if isinstance(obj, User):
            obj.id = mock_user_id

    mock_session.add.side_effect = add_side_effect

    def refresh_side_effect(obj):  # Simulate refresh
        if isinstance(obj, User):
            obj.created_at = datetime.utcnow()

    mock_session.refresh.side_effect = refresh_side_effect

    # Mock query for roles after commit/refresh
    mock_session.query.return_value.filter_by.return_value.all.return_value = [
        ("admin",),
        ("editor",),
    ]

    result = create_user(username, email, password, roles=roles)

    # Assertions
    assert len(added_objects) == 3  # User + 2 UserRoleAssociation
    added_user = next(o for o in added_objects if isinstance(o, User))
    role_assocs = [o for o in added_objects if isinstance(o, UserRoleAssociation)]
    assert len(role_assocs) == 2
    assert {assoc.role for assoc in role_assocs} == set(roles)
    for assoc in role_assocs:
        assert assoc.user_id == mock_user_id

    mock_session.flush.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once_with(added_user)

    assert result["id"] == str(mock_user_id)
    assert result["username"] == username
    assert set(result["roles"]) == set(roles)


def test_create_user_already_exists(mock_session, mock_bcrypt):
    """Test user creation failure due to uniqueness constraint (IntegrityError)."""
    username = "existinguser"
    email = "exists@example.com"
    password = "password"

    # Simulate IntegrityError on commit (or flush)
    # The decorator catches IntegrityError and raises DuplicateEntryError
    mock_session.commit.side_effect = IntegrityError(
        "duplicate key value violates unique constraint \"ix_users_email\"",
        params={}, orig=Exception("DUPLICATE KEY") # Simulate original DB error message 
    )

    with pytest.raises(DuplicateEntryError): # Expect the wrapped error
        create_user(username, email, password)
    
    mock_session.rollback.assert_called_once() # Verify rollback happened in decorator


def test_create_user_other_db_error(mock_session, mock_bcrypt):
    """Test user creation failure due to a generic database error."""
    username = "erroruser"
    email = "error@example.com"
    password = "password"

    # Simulate a generic SQLAlchemyError on commit
    # The decorator catches it and raises QueryError
    mock_session.commit.side_effect = SQLAlchemyError("DB connection lost")

    with pytest.raises(QueryError): # Expect the wrapped error
        create_user(username, email, password)
        
    mock_session.rollback.assert_called_once() # Verify rollback happened in decorator


# --- authenticate_user Tests ---


def test_authenticate_user_success(mock_session, mock_bcrypt):
    """Test successful user authentication."""
    username = "authuser"
    password = "correctpassword"
    user_id = uuid.uuid4()
    # Predictable hashed password based on mock_bcrypt fixture
    hashed_pw_bytes = b"$2b$12$abcdefghijklmnopqrstuvwx:" + password.encode("utf-8")
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    # Mock User object returned by query
    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.username = username
    mock_user.email = "auth@test.com"
    mock_user.password_hash = hashed_pw_str
    mock_user.is_active = True
    mock_user.created_at = datetime(2023, 1, 1)
    mock_user.updated_at = datetime(2023, 1, 1)
    mock_user.last_login = None  # Will be updated

    # Configure session.query(...).first() to return the mock user
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    # Mock query for roles after authentication
    mock_session.query.return_value.filter_by.return_value.all.return_value = [
        ("user",)
    ]

    result = user_db_module.authenticate_user(username, password)

    # Verify bcrypt checkpw was called
    mock_bcrypt["checkpw"].assert_called_once_with(
        password.encode("utf-8"), hashed_pw_bytes
    )

    # Verify last_login was updated (commit called within authenticate_user)
    mock_session.commit.assert_called_once()
    assert mock_user.last_login is not None
    assert isinstance(mock_user.last_login, datetime)

    # Verify the returned dictionary
    assert result["id"] == str(user_id)
    assert result["username"] == username
    assert result["is_active"] is True
    assert result["roles"] == ["user"]


def test_authenticate_user_not_found(mock_session, mock_bcrypt):
    """Test authentication failure when user is not found."""
    username = "nouser"
    password = "password"

    # Configure session.query(...).first() to return None
    mock_session.query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(user_db_module.InvalidCredentialsError):
        user_db_module.authenticate_user(username, password)

    # Ensure checkpw was not called
    mock_bcrypt["checkpw"].assert_not_called()
    # Ensure commit (for last_login) was not called
    mock_session.commit.assert_not_called()


def test_authenticate_user_incorrect_password(mock_session, mock_bcrypt):
    """Test authentication failure due to incorrect password."""
    username = "wrongpassuser"
    password = "wrongpassword"
    correct_password = "correctpassword"
    user_id = uuid.uuid4()
    # Hash the *correct* password
    hashed_pw_bytes = mock_bcrypt["hashpw"](
        correct_password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.username = username
    mock_user.password_hash = hashed_pw_str
    mock_user.is_active = True

    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    # Mock checkpw to return False
    mock_bcrypt["checkpw"].return_value = False

    with pytest.raises(user_db_module.InvalidCredentialsError):
        user_db_module.authenticate_user(username, password)

    # Verify checkpw *was* called
    mock_bcrypt["checkpw"].assert_called_once_with(
        password.encode("utf-8"), hashed_pw_bytes
    )
    # Ensure commit (for last_login) was not called
    mock_session.commit.assert_not_called()


def test_authenticate_user_inactive(mock_session, mock_bcrypt):
    """Test authentication failure for an inactive user."""
    username = "inactiveuser"
    password = "correctpassword"
    user_id = uuid.uuid4()
    hashed_pw_bytes = mock_bcrypt["hashpw"](
        password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.username = username
    mock_user.password_hash = hashed_pw_str
    mock_user.is_active = False  # User is inactive

    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    # Mock checkpw to return True (password is correct)
    mock_bcrypt["checkpw"].return_value = True

    with pytest.raises(
        user_db_module.InvalidCredentialsError, match="User account is not active"
    ):
        user_db_module.authenticate_user(username, password)

    # Verify checkpw *was* called
    mock_bcrypt["checkpw"].assert_called_once_with(
        password.encode("utf-8"), hashed_pw_bytes
    )
    # Ensure commit (for last_login) was not called
    mock_session.commit.assert_not_called()


def test_authenticate_user_last_login_update_fails(mock_session, mock_bcrypt):
    """Test authentication succeeds even if last_login update fails."""
    username = "loginupdatefail"
    password = "correctpassword"
    user_id = uuid.uuid4()
    hashed_pw_bytes = mock_bcrypt["hashpw"](
        password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    hashed_pw_str = hashed_pw_bytes.decode("utf-8")

    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.username = username
    mock_user.password_hash = hashed_pw_str
    mock_user.is_active = True
    mock_user.created_at = datetime(2023, 1, 1)
    mock_user.updated_at = datetime(2023, 1, 1)
    original_last_login = datetime(2023, 1, 10)  # Previous login time
    mock_user.last_login = original_last_login

    mock_session.query.return_value.filter.return_value.first.return_value = mock_user
    mock_bcrypt["checkpw"].return_value = True

    # Simulate failure during the commit for last_login update
    mock_session.commit.side_effect = IntegrityError(
        "DB Error during commit", params={}, orig=None
    )

    # Mock query for roles after authentication attempt
    mock_session.query.return_value.filter_by.return_value.all.return_value = [
        ("tester",)
    ]

    result = user_db_module.authenticate_user(
        username, password
    )  # Should not raise InvalidCredentialsError

    # Verify checkpw was called
    mock_bcrypt["checkpw"].assert_called_once()
    # Verify commit was attempted and rollback occurred (due to the commit side_effect)
    mock_session.commit.assert_called_once()
    mock_session.rollback.assert_called_once()  # Rollback should be called within the function's except block

    # Verify authentication still succeeded and returned user data
    assert result["id"] == str(user_id)
    assert result["username"] == username
    assert result["roles"] == ["tester"]
    # last_login should *not* be updated if commit failed
    # Note: The exact value depends on if the mock_user object was modified in-place before commit failed.
    # Check it's not None and maybe not the original value if modification happened.
    # Re-reading the code: it updates mock_user.last_login *before* commit.
    assert (
        mock_user.last_login != original_last_login
    )  # Modified in-place before failed commit
    # The returned result uses the (potentially modified) mock_user object's state
    assert result["last_login"] == mock_user.last_login


# --- get_user_by_id Tests ---


def test_get_user_by_id_success(mock_session):
    """Test successfully getting a user by ID."""
    user_id = uuid.uuid4()
    username = "getme"
    roles = ["viewer"]

    # Mock User object with roles relationship correctly mocked
    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.username = username
    mock_user.email = "getme@example.com"
    mock_user.is_active = True
    mock_user.created_at = datetime.utcnow()
    mock_user.updated_at = datetime.utcnow()
    mock_user.last_login = None
    # Mock the roles relationship (list of association objects)
    mock_user.roles = [MagicMock(spec=UserRoleAssociation, role=r) for r in roles]

    # Configure session query to return the mock user
    mock_session.query.return_value.options.return_value.filter.return_value.first.return_value = (
        mock_user
    )

    result = user_db_module.get_user_by_id(str(user_id))

    # Verify query was called with options and filter
    mock_session.query.assert_called_once_with(User)
    query_mock = mock_session.query.return_value
    query_mock.options.assert_called_once()  # Check that options() was called
    # Check filter was called with the correct UUID
    query_mock.options.return_value.filter.assert_called_once()
    filter_arg = query_mock.options.return_value.filter.call_args[0][0]
    # This comparison works because SQLAlchemy overloads == for Column comparisons
    # We need to simulate this or check the filter argument structure
    # assert filter_arg == (User.id == user_id) # This won't work directly
    # Check the structure of the filter argument (BinaryExpression)
    assert str(filter_arg) == "users.id = :id_1"  # Check SQL representation
    assert filter_arg.right.value == user_id  # Check the bound value

    # Verify result content
    assert result["id"] == str(user_id)
    assert result["username"] == username
    assert result["roles"] == roles
    assert result["permissions"] == []  # Permissions not handled


def test_get_user_by_id_not_found(mock_session):
    """Test getting a user by ID when the user does not exist."""
    user_id = uuid.uuid4()

    # Configure query to return None
    mock_session.query.return_value.options.return_value.filter.return_value.first.return_value = (
        None
    )

    with pytest.raises(
        user_db_module.UserNotFoundError, match=f"User with ID {str(user_id)} not found"
    ):
        user_db_module.get_user_by_id(str(user_id))


def test_get_user_by_id_invalid_uuid_format(mock_session):
    """Test getting user by ID with an invalid UUID string."""
    invalid_user_id = "not-a-uuid"

    with pytest.raises(
        user_db_module.UserNotFoundError, match="Invalid user ID format"
    ):
        user_db_module.get_user_by_id(invalid_user_id)

    # Ensure query was not even attempted with invalid UUID
    mock_session.query.assert_not_called()


# --- update_user Tests ---


def test_update_user_success(mock_session):
    """Test successfully updating user data."""
    user_id = uuid.uuid4()
    username = "update_me"
    original_email = "original@test.com"
    new_email = "new@test.com"
    new_is_active = False

    # Mock existing User object
    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.username = username
    mock_user.email = original_email
    mock_user.is_active = True
    mock_user.password_hash = "some_hash"  # Not updated here
    mock_user.created_at = datetime(2023, 1, 1)
    mock_user.updated_at = datetime(2023, 1, 1)
    mock_user.last_login = None
    mock_user.roles = [MagicMock(spec=UserRoleAssociation, role="user")]

    # Configure session query to return the mock user
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    update_data = {"email": new_email, "is_active": new_is_active}
    result = user_db_module.update_user(str(user_id), update_data)

    # Verify user attributes were updated *before* commit (remains same)
    assert mock_user.email == new_email
    assert mock_user.is_active == new_is_active
    # Cannot reliably assert updated_at on mock object as it's DB managed
    # assert mock_user.updated_at > datetime(2023, 1, 1) # Removed assertion

    # Verify commit was called
    mock_session.commit.assert_called_once()

    # Verify returned data reflects updates (remains same)
    assert result["id"] == str(user_id)
    assert result["email"] == new_email
    assert result["is_active"] == new_is_active
    assert result["roles"] == ["user"]  # Roles not changed


def test_update_user_only_one_field(mock_session):
    """Test updating only a single user field."""
    user_id = uuid.uuid4()
    new_username = "new_username"

    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.username = "old_username"
    mock_user.email = "email@test.com"
    mock_user.is_active = True
    mock_user.roles = []

    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    update_data = {"username": new_username}
    result = user_db_module.update_user(str(user_id), update_data)

    assert mock_user.username == new_username
    assert mock_user.email == "email@test.com"  # Email should be unchanged
    mock_session.commit.assert_called_once()
    assert result["username"] == new_username


def test_update_user_no_valid_fields(mock_session):
    """Test updating with no valid fields in the data dictionary."""
    user_id = uuid.uuid4()
    original_username = "original_user"

    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.username = original_username
    mock_user.roles = []

    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    # Data contains fields not directly updatable by this function
    update_data = {"password": "newpass", "roles": ["admin"]}
    result = user_db_module.update_user(str(user_id), update_data)

    # User object should remain unchanged
    assert mock_user.username == original_username
    # Commit might still be called if updated_at is always touched, check implementation
    # Assuming updated_at is only touched if other fields change:
    mock_session.commit.assert_not_called()  # Or called once if updated_at changes anyway
    assert result["username"] == original_username


def test_update_user_not_found(mock_session):
    """Test updating a user that does not exist."""
    user_id = uuid.uuid4()

    # Configure query to return None
    mock_session.query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(user_db_module.UserNotFoundError):
        user_db_module.update_user(str(user_id), {"email": "new@test.com"})

    mock_session.commit.assert_not_called()


# --- update_user_roles Tests ---


def test_update_user_roles_add_and_remove(mock_session):
    """Test adding new roles and removing existing ones."""
    user_id = uuid.uuid4()
    existing_roles = ["user", "reader"]
    new_roles_input = ["user", "editor"]  # Add editor, remove reader

    mock_user = MagicMock(spec=User, id=user_id)
    # Mock the query to find the user
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    # Mock the query to get current roles
    mock_current_role_assocs = [MagicMock(role=r) for r in existing_roles]
    mock_session.query.return_value.filter_by.return_value.all.return_value = (
        mock_current_role_assocs
    )

    # Mock the delete operation
    mock_delete_query = MagicMock()
    mock_session.query.return_value.filter.return_value.filter.return_value = (
        mock_delete_query
    )

    added_objects = []
    mock_session.add.side_effect = added_objects.append

    result = user_db_module.update_user_roles(str(user_id), new_roles_input)

    # Verify user was fetched
    mock_session.query.return_value.filter.return_value.first.assert_called_once_with()  # Query User
    # Verify current roles were fetched
    mock_session.query.return_value.filter_by.assert_called_once_with(
        user_id=user_id
    )  # Query UserRoleAssociation

    # Verify delete was called for the role to remove ('reader')
    # This requires checking the filter conditions used before delete
    # The code uses: session.query(UserRoleAssociation).filter(...).filter(...).delete()
    # Let's check the delete call itself on the final query object
    mock_delete_query.delete.assert_called_once_with(synchronize_session=False)
    # Inspect the filters applied before delete - This is complex to mock accurately
    # Easier: check what was *added*

    # Verify add was called for the new role ('editor')
    assert len(added_objects) == 1
    assert isinstance(added_objects[0], UserRoleAssociation)
    assert added_objects[0].user_id == user_id
    assert added_objects[0].role == "editor"

    mock_session.commit.assert_called_once()

    # Verify result (implementation queries roles again after changes)
    # Mock the final query for roles
    mock_session.query.return_value.filter_by.return_value.all.return_value = [
        ("user",),
        ("editor",),
    ]
    result_after_re_query = user_db_module.update_user_roles(
        str(user_id), new_roles_input
    )
    assert set(result_after_re_query["roles"]) == set(new_roles_input)
    assert result_after_re_query["id"] == str(user_id)


def test_update_user_roles_add_only(mock_session):
    """Test adding roles when none are removed."""
    user_id = uuid.uuid4()
    existing_roles = ["user"]
    new_roles_input = ["user", "editor"]

    mock_user = MagicMock(spec=User, id=user_id)
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user
    mock_current_role_assocs = [MagicMock(role=r) for r in existing_roles]
    mock_session.query.return_value.filter_by.return_value.all.return_value = (
        mock_current_role_assocs
    )

    # Mock the delete query path, but expect delete not to be called
    mock_delete_query = MagicMock()
    mock_session.query.return_value.filter.return_value.filter.return_value = (
        mock_delete_query
    )

    added_objects = []
    mock_session.add.side_effect = added_objects.append

    user_db_module.update_user_roles(str(user_id), new_roles_input)

    mock_delete_query.delete.assert_not_called()
    assert len(added_objects) == 1
    assert added_objects[0].role == "editor"
    mock_session.commit.assert_called_once()


def test_update_user_roles_remove_only(mock_session):
    """Test removing roles when none are added."""
    user_id = uuid.uuid4()
    existing_roles = ["user", "editor"]
    new_roles_input = ["user"]

    mock_user = MagicMock(spec=User, id=user_id)
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user
    mock_current_role_assocs = [MagicMock(role=r) for r in existing_roles]
    mock_session.query.return_value.filter_by.return_value.all.return_value = (
        mock_current_role_assocs
    )

    mock_delete_query = MagicMock()
    mock_session.query.return_value.filter.return_value.filter.return_value = (
        mock_delete_query
    )

    added_objects = []
    mock_session.add.side_effect = added_objects.append

    user_db_module.update_user_roles(str(user_id), new_roles_input)

    mock_delete_query.delete.assert_called_once_with(synchronize_session=False)
    assert len(added_objects) == 0
    mock_session.commit.assert_called_once()


def test_update_user_roles_no_change(mock_session):
    """Test updating roles when the input matches existing roles."""
    user_id = uuid.uuid4()
    existing_roles = ["user", "reader"]
    new_roles_input = ["user", "reader"]  # No change

    mock_user = MagicMock(spec=User, id=user_id)
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user
    mock_current_role_assocs = [MagicMock(role=r) for r in existing_roles]
    mock_session.query.return_value.filter_by.return_value.all.return_value = (
        mock_current_role_assocs
    )

    mock_delete_query = MagicMock()
    mock_session.query.return_value.filter.return_value.filter.return_value = (
        mock_delete_query
    )

    added_objects = []
    mock_session.add.side_effect = added_objects.append

    user_db_module.update_user_roles(str(user_id), new_roles_input)

    mock_delete_query.delete.assert_not_called()
    assert len(added_objects) == 0
    # Commit might still be called, depending on implementation details (e.g., touching updated_at)
    # Assuming commit is skipped if no roles added/deleted:
    mock_session.commit.assert_not_called()


def test_update_user_roles_user_not_found(mock_session):
    """Test updating roles for a user that does not exist."""
    user_id = uuid.uuid4()

    # Configure query to return None for the user
    mock_session.query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(user_db_module.UserNotFoundError):
        user_db_module.update_user_roles(str(user_id), ["admin"])

    mock_session.commit.assert_not_called()


# --- update_user_password Tests ---


def test_update_user_password_success(mock_session, mock_bcrypt):
    """Test successfully updating a user's password."""
    user_id = uuid.uuid4()
    new_password = "newSecurePassword!"
    # Predictable new hash
    new_hashed_pw_bytes = mock_bcrypt["hashpw"](
        new_password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    new_hashed_pw_str = new_hashed_pw_bytes.decode("utf-8")

    # Mock existing User object
    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.password_hash = "old_hash"

    # Configure session query to return the mock user
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    result = user_db_module.update_user_password(str(user_id), new_password)

    # Verify user was fetched
    mock_session.query.return_value.filter.return_value.first.assert_called_once()

    # Verify password hash was updated on the mock user object
    assert mock_user.password_hash == new_hashed_pw_str
    assert mock_user.updated_at is not None  # Should be updated

    # Verify commit was called
    mock_session.commit.assert_called_once()

    # Verify result
    assert result is True


def test_update_user_password_user_not_found(mock_session, mock_bcrypt):
    """Test updating password for a non-existent user."""
    user_id = uuid.uuid4()
    new_password = "newpassword"

    # Configure query to return None
    mock_session.query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(user_db_module.UserNotFoundError):
        user_db_module.update_user_password(str(user_id), new_password)

    # Verify bcrypt hashpw was not called
    mock_bcrypt["hashpw"].assert_not_called()
    # Verify commit was not called
    mock_session.commit.assert_not_called()


def test_update_user_password_db_error(mock_session, mock_bcrypt):
    """Test handling database error during password update commit."""
    user_id = uuid.uuid4()
    new_password = "newSecurePassword!"

    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.password_hash = "old_hash"

    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    # Simulate error on commit
    # The decorator will catch IntegrityError and raise QueryError (or specific subclass if matched)
    mock_session.commit.side_effect = IntegrityError("Commit failed", params={}, orig=None)

    with pytest.raises(QueryError): # Expect wrapped QueryError
        user_db_module.update_user_password(str(user_id), new_password)

    # Verify password hash was updated *before* the failed commit (remains same)
    new_hashed_pw_bytes = mock_bcrypt["hashpw"](
        new_password.encode("utf-8"), mock_bcrypt["gensalt"]()
    )
    assert mock_user.password_hash == new_hashed_pw_bytes.decode("utf-8")
    # Verify rollback was called by the decorator
    mock_session.rollback.assert_called_once()


# --- API Key Tests ---


# Need to mock auth_utils.hash_api_key and potentially generate_api_key if not mocked elsewhere
@patch("services.utils.auth_utils.hash_api_key")
@patch("services.db.user_db.uuid.uuid4")  # Mock UUID generation for predictable key_id
def test_create_api_key_success(mock_uuid, mock_hash_api_key, mock_session):
    """Test successful API key creation."""
    user_id_uuid = uuid.uuid4()
    user_id_str = str(user_id_uuid)
    key_name = "My Test Key"
    key_prefix = (
        "testprefix"  # Assuming generate_api_key might be involved or prefix is static
    )
    generated_key_suffix = "_generatedpart123"  # Simulate generated part
    generated_api_key = f"{key_prefix}{generated_key_suffix}"
    hashed_key = "hashed_" + generated_api_key  # Predictable hash for testing
    mock_hash_api_key.return_value = hashed_key
    # Mock UUID for the APIKey primary key
    api_key_id_uuid = uuid.uuid4()
    mock_uuid.return_value = api_key_id_uuid

    # Mock the user object found by the session
    mock_user = MagicMock(spec=User, id=user_id_uuid)
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    # Capture the added APIKey object
    added_key_obj = None

    def add_side_effect(obj):
        nonlocal added_key_obj
        if isinstance(obj, user_db_module.APIKey):
            added_key_obj = obj
            # Simulate setting defaults if necessary (although defaults are in model)
            obj.id = api_key_id_uuid
            obj.created_at = datetime.utcnow()
            obj.updated_at = datetime.utcnow()

    mock_session.add.side_effect = add_side_effect

    # Call the function
    # Note: create_api_key generates the key internally, we don't provide it
    result = user_db_module.create_api_key(user_id_str, key_name)

    # Verify user was fetched
    mock_session.query.return_value.filter.return_value.first.assert_called_once()

    # Verify hash_api_key was called (on the internally generated key)
    # We need to capture the argument passed to hash_api_key
    assert mock_hash_api_key.call_count == 1
    actual_key_generated = mock_hash_api_key.call_args[0][0]
    assert actual_key_generated.startswith("sk_")  # Check default prefix

    # Verify the APIKey object added to session
    assert added_key_obj is not None
    assert added_key_obj.id == api_key_id_uuid
    assert added_key_obj.user_id == user_id_uuid
    assert added_key_obj.name == key_name
    assert added_key_obj.key_hash == hashed_key
    assert added_key_obj.prefix == actual_key_generated[:3]  # Default prefix sk_
    assert added_key_obj.expires_at is None
    assert added_key_obj.last_used is None
    assert added_key_obj.is_active is True

    # Verify commit was called
    mock_session.commit.assert_called_once()

    # Verify the returned dictionary
    assert result["key_id"] == str(api_key_id_uuid)
    # The result should contain the *unhashed* generated key
    assert result["api_key"] == actual_key_generated
    assert result["name"] == key_name
    assert result["user_id"] == user_id_str
    assert result["prefix"] == actual_key_generated[:3]
    assert result["created_at"] is not None
    assert result["expires_at"] is None


@patch("services.utils.auth_utils.hash_api_key")
def test_create_api_key_user_not_found(mock_hash_api_key, mock_session):
    """Test API key creation fails if the user is not found."""
    user_id_str = str(uuid.uuid4())
    key_name = "Test Key Fail"

    # Configure session query to return None for the user
    mock_session.query.return_value.filter.return_value.first.return_value = None

    with pytest.raises(user_db_module.UserNotFoundError):
        user_db_module.create_api_key(user_id_str, key_name)

    mock_hash_api_key.assert_not_called()
    mock_session.add.assert_not_called()
    mock_session.commit.assert_not_called()


@patch("services.utils.auth_utils.hash_api_key")
def test_create_api_key_db_error(mock_hash_api_key, mock_session):
    """Test handling DB error during API key creation commit."""
    user_id_uuid = uuid.uuid4()
    user_id_str = str(user_id_uuid)
    key_name = "Test Key DB Error"

    mock_user = MagicMock(spec=User, id=user_id_uuid)
    mock_session.query.return_value.filter.return_value.first.return_value = mock_user

    mock_hash_api_key.return_value = "some_hash"

    # Simulate error on commit
    mock_session.commit.side_effect = IntegrityError(
        "Commit failed", params={}, orig=None
    )

    with pytest.raises(IntegrityError):  # Assume decorator re-raises or original error
        user_db_module.create_api_key(user_id_str, key_name)

    # Verify add was called, but commit failed
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    # Rollback handled by decorator


@patch("services.utils.auth_utils.verify_api_key_hash")
def test_verify_api_key_success_cache_miss(mock_verify_hash, mock_session, mock_redis):
    """Test successful API key verification with cache miss."""
    key_prefix = "sk"
    key_id_str = str(uuid.uuid4())
    api_key_str = f"{key_prefix}_{key_id_str}"
    stored_hash = "correct_hash_for_key"
    user_id_uuid = uuid.uuid4()
    cache_key = f"{user_db_module.API_KEY_CACHE_PREFIX}{hashlib.sha256(key_id_str.encode()).hexdigest()}"  # Key used in cache
    hashed_key_id_str = hashlib.sha256(key_id_str.encode()).hexdigest()

    # Mock APIKey object found in DB
    mock_api_key_obj = MagicMock(spec=user_db_module.APIKey)
    mock_api_key_obj.id = uuid.UUID(key_id_str)
    mock_api_key_obj.key_hash = stored_hash
    mock_api_key_obj.is_active = True
    mock_api_key_obj.expires_at = None
    mock_api_key_obj.user_id = user_id_uuid
    mock_api_key_obj.user = MagicMock(
        spec=User, id=user_id_uuid, is_active=True
    )  # Mock associated user

    # Configure session query to find the key
    mock_session.query.return_value.options.return_value.filter.return_value.first.return_value = (
        mock_api_key_obj
    )

    # Mock verify_api_key_hash to return True
    mock_verify_hash.return_value = True

    # Mock redis get to simulate cache miss
    mock_redis.get.return_value = None

    result = user_db_module.verify_api_key(api_key_str)

    # Verify cache check
    mock_redis.get.assert_called_once_with(cache_key)

    # Verify DB query was made
    mock_session.query.return_value.options.return_value.filter.return_value.first.assert_called_once()

    # Verify hash verification
    mock_verify_hash.assert_called_once_with(api_key_str, stored_hash)

    # Verify result is correct
    assert result is not None
    assert result["key_id"] == key_id_str
    assert result["user_id"] == str(user_id_uuid)
    assert result["is_active"] is True

    # Verify cache was updated with valid result
    expected_cache_data = json.dumps(
        {
            "key_hash": stored_hash,
            "user_id": str(user_id_uuid),
            "is_active": True,
            "user_is_active": True,
            "expires_at": None,
        }
    )
    mock_redis.setex.assert_called_once_with(
        cache_key, user_db_module.API_KEY_CACHE_TTL, expected_cache_data
    )

    # Verify last_used update commit
    mock_session.commit.assert_called_once()
    assert mock_api_key_obj.last_used is not None


@patch("services.utils.auth_utils.verify_api_key_hash")
def test_verify_api_key_success_cache_hit_valid(
    mock_verify_hash, mock_session, mock_redis
):
    """Test successful API key verification with valid cache hit."""
    key_prefix = "sk"
    key_id_str = str(uuid.uuid4())
    api_key_str = f"{key_prefix}_{key_id_str}"
    stored_hash = "correct_hash_cached"
    user_id_str = str(uuid.uuid4())
    hashed_key_id_str = hashlib.sha256(key_id_str.encode()).hexdigest()
    cache_key = f"{user_db_module.API_KEY_CACHE_PREFIX}{hashed_key_id_str}"
    # Simulate cached data
    cached_data = {
        "key_hash": stored_hash,
        "user_id": user_id_str,
        "is_active": True,
        "user_is_active": True,
        "expires_at": None,
    }
    mock_redis.get.return_value = json.dumps(cached_data).encode("utf-8")

    # Mock verify_api_key_hash to return True
    mock_verify_hash.return_value = True

    result = user_db_module.verify_api_key(api_key_str)

    # Verify cache check
    mock_redis.get.assert_called_once_with(cache_key)

    # Verify DB query was NOT made
    mock_session.query.assert_not_called()

    # Verify hash verification (using cached hash)
    mock_verify_hash.assert_called_once_with(api_key_str, stored_hash)

    # Verify result
    assert result is not None
    assert result["key_id"] == key_id_str
    assert result["user_id"] == user_id_str

    # Verify cache was NOT updated again
    mock_redis.setex.assert_not_called()

    # Verify last_used was NOT updated (no DB object)
    mock_session.commit.assert_not_called()


@patch("services.utils.auth_utils.verify_api_key_hash")
def test_verify_api_key_cache_hit_invalid_marker(
    mock_verify_hash, mock_session, mock_redis
):
    """Test API key verification fails quickly with invalid cache marker hit."""
    key_prefix = "sk"
    key_id_str = str(uuid.uuid4())
    api_key_str = f"{key_prefix}_{key_id_str}"
    hashed_key_id_str = hashlib.sha256(key_id_str.encode()).hexdigest()
    cache_key = f"{user_db_module.API_KEY_CACHE_PREFIX}{hashed_key_id_str}"

    # Simulate invalid marker in cache
    mock_redis.get.return_value = user_db_module.API_KEY_CACHE_INVALID_MARKER.encode(
        "utf-8"
    )

    result = user_db_module.verify_api_key(api_key_str)

    # Verify cache check
    mock_redis.get.assert_called_once_with(cache_key)

    # Verify DB query, hash check etc. were NOT done
    mock_session.query.assert_not_called()
    mock_verify_hash.assert_not_called()
    mock_redis.setex.assert_not_called()

    # Verify result is None (failure)
    assert result is None


@patch("services.utils.auth_utils.verify_api_key_hash")
def test_verify_api_key_fail_hash_mismatch_cache_miss(
    mock_verify_hash, mock_session, mock_redis
):
    """Test verification failure on hash mismatch (cache miss scenario). Cache invalid marker."""
    key_prefix = "sk"
    key_id_str = str(uuid.uuid4())
    api_key_str = f"{key_prefix}_{key_id_str}"
    stored_hash = "correct_hash"
    hashed_key_id_str = hashlib.sha256(key_id_str.encode()).hexdigest()
    cache_key = f"{user_db_module.API_KEY_CACHE_PREFIX}{hashed_key_id_str}"

    mock_api_key_obj = MagicMock(spec=user_db_module.APIKey)
    mock_api_key_obj.id = uuid.UUID(key_id_str)
    mock_api_key_obj.key_hash = stored_hash
    mock_api_key_obj.is_active = True
    mock_api_key_obj.expires_at = None
    mock_api_key_obj.user = MagicMock(spec=User, is_active=True)

    mock_session.query.return_value.options.return_value.filter.return_value.first.return_value = (
        mock_api_key_obj
    )
    mock_redis.get.return_value = None  # Cache miss

    # Mock verify_api_key_hash to return False (hash mismatch)
    mock_verify_hash.return_value = False

    result = user_db_module.verify_api_key(api_key_str)

    assert result is None
    mock_redis.get.assert_called_once_with(cache_key)
    mock_session.query.assert_called_once()
    mock_verify_hash.assert_called_once_with(api_key_str, stored_hash)
    # Verify cache was updated with INVALID marker
    mock_redis.setex.assert_called_once_with(
        cache_key,
        user_db_module.API_KEY_CACHE_TTL,
        user_db_module.API_KEY_CACHE_INVALID_MARKER,
    )
    mock_session.commit.assert_not_called()  # No last_used update on failure


def test_verify_api_key_fail_key_not_found_in_db(mock_session, mock_redis):
    """Test verification failure when key is not found in DB (cache miss). Cache invalid marker."""
    key_prefix = "sk"
    key_id_str = str(uuid.uuid4())
    api_key_str = f"{key_prefix}_{key_id_str}"
    hashed_key_id_str = hashlib.sha256(key_id_str.encode()).hexdigest()
    cache_key = f"{user_db_module.API_KEY_CACHE_PREFIX}{hashed_key_id_str}"

    # Configure query to return None
    mock_session.query.return_value.options.return_value.filter.return_value.first.return_value = (
        None
    )
    mock_redis.get.return_value = None  # Cache miss

    result = user_db_module.verify_api_key(api_key_str)

    assert result is None
    mock_redis.get.assert_called_once_with(cache_key)
    mock_session.query.assert_called_once()
    # Verify cache was updated with INVALID marker
    mock_redis.setex.assert_called_once_with(
        cache_key,
        user_db_module.API_KEY_CACHE_TTL,
        user_db_module.API_KEY_CACHE_INVALID_MARKER,
    )
    mock_session.commit.assert_not_called()


@patch("services.utils.auth_utils.verify_api_key_hash")
def test_verify_api_key_fail_key_inactive(mock_verify_hash, mock_session, mock_redis):
    """Test verification failure for an inactive API key. Cache invalid marker."""
    key_prefix = "sk"
    key_id_str = str(uuid.uuid4())
    api_key_str = f"{key_prefix}_{key_id_str}"
    stored_hash = "inactive_hash"
    hashed_key_id_str = hashlib.sha256(key_id_str.encode()).hexdigest()
    cache_key = f"{user_db_module.API_KEY_CACHE_PREFIX}{hashed_key_id_str}"

    mock_api_key_obj = MagicMock(spec=user_db_module.APIKey)
    mock_api_key_obj.id = uuid.UUID(key_id_str)
    mock_api_key_obj.key_hash = stored_hash
    mock_api_key_obj.is_active = False  # Key is inactive
    mock_api_key_obj.expires_at = None
    mock_api_key_obj.user = MagicMock(spec=User, is_active=True)

    mock_session.query.return_value.options.return_value.filter.return_value.first.return_value = (
        mock_api_key_obj
    )
    mock_redis.get.return_value = None
    mock_verify_hash.return_value = True  # Assume hash matches

    result = user_db_module.verify_api_key(api_key_str)

    assert result is None
    mock_redis.get.assert_called_once_with(cache_key)
    mock_session.query.assert_called_once()
    mock_verify_hash.assert_called_once()  # Hash check happens before active check
    mock_redis.setex.assert_called_once_with(
        cache_key,
        user_db_module.API_KEY_CACHE_TTL,
        user_db_module.API_KEY_CACHE_INVALID_MARKER,
    )
    mock_session.commit.assert_not_called()


# --- get_user_api_keys Tests ---


def test_get_user_api_keys_success(mock_session):
    """Test retrieving API keys for a specific user."""
    user_id_uuid = uuid.uuid4()
    user_id_str = str(user_id_uuid)

    # Mock APIKey objects returned by the query
    mock_key_1 = MagicMock(spec=user_db_module.APIKey)
    mock_key_1.id = uuid.uuid4()
    mock_key_1.name = "Key One"
    mock_key_1.prefix = "sk"
    mock_key_1.created_at = datetime.utcnow()
    mock_key_1.expires_at = None
    mock_key_1.last_used = None
    mock_key_1.is_active = True

    mock_key_2 = MagicMock(spec=user_db_module.APIKey)
    mock_key_2.id = uuid.uuid4()
    mock_key_2.name = "Key Two - Inactive"
    mock_key_2.prefix = "tk"
    mock_key_2.created_at = datetime.utcnow()
    mock_key_2.expires_at = datetime.utcnow() + timedelta(days=30)
    mock_key_2.last_used = datetime.utcnow()
    mock_key_2.is_active = False

    mock_keys = [mock_key_1, mock_key_2]

    # Configure session query
    mock_session.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = (
        mock_keys
    )

    results = user_db_module.get_user_api_keys(user_id_str)

    # Verify query
    mock_session.query.assert_called_once_with(user_db_module.APIKey)
    query_mock = mock_session.query.return_value
    query_mock.filter_by.assert_called_once_with(user_id=user_id_uuid)
    query_mock.filter_by.return_value.order_by.assert_called_once()
    query_mock.filter_by.return_value.order_by.return_value.all.assert_called_once()

    # Verify results
    assert len(results) == 2
    assert results[0]["key_id"] == str(mock_key_1.id)
    assert results[0]["name"] == mock_key_1.name
    assert results[0]["prefix"] == mock_key_1.prefix
    assert results[0]["is_active"] is True
    assert results[1]["key_id"] == str(mock_key_2.id)
    assert results[1]["name"] == mock_key_2.name
    assert results[1]["prefix"] == mock_key_2.prefix
    assert results[1]["is_active"] is False
    assert results[1]["expires_at"] == mock_key_2.expires_at
    assert results[1]["last_used"] == mock_key_2.last_used


def test_get_user_api_keys_none_found(mock_session):
    """Test retrieving API keys when the user has none."""
    user_id_uuid = uuid.uuid4()
    user_id_str = str(user_id_uuid)

    # Configure query to return empty list
    mock_session.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = (
        []
    )

    results = user_db_module.get_user_api_keys(user_id_str)

    # Verify query was made
    mock_session.query.return_value.filter_by.assert_called_once_with(
        user_id=user_id_uuid
    )
    # Verify result is empty list
    assert results == []


def test_get_user_api_keys_invalid_user_id_format(mock_session):
    """Test get_user_api_keys with invalid UUID format."""
    invalid_user_id = "not-a-valid-uuid"

    # The function currently converts to UUID internally and might raise ValueError
    # or potentially DatabaseError if the query fails due to type mismatch.
    # Let's test for ValueError during UUID conversion.
    with pytest.raises(ValueError):
        user_db_module.get_user_api_keys(invalid_user_id)

    # Ensure query was not made with invalid UUID
    mock_session.query.assert_not_called()


# TODO: Add tests for expired keys, invalid key format, user inactive, Redis errors
