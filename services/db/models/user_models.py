import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (Boolean, Column, DateTime, ForeignKey, String, Table,
                        func, Uuid)
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base

# Association table for User Roles -> REMOVED, defined via class below
# user_roles_table = Table(...)

# Association table for User Permissions -> REMOVED, defined via class below
# user_permissions_table = Table(...)

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    # Use the association *class* name now
    roles: Mapped[List["UserRoleAssociation"]] = relationship(
        "UserRoleAssociation", 
        back_populates="user",
        cascade="all, delete-orphan"
        # Removed secondary=, viewonly=True
    )
    permissions: Mapped[List["UserPermissionAssociation"]] = relationship(
        "UserPermissionAssociation", 
        back_populates="user",
        cascade="all, delete-orphan"
        # Removed secondary=, viewonly=True
    )
    api_keys: Mapped[List["APIKey"]] = relationship("APIKey", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


# Keep the Association Classes
class UserRoleAssociation(Base):
    __tablename__ = 'user_roles' # This defines the table name
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey('users.id'), primary_key=True)
    role: Mapped[str] = mapped_column(String(50), primary_key=True)
    # Example: granted_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Add back-populates to link back to User
    user: Mapped["User"] = relationship("User", back_populates="roles")

class UserPermissionAssociation(Base):
    __tablename__ = 'user_permissions' # This defines the table name
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey('users.id'), primary_key=True)
    permission: Mapped[str] = mapped_column(String(100), primary_key=True)

    # Add back-populates to link back to User
    user: Mapped["User"] = relationship("User", back_populates="permissions")


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationship
    user: Mapped["User"] = relationship("User", back_populates="api_keys")

    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>" 