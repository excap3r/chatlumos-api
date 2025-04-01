from datetime import datetime
from typing import List, Optional

from sqlalchemy import (Boolean, Column, DateTime, ForeignKey, Integer, String,
                        Text, func, Enum, UniqueConstraint, Index, BigInteger)
# Import MySQL specific types
from sqlalchemy.dialects.mysql import LONGTEXT, TIMESTAMP, ENUM
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .base import Base # Import Base from the common base file

# Define ENUM types using mysql.ENUM
status_enum_doc = ENUM('processing', 'completed', 'failed', name='document_status')
status_enum_chunk = ENUM('pending', 'processing', 'completed', 'failed', name='chunk_status')

class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    author: Mapped[Optional[str]] = mapped_column(String(255))
    processed_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, server_default=func.now())
    full_text: Mapped[Optional[str]] = mapped_column(LONGTEXT)
    total_chunks: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[Optional[str]] = mapped_column(status_enum_doc, server_default='processing')
    file_path: Mapped[Optional[str]] = mapped_column(String(512))

    # Relationships
    chunks: Mapped[List["DocumentChunk"]] = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    concepts: Mapped[List["Concept"]] = relationship("Concept", back_populates="document", cascade="all, delete-orphan")
    qa_pairs: Mapped[List["QAPair"]] = relationship("QAPair", back_populates="document", cascade="all, delete-orphan")
    summary: Mapped[Optional["Summary"]] = relationship("Summary", back_populates="document", uselist=False, cascade="all, delete-orphan")

    # Adjust index to match DB (unique=True)
    __table_args__ = (Index('idx_filename', 'filename', unique=True),)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.document_id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    processed_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, server_default=func.now())
    status: Mapped[Optional[str]] = mapped_column(status_enum_chunk, server_default='pending')

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    concepts: Mapped[List["Concept"]] = relationship("Concept", back_populates="chunk", cascade="all, delete-orphan")
    qa_pairs: Mapped[List["QAPair"]] = relationship("QAPair", back_populates="chunk", cascade="all, delete-orphan")

    # Adjust index to match DB (unique=True) and remove added UniqueConstraint
    __table_args__ = (Index('idx_doc_chunk', 'document_id', 'chunk_index', unique=True),)

class Concept(Base):
    __tablename__ = "concepts"

    concept_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.document_id", ondelete="CASCADE"))
    chunk_id: Mapped[int] = mapped_column(Integer, ForeignKey("document_chunks.chunk_id", ondelete="CASCADE"))
    concept_name: Mapped[str] = mapped_column(String(255), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="concepts")
    chunk: Mapped["DocumentChunk"] = relationship("DocumentChunk", back_populates="concepts")

    # Adjust index to match DB (unique=True)
    __table_args__ = (Index('idx_doc_concept', 'document_id', 'concept_name', unique=True),)

class QAPair(Base):
    __tablename__ = "qa_pairs"
    qa_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.document_id", ondelete="CASCADE"))
    chunk_id: Mapped[int] = mapped_column(Integer, ForeignKey("document_chunks.chunk_id", ondelete="CASCADE"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="qa_pairs")
    chunk: Mapped["DocumentChunk"] = relationship("DocumentChunk", back_populates="qa_pairs")

class Summary(Base):
    __tablename__ = "summaries"

    summary_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.document_id", ondelete="CASCADE"))
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    generated_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, server_default=func.now())

    # Relationship
    document: Mapped["Document"] = relationship("Document", back_populates="summary") 