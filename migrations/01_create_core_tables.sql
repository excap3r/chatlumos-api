-- Migrations for Core Document Processing Tables
-- Order: 01 (Must run before auth/analytics if they reference documents)

-- Ensure database exists (optional, depends on setup process)
-- CREATE DATABASE IF NOT EXISTS your_database_name CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- USE your_database_name;

-- Table: documents
-- Stores information about uploaded/processed documents.
CREATE TABLE IF NOT EXISTS documents (
    document_id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique identifier for the document',
    filename VARCHAR(255) NOT NULL UNIQUE COMMENT 'Original filename, used as a unique identifier for upserting',
    title VARCHAR(255) NULL COMMENT 'Document title (optional)',
    author VARCHAR(255) NULL COMMENT 'Document author (optional)',
    file_path VARCHAR(1024) NULL COMMENT 'Path where the original file is stored (optional)',
    status VARCHAR(50) NOT NULL DEFAULT 'pending' COMMENT 'Processing status (e.g., pending, processing, completed, failed)',
    total_chunks INT NULL COMMENT 'Total number of chunks generated for this document',
    full_text MEDIUMTEXT NULL COMMENT 'Extracted full text of the document (optional)',
    processed_date TIMESTAMP NULL COMMENT 'Timestamp when processing was last updated or completed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the record was created',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Timestamp when the record was last updated',
    INDEX idx_doc_status (status),
    INDEX idx_doc_processed_date (processed_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores metadata about processed documents';

-- Table: document_chunks
-- Stores individual text chunks extracted from documents.
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique identifier for the chunk',
    document_id INT NOT NULL COMMENT 'Foreign key referencing the documents table',
    chunk_index INT NOT NULL COMMENT 'Sequential index of the chunk within the document',
    chunk_text MEDIUMTEXT NOT NULL COMMENT 'The text content of the chunk',
    status VARCHAR(50) NOT NULL DEFAULT 'pending' COMMENT 'Processing status for this specific chunk (e.g., pending, processed, failed)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the record was created',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Timestamp when the record was last updated',
    UNIQUE KEY uq_doc_chunk (document_id, chunk_index) COMMENT 'Ensure chunk indices are unique per document',
    INDEX idx_chunk_status (status),
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE COMMENT 'Link to the parent document, cascade delete chunks if document is deleted'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores text chunks extracted from documents';

-- Table: concepts
-- Stores key concepts extracted from document chunks.
CREATE TABLE IF NOT EXISTS concepts (
    concept_id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique identifier for the concept',
    document_id INT NOT NULL COMMENT 'Foreign key referencing the documents table',
    chunk_id INT NOT NULL COMMENT 'Foreign key referencing the document_chunks table',
    concept_name VARCHAR(255) NOT NULL COMMENT 'The name or title of the concept',
    explanation TEXT NOT NULL COMMENT 'Explanation or definition of the concept',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the record was created',
    INDEX idx_concept_doc_chunk (document_id, chunk_id) COMMENT 'Index for looking up concepts by document and chunk',
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE COMMENT 'Link to the parent document',
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(chunk_id) ON DELETE CASCADE COMMENT 'Link to the parent chunk'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores key concepts extracted from document chunks';

-- Table: qa_pairs
-- Stores question-answer pairs generated from document chunks.
CREATE TABLE IF NOT EXISTS qa_pairs (
    qa_id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique identifier for the QA pair',
    document_id INT NOT NULL COMMENT 'Foreign key referencing the documents table',
    chunk_id INT NOT NULL COMMENT 'Foreign key referencing the document_chunks table',
    question TEXT NOT NULL COMMENT 'The generated question',
    answer TEXT NOT NULL COMMENT 'The generated answer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the record was created',
    INDEX idx_qa_doc_chunk (document_id, chunk_id) COMMENT 'Index for looking up QA pairs by document and chunk',
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE COMMENT 'Link to the parent document',
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(chunk_id) ON DELETE CASCADE COMMENT 'Link to the parent chunk'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores question-answer pairs generated from chunks';

-- Table: summaries
-- Stores generated summaries for documents.
CREATE TABLE IF NOT EXISTS summaries (
    summary_id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique identifier for the summary',
    document_id INT NOT NULL COMMENT 'Foreign key referencing the documents table',
    summary_text LONGTEXT NOT NULL COMMENT 'The text content of the summary',
    generated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the summary was generated',
    INDEX idx_summary_doc_date (document_id, generated_date) COMMENT 'Index for fetching the latest summary for a document',
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE COMMENT 'Link to the parent document'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores generated summaries for documents'; 