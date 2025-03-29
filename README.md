# PDF Wisdom Extractor and Vector Database Tool

This project consists of three main components:

1. **PDF Wisdom Extractor** - Extracts concepts and Q&A pairs from PDF lecture transcripts
2. **MySQL to Vector Database Converter** - Converts extracted data to vector embeddings for AI-powered search
3. **Wisdom QA System** - Command-line interface for asking questions to the vector database

## Installation

### Prerequisites

- Python 3.8+
- MySQL database
- [Pinecone](https://www.pinecone.io/) account for vector database

### Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.template` to `.env` and fill in your API keys and database configuration:

```bash
cp .env.template .env
# Edit .env with your credentials
```

## PDF Wisdom Extractor Usage

The PDF Wisdom Extractor processes PDF lecture transcripts, extracts key concepts and Q&A pairs, and stores them in a MySQL database.

### Basic Usage

```bash
python pdf_wisdom_extractor.py --pdf_path path/to/lecture.pdf
```

### Options

- `--pdf_path`: Path to the PDF file to process
- `--chunk_size`: Number of words per chunk (default: 1000)
- `--chunk_overlap`: Number of overlapping words between chunks (default: 200)
- `--batch_size`: Number of chunks to process in parallel (default: 3)
- `--author`: Document author (default: "Iva Adamcová")
- `--title`: Document title (default: filename)
- `--translate_to_english`: Translate concepts and Q&A pairs to English
- `--list_documents`: List all documents in the database
- `--document_id ID`: Specify a document ID for operations
- `--export_data`: Export data to JSON files
- `--create_summary`: Create a summary document
- `--create_visualization`: Create a concept visualization

### Examples

Process a PDF with translation to English:
```bash
python pdf_wisdom_extractor.py --pdf_path lectures/wisdom.pdf --translate_to_english
```

List processed documents:
```bash
python pdf_wisdom_extractor.py --list_documents
```

Create a summary document:
```bash
python pdf_wisdom_extractor.py --document_id 1 --create_summary
```

## MySQL to Vector Database Converter

This tool converts concepts and Q&A pairs from the MySQL database to vector embeddings stored in Pinecone for efficient semantic search.

### Basic Usage

```bash
python mysql_to_vector.py
```

### Options

- `--document_id ID`: Process only a specific document
- `--model`: Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)
- `--index_name`: Name for the Pinecone index (default: wisdom-embeddings)
- `--list_documents`: List all documents in the database

### Examples

Convert all documents to vectors:
```bash
python mysql_to_vector.py
```

Process a specific document:
```bash
python mysql_to_vector.py --document_id 1
```

List documents:
```bash
python mysql_to_vector.py --list_documents
```

## Wisdom QA System

The Wisdom QA system provides a command-line interface for asking questions against your vector database. It uses DeepSeek to break down complex questions into sub-questions and concepts, then searches for relevant information in Pinecone.

### Basic Usage

Run the script with a question:
```bash
python wisdom_qa.py --question "What is the nature of reality according to Iva Adamcová?"
```

Or use interactive mode for continuous questioning:
```bash
python wisdom_qa.py --interactive
```

### Options

- `--question`: The question to ask
- `--model`: Embedding model to use (default: all-MiniLM-L6-v2)
- `--index`: Pinecone index name (default: wisdom-embeddings)
- `--top_k`: Number of results to retrieve per query (default: 10)
- `--interactive`: Run in interactive mode

### Examples

Ask a specific question:
```bash
python wisdom_qa.py --question "What is the law of attraction?"
```

Use a specific index and more results:
```bash
python wisdom_qa.py --question "How can I achieve inner peace?" --index my-custom-index --top_k 20
```

Run in interactive mode:
```bash
python wisdom_qa.py --interactive
```

## Complete Workflow

For a complete extraction-to-search workflow:

1. Extract wisdom from PDF:
```bash
python pdf_wisdom_extractor.py --pdf_path lectures/wisdom.pdf --translate_to_english
```

2. Convert to vector database:
```bash
python mysql_to_vector.py
```

3. Ask questions using the QA system:
```bash
python wisdom_qa.py --interactive
```

## API Authentication

The PDF Wisdom Extractor API now includes a complete authentication system for secure access to its endpoints. This system supports:

- User registration and login
- JWT-based authentication
- API key management 
- Role-based access control
- Permission-based authorization

### Authentication Endpoints

The API provides the following authentication endpoints:

- `POST /api/v1/auth/register` - Register a new user
- `POST /api/v1/auth/login` - Authenticate and get access tokens
- `POST /api/v1/auth/refresh` - Refresh expired access tokens
- `POST /api/v1/auth/api-keys` - Create a new API key
- `GET /api/v1/auth/api-keys` - List all API keys for the current user
- `DELETE /api/v1/auth/api-keys/{key_id}` - Revoke an API key
- `GET /api/v1/auth/users` - List all users (admin only)
- `PUT /api/v1/auth/users/{user_id}/roles` - Update user roles (admin only)

### Authentication Methods

The API supports two authentication methods:

1. **JWT tokens** - For interactive sessions
   - Obtained via login endpoint
   - Must be included in `Authorization` header as `Bearer <token>`
   - Access tokens expire after 1 hour by default
   - Refresh tokens can be used to get new access tokens

2. **API keys** - For programmatic access
   - Created via the API key endpoint
   - Can be included in request header as `X-API-Key` or `Authorization: ApiKey <key>`
   - Can be included as a query parameter `?api_key=<key>`
   - No expiration by default (can be revoked)

### User Roles and Permissions

The authentication system supports role-based access control:

- **User** - Basic access to the API
- **Admin** - Full access to all endpoints
- **Custom roles** - Can be defined and assigned by admins

### Environment Configuration

Authentication-related environment variables:

```
# Authentication settings
JWT_SECRET=<secure-random-string>
JWT_ACCESS_TOKEN_EXPIRE_SECONDS=3600
JWT_REFRESH_TOKEN_EXPIRE_SECONDS=2592000

# Database configuration for user storage
DB_HOST=localhost
DB_PORT=3306
DB_USER=wisdom_user
DB_PASSWORD=wisdom_password
DB_NAME=wisdom_db
DB_POOL_SIZE=10
```

### Setup with Docker

The authentication system is fully integrated with the Docker setup:

```bash
# Generate a secure JWT secret
export JWT_SECRET=$(openssl rand -hex 32)

# Start the services with Docker Compose
docker-compose up -d
```

### Authentication Example

```bash
# Register a new user
curl -X POST http://localhost:5000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "password123", "email": "user1@example.com"}'

# Login and get tokens
curl -X POST http://localhost:5000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "password123"}'

# Use the access token to make authenticated requests
curl -X POST http://localhost:5000/api/v1/auth/api-keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{"name": "My API Key"}'

# Use the API key to make authenticated requests
curl -X POST http://localhost:5000/api/v1/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <api_key>" \
  -d '{"question": "What is the nature of reality?"}'
```

## Required Packages

Create a `requirements.txt` file with:

```
PyPDF2>=3.0.0
mysql-connector-python>=8.0.0
python-dotenv>=0.20.0
requests>=2.27.1
tqdm>=4.64.0
tiktoken>=0.4.0
pinecone-client>=2.2.1
torch>=1.13.0
sentence-transformers>=2.2.2
numpy>=1.23.0
```

## License

MIT 