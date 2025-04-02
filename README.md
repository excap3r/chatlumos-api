# PDF Wisdom Extraction API

This project provides an API for extracting information from PDF documents, storing it, and allowing users to ask questions against the extracted knowledge. It features a Flask-based API frontend, asynchronous task processing using Celery workers, and interacts with various internal services for core functionalities like text extraction, vector search, and language model interactions.

## Core Features

*   **PDF Processing:** Upload PDF documents via the API for asynchronous processing (chunking, text extraction).
*   **Knowledge Extraction:** Automatically extracts key concepts and generates Q&A pairs from processed documents using LLMs (handled by background workers).
*   **Vector Search:** Stores extracted information as vector embeddings (using Pinecone by default, configured via `services.vector_search`) for semantic search.
*   **Question Answering:** Provides an API endpoint (`/ask`) to query the knowledge base, leveraging vector search and LLMs.
*   **Authentication:** Secure API access using JWT and API keys, with role-based access control.
*   **Asynchronous Processing:** Uses Celery and Redis for handling long-running tasks without blocking API requests.
*   **Service-Oriented:** Designed with distinct service components for modularity (e.g., LLM service, vector service, PDF processing).

## Architecture Overview

The system employs a service-oriented architecture:

1.  **Flask API Frontend (`app.py`):** Handles incoming HTTP requests, authentication, request validation, and basic routing. It acts as the main entry point for external clients.
2.  **API Gateway (`services/api_gateway`):** Used by some Flask API routes to communicate with potential downstream internal services (like a dedicated vector search or LLM inference service). This provides a level of indirection.
3.  **Celery Workers (`celery_app.py`, `services/tasks/*`):** Execute long-running background tasks (e.g., `process_pdf_task`, `process_question_task`) asynchronously. They interact with Redis for task queuing and status updates.
4.  **Internal Services (`services/*`):** Contain the core logic for specific domains:
    *   `services/llm_service`: Interacts with external Large Language Models.
    *   `services/vector_search`: Manages vector embeddings and search (e.g., Pinecone).
    *   `services/db`: Handles database interactions (metadata, users, etc.).
    *   `services/pdf_processor`: Extracts text and chunks PDFs.
    *   `services/analytics`: Manages event tracking and webhooks.
    *   *Interaction:* Celery tasks often initialize and use these service classes directly, while some Flask routes might use the API Gateway pattern to interact with them (or placeholder API calls).
5.  **Dependencies:**
    *   **MySQL:** Stores relational data (users, document metadata, auth tokens, etc.).
    *   **Redis:** Used as the Celery message broker, task result backend, and for caching/rate limiting.
    *   **Pinecone/Vector DB:** Stores and searches vector embeddings.

## Installation

### Prerequisites

- Python 3.8+
- MySQL database (for metadata and authentication)
- Redis (for Celery broker/backend and caching)
- [Pinecone](https://www.pinecone.io/) account (or other vector DB, requires code changes)
- API Keys for required LLM providers (e.g., OpenAI, Groq, Anthropic)

### Setup

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Copy `.env.template` to `.env` and fill in your API keys, database credentials, Redis URL, and JWT secret:
    ```bash
    cp .env.template .env
    # Edit .env with your specific configuration
    ```
    *Ensure you set a strong `JWT_SECRET`.*

## Running the Application (API + Worker)

This project runs as a web API server with background task processing using Celery.

### Prerequisites

Ensure you have completed the [Installation](#installation) steps, including setting up your `.env` file with database, Redis, Pinecone, and LLM credentials.

### Using Docker Compose (Recommended)

The easiest way to run the application and its dependencies (MySQL, Redis) is using Docker Compose:

1.  Make sure Docker is installed and running.
2.  Ensure your `.env` file is correctly configured.
3.  Generate a secure JWT secret (if not already set in `.env`):
    ```bash
    export JWT_SECRET=$(openssl rand -hex 32) 
    # You might need to manually add this to your .env file depending on your setup
    ```
4.  Build and start the services:
    ```bash
    docker-compose up --build -d
    ```
    This will start the Flask API server (via Gunicorn), the Celery worker, a MySQL database, and a Redis instance.

### Running Manually

Alternatively, you can run the components manually after ensuring MySQL and Redis servers are running and accessible based on your `.env` configuration.

**1. Run Database Migrations:**
Apply the necessary database schemas:
```bash
# Assuming you have a MySQL client installed and configured access
mysql -u<user> -p<password> -h<host> <database_name> < migrations/01_create_core_tables.sql
mysql -u<user> -p<password> -h<host> <database_name> < migrations/02_create_auth_tables.sql
mysql -u<user> -p<password> -h<host> <database_name> < migrations/03_create_analytics_tables.sql
# Add any subsequent migration files here
```
*(Note: A proper migration tool like Alembic or Flyway is recommended for more complex scenarios)*

**2. Run the API Server:**
You can run the Flask API server directly for development or using Gunicorn for production:

*Development Server:*
```bash
export FLASK_APP=app.py 
# export FLASK_DEBUG=1 # Optional for development
flask run --host=0.0.0.0 --port=5000 
```

*Production Server (using Gunicorn):*
```bash
gunicorn --config gunicorn_config.py app:app
```

**3. Run the Celery Worker:**
In a separate terminal, start the Celery worker to process background tasks:
```bash
celery -A celery_app worker --loglevel=info
```
*(Make sure your current directory is the project root where `celery_app.py` resides)*

## API Usage

Interact with the service through its REST API endpoints. Authentication is required for most endpoints.

### Core Endpoints

*   `POST /api/v1/pdf/upload`: Upload a PDF for processing. Returns a `task_id`.
    *   Requires form-data with `file` (the PDF), `author`, `title` (optional), `language` (optional), `translate_to_english` (optional boolean string 'true'/'false').
*   `GET /api/v1/progress/{task_id}`: Stream progress updates for a background task (like PDF processing or asking a question with `stream=true`) using Server-Sent Events (SSE).
*   `POST /api/v1/ask`: Ask a question against the knowledge base.
    *   Requires JSON body with `question`. Optional: `stream` (boolean, defaults to `false`), `index_name`, `top_k`.
    *   If `stream=true`, returns a `task_id` to use with the `/progress` endpoint.
    *   If `stream=false`, returns the answer and context directly (blocking request).
*   `POST /api/v1/search`: Perform a direct vector search.
    *   Requires JSON body with `query`. Optional: `index_name`, `top_k`, `metadata_filter`.
*   `POST /api/v1/translate`: Translate text.
    *   Requires JSON body with `text`. Optional: `target_lang`, `source_lang`.
*   `GET /api/v1/health`: Check the health status of the API and its backend services.

*(See `auth_routes.py`, `pdf.py`, `ask.py`, etc. for detailed request/response formats)*

### Example API Workflow (using curl)

1.  **Register and Login (Get Auth Token - see Authentication section below)**
    ```bash
    # Assuming you have registered and logged in to get an <access_token>
    export AUTH_TOKEN="<your_access_token>" 
    ```

2.  **Upload a PDF:**
    ```bash
    curl -X POST http://localhost:5000/api/v1/pdf/upload \
      -H "Authorization: Bearer $AUTH_TOKEN" \
      -F "file=@/path/to/your/document.pdf" \
      -F "author=Document Author" \
      -F "title=Document Title" 
    # This will return a JSON response like: {"task_id": "some-uuid", "status": "PDF processing task queued", ...}
    export PDF_TASK_ID="some-uuid" 
    ```

3.  **Monitor PDF Processing Progress (Optional):**
    ```bash
    curl -N -H "Authorization: Bearer $AUTH_TOKEN" http://localhost:5000/api/v1/progress/$PDF_TASK_ID
    # This will stream Server-Sent Events until the task is complete or fails.
    ```

4.  **Ask a Question (Streaming):**
    ```bash
    curl -X POST http://localhost:5000/api/v1/ask \
      -H "Authorization: Bearer $AUTH_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"question": "What is the main topic of the document?", "stream": true}'
    # This will return a JSON response like: {"task_id": "another-uuid", "status": "Processing started"}
    export ASK_TASK_ID="another-uuid"
    ```

5.  **Monitor Question Answering Progress:**
    ```bash
    curl -N -H "Authorization: Bearer $AUTH_TOKEN" http://localhost:5000/api/v1/progress/$ASK_TASK_ID
    # Streams SSE progress, including the final answer when status is 'Completed'.
    ```

6.  **Ask a Question (Synchronous):**
    ```bash
    curl -X POST http://localhost:5000/api/v1/ask \
      -H "Authorization: Bearer $AUTH_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"question": "What is the main topic of the document?", "stream": false}'
    # Returns the answer directly: {"answer": "The main topic is...", "context": [...]} 
    ```

## Legacy Command-Line Tools (Deprecated)

Previous versions of this project included command-line tools (`pdf_wisdom_extractor.py`, `mysql_to_vector.py`, `wisdom_qa.py`). These tools are **no longer the primary way** to interact with the system and may not be actively maintained.

The core functionality provided by these tools (PDF processing, vectorization, Q&A) is now handled by the **API server and background Celery workers**. Users should interact with the system via the API endpoints described above.

The CLI scripts might remain in the codebase for historical reference but should not be relied upon for current usage.

## API Authentication

The API uses a robust authentication system supporting JWT tokens and API keys.

### Authentication Endpoints

*   `POST /api/v1/auth/register`: Register a new user.
*   `POST /api/v1/auth/login`: Authenticate and get JWT access/refresh tokens.
*   `POST /api/v1/auth/refresh`: Use a refresh token to get a new access token.
*   `POST /api/v1/auth/keys`: Create a new API key (requires JWT auth).
*   `GET /api/v1/auth/keys`: List API keys for the current user (requires JWT auth).
*   `DELETE /api/v1/auth/keys/{key_id}`: Revoke an API key (requires JWT auth).
*   `GET /api/v1/auth/users`: List all users (admin only).
*   `PUT /api/v1/auth/users/{user_id}/roles`: Update user roles (admin only).
*   `GET /api/v1/auth/validate`: Validate the current JWT token or API key.

*(Note: API key management paths confirmed as `/auth/keys`)*

### Authentication Methods

1.  **JWT tokens:** For interactive sessions or user-based applications.
    *   Obtained via `/login`.
    *   Include in `Authorization: Bearer <access_token>` header.
    *   Access tokens are short-lived; use the refresh token with `/refresh` to get a new one.
2.  **API keys:** For programmatic access or server-to-server communication.
    *   Created via the API key endpoint (e.g., `/auth/keys`).
    *   Include in request header: `X-API-Key: <api_key>` or `Authorization: ApiKey <api_key>`.
    *   Can also be included as a query parameter: `?api_key=<api_key>`.

### User Roles and Permissions

*   **user:** Basic access (e.g., upload PDFs, ask questions).
*   **admin:** Full access, including user management.
*   Custom roles/permissions can be configured in the database.

### Environment Configuration for Auth

Ensure these are set in your `.env` file:
```dotenv
# Authentication settings
JWT_SECRET=<generate_a_strong_random_string>
JWT_ACCESS_TOKEN_EXPIRE_SECONDS=3600 # 1 hour
JWT_REFRESH_TOKEN_EXPIRE_SECONDS=2592000 # 30 days

# Database configuration (used for user/auth storage)
MYSQL_HOST=localhost 
MYSQL_PORT=3306
MYSQL_USER=wisdom_user
MYSQL_PASSWORD=wisdom_password
MYSQL_DB=wisdom_db 
MYSQL_POOL_SIZE=10 
```
+*(Note: When using Docker Compose, the `MYSQL_*` variables above are also used to configure the MySQL service container itself.)*

### Authentication Example (`curl`)

1.  **Register:**
    ```bash
    curl -X POST http://localhost:5000/api/v1/auth/register \
      -H "Content-Type: application/json" \
      -d '{"username": "testuser", "password": "password123", "email": "test@example.com"}'
    ```
2.  **Login:**
    ```bash
    curl -X POST http://localhost:5000/api/v1/auth/login \
      -H "Content-Type: application/json" \
      -d '{"username": "testuser", "password": "password123"}'
    # This returns access_token and refresh_token
    export AUTH_TOKEN=$(...) # Store the access_token
    ```
3.  **Create an API Key (using JWT Token):**
    ```bash
    curl -X POST http://localhost:5000/api/v1/auth/keys \
      -H "Authorization: Bearer $AUTH_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"name": "MyScriptKey"}'
    # This returns the new API key (save it securely!)
    export API_KEY=$(...) # Store the actual api_key value returned
    ```
4.  **Make an Authenticated Request using API Key:**
    ```bash
    curl -X POST http://localhost:5000/api/v1/ask \
      -H "X-API-Key: $API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"question": "What is the law of attraction?"}'
    ```

## License

MIT 