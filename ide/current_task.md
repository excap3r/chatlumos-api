# Codebase Analysis and Refactoring Task

## Goal
Analyze the entire codebase, identify potential refactoring opportunities and improvements, and document them as a checklist. Ensure the final code meets high standards of quality, maintainability, and performance.

## Checklist

### Phase 1: Project Structure and Dependency Analysis
- [x] Map out the complete project file structure (Top-level and key directories listed).
- [x] Analyze dependencies listed in `requirements.txt`.

### Phase 2: Core Application Analysis
- [x] Analyze `app.py` for structure, routing, potential refactoring, and error handling.
- [x] Analyze `gunicorn_config.py` for configuration best practices.
- [x] Analyze `Dockerfile` for build efficiency and security.
- [x] Analyze `docker-compose.yml` for service configuration and orchestration.

### Phase 3: Service Layer Analysis
- [X] Analyze files within the `services/` directory (Sub-directory structure mapped).
  - [X] Analyze `services/api/` (Files listed: `auth_routes.py` [Done], `analytics_routes.py` [Done], `webhook_routes.py` [Done], `middleware/auth_middleware.py` [Done], `routes/*` (`health.py` [Done], `ask.py` [Done], `progress.py` [Done], `translate.py` [Done], `search.py` [Done], `pdf.py` [Done], `question.py` [Done], `docs.py` [Done], `root.py` [Done]))
  - [X] Analyze `services/tasks/` (Files listed: `pdf_processing.py` [Done], `question_processing.py` [Done])
  - [X] Analyze `services/utils/` (Files listed: `api_helpers.py` [Done], `auth_utils.py` [Done], `log_utils.py` [Done], `error_utils.py` [Done], `env_utils.py` [Done], `__init__.py` [Done])
  - [X] Analyze `services/analytics/` (Files listed: `analytics_service.py` [Done], `analytics_middleware.py` [Done], `webhooks/webhook_service.py` [Done])
  - [X] Analyze `services/db/` (Files listed: `user_db.py` [Done], `exceptions.py` [Done], `schema/users.sql` [Done])
  - [X] Analyze `services/api_gateway/` (Files listed: `api_gateway.py` [Done], `__init__.py` [Done])
  - [X] Analyze `services/db_service/` (Files listed: `db_service.py` [Done], `__init__.py` [Done])
  - [X] Analyze `services/llm_service/` (Files listed: `llm_service.py` [Done], `providers/*` [Done])
  - [X] Analyze `services/vector_search/` (Files listed: `vector_search.py` [Done])
  - [X] Analyze `services/pdf_processor/` (Files listed: `pdf_processor.py` [Done])

### Phase 4: Testing Analysis
- [ ] Analyze files within the `tests/` directory.
- [ ] Implement a comprehensive test suite using `pytest` or `unittest`.
- [ ] Add unit tests for service logic and utility functions.
- [ ] Add integration tests (mocking external services) for component interactions.
- [ ] Add API tests for endpoint validation, request/response formats, and auth.

### Phase 5: Database Migrations Analysis
- [X] Analyze files within the `migrations/` directory (`02_create_auth_tables.sql` analyzed).
  - Finding: Defines the *actual* auth schema, including the critical `password_salt` column.
  - Discrepancy: Differs significantly from unused `services/db/schema/users.sql`.

### Phase 6: Documentation, Configuration, and Dependency Analysis
- [ ] Analyze files within the `docs/` directory.
- [X] Analyze `README.md` for clarity and completeness.
  - [ ] Update README to accurately reflect the current service-oriented architecture (API Gateway, Flask services) vs. the older CLI tool description.
  - [ ] Clarify the relationship/status of the described CLI tools (`pdf_wisdom_extractor.py`, `mysql_to_vector.py`, `wisdom_qa.py`).
- [X] Analyze `.gitignore` for comprehensive coverage.
- [X] Analyze `.env.example` and `.env.template` for clarity and consistency.
  - [ ] Reconcile `MYSQL_DATABASE` name inconsistency (`IAAI` vs `wisdom_db`).
  - [ ] Remove or clarify the purpose of `*_SERVICE_URL` variables given the single-container setup.
  - [ ] Add missing LLM provider API key variables (Groq, OpenRouter) to the template.
- [X] Analyze `requirements.txt`.
  - [ ] Consider pinning dependency versions (`==`) for reproducibility.
  - [ ] Check for any outdated / vulnerable dependencies.

### Phase 7: Refactoring and Improvement Implementation (Requires Approval)
- [ ] Define priority levels (Critical, High, Medium, Low).
- [ ] Prioritize tasks based on severity (security > correctness > consistency > improvements).
- [ ] Create a detailed, step-by-step refactoring plan based on prioritized items below.

- **Architecture & Cross-Cutting Concerns:**
  - [ ] **CRITICAL:** Address state management: Move `app.progress_events` from memory to Redis/shared store (Affects `app.py`, `services/api/routes/ask.py`, `services/api/routes/progress.py`, `services/api/routes/pdf.py`, `services/tasks/pdf_processing.py`, `services/tasks/question_processing.py`).
  - [ ] **CRITICAL:** Refactor async processing/progress tracking in `/api/routes/ask` and `/api/routes/pdf` to use a scalable task queue (e.g., Celery) and pub/sub mechanism (e.g., Redis Pub/Sub) instead of `threading` and in-memory state (coordinate with `progress.py` and `tasks/*` refactors).
  - [ ] Refactor internal communication: Avoid using `APIGateway` client for intra-app calls in `app.py`; call service functions directly.
  - [ ] Standardize Logging: Switch to fully structured logging (e.g., `structlog`) consistently across all services (`app.py`, `gunicorn_config.py`, `services/utils/log_utils.py`, `services/db/user_db.py`, `services/db_service/db_service.py`, `services/llm_service/llm_service.py`, `services/vector_search/vector_search.py`, `services/pdf_processor/pdf_processor.py`, `services/analytics/*`, etc.).
  - [ ] Standardize Error Handling: Use centralized Flask `@app.errorhandler` registration (`app.py`, `services/utils/error_utils.py`), define specific `APIError` subclasses, integrate with structured logging.
  - [ ] Centralize application configuration loading and management (`app.py`, `services/utils/env_utils.py`).
  - [ ] Clarify/Implement reliable service initialization at startup (`app.py`, `gunicorn_config.py`, consider `preload_app` interaction).
  - [ ] **CRITICAL:** Resolve structural ambiguity for `services/analytics/analytics_service.py` and `services/analytics/webhooks/webhook_service.py`: Decide if standalone services or utility modules; refactor accordingly.
  - [ ] Refactor webhook sending (`services/analytics/webhooks/webhook_service.py`) to use a robust task queue instead of `threading`.

- **Database (`services/db/`, `migrations/`):**
  - [ ] **CRITICAL:** Update `services/db/user_db.py` to align with `migrations/02_create_auth_tables.sql` schema (UUIDs, `password_salt`, roles/permissions).
  - [ ] **CRITICAL:** Fix `authenticate_user` in `services/db/user_db.py` to use secure password hashing and verification (e.g., `bcrypt.checkpw` with salt) instead of direct hash comparison in SQL.
  - [ ] **CRITICAL:** Ensure `update_user_password` in `services/db/user_db.py` also uses secure hashing with salt.
  - [ ] **CRITICAL:** Secure or remove default admin user creation in `migrations/02_create_auth_tables.sql`.
  - [ ] Delete unused/incorrect `services/db/schema/users.sql`.
  - [ ] Consolidate all schema management into `migrations/` or use a migration tool (e.g., Alembic).
  - [ ] Consider adding a `key_salt` column to the `api_keys` table (`migrations/` and `services/db/user_db.py`).
  - [ ] Consider ORM (like SQLAlchemy) for `services/db/user_db.py` for maintainability, type safety, and abstraction.
  - [ ] Centralize DB pool initialization (maybe in `app.py` or a dedicated config module).
  - [ ] Manage DB connections/cursors via context managers (`with`) or dependency injection (`services/db/user_db.py`, `services/db_service/db_service.py`).
  - [ ] Improve DB error handling specificity (`services/db/user_db.py` to raise custom exceptions from `services/db/exceptions.py`).
  - [ ] Add necessary DB indexes for performance (e.g., on `username`, `email`, `api_key` in `migrations/`).
  - [ ] Review `ON DELETE CASCADE` behavior in `migrations/`.
  - [ ] Ensure application logic enforces `daily_rate_limit` from schema (if applicable).

- **API (`services/api/`):**
  - **`auth_routes.py`:**
    - [ ] Enhance input validation (e.g., email format, password complexity) possibly with a library.
    - [ ] Consider implementing refresh token rotation.
    - [ ] Make token `expires_in` value reflect actual configured expiry.
    - [ ] Review necessity of all user details returned on login.
    - [ ] Consider abstracting direct `user_db.py` calls.
  - **`middleware/auth_middleware.py`:**
    - [ ] Refactor/consolidate role/permission checking logic.
    - [ ] Ensure consistency in permission check logic (utility vs. list lookup).
    - [ ] Handle potential `verify_api_key` database exceptions explicitly.
    - [ ] Standardize `g.user` structure across auth types.
  - **`analytics_routes.py`:**
    - [ ] Extract duplicated date parsing logic into a helper function.
    - [ ] Move `/user-stats` calculation logic into `services/analytics/analytics_service.py`.
    - [ ] Make hardcoded limits (defaults, export) configurable.
    - [ ] Abstract CSV generation logic in `/export`.
    - [ ] Improve error handling specificity (catch specific exceptions for 400 errors).
    - [ ] Consider implementing proper pagination for `/events` and `/export`.
  - **`webhook_routes.py`:**
    - [ ] Enhance input validation (URL format, known event types).
    - [ ] Refactor repeated ownership/admin check logic.
    - [ ] Consider using specific exceptions in `webhook_service` instead of boolean return.
    - [ ] Review PUT vs. PATCH semantics for updates.
    - [ ] Analyze `/test/<webhook_id>` endpoint implementation.
    - [ ] Ensure integration with refactored `services/analytics/webhooks/webhook_service.py`.
  - **`routes/health.py`:**
    - [ ] Refactor/verify `rate_limit` decorator interaction with `current_app.redis_client`.
    - [ ] Break down long `health_check` function into smaller parts.
    - [ ] Evaluate potential consolidation of API Gateway calls (if external gateway is used, otherwise remove internal calls).
    - [ ] Simplify logic for fetching/handling individual service details.
    - [ ] Improve error handling specificity (e.g., network errors).
    - [ ] Review clarity of sub-service status derivation in response.
  - **`routes/ask.py`:**
    - [ ] **CRITICAL:** Implement scalable streaming/task queue refactor (see Architecture section).
    - [ ] Consolidate synchronous RAG logic into a shared service/task function if applicable.
    - [ ] Improve error reporting for async path initiation.
    - [ ] Verify/refactor rate limiter interaction with `current_app.redis_client`.
    - [ ] Ensure configuration (e.g., `top_k`) is consistently used.
  - **`routes/progress.py`:**
    - [ ] **CRITICAL:** Implement scalable SSE/PubSub refactor (see Architecture section).
    - [ ] Make stream timeout configurable.
    - [ ] Review and improve session data cleanup mechanism, especially with Redis.
  - **`routes/translate.py`:**
    - [ ] Consider replacing LLM-based translation with the dedicated `deep-translator` library (from `requirements.txt`) for efficiency/cost.
    - [ ] If using LLM, refine system prompt and max token estimation.
    - [ ] Improve robustness of error status code determination (avoid string checking).
    - [ ] Ensure cache key for `@cache_result` is safe and considers all inputs.
    - [ ] Verify/refactor rate limiter/caching decorator interaction with `current_app.redis_client`.
  - **`routes/search.py`:**
    - [ ] Add input validation for `top_k` and `metadata_filter` structure.
    - [ ] Consolidate configuration loading for default parameters (`index_name`, `top_k`).
    - [ ] Verify/refactor rate limiter/caching decorator interaction with `current_app.redis_client`.
    - [ ] Improve robustness of downstream error handling (avoid relying on `status_code` in JSON body).
  - **`routes/pdf.py`:**
    - [ ] **CRITICAL:** Implement scalable async processing/task queue refactor (see Architecture section).
    - [ ] Use a configurable upload location (or object storage) instead of hardcoding `/tmp`.
    - [ ] Replace delayed cleanup thread with robust state management via task queue/store.
    - [ ] Use `g.user` for auth context consistency instead of `request.user`.
    - [ ] Consider using `secure_filename`.
    - [ ] Make `UPLOAD_FOLDER`, `ALLOWED_EXTENSIONS` configurable.
  - **`routes/question.py`:**
    - [ ] Clarify purpose/use case vs. `/ask` endpoint.
    - [ ] Add validation/documentation for expected `context` format.
    - [ ] Validate `model` parameter against known generative models; correct the default value (currently an embedding model).
    - [ ] Use `current_app.logger` consistently.
    - [ ] Verify decorator interaction with `current_app.redis_client`.
  - **`routes/docs.py`:**
    - [ ] Replace manual OpenAPI definition with automated generation (e.g., using `apispec`, `Flask-RESTX`, or `Flask-Smorest`).
    - [ ] If manual: Add missing endpoints (search, auth, analytics, webhooks) and define reusable schemas.
    - [ ] Ensure security requirements are correctly applied to all relevant endpoints in the spec.

- **Task Queue Services (`services/tasks/`):**
  - **`pdf_processing.py`:**
    - [ ] **CRITICAL:** Integrate with scalable task queue (e.g., Celery), remove `progress_events` dependency, adapt progress reporting.
    - [ ] Adapt file handling to use identifiers (e.g., S3 keys) instead of local paths.
    - [ ] Integrate proper translation logic (replace placeholder).
    - [ ] Make pipeline parameters (chunk size, etc.) configurable.
    - [ ] Refactor dependency handling (logger, gateway) for task queue context.
    - [ ] Review error handling strategy for partial failures (e.g., mark doc metadata).
    - [ ] Consider explicit batching for vector storage if needed.
  - **`question_processing.py`:**
    - [ ] **CRITICAL:** Integrate with scalable task queue (e.g., Celery), remove `progress_events` dependency, adapt progress reporting (e.g., Redis Pub/Sub).
    - [ ] Make `DEFAULT_TOP_K` and cache TTLs configurable.
    - [ ] Refactor dependency handling (logger, gateway, redis) for task queue context.
    - [ ] Review `get_cache_key` logic for robustness (especially answer cache).
    - [ ] Refine error handling for partial search failures and context aggregation.
    - [ ] Ensure context aggregation for final answer generation is effective and respects limits.

- **Utility Services (`services/utils/`):**
  - **`api_helpers.py`:**
    - [ ] Consider hashing request components for `cache_result` key robustness.
    - [ ] Refine/document `cache_result` response handling logic for non-standard types.
    - [ ] Allow global configuration defaults for `rate_limit` parameters.
    - [ ] Add `Retry-After` header to `rate_limit` 429 response.
    - [ ] Ensure consistent usage and context handling for `redis_client_provider`.
  - **`auth_utils.py`:**
    - [ ] Make JWT expiry times configurable.
    - [ ] Consider making PBKDF2 iteration count configurable.
    - [ ] Consider supporting configurable/asymmetric JWT algorithms (e.g., RS256).
    - [ ] Consider adding salt/pepper to API key hashing (related to `key_salt` in DB).
    - [ ] If complexity grows, refactor hardcoded 'admin' role logic to a more structured RBAC system.
    - [ ] Consider migrating password hashing from PBKDF2 to Argon2.
  - **`log_utils.py`:**
    - [ ] Switch to fully structured logging (e.g., `structlog`), integrating with `app.py`/`gunicorn` refactor goal.
    - [ ] Make logging level and file path configurable (env/config).
    - [ ] Ensure consistent propagation and use of `request_id` in logging context.
    - [ ] Enhance sensitive data redaction (beyond just 'Authorization').
    - [ ] Refine error handling for parsing request/response bodies in logs.
    - [ ] Consider adding configuration to disable/truncate body logging for performance.
    - [ ] Use named loggers consistently instead of root logger default.
  - **`error_utils.py`:**
    - [ ] Replace `handle_error` decorator with centralized Flask `@app.errorhandler` registration.
    - [ ] Remove or strictly control traceback exposure in `format_error_response` (via debug flag).
    - [ ] Integrate logging with structured logging setup (e.g., `structlog`), including context.
    - [ ] Define more specific `APIError` subclasses (e.g., `NotFoundError`, `AuthenticationError`).
    - [ ] Clarify intended usage pattern (decorator vs. Flask handlers).
  - **`env_utils.py`:**
    - [ ] Integrate into a centralized configuration system (e.g., Flask `app.config`, Pydantic `BaseSettings`) instead of direct calls.
    - [ ] Move `load_dotenv()` call to application entry point (`app.py`).
    - [ ] Add explicit type conversion capabilities (int, bool, etc.).
    - [ ] Make JSON parsing in `get_config` more explicit (e.g., flag or separate function).
    - [ ] Use more specific custom exceptions for configuration errors.
  - **`__init__.py`:**
    - [ ] Ensure `__all__` is kept synchronized with refactoring of utility modules.

- **Analytics Services (`services/analytics/`):**
  - **`analytics_service.py`:**
    - [ ] **CRITICAL:** Resolve structural ambiguity (see Architecture section).
    - [ ] Optimize Redis data model: Avoid event duplication (e.g., use HASH), use efficient structures for querying (e.g., ZSET/TimeSeries), improve counters.
    - [ ] Make analytics TTL configurable.
    - [ ] Refactor `track_api_call` to use standard `after_request` handler and avoid non-standard `request` attributes.
    - [ ] Improve `get_analytics` function: Implement pagination, enhance query efficiency (linked to data model), add input validation.
    - [ ] Integrate with centralized structured logging.
    - [ ] Improve Redis error handling and endpoint input validation (if endpoints are kept).
  - **`analytics_middleware.py`:**
    - [ ] Refactor integration based on `analytics_service.py` structure (HTTP calls vs. direct imports).
    - [ ] Fix `after_request` logic: use `response` object directly, pass status/error to refactored `track_api_call`.
    - [ ] Consolidate `errorhandler(Exception)` with centralized application error handling.
    - [ ] Make payload redaction in `track_specific_event` configurable.
    - [ ] Integrate logging with centralized structured logging (`structlog`).
    - [ ] Clarify/document dependency on `g.user` from auth middleware.
  - **`webhooks/webhook_service.py`:**
    - [ ] **CRITICAL:** Resolve structural ambiguity (see Architecture section).
    - [ ] Replace webhook `threading` with a robust task queue (see Architecture section).
    - [ ] Fix success/failure count logic (update after send attempt in task).
    - [ ] Make TTL, retry count, timeout configurable.
    - [ ] Implement exponential backoff for retries (via task queue).
    - [ ] Consider Redis transactions for atomic updates/deletes.
    - [ ] Integrate logging with centralized structured logging (`structlog`).
    - [ ] Improve Redis/request error handling.

- **Other Services:**
  - **`services/api_gateway/api_gateway.py`:** (Note: This seems to be an *internal* client, not a standalone gateway service based on docker-compose)
    - [ ] If intended as external gateway (needs separate service/container):
        - Implement Authentication/Authorization (e.g., JWT/API Key validation) before proxying.
        - Implement Rate Limiting (per user/key).
        - Consider dynamic Service Discovery (e.g., Consul) instead of static config.
        - Implement periodic background health checks for services.
        - Add resilience patterns (e.g., Circuit Breaker, Retries).
        - Enhance security (header filtering, input validation).
    - [ ] If only internal client: Rename/refactor to avoid "Gateway" confusion. Decouple client helper methods (search, ask) from hardcoded service/endpoint names. Ensure client URL construction aligns with actual routing logic.
  - **`services/db_service/db_service.py`:** (Note: Appears to be unused based on current structure - DB access via `services/db/user_db.py`)
    - [ ] Clarify architectural intent/usage. If kept:
        - Standardize logging using `log_utils`.
        - Standardize API error handling using `error_utils`.
        - Use context managers for DB connection handling.
        - Consider a DB migration tool (e.g., Alembic) for schema management.
        - Add input validation to API endpoints.
        - Review transaction management for batch inserts (`store_concepts`, `store_qa_pairs`).
  - **`services/llm_service/llm_service.py`:**
    - [ ] Standardize logging using `log_utils`.
    - [ ] Standardize API error handling using `error_utils`.
    - [ ] Implement robust input validation (e.g., Pydantic/Marshmallow).
    - [ ] Review security of API key handling (consider headers or secure config instead of body).
    - [ ] Consider externalizing prompt templates.
    - **`providers/base.py`:** Improve retry logic (more specific exceptions, use logging).
    - **`providers/openai.py` (and others):** Use standard logging instead of print, enhance error handling based on specific API errors.
  - **`services/vector_search/vector_search.py`:**
    - [ ] Standardize logging using `log_utils`.
    - [ ] Standardize API error handling using `error_utils`.
    - [ ] Refine initialization strategy (e.g., on startup/lazy load) instead of relying solely on `/initialize` API call.
    - [ ] Add readiness checks to `/health` based on initialization status.
    - [ ] Consider caching Pinecone index object instance.
    - [ ] Investigate Pinecone SDK for efficient batch query operations.
    - [ ] Implement robust input validation.
  - **`services/pdf_processor/pdf_processor.py`:**
    - [ ] Standardize logging using `log_utils`.
    - [ ] Standardize API error handling using `error_utils`.
    - [ ] Consider more robust PDF extraction library (e.g., PyMuPDF) if needed.
    - [ ] Implement token-based chunking (using tiktoken) instead of word-based.
    - [ ] Refine temporary file handling (e.g., use `tempfile` module or in-memory processing if possible).
    - [ ] Remove unused code/imports (rate limiting, etc.).

- **Build & Deployment (`Dockerfile`, `docker-compose.yml`, `gunicorn_config.py`):**
  - **`Dockerfile`:**
    - [ ] Add `.dockerignore` file.
    - [ ] Implement multi-stage build to reduce final image size and remove build dependencies.
  - **`docker-compose.yml`:**
    - [ ] Add container healthchecks for `redis` and `mysql`.
    - [ ] Define resource limits (memory, CPU) for services.
    - [ ] Review secret management strategy (source sensitive data from env/`.env`, avoid mounting whole `.env`).
    - [ ] Define API environment variables directly in the service definition, using `${VAR}` substitution from `.env`.
    - [ ] Consider separate `docker-compose.override.yml` or profiles for development vs. production.
  - **`gunicorn_config.py`:**
    - [ ] Integrate application's structured logging (`structlog`) with Gunicorn logging.
    - [ ] Ensure application initialization strategy works correctly with `preload_app=True` and `gevent` workers (e.g., using `post_fork` hook if needed for non-fork-safe resources).
    - [ ] Consider making more settings (e.g., `backlog`, `timeout`, `max_requests`) configurable via environment variables.

### Phase 8: Verification
- [ ] Review all implemented changes.
- [ ] Ensure all checklist items are addressed.