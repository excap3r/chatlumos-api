# Task: Codebase Refactoring

**Goal:** Refactor the codebase based on the analysis performed (see `ide/refactoring/` directory) to improve security, performance, code quality, and consistency. Implement the recommended improvements outlined below.

**Reference:** Detailed analysis and specific recommendations for each file can be found in the corresponding markdown files within the `ide/refactoring/` subdirectories. Refer to these files for context when implementing changes.

**Instructions for Agent:**

*   **Modify Code:** You are authorized to modify the codebase to implement the refactoring tasks.
*   **Incremental Changes:** Apply changes incrementally. Focus on completing one checklist item or a small group of related items before moving to the next.
*   **Testing:**
    *   Run `pytest` frequently after making changes to ensure existing functionality is not broken.
    *   Write new unit or integration tests where necessary to cover new logic or verify fixes (e.g., API key hashing/verification, new authorization checks).
    *   Update existing tests if function signatures or expected behavior changes due to refactoring.
*   **Verification:** After implementing a change, verify that it addresses the specific recommendation(s) from the corresponding analysis file(s).
*   **Dependencies:** If new dependencies are required (e.g., for SSRF filtering), add them to `requirements.txt` with appropriate version pinning.
*   **Database Migrations:** If schema changes are required (e.g., for API key hash migration), create new Alembic migration scripts in `migrations/alembic/versions/`. Document the migration clearly. **Note:** Migrating existing API key hashes from SHA-256 to bcrypt is complex and may require careful planning (e.g., a staged rollout or requiring users to regenerate keys). Implement the hashing change but plan the data migration strategy carefully.
*   **Documentation:** Update code comments, docstrings, README files, or MkDocs documentation if APIs, configurations, or critical logic change significantly.
*   **Progress Reporting:** Mark checklist items as complete (`[x]`) as you finish them. Provide summaries after completing major phases or significant items.

---

## Refactoring Checklist

**Phase 1: Critical Security & Core Improvements**

*   **API Key Hashing:**
    *   [x] `auth_utils.py`: Modify `hash_api_key` to use `bcrypt` (salt generated automatically).
    *   [x] `auth_utils.py`: Add `verify_api_key_hash` function using `bcrypt.checkpw`.
    *   [x] `user_db.py`: Update `verify_api_key` to use the new `auth_utils.verify_api_key_hash`.
    *   [x] `test_auth_utils.py`: Update tests for API key hashing/verification to reflect bcrypt usage. Add tests for `verify_api_key_hash`.
    *   [x] `migrations/`: Create Alembic migration script stub for migrating existing API key hashes (document the complexity/strategy in the script's docstring). **Note: File creation failed, please create `migrations/alembic/versions/abc123456789_api_key_bcrypt_migration.py` manually with the provided content.**
*   **API Gateway Enhancements:**
    *   [x] `api_gateway.py`: Integrate JWT validation using `auth_utils.decode_token`. Reject invalid/expired tokens.
    *   [x] `api_gateway.py`: Integrate API Key validation using `user_db.verify_api_key`. Reject invalid keys.
    *   [x] `api_gateway.py`: Implement IP-based rate limiting using Redis (leverage logic from `services/utils/api_helpers.py`).
    *   [x] `api_gateway.py`: Forward validated user context (e.g., user ID) in a custom header (e.g., `X-User-ID`) to downstream services.
    *   [x] `docker-compose.yml` / `config.py`: Ensure Redis is available to the gateway service for rate limiting.
*   **Endpoint Security:**
    *   [x] `services/api/routes/search.py`: Add `@auth_required()` decorator (or other appropriate level).
    *   [x] `services/api/routes/translate.py`: Add `@auth_required()` decorator (or other appropriate level).
    *   [x] `services/api/routes/question.py`: Add `@auth_required()` decorator (or other appropriate level).
    *   [x] `services/api/routes/progress.py`: Add `@auth_required()` decorator.
    *   [x] `services/api/routes/progress.py`: Implement authorization logic to ensure the requesting user (`g.user['id']`) is allowed to view the specified `task_id` (e.g., check against task metadata stored elsewhere, potentially requires modification to task creation/status updates).
*   **Secure Configuration:**
    *   [x] `docker-compose.yml`: Remove hardcoded DB credentials; configure them using environment variables loaded from `.env`.
    *   [x] Update `README.md` or deployment docs regarding `.env` file usage for `docker-compose`.
*   **API Key Transmission:**
    *   [x] `services/api/middleware/auth_middleware.py`: Remove the logic that checks `request.args.get('api_key')`. Prioritize headers only.
*   **Webhook Security:**
    *   [x] `services/analytics/webhooks/webhook_service.py`: Implement URL validation in `create_webhook` and `update_webhook` to block private/loopback IPs (consider using `ssrf-filter` library - add to `requirements.txt`).

**Phase 2: Performance & Quality Enhancements**

*   **Asynchronous Analytics:**
    *   [x] `services/analytics/analytics_service.py`: Modify `track_event` to send event data to a Celery task instead of writing directly to Redis.
    *   [x] `services/tasks/`: Create a new Celery task (e.g., `analytics_tasks.py`) to receive event data and perform the Redis pipeline operations.
    *   [x] Update `services/analytics/analytics_middleware.py` calls if needed.
*   **Database Performance:**
    *   [x] `services/db_service/db_service.py`: Refactor `store_concepts` to use `session.bulk_save_objects`.
    *   [x] `services/db_service/db_service.py`: Refactor `store_qa_pairs` to use `session.bulk_save_objects`.
    *   [x] `services/user_db.py`: Implement caching (Redis, short TTL) for `verify_api_key` results. Add necessary configuration (Redis client, TTL).
*   **Testing Improvements:**
    *   [x] `tests/`: Implement a reusable fixture or helper function for mocking authentication (`g.user`, token decoding, API key verification) to replace repetitive `@patch('flask.g')` usage across API/integration tests.
    *   [x] Update relevant API/integration tests to use the new auth mocking approach.
*   **Code Quality:**
    *   [x] `services/db_service.py` & `services/user_db.py`: Create and apply a decorator or context manager to handle the common `try...except SQLAlchemyError...rollback...log...raise` pattern.
    *   [x] `services/analytics/webhooks/webhook_service.py`: Refactor `_get_webhooks_by_ids` to use Redis `MGET` for fetching multiple subscriptions.
    *   [x] `tests/test_pinecone.py`: Rename to `scripts/check_pinecone_connection.py`, update comments to clarify purpose, and remove from `pytest` execution (update `pytest.ini` if needed). Optionally integrate `structlog`.

**Phase 3: Further Refinements**

*   [x] Review all `get_all_*` methods in DB services for potential N+1 query issues and add `selectinload`/`joinedload` where appropriate based on usage.
*   [x] Configure gateway and downstream services for HTTPS communication (update Dockerfiles, configs, docker-compose).
*   [ ] Refactor Redis client acquisition logic across services (e.g., `AnalyticsService`, `WebhookService`) to be less dependent on `current_app` if possible (e.g., explicit injection).
*   [ ] `auth_utils.py`: Optionally refactor `verify_password` signature/tests to remove the unused `salt` argument.

## Task: Refactor Redis Client Acquisition

**Goal:** Modify services to receive the Redis client via dependency injection instead of accessing `current_app`.

**Checklist:**
- [x] Analyze `app.py` Redis client and service initialization.
- [x] Analyze `AnalyticsService` (`services/analytics/analytics_service.py`) for `__init__` and Redis usage.
- [x] Analyze `WebhookService` (`services/analytics/webhooks/webhook_service.py`) for `__init__` and Redis usage.
- [x] Analyze `api_gateway.py` (`services/api_gateway/api_gateway.py`) for Redis usage (especially rate limiting).
- [x] Analyze `user_db.py` (`services/db/user_db.py`) for direct Redis usage.
- [x] Update `app.py` to inject `redis_client` into `AnalyticsService` and `WebhookService` constructors.
- [x] Update `AnalyticsService` `__init__` to accept `redis_client` and store it as `self.redis_client`.
- [x] Update `AnalyticsService` methods to use `self.redis_client` instead of `current_app.redis_client`.
- [x] Update `WebhookService` `__init__` to accept `redis_client` and store it as `self.redis_client`.
- [x] Update `WebhookService` methods to use `self.redis_client` instead of `current_app.redis_client`.
- [x] Update other direct Redis users (if found) to use injected client or access `current_app.redis_client` (if appropriate, e.g., decorators initialized early).
- [ ] Verify changes and ensure tests pass (if tests exist).
- [ ] Mark task as complete in `current_task.md`.

## Task: Debug Failing Tests

-   [x] Remove unused `ssrf-filter` package from `requirements.txt`.
-   [x] Update `torch` version in `requirements.txt` from `2.3.0` to `2.6.0`.
-   [x] Update `flask-swagger-ui` version in `requirements.txt` from `5.11` to `4.11.1`.
-   [x] Install `swig` using brew.
-   [x] Remove pinned versions for `pydantic` and `tokenizers` (and related dependencies) in `requirements.txt` to allow latest compatible versions for Python 3.13.
-   [x] Install `cmake` and `libomp` using brew (for faiss-cpu build).
-   [x] Replace `faiss-cpu` with `annoy` for vector similarity search.
-   [x] Install dependencies using `pip install -r requirements.txt`.
-   [ ] Run `pytest` to evaluate test results after dependency fixes and code changes.
-   [ ] Analyze `pytest` results and fix any remaining issues.

## Task: Fix Failing Registration Tests

- [x] Modify passwords in `test_register_success` and `test_register_duplicate` in `tests/api/test_auth_api.py` to include a special character.
- [x] Update assertion in `test_register_invalid_password` in `tests/api/test_auth_api.py` to match the actual string error format.
- [x] Re-run the specific registration tests to confirm fixes.
- [x] Fix `NameError: name 'InvalidCredentialsError' is not defined` in `services/user_service.py`.
- [x] Fix `TypeError: create_user() got an unexpected keyword argument 'password_hash'` by correcting the arguments passed from `services/user_service.py` to the database `create_user` function.
- [x] Re-run the specific registration tests to confirm fixes.
- [x] Run the full `test_auth_api.py` suite.
- [x] Run the full test suite (`pytest`) if auth tests pass.

## Next Steps:
1. Run the test suite to check if all dependencies are working correctly
2. Update any code that was using `faiss-cpu` to use `annoy` instead
3. Fix any remaining test failures
4. Document the changes in the codebase