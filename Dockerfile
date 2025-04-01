# Stage 1: Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-warn-script-location --prefix=/install -r requirements.txt

# Stage 2: Final stage
FROM python:3.10-slim

# Create a non-root user to run the application
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/sh -m appuser # Use /bin/sh for smaller footprint

WORKDIR /app

# Add installed packages to PATH
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH"

# Copy installed dependencies from builder stage
COPY --from=builder /install /install

# Copy application code (be specific)
COPY alembic.ini .
COPY app.py .
COPY celery_app.py .
COPY gunicorn_config.py .
COPY services/ services/
# Add other necessary files/dirs like migrations if needed at runtime
COPY migrations/ migrations/

# Set proper permissions
# Ensure ownership is changed before switching user
RUN chown -R appuser:appuser /app /install

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Start the API server using Gunicorn
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"] 