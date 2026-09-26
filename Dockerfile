# Production Dockerfile for NADI-9 Agent API Service
FROM python:3.11-slim

# Set working directory & environment variables
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NADI9_DATABASE_URL=sqlite:////app/data/nadi9.db \
    NADI9_CHECKPOINT_DB_PATH=/app/data/nadi9_checkpoints.db

# Install system build dependencies if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml /app/
COPY README.md /app/

# Copy source code and default sample dataset
COPY src /app/src
COPY data /app/sample_data

# Install dependencies and package
RUN pip install --no-cache-dir .

# Create data output directory for SQLite persistence
RUN mkdir -p /app/data /app/outputs

# Expose HTTP REST API port
EXPOSE 8000

# Healthcheck using curl on /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI server entrypoint
CMD ["uvicorn", "nadi9.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
