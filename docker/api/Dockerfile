# ── ModelMesh API Dockerfile ──────────────────────────────────────────────────
# Multi-stage build: keeps final image lean (~600MB vs ~2GB single stage)
# Stage 1: dependency builder
# Stage 2: runtime with non-root user

# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some Python packages (e.g. psycopg2, lxml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/api.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# System runtime deps only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \         
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/reference/ ./data/reference/

# Create writable dirs for mlruns artifacts (when MLflow is local)
RUN mkdir -p /app/mlruns /app/logs \
    && chown -R appuser:appuser /app/mlruns /app/logs

USER appuser

# Environment defaults (override via K8s ConfigMap/Secret)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO

EXPOSE 8000

# Health check — K8s will also use /health/live, but useful for docker run
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Use exec form (no shell) — signals propagate correctly to Python
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-config", "/dev/null", \
     "--access-log"]
