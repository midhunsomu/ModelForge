# ── ModelMesh Drift Monitor Dockerfile ───────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
COPY requirements/drift.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
RUN groupadd -r driftmon && useradd -r -g driftmon -d /app driftmon
WORKDIR /app
COPY --from=builder /install /usr/local
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=driftmon:driftmon config/ ./config/
COPY --chown=driftmon:driftmon src/monitoring/ ./src/monitoring/
COPY --chown=driftmon:driftmon src/database/ ./src/database/
COPY --chown=driftmon:driftmon src/registry/ ./src/registry/
COPY --chown=driftmon:driftmon src/streaming/ ./src/streaming/
COPY --chown=driftmon:driftmon data/reference/ ./data/reference/

USER driftmon
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app LOG_FORMAT=json

CMD ["python", "-m", "src.monitoring.drift_monitor"]
