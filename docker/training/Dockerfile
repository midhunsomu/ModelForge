# ── ModelMesh Training Dockerfile ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements/training.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
RUN groupadd -r trainer && useradd -r -g trainer -d /app trainer
WORKDIR /app
COPY --from=builder /install /usr/local
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=trainer:trainer config/ ./config/
COPY --chown=trainer:trainer src/training/ ./src/training/
COPY --chown=trainer:trainer src/registry/ ./src/registry/
COPY --chown=trainer:trainer src/database/ ./src/database/
COPY --chown=trainer:trainer pipelines/ ./pipelines/
COPY --chown=trainer:trainer data/ ./data/

RUN mkdir -p /app/mlruns /app/data/reference \
    && chown -R trainer:trainer /app/mlruns /app/data

USER trainer

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app LOG_FORMAT=json

# Default: run training. Override CMD for retraining or Prefect worker.
CMD ["python", "-m", "src.training.train"]
