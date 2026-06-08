"""
ModelMesh — FastAPI Inference Server
--------------------------------------
Production-grade REST API for model serving.

Features:
- Async lifespan: loads models, connects DB, starts Kafka on startup
- Request tracing via X-Trace-ID header
- Prometheus metrics at /metrics
- Structured JSON logging
- Rate limiting (sliding window)
- CORS + security headers
- Graceful shutdown
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
)

from config.logging_config import configure_logging, get_logger, set_trace_id
from config.settings import get_settings
from src.api.routers import health, inference, metrics, models
from src.api.middleware.rate_limiter import RateLimitMiddleware
from src.database.connection import close_db, init_db
from src.serving.model_loader import ModelLoader

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
# Defined at module level so they're shared across all requests

REQUEST_COUNT = Counter(
    "modelmesh_requests_total",
    "Total inference requests",
    ["endpoint", "model_version", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "modelmesh_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
ACTIVE_REQUESTS = Gauge(
    "modelmesh_active_requests",
    "Number of in-flight requests",
)
MODEL_PREDICTIONS = Counter(
    "modelmesh_predictions_total",
    "Total predictions made",
    ["model_name", "model_version", "model_stage", "prediction_label"],
)
DRIFT_ALERTS = Counter(
    "modelmesh_drift_alerts_total",
    "Total drift alerts triggered",
    ["model_name"],
)


# ── Application Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Handles startup and shutdown lifecycle.
    All stateful resources (DB pool, ML models, Kafka) are
    initialized here so they're ready before requests arrive.
    """
    logger.info(
        "modelmesh_starting",
        version=settings.app_version,
        environment=settings.environment,
    )

    # 1. Initialize database schema
    await init_db()

    # 2. Load models from MLflow registry
    model_loader = ModelLoader()
    app.state.model_loader = model_loader
    await model_loader.load_all()

    # 3. Start Kafka producer (if enabled)
    if settings.enable_kafka_streaming:
        from src.streaming.producer import KafkaPredictionProducer
        producer = KafkaPredictionProducer()
        app.state.kafka_producer = producer
        logger.info("kafka_producer_started")
    else:
        app.state.kafka_producer = None

    logger.info("modelmesh_startup_complete")

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("modelmesh_shutting_down")

    if settings.enable_kafka_streaming and hasattr(app.state, "kafka_producer"):
        app.state.kafka_producer.close()

    await close_db()
    logger.info("modelmesh_shutdown_complete")


# ── App Factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="ModelMesh Inference API",
        description="Production MLOps platform — real-time model serving with drift detection",
        version=settings.app_version,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.security.cors_origins],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.security.rate_limit_per_minute,
        window_seconds=60,
    )

    # ── Tracing + Metrics Middleware ──────────────────────────────────────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        # Assign or forward trace ID
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        set_trace_id(trace_id)

        # Track active requests
        ACTIVE_REQUESTS.inc()
        start = time.perf_counter()

        try:
            response: Response = await call_next(request)
            duration = time.perf_counter() - start

            # Inject trace ID into response headers
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Response-Time-Ms"] = str(round(duration * 1000, 2))

            # Record metrics
            endpoint = request.url.path
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
            REQUEST_COUNT.labels(
                endpoint=endpoint,
                model_version=response.headers.get("X-Model-Version", "unknown"),
                status_code=response.status_code,
            ).inc()

            logger.info(
                "http_request_completed",
                method=request.method,
                path=endpoint,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )
            return response

        except Exception as exc:
            duration = time.perf_counter() - start
            logger.error(
                "http_request_failed",
                path=request.url.path,
                error=str(exc),
                duration_ms=round(duration * 1000, 2),
            )
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "trace_id": trace_id},
            )
        finally:
            ACTIVE_REQUESTS.dec()

    # ── Security Headers ──────────────────────────────────────────────────────
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(inference.router, prefix="/v1", tags=["Inference"])
    app.include_router(metrics.router, prefix="/v1", tags=["Metrics"])
    app.include_router(models.router, prefix="/v1/models", tags=["Models"])

    # ── Prometheus scrape endpoint ────────────────────────────────────────────
    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,          # use 1 worker; model is loaded in process memory
        log_config=None,    # structlog handles logging
        access_log=False,   # our middleware handles access logs
        reload=settings.debug,
    )
