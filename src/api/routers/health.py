"""ModelMesh — Health Check Endpoints (K8s liveness + readiness probes)"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config.settings import get_settings
from src.api.schemas.inference import HealthResponse, ReadinessResponse, ServiceStatus
from src.database.connection import check_db_health

settings = get_settings()
router = APIRouter()


@router.get("/live", summary="Liveness probe", status_code=200)
async def liveness() -> dict:
    """K8s liveness probe — returns 200 if process is alive."""
    return {"status": "alive"}


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness(request: Request) -> ReadinessResponse:
    """
    K8s readiness probe — returns 200 only when all critical deps are ready.
    Pod won't receive traffic until this passes.
    """
    model_loader = getattr(request.app.state, "model_loader", None)
    champion_loaded = (
        model_loader is not None and model_loader.champion_version is not None
    )
    challenger_loaded = (
        model_loader is not None and model_loader.challenger_version is not None
    )

    db_health = await check_db_health()
    db_ok = db_health["status"] == "healthy"

    ready = champion_loaded and db_ok

    return ReadinessResponse(
        ready=ready,
        champion_model_loaded=champion_loaded,
        challenger_model_loaded=challenger_loaded,
        database_connected=db_ok,
    )


@router.get("/", response_model=HealthResponse, summary="Detailed health status")
async def health_check(request: Request) -> HealthResponse:
    """Full health report with per-service status."""
    services = []

    # Database
    db_health = await check_db_health()
    services.append(ServiceStatus(
        name="postgresql",
        status=db_health["status"],
        details=db_health.get("error"),
    ))

    # Kafka
    kafka_producer = getattr(request.app.state, "kafka_producer", None)
    kafka_status = "healthy" if kafka_producer else "disabled"
    services.append(ServiceStatus(name="kafka", status=kafka_status))

    # Model Registry
    model_loader = getattr(request.app.state, "model_loader", None)
    services.append(ServiceStatus(
        name="model_registry",
        status="healthy" if model_loader and model_loader.champion_version else "degraded",
        details=f"champion={model_loader.champion_version}" if model_loader else None,
    ))

    overall = (
        "healthy"
        if all(s.status in ("healthy", "disabled") for s in services)
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.utcnow(),
        services=services,
    )
