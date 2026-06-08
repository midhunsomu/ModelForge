"""ModelMesh — Model Management Router"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from config.logging_config import get_logger
from config.settings import get_settings
from src.api.middleware.auth import verify_api_key
from src.api.schemas.inference import ModelInfo, ModelListResponse
from src.registry.model_registry import ModelRegistryClient

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()
registry = ModelRegistryClient()


@router.get("/", response_model=ModelListResponse)
async def list_models(_: str = Depends(verify_api_key)) -> ModelListResponse:
    versions = registry.list_versions()
    models = []
    for v in versions:
        metrics = {}
        try:
            metrics = registry.get_run_metrics(v.run_id)
        except Exception:
            pass
        models.append(ModelInfo(
            name=v.name,
            version=v.version,
            stage=v.current_stage,
            mlflow_run_id=v.run_id,
            metrics=metrics,
            created_at=None,
        ))
    return ModelListResponse(models=models, total=len(models))


@router.post("/{version}/promote")
async def promote_model(
    version: str,
    request: Request,
    _: str = Depends(verify_api_key),
) -> dict:
    """Manually promote a specific version to champion."""
    try:
        registry.promote_challenger_to_champion(
            challenger_version=version,
            reason="manual_promotion",
        )
        # Hot-reload champion in serving layer
        model_loader = request.app.state.model_loader
        await model_loader.reload_champion()
        logger.info("manual_promotion", version=version)
        return {"message": f"Version {version} promoted to champion"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{version}/rollback")
async def rollback_model(
    version: str,
    request: Request,
    _: str = Depends(verify_api_key),
) -> dict:
    """Emergency rollback to a specific version."""
    try:
        registry.rollback_to_version(version)
        model_loader = request.app.state.model_loader
        await model_loader.reload_champion()
        logger.warning("manual_rollback", version=version)
        return {"message": f"Rolled back to version {version}"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/shadow/pct")
async def set_shadow_pct(
    pct: float,
    request: Request,
    _: str = Depends(verify_api_key),
) -> dict:
    """Dynamically adjust shadow traffic percentage (0.0 – 1.0)."""
    if not 0.0 <= pct <= 1.0:
        raise HTTPException(status_code=400, detail="pct must be 0.0–1.0")
    model_loader = request.app.state.model_loader
    model_loader.shadow_router.set_shadow_pct(pct)
    return {"message": f"Shadow traffic set to {pct:.0%}"}


@router.get("/metrics/router")
async def router_metrics(
    request: Request,
    _: str = Depends(verify_api_key),
) -> dict:
    """Live shadow router A/B metrics snapshot."""
    model_loader = request.app.state.model_loader
    return model_loader.shadow_router.get_shadow_metrics_snapshot()
