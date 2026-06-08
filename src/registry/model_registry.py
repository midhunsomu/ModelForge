"""
ModelMesh — MLflow Model Registry Client
-----------------------------------------
Wraps MLflow's model registry API with:
- Champion/challenger alias management
- Automatic staging → production promotion
- Rollback support
- Health-check based loading
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from config.logging_config import get_logger
from config.settings import ModelStage, get_settings

logger = get_logger(__name__)
settings = get_settings()


class ModelRegistryClient:
    """
    Wraps MLflow's client with ModelMesh-specific promotion logic.

    Champion/Challenger pattern:
    - 'champion' alias → the currently live production model
    - 'challenger' alias → the most recent staged model for shadow testing
    When challenger beats champion by > min_improvement_pct, it can be
    auto-promoted to champion.
    """

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        self.client = MlflowClient(
            tracking_uri=settings.mlflow.tracking_uri,
            registry_uri=settings.mlflow.registry_uri,
        )
        self.model_name = settings.mlflow.model_name

    # ── Version Retrieval ──────────────────────────────────────────────────────

    def get_champion_version(self) -> Optional[ModelVersion]:
        """Fetch the version tagged with the 'champion' alias."""
        try:
            return self.client.get_model_version_by_alias(
                name=self.model_name,
                alias=settings.serving.champion_model_alias,
            )
        except Exception as exc:
            # Fallback: get latest Production version
            logger.warning("champion_alias_missing", error=str(exc))
            return self._get_latest_by_stage(ModelStage.PRODUCTION)

    def get_challenger_version(self) -> Optional[ModelVersion]:
        """Fetch the version tagged with the 'challenger' alias."""
        try:
            return self.client.get_model_version_by_alias(
                name=self.model_name,
                alias=settings.serving.challenger_model_alias,
            )
        except Exception:
            return self._get_latest_by_stage(ModelStage.STAGING)

    def _get_latest_by_stage(self, stage: ModelStage) -> Optional[ModelVersion]:
        versions = self.client.get_latest_versions(
            name=self.model_name, stages=[stage.value]
        )
        return versions[0] if versions else None

    # ── Model Loading ──────────────────────────────────────────────────────────

    def load_model(self, version: ModelVersion) -> Any:
        """Load a sklearn/pyfunc model by version."""
        model_uri = f"models:/{self.model_name}/{version.version}"
        t0 = time.perf_counter()
        model = mlflow.sklearn.load_model(model_uri)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "model_loaded",
            model=self.model_name,
            version=version.version,
            stage=version.current_stage,
            load_time_ms=round(elapsed, 2),
        )
        return model

    def load_champion(self) -> Tuple[Any, str]:
        """Returns (model, version_str)."""
        version = self.get_champion_version()
        if not version:
            raise RuntimeError(f"No champion model found for '{self.model_name}'")
        return self.load_model(version), version.version

    def load_challenger(self) -> Tuple[Optional[Any], Optional[str]]:
        """Returns (model, version_str) or (None, None)."""
        version = self.get_challenger_version()
        if not version:
            logger.info("no_challenger_model_found")
            return None, None
        return self.load_model(version), version.version

    # ── Promotion Logic ────────────────────────────────────────────────────────

    def promote_challenger_to_champion(
        self,
        challenger_version: str,
        reason: str = "auto_promotion",
    ) -> None:
        """
        Demote current champion → Archived, promote challenger → Production.
        Also swaps the 'champion'/'challenger' aliases.
        """
        # 1. Archive current champion
        current_champion = self.get_champion_version()
        if current_champion:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=current_champion.version,
                stage=ModelStage.ARCHIVED.value,
                archive_existing_versions=False,
            )
            logger.info(
                "champion_archived",
                version=current_champion.version,
            )

        # 2. Promote challenger → Production
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=challenger_version,
            stage=ModelStage.PRODUCTION.value,
        )

        # 3. Update aliases
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=settings.serving.champion_model_alias,
            version=challenger_version,
        )
        if current_champion:
            try:
                self.client.delete_registered_model_alias(
                    name=self.model_name,
                    alias=settings.serving.challenger_model_alias,
                )
            except Exception:
                pass

        # 4. Add promotion tag
        self.client.set_model_version_tag(
            name=self.model_name,
            version=challenger_version,
            key="promotion_reason",
            value=reason,
        )
        logger.info(
            "challenger_promoted_to_champion",
            new_champion_version=challenger_version,
            reason=reason,
        )

    def rollback_to_version(self, version: str) -> None:
        """Emergency rollback: set a specific version as champion."""
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=settings.serving.champion_model_alias,
            version=version,
        )
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=ModelStage.PRODUCTION.value,
        )
        logger.warning("model_rollback_executed", target_version=version)

    # ── Registration ──────────────────────────────────────────────────────────

    def register_new_model(
        self,
        run_id: str,
        artifact_path: str = "model",
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        """Register a trained model from an MLflow run."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_details = mlflow.register_model(
            model_uri=model_uri, name=self.model_name
        )
        if tags:
            for k, v in tags.items():
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=model_details.version,
                    key=k,
                    value=str(v),
                )

        # Auto-transition to Staging
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=model_details.version,
            stage=ModelStage.STAGING.value,
        )
        # Set challenger alias
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=settings.serving.challenger_model_alias,
            version=model_details.version,
        )
        logger.info(
            "model_registered",
            version=model_details.version,
            run_id=run_id,
        )
        return model_details

    def list_versions(self, stages: Optional[List[str]] = None) -> List[ModelVersion]:
        if stages is None:
            stages = ["None", "Staging", "Production", "Archived"]
        return self.client.get_latest_versions(name=self.model_name, stages=stages)

    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        run = self.client.get_run(run_id)
        return dict(run.data.metrics)
