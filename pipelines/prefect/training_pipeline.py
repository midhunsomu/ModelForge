"""
ModelMesh — Prefect Training + Retraining Pipeline
----------------------------------------------------
Two flows:
1. modelmesh_train_flow       — initial / scheduled training
2. modelmesh_retrain_flow     — triggered by drift alert

Each flow is broken into Prefect tasks for:
- Granular retry control
- Per-task logging + caching
- Parallel execution where safe
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Dict, Optional

from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash

from config.settings import get_settings

settings = get_settings()


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(
    name="load-and-prepare-data",
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=6),
)
def task_load_data(data_path: Optional[str] = None) -> Dict:
    from src.training.train import load_and_prepare_data
    import hashlib, pandas as pd

    X_train, X_test, y_train, y_test = load_and_prepare_data(data_path)
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(X_train).values.tobytes()
    ).hexdigest()
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "data_hash": data_hash,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


@task(
    name="train-model",
    retries=1,
    retry_delay_seconds=60,
    timeout_seconds=3600,
)
def task_train_model(
    data: Dict,
    model_type: str = "xgboost",
    n_trials: int = 20,
) -> str:
    from src.training.train import ModelTrainer

    logger = get_run_logger()
    logger.info(f"Training {model_type} with {n_trials} HPO trials on {data['n_train']} rows")

    trainer = ModelTrainer(model_type=model_type, n_optuna_trials=n_trials)
    run_id = trainer.train(
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
        data_hash=data["data_hash"],
        run_name=f"prefect-{model_type}",
    )
    logger.info(f"MLflow run_id: {run_id}")
    return run_id


@task(
    name="register-model",
    retries=3,
    retry_delay_seconds=10,
)
def task_register_model(run_id: str, data_hash: str, model_type: str) -> str:
    from src.registry.model_registry import ModelRegistryClient

    logger = get_run_logger()
    registry = ModelRegistryClient()
    version = registry.register_new_model(
        run_id=run_id,
        tags={"data_hash": data_hash, "model_type": model_type},
    )
    logger.info(f"Registered model version: {version.version}")
    return version.version


@task(name="evaluate-challenger-vs-champion", retries=1)
def task_compare_models(
    new_version: str,
    data: Dict,
    min_improvement_pct: float = 0.02,
) -> Dict[str, Any]:
    """
    Compare challenger (new_version) vs champion (current Production).
    Returns comparison result and promotion decision.
    """
    import mlflow
    from sklearn.metrics import f1_score

    logger = get_run_logger()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    from src.registry.model_registry import ModelRegistryClient
    registry = ModelRegistryClient()

    # Load challenger
    challenger_model, _ = registry.load_challenger()
    challenger_preds = challenger_model.predict(data["X_test"])
    challenger_f1 = float(f1_score(data["y_test"], challenger_preds))

    # Load champion (may not exist on first run)
    champion_f1 = 0.0
    try:
        champion_model, _ = registry.load_champion()
        champion_preds = champion_model.predict(data["X_test"])
        champion_f1 = float(f1_score(data["y_test"], champion_preds))
    except Exception:
        logger.warning("No champion model found — challenger auto-wins")
        champion_f1 = 0.0

    improvement = challenger_f1 - champion_f1
    improvement_pct = improvement / max(champion_f1, 0.0001)
    should_promote = improvement_pct >= min_improvement_pct

    logger.info(
        f"Champion F1={champion_f1:.4f}, Challenger F1={challenger_f1:.4f}, "
        f"Δ={improvement_pct:.2%} → promote={should_promote}"
    )

    return {
        "champion_f1": champion_f1,
        "challenger_f1": challenger_f1,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "should_promote": should_promote,
        "new_version": new_version,
    }


@task(name="promote-model", retries=2)
def task_promote_model(comparison: Dict, trigger_type: str = "pipeline") -> bool:
    from src.registry.model_registry import ModelRegistryClient

    logger = get_run_logger()

    if not comparison["should_promote"]:
        logger.info(
            f"Promotion skipped: improvement {comparison['improvement_pct']:.2%} "
            f"below threshold"
        )
        return False

    registry = ModelRegistryClient()
    registry.promote_challenger_to_champion(
        challenger_version=comparison["new_version"],
        reason=f"{trigger_type}_auto_promotion_f1+{comparison['improvement_pct']:.2%}",
    )
    logger.info(
        f"✅ Promoted version {comparison['new_version']} to champion "
        f"(F1: {comparison['champion_f1']:.4f} → {comparison['challenger_f1']:.4f})"
    )
    return True


@task(name="notify-slack", retries=1)
def task_notify(
    comparison: Dict,
    promoted: bool,
    trigger_type: str,
    webhook_url: Optional[str] = None,
) -> None:
    """Send Slack notification on retrain completion."""
    import httpx

    logger = get_run_logger()
    status = "✅ PROMOTED" if promoted else "⏭️ SKIPPED PROMOTION"
    message = (
        f"*ModelMesh Retrain [{trigger_type}]* — {status}\n"
        f"Champion F1: `{comparison['champion_f1']:.4f}` → "
        f"Challenger F1: `{comparison['challenger_f1']:.4f}` "
        f"(Δ {comparison['improvement_pct']:+.2%})\n"
        f"New version: `{comparison['new_version']}`"
    )

    logger.info(message)

    slack_url = webhook_url or settings.retraining.__dict__.get("slack_webhook_url")
    if slack_url:
        try:
            httpx.post(slack_url, json={"text": message}, timeout=5)
        except Exception as exc:
            logger.warning(f"Slack notify failed: {exc}")


# ── Flows ──────────────────────────────────────────────────────────────────────

@flow(
    name="modelmesh-train",
    description="Initial / scheduled full training pipeline",
    retries=1,
    retry_delay_seconds=120,
)
def modelmesh_train_flow(
    data_path: Optional[str] = None,
    model_type: str = "xgboost",
    n_trials: int = 20,
    auto_promote: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end training pipeline:
    data → train → register → compare → (optionally) promote
    """
    logger = get_run_logger()
    logger.info(f"🚀 Starting train flow: model_type={model_type}")

    data = task_load_data(data_path)
    run_id = task_train_model(data, model_type, n_trials)
    version = task_register_model(run_id, data["data_hash"], model_type)
    comparison = task_compare_models(version, data)

    promoted = False
    if auto_promote or settings.retraining.auto_promote:
        promoted = task_promote_model(comparison, trigger_type="scheduled")

    task_notify(comparison, promoted, trigger_type="scheduled")

    return {
        "run_id": run_id,
        "version": version,
        "promoted": promoted,
        "champion_f1": comparison["champion_f1"],
        "challenger_f1": comparison["challenger_f1"],
    }


@flow(
    name="modelmesh-retrain",
    description="Drift-triggered automated retraining pipeline",
    retries=1,
    retry_delay_seconds=300,
)
def modelmesh_retrain_flow(
    model_name: str = "churn-classifier",
    drift_report_id: Optional[str] = None,
    triggered_by: str = "drift_alert",
    data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Automated retraining triggered by Evidently drift alert.
    More conservative than train_flow: promotes ONLY if F1 improvement > threshold.
    """
    logger = get_run_logger()
    logger.info(
        f"🔄 Retrain triggered by: {triggered_by}, "
        f"drift_report_id={drift_report_id}"
    )

    data = task_load_data(data_path)
    run_id = task_train_model(data, "xgboost", n_trials=15)  # fewer trials for speed
    version = task_register_model(run_id, data["data_hash"], "xgboost")

    comparison = task_compare_models(
        version,
        data,
        min_improvement_pct=settings.retraining.min_improvement_pct,
    )

    # Retrain flow is more aggressive — always promote if improvement exists
    promoted = task_promote_model(comparison, trigger_type=triggered_by)
    task_notify(comparison, promoted, trigger_type=triggered_by)

    return {
        "run_id": run_id,
        "version": version,
        "promoted": promoted,
        "drift_report_id": drift_report_id,
        "challenger_f1": comparison["challenger_f1"],
        "improvement_pct": comparison["improvement_pct"],
    }


if __name__ == "__main__":
    # Local test run
    result = modelmesh_train_flow(n_trials=3, auto_promote=False)
    print(result)
