#!/usr/bin/env python3
"""
ModelMesh — Bootstrap Script
------------------------------
One-command setup for first-time local development:
  1. Generate sample datasets
  2. Train initial model with MLflow tracking
  3. Register model to MLflow registry
  4. Promote first version as champion
  5. Deploy Prefect pipeline schedules

Run: python scripts/bootstrap.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("PREFECT_API_URL", "http://localhost:4200/api")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_USER", "modelmesh")
os.environ.setdefault("POSTGRES_PASSWORD", "modelmesh_secret")
os.environ.setdefault("POSTGRES_DB", "modelmesh_db")


def step(n: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {n}: {title}")
    print(f"{'='*60}")


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=False)
    if check and result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def wait_for_service(name: str, url: str, max_wait: int = 60) -> bool:
    import httpx
    print(f"  Waiting for {name} at {url}...")
    for i in range(max_wait):
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code < 500:
                print(f"  ✅ {name} is ready")
                return True
        except Exception:
            pass
        time.sleep(1)
        if i % 10 == 9:
            print(f"  Still waiting... ({i+1}s)")
    print(f"  ⚠️  {name} not ready after {max_wait}s — continuing anyway")
    return False


if __name__ == "__main__":
    print("\n🚀 ModelMesh Bootstrap")
    print("   Setting up your local MLOps environment...\n")

    # ── Step 1: Generate sample data ──────────────────────────────────────────
    step(1, "Generate sample datasets")
    run("python scripts/generate_data.py")

    # ── Step 2: Wait for services ─────────────────────────────────────────────
    step(2, "Check infrastructure services")
    wait_for_service("MLflow", "http://localhost:5000/health", max_wait=30)
    wait_for_service("PostgreSQL", "http://localhost:5432", max_wait=5)
    wait_for_service("Prefect", "http://localhost:4200/api/health", max_wait=30)

    # ── Step 3: Initialize database ───────────────────────────────────────────
    step(3, "Initialize database schema")
    import asyncio
    from src.database.connection import init_db
    asyncio.run(init_db())
    print("  ✅ Database schema ready")

    # ── Step 4: Create MLflow experiment ──────────────────────────────────────
    step(4, "Set up MLflow experiment")
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.set_experiment("modelmesh-churn-detection")
    print(f"  ✅ Experiment ID: {experiment.experiment_id}")

    # ── Step 5: Train initial model ────────────────────────────────────────────
    step(5, "Train initial champion model")
    from src.training.train import train_and_register
    print("  Training XGBoost model with 5 Optuna trials (fast bootstrap)...")
    result = train_and_register(
        data_path=str(ROOT / "data" / "raw" / "train.csv"),
        model_type="xgboost",
        n_trials=5,
        auto_register=True,
    )
    print(f"  ✅ MLflow run_id : {result['run_id']}")
    print(f"  ✅ Model version : {result.get('model_version', 'N/A')}")

    # ── Step 6: Promote first version to champion ─────────────────────────────
    step(6, "Promote model version 1 → champion")
    from src.registry.model_registry import ModelRegistryClient
    registry = ModelRegistryClient()
    version = result.get("model_version", "1")
    try:
        registry.promote_challenger_to_champion(
            challenger_version=version,
            reason="bootstrap_initial_champion",
        )
        print(f"  ✅ Version {version} is now champion")
    except Exception as exc:
        print(f"  ⚠️  Promotion note: {exc}")

    # ── Step 7: Register Prefect deployments ──────────────────────────────────
    step(7, "Register Prefect pipeline deployments")
    run("python pipelines/prefect/deploy.py", check=False)

    # ── Done ───────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ✅ Bootstrap complete!")
    print("="*60)
    print("""
  Services:
    API docs     → http://localhost:8000/docs
    MLflow UI    → http://localhost:5000
    Prefect UI   → http://localhost:4200
    Grafana      → http://localhost:3000  (admin/admin)
    Prometheus   → http://localhost:9090

  Quick test:
    curl -X POST http://localhost:8000/v1/predict \\
      -H "X-API-Key: dev-key-1234" \\
      -H "Content-Type: application/json" \\
      -d @data/processed/test_payloads.json | python -m json.tool
""")
