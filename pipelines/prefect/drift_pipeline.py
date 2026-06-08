"""ModelMesh — Prefect Drift Check Flow"""

from __future__ import annotations

import asyncio
from typing import Dict, Any

from prefect import flow, get_run_logger, task


@task(name="run-drift-check", retries=2, retry_delay_seconds=30)
def task_run_drift_check() -> Dict[str, Any]:
    result = asyncio.run(_async_drift_check())
    logger = get_run_logger()
    logger.info(f"Drift check result: {result}")
    return result


async def _async_drift_check() -> Dict:
    from src.database.connection import init_db
    from src.monitoring.drift_monitor import DriftMonitorService
    await init_db()
    service = DriftMonitorService()
    return await service.run_check()


@flow(
    name="modelmesh-drift-check",
    description="Periodic data drift monitoring",
    retries=1,
)
def modelmesh_drift_check_flow() -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("🔍 Running drift check...")
    result = task_run_drift_check()
    if result.get("alert_triggered"):
        logger.warning(f"⚠️ Drift alert! share={result.get('drift_share'):.2%}")
    return result


if __name__ == "__main__":
    modelmesh_drift_check_flow()
