"""
ModelMesh — Prefect Deployment Configuration
---------------------------------------------
Registers flows as deployments with schedules.
Run once: python pipelines/prefect/deploy.py
"""

from prefect import serve
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
from datetime import timedelta

from pipelines.prefect.training_pipeline import (
    modelmesh_retrain_flow,
    modelmesh_train_flow,
)
from pipelines.prefect.drift_pipeline import modelmesh_drift_check_flow


def deploy_all():
    # Weekly scheduled retraining (Sundays at 2am)
    train_deployment = Deployment.build_from_flow(
        flow=modelmesh_train_flow,
        name="production",
        schedule=CronSchedule(cron="0 2 * * 0"),
        parameters={
            "model_type": "xgboost",
            "n_trials": 30,
            "auto_promote": False,
        },
        tags=["production", "training"],
        description="Weekly scheduled retraining of churn model",
    )

    # On-demand retrain (triggered by drift alerts)
    retrain_deployment = Deployment.build_from_flow(
        flow=modelmesh_retrain_flow,
        name="production",
        parameters={
            "model_name": "churn-classifier",
            "triggered_by": "drift_alert",
        },
        tags=["production", "retraining"],
        description="Drift-triggered automated retraining",
    )

    # Drift check every hour
    drift_deployment = Deployment.build_from_flow(
        flow=modelmesh_drift_check_flow,
        name="production",
        schedule=IntervalSchedule(interval=timedelta(minutes=60)),
        tags=["production", "monitoring"],
        description="Hourly drift monitoring check",
    )

    train_deployment.apply()
    retrain_deployment.apply()
    drift_deployment.apply()

    print("✅ All deployments registered with Prefect server")


if __name__ == "__main__":
    deploy_all()
