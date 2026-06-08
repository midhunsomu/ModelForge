"""
ModelMesh — ML Training Pipeline
----------------------------------
Trains a churn-classification model with:
- Feature engineering
- Hyperparameter optimization (Optuna)
- MLflow experiment tracking (metrics, params, artifacts)
- Cross-validation
- Model registration to MLflow registry

Run standalone or called from Prefect pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.logging_config import configure_logging, get_logger
from config.settings import get_settings

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-ready features from raw telco churn dataset.
    All transformations are deterministic — no leakage.
    """
    df = df.copy()

    # Encode binary categoricals
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["paperless_billing"] = (df["PaperlessBilling"] == "Yes").astype(int)
    df["tech_support"] = (df["TechSupport"] == "Yes").astype(int)
    df["online_security"] = (df["OnlineSecurity"] == "Yes").astype(int)

    # Encode multi-class categoricals as ordinal
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_map = {
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3,
    }
    df["contract_type_encoded"] = df["Contract"].map(contract_map).fillna(0).astype(int)
    df["payment_method_encoded"] = df["PaymentMethod"].map(payment_map).fillna(0).astype(int)

    # Numeric cleanup
    df["total_charges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["monthly_charges"] = df["MonthlyCharges"].astype(float)
    df["tenure"] = df["tenure"].astype(int)
    df["num_products"] = df.get("NumProducts", pd.Series(1, index=df.index)).astype(int)

    # Target
    df["churn"] = (df["Churn"] == "Yes").astype(int)

    feature_cols = [
        "tenure", "monthly_charges", "total_charges", "num_products",
        "has_internet", "contract_type_encoded", "payment_method_encoded",
        "paperless_billing", "tech_support", "online_security",
    ]
    return df[feature_cols + ["churn"]]


def load_and_prepare_data(
    data_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load data, engineer features, train/test split."""
    if data_path and os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        logger.info("data_loaded", rows=len(raw_df), path=data_path)
    else:
        logger.info("generating_synthetic_training_data")
        raw_df = _generate_synthetic_dataset(n=7043)

    df = engineer_features(raw_df)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(
        "data_split",
        train_rows=len(X_train),
        test_rows=len(X_test),
        churn_rate=round(y.mean(), 3),
    )
    return X_train, X_test, y_train, y_test


def _generate_synthetic_dataset(n: int = 7043) -> pd.DataFrame:
    """Generate realistic synthetic telco churn dataset."""
    np.random.seed(42)
    contracts = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )
    tenure = np.where(
        contracts == "Month-to-month",
        np.random.randint(1, 24, n),
        np.where(contracts == "One year",
                 np.random.randint(12, 48, n),
                 np.random.randint(24, 72, n)),
    )
    monthly = np.random.uniform(20, 120, n)
    total = tenure * monthly + np.random.normal(0, 50, n)
    total = np.clip(total, 0, None)

    # Churn probability increases with month-to-month, high charges, low tenure
    churn_prob = (
        0.05
        + 0.30 * (contracts == "Month-to-month").astype(float)
        + 0.15 * (monthly > 80).astype(float)
        - 0.20 * (tenure > 36).astype(float)
        + np.random.normal(0, 0.05, n)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total.astype(str),
        "NumProducts": np.random.randint(1, 5, n),
        "InternetService": np.random.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22]),
        "Contract": contracts,
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.34, 0.22, 0.22, 0.22]
        ),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
        "Churn": np.where(churn == 1, "Yes", "No"),
    })


# ── Model Definition ───────────────────────────────────────────────────────────

def get_model_and_params(trial: optuna.Trial, model_type: str = "xgboost") -> Any:
    """Build model with Optuna-suggested hyperparameters."""
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            scale_pos_weight=2.5,    # handle class imbalance
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 4, 16),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        return LogisticRegression(
            C=trial.suggest_float("C", 0.001, 100, log=True),
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )


# ── Training Orchestrator ──────────────────────────────────────────────────────

class ModelTrainer:
    """
    Orchestrates training with MLflow experiment tracking.
    Supports hyperparameter optimization with Optuna.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        n_optuna_trials: int = 20,
        cv_folds: int = 5,
    ) -> None:
        self.model_type = model_type
        self.n_trials = n_optuna_trials
        self.cv_folds = cv_folds
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> float:
        """Optuna objective: maximize cross-validated F1."""
        model = get_model_and_params(trial, self.model_type)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="f1", n_jobs=-1
        )
        return float(scores.mean())

    def train(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        data_hash: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> str:
        """
        Full training run. Returns MLflow run_id.
        Logs all metrics, params, and model artifact to MLflow.
        """
        logger.info("training_started", model_type=self.model_type, n_trials=self.n_trials)

        # ── Optuna HPO ────────────────────────────────────────────────────────
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        logger.info(
            "hpo_complete",
            best_f1=round(study.best_value, 4),
            best_params=best_params,
        )

        # ── Final training with best params ───────────────────────────────────
        with mlflow.start_run(run_name=run_name or f"train-{self.model_type}") as run:
            run_id = run.info.run_id

            # Build and train final model
            final_model = get_model_and_params(
                optuna.trial.FixedTrial(best_params), self.model_type
            )
            t0 = time.time()
            final_model.fit(X_train, y_train)
            train_time = time.time() - t0

            # ── Evaluate ──────────────────────────────────────────────────────
            y_pred = final_model.predict(X_test)
            y_proba = final_model.predict_proba(X_test)[:, 1]

            metrics = {
                "f1_score": float(f1_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "train_time_seconds": round(train_time, 2),
                "hpo_best_cv_f1": round(study.best_value, 4),
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "churn_rate_train": round(float(y_train.mean()), 4),
            }

            logger.info("model_evaluation", **{k: round(v, 4) for k, v in metrics.items()})

            # ── MLflow Logging ────────────────────────────────────────────────
            mlflow.log_params({
                "model_type": self.model_type,
                "n_optuna_trials": self.n_trials,
                "cv_folds": self.cv_folds,
                "training_data_hash": data_hash or "unknown",
                **best_params,
            })
            mlflow.log_metrics(metrics)

            # Log feature importance
            if hasattr(final_model, "feature_importances_"):
                importance_dict = dict(
                    zip(X_train.columns.tolist(), final_model.feature_importances_)
                )
                mlflow.log_dict(importance_dict, "feature_importance.json")

            # Log HPO study summary
            optuna_summary = {
                "n_trials": len(study.trials),
                "best_trial": study.best_trial.number,
                "best_cv_f1": study.best_value,
                "all_params": best_params,
            }
            mlflow.log_dict(optuna_summary, "hpo_summary.json")

            # Log model + input signature
            from mlflow.models import infer_signature
            signature = infer_signature(X_train, y_proba)
            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(3),
                registered_model_name=None,   # register separately
            )

            # Save reference features for drift monitoring
            ref_path = DATA_DIR / "reference" / "reference_features.parquet"
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            X_train.to_parquet(ref_path, index=False)
            mlflow.log_artifact(str(ref_path), "reference_data")

            mlflow.set_tags({
                "model_type": self.model_type,
                "environment": settings.environment,
                "feature_count": len(X_train.columns),
            })

            logger.info("mlflow_run_complete", run_id=run_id, metrics=metrics)
            return run_id


def train_and_register(
    data_path: Optional[str] = None,
    model_type: str = "xgboost",
    n_trials: int = 20,
    auto_register: bool = True,
) -> Dict[str, Any]:
    """
    Entry point for Prefect task / CLI.
    Trains model and optionally registers to MLflow model registry.
    """
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_path)

    # Compute data fingerprint for reproducibility tracking
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(X_train).values.tobytes()
    ).hexdigest()

    trainer = ModelTrainer(model_type=model_type, n_optuna_trials=n_trials)
    run_id = trainer.train(
        X_train, X_test, y_train, y_test,
        data_hash=data_hash,
        run_name=f"modelmesh-{model_type}-{data_hash[:6]}",
    )

    result = {"run_id": run_id, "registered": False}

    if auto_register:
        from src.registry.model_registry import ModelRegistryClient
        registry = ModelRegistryClient()
        version = registry.register_new_model(
            run_id=run_id,
            artifact_path="model",
            tags={"data_hash": data_hash, "model_type": model_type},
        )
        result["registered"] = True
        result["model_version"] = version.version
        logger.info("model_registered_post_training", version=version.version)

    return result


if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    result = train_and_register(data_path=data_file, n_trials=5)
    print(json.dumps(result, indent=2))
