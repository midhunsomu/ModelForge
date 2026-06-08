"""
ModelMesh — Sample Dataset Generator
--------------------------------------
Generates realistic synthetic telco churn datasets for:
  1. Initial model training  (data/raw/train.csv)
  2. Drift simulation        (data/raw/drifted.csv)  ← shifted distribution
  3. Reference snapshot      (data/reference/reference_features.parquet)
  4. API test payloads       (data/processed/test_payloads.json)

Run: python scripts/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_REF = ROOT / "data" / "reference"

for d in [DATA_RAW, DATA_PROCESSED, DATA_REF]:
    d.mkdir(parents=True, exist_ok=True)


def generate_base_dataset(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Mimics the real IBM Telco Customer Churn dataset distribution."""
    np.random.seed(seed)

    contracts = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n, p=[0.55, 0.25, 0.20],
    )
    tenure = np.where(
        contracts == "Month-to-month", np.random.randint(1, 24, n),
        np.where(contracts == "One year", np.random.randint(12, 48, n),
                 np.random.randint(24, 72, n))
    )
    internet = np.random.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22])
    monthly = np.where(
        internet == "Fiber optic", np.random.uniform(70, 120, n),
        np.where(internet == "DSL", np.random.uniform(25, 70, n),
                 np.random.uniform(18, 30, n))
    )
    total = tenure * monthly + np.random.normal(0, 30, n)
    total = np.clip(total, 0, None)

    churn_prob = (
        0.08
        + 0.28 * (contracts == "Month-to-month").astype(float)
        + 0.18 * (internet == "Fiber optic").astype(float)
        + 0.15 * (monthly > 80).astype(float)
        - 0.25 * (tenure > 36).astype(float)
        - 0.10 * (contracts == "Two year").astype(float)
        + np.random.normal(0, 0.05, n)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:06d}" for i in range(n)],
        "tenure": tenure.astype(int),
        "MonthlyCharges": np.round(monthly, 2),
        "TotalCharges": np.round(total, 2).astype(str),
        "NumProducts": np.random.randint(1, 5, n),
        "InternetService": internet,
        "Contract": contracts,
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.34, 0.22, 0.22, 0.22]
        ),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "TechSupport": np.where(
            internet == "No", "No internet service",
            np.random.choice(["Yes", "No"], n, p=[0.37, 0.63])
        ),
        "OnlineSecurity": np.where(
            internet == "No", "No internet service",
            np.random.choice(["Yes", "No"], n, p=[0.29, 0.71])
        ),
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    print(f"[train] rows={len(df)}, churn_rate={churn.mean():.2%}")
    return df


def generate_drifted_dataset(n: int = 2000, seed: int = 99) -> pd.DataFrame:
    """
    Simulate production drift:
    - More month-to-month customers (covariate shift)
    - Higher charges (price increase simulation)
    - Lower tenure (newer customer base)
    This should trigger Evidently drift detection.
    """
    np.random.seed(seed)

    contracts = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n, p=[0.80, 0.15, 0.05],  # ← shifted: more month-to-month
    )
    tenure = np.random.randint(1, 12, n)  # ← shifted: all new customers

    internet = np.random.choice(["Fiber optic", "DSL", "No"], n, p=[0.75, 0.20, 0.05])

    # Charges shifted up — simulating a price increase
    monthly = np.where(
        internet == "Fiber optic", np.random.uniform(100, 150, n),
        np.where(internet == "DSL", np.random.uniform(60, 100, n),
                 np.random.uniform(30, 50, n))
    )
    total = tenure * monthly + np.random.normal(0, 20, n)

    churn_prob = np.clip(0.35 + np.random.normal(0, 0.08, n), 0.1, 0.95)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"DRIFT-{i:06d}" for i in range(n)],
        "tenure": tenure.astype(int),
        "MonthlyCharges": np.round(monthly, 2),
        "TotalCharges": np.round(total, 2).astype(str),
        "NumProducts": np.random.randint(1, 3, n),
        "InternetService": internet,
        "Contract": contracts,
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.55, 0.15, 0.15, 0.15]  # more electronic check
        ),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n, p=[0.80, 0.20]),
        "TechSupport": np.where(
            internet == "No", "No internet service",
            np.random.choice(["Yes", "No"], n, p=[0.15, 0.85])
        ),
        "OnlineSecurity": np.where(
            internet == "No", "No internet service",
            np.random.choice(["Yes", "No"], n, p=[0.12, 0.88])
        ),
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    print(f"[drifted] rows={len(df)}, churn_rate={churn.mean():.2%}")
    return df


def generate_reference_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Save encoded feature matrix (post-engineering) as reference for Evidently.
    This is loaded by DriftMonitor at runtime.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from src.training.train import engineer_features

    df = engineer_features(train_df)
    feature_cols = [
        "tenure", "monthly_charges", "total_charges", "num_products",
        "has_internet", "contract_type_encoded", "payment_method_encoded",
        "paperless_billing", "tech_support", "online_security",
    ]
    return df[feature_cols].head(5000)


def generate_api_test_payloads(n: int = 10) -> list:
    """Generate ready-to-use JSON payloads for API testing."""
    np.random.seed(0)
    payloads = []
    contracts = ["Month-to-month", "One year", "Two year"]
    payments = ["Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"]

    for i in range(n):
        tenure = int(np.random.randint(1, 72))
        monthly = round(float(np.random.uniform(20, 120)), 2)
        payloads.append({
            "instance": {
                "tenure": tenure,
                "monthly_charges": monthly,
                "total_charges": round(tenure * monthly, 2),
                "num_products": int(np.random.randint(1, 4)),
                "has_internet": int(np.random.randint(0, 2)),
                "contract_type": contracts[i % 3],
                "payment_method": payments[i % 4],
                "paperless_billing": int(np.random.randint(0, 2)),
                "tech_support": int(np.random.randint(0, 2)),
                "online_security": int(np.random.randint(0, 2)),
            },
            "include_shap": i % 3 == 0,  # every 3rd request includes SHAP
            "shadow_eligible": True,
        })
    return payloads


if __name__ == "__main__":
    print("Generating ModelMesh sample datasets...\n")

    # 1. Training data
    train_df = generate_base_dataset(n=7043)
    train_path = DATA_RAW / "train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"  ✅ Training data → {train_path}")

    # 2. Drifted data (for drift simulation)
    drift_df = generate_drifted_dataset(n=2000)
    drift_path = DATA_RAW / "drifted.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"  ✅ Drifted data  → {drift_path}")

    # 3. Reference features parquet
    ref_df = generate_reference_features(train_df)
    ref_path = DATA_REF / "reference_features.parquet"
    ref_df.to_parquet(ref_path, index=False)
    print(f"  ✅ Reference     → {ref_path}  (shape={ref_df.shape})")

    # 4. API test payloads
    payloads = generate_api_test_payloads(n=20)
    payloads_path = DATA_PROCESSED / "test_payloads.json"
    with open(payloads_path, "w") as f:
        json.dump(payloads, f, indent=2)
    print(f"  ✅ Test payloads → {payloads_path}  ({len(payloads)} examples)")

    print("\n✅ All datasets generated successfully!")
    print(f"\nDataset summary:")
    print(f"  Training rows  : {len(train_df):,}")
    print(f"  Drifted rows   : {len(drift_df):,}")
    print(f"  Reference rows : {len(ref_df):,}")
    print(f"  Test payloads  : {len(payloads)}")
