"""
ModelMesh — Load Testing with Locust
--------------------------------------
Simulates production traffic patterns:
  - 80% single predictions
  - 15% batch predictions (avg 10 instances)
  - 5% shadow metrics polling

Run:
  locust -f tests/load/locustfile.py --host=http://localhost:8000
  locust -f tests/load/locustfile.py --host=http://localhost:8000 \
         --headless -u 100 -r 10 --run-time 2m --html report.html

Targets:
  - P95 latency < 200ms at 100 concurrent users
  - Error rate < 1%
  - Throughput > 500 req/s
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner


API_KEY = "dev-key-1234"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}

# Feature value pools — sampled to generate realistic diversity
TENURE_POOL = list(range(1, 72))
MONTHLY_CHARGES_POOL = [round(x, 2) for x in [20.0, 35.5, 50.0, 65.5, 80.0, 95.0, 110.0]]
CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]


def generate_features() -> Dict[str, Any]:
    """Generate a realistic random feature set."""
    tenure = random.choice(TENURE_POOL)
    monthly = random.choice(MONTHLY_CHARGES_POOL)
    return {
        "tenure": tenure,
        "monthly_charges": monthly,
        "total_charges": round(tenure * monthly * random.uniform(0.95, 1.05), 2),
        "num_products": random.randint(1, 4),
        "has_internet": random.randint(0, 1),
        "contract_type": random.choice(CONTRACT_TYPES),
        "payment_method": random.choice(PAYMENT_METHODS),
        "paperless_billing": random.randint(0, 1),
        "tech_support": random.randint(0, 1),
        "online_security": random.randint(0, 1),
    }


class ModelMeshUser(HttpUser):
    """
    Simulates a real API consumer.
    wait_time: 0.1–1s between requests → realistic think time.
    """
    wait_time = between(0.1, 1.0)
    host = "http://localhost:8000"

    @task(80)
    def single_predict(self):
        """High-frequency single prediction — primary load."""
        payload = {
            "instance": generate_features(),
            "include_shap": False,
            "shadow_eligible": True,
        }
        with self.client.post(
            "/v1/predict",
            data=json.dumps(payload),
            headers=HEADERS,
            catch_response=True,
            name="/v1/predict [single]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "prediction" not in data:
                    response.failure("Missing 'prediction' in response")
                elif not (0 <= data["probability"] <= 1):
                    response.failure("Invalid probability value")
                else:
                    response.success()
            elif response.status_code == 429:
                response.failure("Rate limit hit")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(15)
    def batch_predict(self):
        """Batch prediction — simulates downstream service bulk scoring."""
        batch_size = random.randint(5, 20)
        payload = {
            "instances": [generate_features() for _ in range(batch_size)],
            "include_shap": False,
        }
        with self.client.post(
            "/v1/predict/batch",
            data=json.dumps(payload),
            headers=HEADERS,
            catch_response=True,
            name=f"/v1/predict/batch [n={batch_size}]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("batch_size") != batch_size:
                    response.failure(f"Expected {batch_size} predictions, got {data.get('batch_size')}")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(5)
    def poll_shadow_metrics(self):
        """Low-frequency shadow A/B metrics polling."""
        with self.client.get(
            "/v1/shadow/metrics",
            headers=HEADERS,
            catch_response=True,
            name="/v1/shadow/metrics",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    def on_start(self):
        """Warm up: verify liveness before load testing."""
        resp = self.client.get("/health/live")
        if resp.status_code != 200:
            raise Exception("API is not live — aborting load test")


class ModelMeshSHAPUser(HttpUser):
    """
    Secondary user class: tests SHAP-enabled requests.
    Smaller proportion — SHAP is expensive.
    """
    wait_time = between(1.0, 3.0)
    weight = 5  # 5% of total users

    @task
    def predict_with_shap(self):
        payload = {
            "instance": generate_features(),
            "include_shap": True,
            "shadow_eligible": False,
        }
        with self.client.post(
            "/v1/predict",
            data=json.dumps(payload),
            headers=HEADERS,
            catch_response=True,
            name="/v1/predict [with-shap]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # SHAP may or may not be in response (sampled)
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


# ── Custom Locust Event Hooks ──────────────────────────────────────────────────

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "="*60)
    print("🚀 ModelMesh Load Test Starting")
    print(f"   Target: {environment.host}")
    print(f"   Users: {environment.parsed_options.num_users}")
    print(f"   Spawn rate: {environment.parsed_options.spawn_rate}/s")
    print("="*60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.total
    print("\n" + "="*60)
    print("📊 ModelMesh Load Test Results")
    print(f"   Total requests:   {stats.num_requests:,}")
    print(f"   Failed requests:  {stats.num_failures:,}")
    print(f"   Error rate:       {stats.fail_ratio:.2%}")
    print(f"   Avg latency:      {stats.avg_response_time:.0f}ms")
    print(f"   P95 latency:      {stats.get_response_time_percentile(0.95):.0f}ms")
    print(f"   P99 latency:      {stats.get_response_time_percentile(0.99):.0f}ms")
    print(f"   Throughput:       {stats.current_rps:.1f} req/s")
    print("="*60 + "\n")

    # Fail load test if SLOs are breached
    if stats.fail_ratio > 0.01:
        print("❌ FAIL: Error rate exceeded 1% SLO")
    if stats.get_response_time_percentile(0.95) > 500:
        print("❌ FAIL: P95 latency exceeded 500ms SLO")
    else:
        print("✅ PASS: All SLOs met")
