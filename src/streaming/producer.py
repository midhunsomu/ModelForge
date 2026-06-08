"""
ModelMesh — Kafka Prediction Producer
---------------------------------------
Publishes every inference result to the 'modelmesh.predictions' topic.
The downstream Evidently drift monitor consumes from this topic.

Uses confluent-kafka (librdkafka bindings) for production-grade throughput.
Messages are JSON-encoded with a schema envelope for future schema registry.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from confluent_kafka import KafkaException, Producer

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class KafkaPredictionProducer:
    """
    Wraps confluent_kafka.Producer with:
    - Retry logic on transient errors
    - Delivery report callbacks for observability
    - Graceful flush on shutdown
    """

    def __init__(self) -> None:
        conf = {
            "bootstrap.servers": settings.kafka.bootstrap_servers,
            "client.id": "modelmesh-api-producer",
            "acks": "1",                    # leader ack only — balance durability vs latency
            "linger.ms": 5,                 # micro-batching for throughput
            "batch.size": 16384,
            "compression.type": "snappy",
            "retries": 3,
            "retry.backoff.ms": 200,
            "max.block.ms": settings.kafka.max_block_ms,
        }
        self._producer = Producer(conf)
        self._topic = settings.kafka.prediction_topic
        logger.info(
            "kafka_producer_initialized",
            servers=settings.kafka.bootstrap_servers,
            topic=self._topic,
        )

    def _delivery_report(self, err, msg) -> None:
        """Callback fired by librdkafka on message delivery or failure."""
        if err:
            logger.error(
                "kafka_delivery_failed",
                topic=msg.topic(),
                partition=msg.partition(),
                error=str(err),
            )
        else:
            logger.debug(
                "kafka_message_delivered",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )

    def send_prediction(
        self,
        *,
        trace_id: str,
        model_version: str,
        prediction: float,
        label: int,
        features: Dict[str, Any],
        model_stage: str = "champion",
    ) -> None:
        """
        Produce a prediction event to Kafka.
        Non-blocking — uses internal queue; librdkafka batches and sends async.
        """
        event = {
            "event_type": "prediction",
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": settings.mlflow.model_name,
            "model_version": model_version,
            "model_stage": model_stage,
            "prediction": prediction,
            "prediction_label": label,
            "features": features,
        }

        try:
            self._producer.produce(
                topic=self._topic,
                key=trace_id.encode(),
                value=json.dumps(event).encode(),
                on_delivery=self._delivery_report,
            )
            # Poll to trigger delivery callbacks — non-blocking
            self._producer.poll(0)
        except KafkaException as exc:
            logger.error("kafka_produce_error", error=str(exc), trace_id=trace_id)
        except BufferError:
            logger.warning("kafka_queue_full_dropping_message", trace_id=trace_id)

    def send_drift_alert(
        self,
        *,
        model_name: str,
        model_version: str,
        drift_share: float,
        drift_report_id: str,
    ) -> None:
        """Publish drift alert event to trigger retraining pipeline."""
        event = {
            "event_type": "drift_alert",
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "drift_share": drift_share,
            "drift_report_id": drift_report_id,
        }
        try:
            self._producer.produce(
                topic=settings.kafka.drift_alert_topic,
                key=model_name.encode(),
                value=json.dumps(event).encode(),
                on_delivery=self._delivery_report,
            )
            self._producer.poll(0)
        except KafkaException as exc:
            logger.error("kafka_drift_alert_error", error=str(exc))

    def close(self) -> None:
        """Flush all pending messages before shutdown."""
        remaining = self._producer.flush(timeout=10)
        if remaining > 0:
            logger.warning("kafka_flush_incomplete", remaining_messages=remaining)
        logger.info("kafka_producer_closed")
