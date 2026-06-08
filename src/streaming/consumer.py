"""
ModelMesh — Kafka Prediction Consumer
---------------------------------------
Consumes prediction events and accumulates them for drift analysis.
Runs as a standalone process (not inside the FastAPI server).

Architecture:
  Kafka topic → Consumer → Feature buffer → Evidently drift check
  On drift → Kafka drift_alert topic → Prefect retrain trigger
"""

from __future__ import annotations

import json
import signal
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from confluent_kafka import Consumer, KafkaException, Message

from config.logging_config import configure_logging, get_logger
from config.settings import get_settings

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)


class PredictionConsumer:
    """
    Consumes from modelmesh.predictions topic.
    Maintains a rolling buffer of recent feature vectors for drift analysis.
    The DriftMonitorService polls this buffer periodically.
    """

    def __init__(
        self,
        buffer_size: int = 5000,
        on_drift_check_interval: int = 100,   # check drift every N messages
    ) -> None:
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)
        self._check_interval = on_drift_check_interval
        self._message_count = 0
        self._running = False

        conf = {
            "bootstrap.servers": settings.kafka.bootstrap_servers,
            "group.id": settings.kafka.consumer_group,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 30000,
        }
        self._consumer = Consumer(conf)
        self._consumer.subscribe([settings.kafka.prediction_topic])
        logger.info(
            "prediction_consumer_initialized",
            topic=settings.kafka.prediction_topic,
            group=settings.kafka.consumer_group,
        )

    def get_recent_features(
        self, n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return most recent N feature dicts for drift analysis."""
        items = list(self._buffer)
        return items[-n:] if n else items

    def run(self, on_window_full=None) -> None:
        """
        Main consume loop. Runs until SIGTERM/SIGINT.

        Args:
            on_window_full: optional async callback when buffer is full
        """
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        logger.info("consumer_loop_started")

        while self._running:
            try:
                msg: Optional[Message] = self._consumer.poll(timeout=1.0)

                if msg is None:
                    continue
                if msg.error():
                    logger.error("kafka_consume_error", error=str(msg.error()))
                    continue

                event = json.loads(msg.value().decode("utf-8"))

                if event.get("event_type") == "prediction":
                    self._buffer.append({
                        "features": event.get("features", {}),
                        "model_version": event.get("model_version"),
                        "timestamp": event.get("timestamp"),
                        "prediction_label": event.get("prediction_label"),
                    })
                    self._message_count += 1

                    if self._message_count % 1000 == 0:
                        logger.info(
                            "consumer_progress",
                            messages_processed=self._message_count,
                            buffer_size=len(self._buffer),
                        )

            except KafkaException as exc:
                logger.error("kafka_poll_exception", error=str(exc))
                time.sleep(1)
            except json.JSONDecodeError as exc:
                logger.warning("message_decode_error", error=str(exc))
            except Exception as exc:
                logger.error("consumer_unexpected_error", error=str(exc))

        self._consumer.close()
        logger.info("consumer_loop_stopped")

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("consumer_shutdown_signal_received", signal=signum)
        self._running = False


if __name__ == "__main__":
    consumer = PredictionConsumer()
    consumer.run()
