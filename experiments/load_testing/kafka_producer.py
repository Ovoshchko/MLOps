import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from confluent_kafka import Consumer, KafkaError, Producer
from locust import User, between, task


BATCH_SIZE = int(os.environ.get("KAFKA_BATCH_SIZE", "1"))
PARQUET_PATH = os.environ.get(
    "KAFKA_PARQUET_PATH",
    str(Path(__file__).parent / "load_date=2025-11-25.parquet"),
)
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REQUEST_TOPIC = os.environ.get("KAFKA_REQUEST_TOPIC", "inference.requests")
RESPONSE_TOPIC_PREFIX = os.environ.get("KAFKA_RESPONSE_TOPIC_PREFIX", "inference.responses")
RESPONSE_TIMEOUT_SECONDS = float(os.environ.get("KAFKA_RESPONSE_TIMEOUT_SECONDS", "60"))


class KafkaProducerUser(User):
    wait_time = between(0.05, 0.2)

    def on_start(self) -> None:
        self.df = pd.read_parquet(PARQUET_PATH).replace({np.nan: None})
        self.batch_size = min(BATCH_SIZE, len(self.df))
        self.response_topic = f"{RESPONSE_TOPIC_PREFIX}.{uuid.uuid4()}"
        self.producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})
        self.consumer = Consumer(
            {
                "bootstrap.servers": BOOTSTRAP_SERVERS,
                "group.id": f"locust-kafka-{uuid.uuid4()}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            }
        )
        self.consumer.subscribe([self.response_topic])

    def on_stop(self) -> None:
        self.producer.flush()
        self.consumer.close()

    def _payload(self, request_id: str) -> bytes:
        idx = np.random.choice(len(self.df), size=self.batch_size, replace=False)
        rows = [self.df.iloc[i].to_dict() for i in idx]
        return json.dumps(
            {
                "request_id": request_id,
                "response_topic": self.response_topic,
                "rows": rows,
            }
        ).encode("utf-8")

    def _delivery_callback(self, error: KafkaError | None, message: Any) -> None:
        if error is not None:
            raise RuntimeError(str(error))

    def _wait_response(self, request_id: str) -> dict[str, Any]:
        deadline = time.perf_counter() + RESPONSE_TIMEOUT_SECONDS
        while time.perf_counter() < deadline:
            message = self.consumer.poll(0.2)
            if message is None:
                continue
            if message.error():
                raise RuntimeError(str(message.error()))

            payload = json.loads(message.value().decode("utf-8"))
            if payload.get("request_id") == request_id:
                return payload

        raise TimeoutError(f"response timeout for request_id={request_id}")

    @task
    def predict(self) -> None:
        request_id = str(uuid.uuid4())
        payload = self._payload(request_id)
        started_at = time.perf_counter()

        try:
            self.producer.produce(
                REQUEST_TOPIC,
                key=request_id.encode("utf-8"),
                value=payload,
                callback=self._delivery_callback,
            )
            self.producer.flush()
            response = self._wait_response(request_id)
            exception = None
            response_length = len(json.dumps(response).encode("utf-8"))
        except Exception as exc:
            exception = exc
            response_length = 0

        self.environment.events.request.fire(
            request_type="KAFKA",
            name="predict",
            response_time=(time.perf_counter() - started_at) * 1000,
            response_length=response_length,
            exception=exception,
        )
