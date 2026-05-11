from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition

from common.yaml import load_yaml_config
from inference.src.predictor import PredictionService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


config = load_yaml_config(str(Path(__file__).parent / "config.yml"))


def _kafka_config() -> dict[str, Any]:
    kafka = dict(config.get("kafka") or {})
    return {
        "bootstrap.servers": _bootstrap_servers(),
        "group.id": kafka.get("group_id", "inference-consumer"),
        "auto.offset.reset": kafka.get("auto_offset_reset", "earliest"),
        "enable.auto.commit": False,
    }


def _bootstrap_servers() -> str:
    kafka = dict(config.get("kafka") or {})
    return str(kafka.get("bootstrap_servers", "localhost:9092"))


def _topic() -> str:
    kafka = dict(config.get("kafka") or {})
    return str(kafka.get("topic", "inference.requests"))


def _response_topic() -> str:
    kafka = dict(config.get("kafka") or {})
    return str(kafka.get("response_topic", "inference.responses"))


def _retry_sleep_seconds() -> float:
    kafka = dict(config.get("kafka") or {})
    return float(kafka.get("retry_sleep_seconds", 5.0))


def _batch_size() -> int:
    kafka = dict(config.get("kafka") or {})
    return int(kafka.get("batch_size", 10))


def _batch_poll_timeout_seconds() -> float:
    kafka = dict(config.get("kafka") or {})
    return float(kafka.get("batch_poll_timeout_seconds", 1.0))


def _max_workers() -> int:
    kafka = dict(config.get("kafka") or {})
    return int(kafka.get("max_workers", 4))


def _commit_offsets(messages: list[Any]) -> list[TopicPartition]:
    offsets: dict[tuple[str, int], int] = {}
    for message in messages:
        key = (message.topic(), message.partition())
        offsets[key] = max(offsets.get(key, -1), message.offset() + 1)
    return [
        TopicPartition(topic, partition, offset)
        for (topic, partition), offset in offsets.items()
    ]


def _rollback_offsets(messages: list[Any]) -> list[TopicPartition]:
    offsets: dict[tuple[str, int], int] = {}
    for message in messages:
        key = (message.topic(), message.partition())
        offsets[key] = min(offsets.get(key, message.offset()), message.offset())
    return [
        TopicPartition(topic, partition, offset)
        for (topic, partition), offset in offsets.items()
    ]


def _rollback(consumer: Consumer, messages: list[Any]) -> None:
    for partition in _rollback_offsets(messages):
        consumer.seek(partition)


def _handle_message(message: Any, service: PredictionService) -> dict[str, Any]:
    payload = json.loads(message.value().decode("utf-8"))
    predictions = service.predict_rows(payload["rows"])
    logger.info(
        "processed topic=%s partition=%s offset=%s predictions=%s",
        message.topic(),
        message.partition(),
        message.offset(),
        predictions,
    )
    return {
        "request_id": payload.get("request_id"),
        "response_topic": payload.get("response_topic"),
        "predictions": predictions,
    }


def _publish_response(
    producer: Producer,
    default_topic: str,
    result: dict[str, Any],
) -> None:
    request_id = result.get("request_id")
    if request_id is None:
        return
    topic = str(result.get("response_topic") or default_topic)
    producer.produce(
        topic,
        key=str(request_id).encode("utf-8"),
        value=json.dumps(result).encode("utf-8"),
    )
    producer.poll(0)


class KafkaInferenceConsumer:
    def __init__(
        self,
        service: PredictionService,
        *,
        stop_event: Event | None = None,
    ):
        self.service = service
        self.stop_event = stop_event or Event()
        self.consumer = Consumer(_kafka_config())
        self.producer = Producer({"bootstrap.servers": _bootstrap_servers()})
        self.topic = _topic()
        self.response_topic = _response_topic()
        self.retry_sleep = _retry_sleep_seconds()
        self.batch_size = _batch_size()
        self.batch_timeout = _batch_poll_timeout_seconds()
        self.max_workers = _max_workers()

    def run(self) -> None:
        self.consumer.subscribe([self.topic])
        logger.info(
            "listening topic=%s batch_size=%s max_workers=%s",
            self.topic,
            self.batch_size,
            self.max_workers,
        )

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                while not self.stop_event.is_set():
                    messages = self.consumer.consume(
                        num_messages=self.batch_size,
                        timeout=self.batch_timeout,
                    )
                    if not messages:
                        continue

                    for message in messages:
                        if message.error():
                            raise KafkaException(message.error())

                    futures = [
                        executor.submit(_handle_message, message, self.service)
                        for message in messages
                    ]
                    results = []

                    try:
                        for future in as_completed(futures):
                            results.append(future.result())
                    except Exception:
                        logger.exception(
                            "failed batch size=%s; rolling back",
                            len(messages),
                        )
                        _rollback(self.consumer, messages)
                        time.sleep(self.retry_sleep)
                        continue

                    for result in results:
                        _publish_response(self.producer, self.response_topic, result)
                    self.producer.flush()

                    offsets = _commit_offsets(messages)
                    self.consumer.commit(offsets=offsets, asynchronous=False)
                    logger.info(
                        "committed batch size=%s offsets=%s",
                        len(messages),
                        offsets,
                    )
        finally:
            self.consumer.close()

    def stop(self) -> None:
        self.stop_event.set()


def run() -> None:
    service = PredictionService.from_config_path(Path(__file__).parent / "config.yml")
    service.start()
    KafkaInferenceConsumer(service).run()


if __name__ == "__main__":
    run()
