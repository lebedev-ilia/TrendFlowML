"""Kafka Producer для отправки событий Fetcher.

Поддерживает отправку событий pipeline в Kafka topics.
Используется для production event streaming (опционально, Celery + Redis остаётся для MVP).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    KafkaProducer = None
    KafkaError = None

from schemas.events import FetcherEvent

logger = logging.getLogger(__name__)


class FetcherKafkaProducer:
    """Producer для отправки событий Fetcher в Kafka."""

    def __init__(
        self,
        bootstrap_servers: str | list[str],
        topic_prefix: str = "fetcher",
        **kwargs,
    ):
        """Инициализировать Kafka producer.

        Args:
            bootstrap_servers: Список Kafka brokers (например, "localhost:9092" или ["kafka1:9092", "kafka2:9092"])
            topic_prefix: Префикс для topic'ов (например, "fetcher" → "fetcher.events", "fetcher.tasks")
            **kwargs: Дополнительные параметры для KafkaProducer
        """
        if KafkaProducer is None:
            raise ImportError("kafka-python is required for KafkaProducer. Install with: pip install kafka-python")

        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            **kwargs,
        )
        logger.info(f"Kafka producer initialized: bootstrap_servers={bootstrap_servers}, topic_prefix={topic_prefix}")

    def publish_event(self, event: FetcherEvent, topic: Optional[str] = None) -> None:
        """Опубликовать событие в Kafka.

        Args:
            event: Событие Fetcher (FetcherEvent)
            topic: Имя topic'а (если None, используется дефолтный)
        """
        if topic is None:
            topic = f"{self.topic_prefix}.events"

        try:
            # Конвертируем событие в dict
            event_dict = event.model_dump(mode="json")
            # Убеждаемся, что ts в правильном формате
            if isinstance(event_dict.get("ts"), datetime):
                event_dict["ts"] = event_dict["ts"].isoformat()

            # Отправляем в Kafka
            future = self.producer.send(topic, value=event_dict)
            # Ждём подтверждения (можно сделать async)
            future.get(timeout=10)

            logger.debug(f"Published event to Kafka: topic={topic}, type={event.type}, run_id={event.run_id}")

        except KafkaError as e:
            logger.error(f"Failed to publish event to Kafka: {e}", exc_info=True)
            # Не поднимаем исключение, чтобы не ломать основной поток

    def publish_task(
        self,
        task_type: str,
        run_id: UUID | str,
        payload: dict,
        topic: Optional[str] = None,
    ) -> None:
        """Опубликовать задачу в Kafka для обработки consumer'ом.

        Args:
            task_type: Тип задачи (fetch_metadata, download_video, fetch_comments, finalize)
            run_id: UUID run'а
            topic: Имя topic'а (если None, используется дефолтный для задач)
        """
        if topic is None:
            # Используем разные topic'и для разных типов задач
            topic = f"{self.topic_prefix}.tasks.{task_type}"

        try:
            task_message = {
                "task_type": task_type,
                "run_id": str(run_id),
                "payload": payload,
                "ts": datetime.now(timezone.utc).isoformat(),
            }

            # Отправляем в Kafka
            future = self.producer.send(topic, value=task_message)
            future.get(timeout=10)

            logger.debug(f"Published task to Kafka: topic={topic}, task_type={task_type}, run_id={run_id}")

        except KafkaError as e:
            logger.error(f"Failed to publish task to Kafka: {e}", exc_info=True)
            # Не поднимаем исключение, чтобы не ломать основной поток

    def close(self) -> None:
        """Закрыть producer."""
        if hasattr(self, "producer"):
            self.producer.close()
            logger.info("Kafka producer closed")


# Глобальный producer instance (опционально)
_producer: Optional[FetcherKafkaProducer] = None


def get_producer() -> Optional[FetcherKafkaProducer]:
    """Получить глобальный Kafka producer instance."""
    return _producer


def init_producer(
    bootstrap_servers: str | list[str],
    topic_prefix: str = "fetcher",
    **kwargs,
) -> FetcherKafkaProducer:
    """Инициализировать глобальный Kafka producer.

    Args:
        bootstrap_servers: Список Kafka brokers
        topic_prefix: Префикс для topic'ов
        **kwargs: Дополнительные параметры для KafkaProducer

    Returns:
        FetcherKafkaProducer instance
    """
    global _producer
    _producer = FetcherKafkaProducer(bootstrap_servers, topic_prefix, **kwargs)
    return _producer


def publish_event(event: FetcherEvent, topic: Optional[str] = None) -> None:
    """Опубликовать событие через глобальный producer (если инициализирован)."""
    producer = get_producer()
    if producer:
        producer.publish_event(event, topic)
    else:
        logger.debug("Kafka producer not initialized, skipping event publication")


__all__ = [
    "FetcherKafkaProducer",
    "get_producer",
    "init_producer",
    "publish_event",
]

