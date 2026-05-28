"""Kafka Consumer для обработки задач Fetcher.

Поддерживает обработку задач из Kafka topics вместо Celery (опционально для production).
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
except ImportError:
    KafkaConsumer = None
    KafkaError = None

logger = logging.getLogger(__name__)


class FetcherKafkaConsumer:
    """Consumer для обработки задач Fetcher из Kafka."""

    def __init__(
        self,
        bootstrap_servers: str | list[str],
        topic: str,
        group_id: str = "fetcher-workers",
        **kwargs,
    ):
        """Инициализировать Kafka consumer.

        Args:
            bootstrap_servers: Список Kafka brokers
            topic: Имя topic'а для подписки
            group_id: Consumer group ID для балансировки нагрузки
            **kwargs: Дополнительные параметры для KafkaConsumer
        """
        if KafkaConsumer is None:
            raise ImportError("kafka-python is required for KafkaConsumer. Install with: pip install kafka-python")

        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id

        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            **kwargs,
        )
        logger.info(f"Kafka consumer initialized: topic={topic}, group_id={group_id}")

    def consume_tasks(self, handler: Callable[[dict], None]) -> None:
        """Обрабатывать задачи из Kafka.

        Args:
            handler: Функция-обработчик для задач (принимает dict с task message)
        """
        logger.info(f"Starting to consume tasks from topic: {self.topic}")

        try:
            for message in self.consumer:
                try:
                    task_message = message.value
                    logger.debug(f"Received task: {task_message}")

                    # Вызываем обработчик
                    handler(task_message)

                except Exception as e:
                    logger.error(f"Error processing task: {e}", exc_info=True)
                    # Продолжаем обработку следующих сообщений

        except KeyboardInterrupt:
            logger.info("Consumer interrupted, shutting down...")
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
        finally:
            self.close()

    def close(self) -> None:
        """Закрыть consumer."""
        if hasattr(self, "consumer"):
            self.consumer.close()
            logger.info("Kafka consumer closed")


def create_consumer_for_task_type(
    task_type: str,
    bootstrap_servers: str | list[str],
    topic_prefix: str = "fetcher",
    group_id: Optional[str] = None,
    **kwargs,
) -> FetcherKafkaConsumer:
    """Создать consumer для конкретного типа задачи.

    Args:
        task_type: Тип задачи (fetch_metadata, download_video, fetch_comments, finalize)
        bootstrap_servers: Список Kafka brokers
        topic_prefix: Префикс для topic'ов
        group_id: Consumer group ID (если None, используется дефолтный)
        **kwargs: Дополнительные параметры для KafkaConsumer

    Returns:
        FetcherKafkaConsumer instance
    """
    topic = f"{topic_prefix}.tasks.{task_type}"
    if group_id is None:
        group_id = f"fetcher-{task_type}-workers"

    return FetcherKafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        group_id=group_id,
        **kwargs,
    )


__all__ = [
    "FetcherKafkaConsumer",
    "create_consumer_for_task_type",
]

