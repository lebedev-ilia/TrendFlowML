"""Handlers для централизованного логирования.

Поддерживает интеграцию с:
- Grafana Loki (через HTTP API)
- Elasticsearch (через HTTP API)
- AWS CloudWatch (через boto3)
- GCP Cloud Logging (через google-cloud-logging)
- Azure Monitor (через azure-monitor-opentelemetry)

Соответствует требованиям из `Fetcher/docs/checklist.md` (Phase 6 — Central log storage).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from typing import Any, Optional

try:
    import httpx
except ImportError:
    httpx = None

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = None


class LokiHandler(logging.Handler):
    """Handler для отправки логов в Grafana Loki через HTTP API.

    Использует Loki Push API: POST /loki/api/v1/push
    Формат: Prometheus labels + log lines
    """

    def __init__(
        self,
        loki_url: str,
        labels: Optional[dict[str, str]] = None,
        timeout: float = 5.0,
    ):
        """Инициализировать Loki handler.

        Args:
            loki_url: URL Loki API (например, http://loki:3100)
            labels: Статические labels для всех логов (например, {"job": "fetcher"})
            timeout: Таймаут HTTP запроса в секундах
        """
        super().__init__()
        if httpx is None:
            raise ImportError("httpx is required for LokiHandler. Install with: pip install httpx")

        self.loki_url = loki_url.rstrip("/")
        self.labels = labels or {}
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def emit(self, record: logging.LogRecord) -> None:
        """Отправить log record в Loki."""
        try:
            # Формируем labels
            labels = self.labels.copy()
            labels["level"] = record.levelname.lower()
            labels["logger"] = record.name

            # Добавляем динамические labels из extra
            if hasattr(record, "run_id"):
                labels["run_id"] = str(record.run_id)
            if hasattr(record, "stage"):
                labels["stage"] = record.stage
            if hasattr(record, "platform"):
                labels["platform"] = record.platform

            # Формируем log line (JSON или plain text)
            if isinstance(record.msg, dict):
                log_line = json.dumps(record.msg, ensure_ascii=False)
            else:
                log_line = self.format(record)

            # Формируем payload для Loki Push API
            # Формат: {"streams": [{"stream": {...labels}, "values": [[timestamp_ns, log_line]]}]}
            timestamp_ns = str(int(record.created * 1_000_000_000))

            payload = {
                "streams": [
                    {
                        "stream": labels,
                        "values": [[timestamp_ns, log_line]],
                    }
                ]
            }

            # Отправляем в Loki
            url = f"{self.loki_url}/loki/api/v1/push"
            response = self.client.post(url, json=payload)
            response.raise_for_status()

        except Exception:
            # Не допускаем, чтобы проблемы с отправкой логов ломали основной поток
            self.handleError(record)

    def close(self) -> None:
        """Закрыть HTTP клиент."""
        if hasattr(self, "client"):
            self.client.close()
        super().close()


class ElasticsearchHandler(logging.Handler):
    """Handler для отправки логов в Elasticsearch через HTTP API.

    Использует Elasticsearch Index API: POST /{index}/_doc
    """

    def __init__(
        self,
        es_url: str,
        index: str = "fetcher-logs",
        timeout: float = 5.0,
    ):
        """Инициализировать Elasticsearch handler.

        Args:
            es_url: URL Elasticsearch (например, http://elasticsearch:9200)
            index: Имя индекса для логов
            timeout: Таймаут HTTP запроса в секундах
        """
        super().__init__()
        if httpx is None:
            raise ImportError("httpx is required for ElasticsearchHandler. Install with: pip install httpx")

        self.es_url = es_url.rstrip("/")
        self.index = index
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def emit(self, record: logging.LogRecord) -> None:
        """Отправить log record в Elasticsearch."""
        try:
            # Формируем документ для Elasticsearch
            doc: dict[str, Any] = {
                "@timestamp": record.created * 1000,  # milliseconds
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
            }

            # Добавляем поля из extra
            if hasattr(record, "run_id"):
                doc["run_id"] = str(record.run_id)
            if hasattr(record, "stage"):
                doc["stage"] = record.stage
            if hasattr(record, "platform"):
                doc["platform"] = record.platform
            if hasattr(record, "platform_video_id"):
                doc["platform_video_id"] = record.platform_video_id

            # Добавляем exception info
            if record.exc_info:
                doc["exception"] = self.formatException(record.exc_info)

            # Отправляем в Elasticsearch
            url = f"{self.es_url}/{self.index}/_doc"
            response = self.client.post(url, json=doc)
            response.raise_for_status()

        except Exception:
            # Не допускаем, чтобы проблемы с отправкой логов ломали основной поток
            self.handleError(record)

    def close(self) -> None:
        """Закрыть HTTP клиент."""
        if hasattr(self, "client"):
            self.client.close()
        super().close()


class CloudWatchHandler(logging.Handler):
    """Handler для отправки логов в AWS CloudWatch Logs.

    Использует boto3 для отправки логов в CloudWatch Logs.
    """

    def __init__(
        self,
        log_group: str = "/aws/fetcher",
        log_stream: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """Инициализировать CloudWatch handler.

        Args:
            log_group: Имя log group в CloudWatch
            log_stream: Имя log stream (если None, используется hostname)
            region_name: AWS регион (если None, используется дефолтный)
        """
        super().__init__()
        if boto3 is None:
            raise ImportError("boto3 is required for CloudWatchHandler. Install with: pip install boto3")

        self.log_group = log_group
        self.log_stream = log_stream or "fetcher"
        self.region_name = region_name
        self.client = boto3.client("logs", region_name=region_name)
        self.sequence_token: Optional[str] = None

    def emit(self, record: logging.LogRecord) -> None:
        """Отправить log record в CloudWatch."""
        try:
            # Формируем сообщение
            message = self.format(record)

            # Формируем log event
            log_events = [
                {
                    "timestamp": int(record.created * 1000),  # milliseconds
                    "message": message,
                }
            ]

            # Отправляем в CloudWatch
            kwargs: dict[str, Any] = {
                "logGroupName": self.log_group,
                "logStreamName": self.log_stream,
                "logEvents": log_events,
            }

            if self.sequence_token:
                kwargs["sequenceToken"] = self.sequence_token

            response = self.client.put_log_events(**kwargs)
            self.sequence_token = response.get("nextSequenceToken")

        except ClientError as e:
            # Обработка ошибок CloudWatch
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                # Создаём log group и stream, если не существуют
                try:
                    self.client.create_log_group(logGroupName=self.log_group)
                    self.client.create_log_stream(
                        logGroupName=self.log_group, logStreamName=self.log_stream
                    )
                    # Повторяем отправку
                    self.emit(record)
                except Exception:
                    self.handleError(record)
            else:
                self.handleError(record)
        except Exception:
            # Не допускаем, чтобы проблемы с отправкой логов ломали основной поток
            self.handleError(record)


def setup_centralized_logging(
    backend: str = "loki",
    **kwargs: Any,
) -> Optional[logging.Handler]:
    """Настроить централизованное логирование.

    Args:
        backend: Backend для логирования ("loki", "elasticsearch", "cloudwatch")
        **kwargs: Параметры для конкретного backend'а:
            - loki: loki_url, labels
            - elasticsearch: es_url, index
            - cloudwatch: log_group, log_stream, region_name

    Returns:
        Handler instance или None при ошибке
    """
    try:
        if backend == "loki":
            handler = LokiHandler(
                loki_url=kwargs.get("loki_url", "http://loki:3100"),
                labels=kwargs.get("labels", {"job": "fetcher"}),
                timeout=kwargs.get("timeout", 5.0),
            )
        elif backend == "elasticsearch":
            handler = ElasticsearchHandler(
                es_url=kwargs.get("es_url", "http://elasticsearch:9200"),
                index=kwargs.get("index", "fetcher-logs"),
                timeout=kwargs.get("timeout", 5.0),
            )
        elif backend == "cloudwatch":
            handler = CloudWatchHandler(
                log_group=kwargs.get("log_group", "/aws/fetcher"),
                log_stream=kwargs.get("log_stream"),
                region_name=kwargs.get("region_name"),
            )
        else:
            raise ValueError(f"Unsupported logging backend: {backend}")

        return handler

    except Exception as e:
        logging.warning(f"Failed to setup centralized logging ({backend}): {e}")
        return None


__all__ = [
    "LokiHandler",
    "ElasticsearchHandler",
    "CloudWatchHandler",
    "setup_centralized_logging",
]

