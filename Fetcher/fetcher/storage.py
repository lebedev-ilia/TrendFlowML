from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

import boto3
from botocore.client import BaseClient

from .config import settings


class StorageClient(Protocol):
    """Интерфейс клиента object storage для Fetcher.

    Лёгкая обёртка над S3/MinIO, соответствующая контракту из `STORAGE_LAYOUT.md`.
    """

    def upload_file(self, local_path: str | Path, bucket: str, key: str) -> None:
        """Загрузить локальный файл в указанный bucket/key."""

    def download_file(self, bucket: str, key: str, local_path: str | Path) -> None:
        """Скачать объект в локальный файл."""

    def delete_object(self, bucket: str, key: str) -> None:
        """Удалить объект из storage.

        Args:
            bucket: Имя bucket'а
            key: Ключ объекта для удаления

        Raises:
            Exception: При ошибке удаления
        """

    def object_exists(self, bucket: str, key: str) -> bool:
        """Проверить, существует ли объект."""

    def list_objects(
        self, bucket: str, prefix: str = "", max_keys: int = 1000
    ) -> list[tuple[str, datetime]]:
        """Перечислить объекты в bucket с опциональным prefix.

        Returns:
            Список пар (key, last_modified_utc).
        """

    def generate_presigned_url(
        self, bucket: str, key: str, expires_in: int = 3600
    ) -> str:
        """Генерировать presigned URL для безопасного доступа к объекту.

        Args:
            bucket: Имя bucket'а
            key: Ключ объекта
            expires_in: Время жизни URL в секундах (default: 1 час, max: 7 дней)

        Returns:
            Presigned URL для скачивания объекта
        """


class S3StorageClient:
    """S3/MinIO реализация StorageClient.

    Настройки берутся из `FetcherSettings` (endpoint, креденшелы, region).
    """

    def __init__(self) -> None:
        # Определяем use_ssl из endpoint_url или настроек
        use_ssl = settings.s3_use_ssl
        if settings.s3_endpoint_url:
            endpoint_str = str(settings.s3_endpoint_url)
            if endpoint_str.startswith("https://"):
                use_ssl = True
            elif endpoint_str.startswith("http://"):
                use_ssl = False

        self._client: BaseClient = boto3.client(
            "s3",
            endpoint_url=str(settings.s3_endpoint_url) if settings.s3_endpoint_url else None,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            use_ssl=use_ssl,
            verify=settings.s3_verify_ssl,
        )

    def upload_file(self, local_path: str | Path, bucket: str, key: str) -> None:
        self._client.upload_file(str(local_path), bucket, key)

    def download_file(self, bucket: str, key: str, local_path: str | Path) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(bucket, key, str(local_path))

    def delete_object(self, bucket: str, key: str) -> None:
        """Удалить объект из S3/MinIO storage."""
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            # Логируем ошибку, но не скрываем её
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to delete object {bucket}/{key}: {e}")
            raise

    def object_exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except self._client.exceptions.NoSuchKey:
            return False
        except Exception:
            # Для MVP любые неожиданные ошибки считаем "нет объекта",
            # детальный error‑handling будет добавлен на этапе observability.
            return False

    def list_objects(
        self, bucket: str, prefix: str = "", max_keys: int = 1000
    ) -> list[tuple[str, datetime]]:
        """Перечислить объекты в bucket с опциональным prefix."""
        result: list[tuple[str, datetime]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=bucket, Prefix=prefix, MaxKeys=max_keys
        ):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                last_mod = obj.get("LastModified")
                if last_mod:
                    # LastModified is timezone-aware from boto3
                    result.append((key, last_mod))
            if len(result) >= max_keys:
                break
        return result[:max_keys]

    def generate_presigned_url(
        self, bucket: str, key: str, expires_in: int = 3600
    ) -> str:
        """Генерировать presigned URL для безопасного доступа к объекту.

        Args:
            bucket: Имя bucket'а
            key: Ключ объекта
            expires_in: Время жизни URL в секундах (default: 1 час, max: 7 дней)

        Returns:
            Presigned URL для скачивания объекта
        """
        # Ограничиваем expires_in до 7 дней (604800 секунд)
        expires_in = min(expires_in, 604800)
        expires_in = max(expires_in, 60)  # Минимум 1 минута

        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to generate presigned URL for {bucket}/{key}: {e}")
            raise


storage_client: StorageClient = S3StorageClient()


__all__ = ["StorageClient", "S3StorageClient", "storage_client"]


