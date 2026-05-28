from __future__ import annotations

from typing import Iterable, Optional, AsyncIterator
import json

# boto3/botocore are required in S3/MinIO mode, but may be absent in some dev envs.
# pyright: ignore[reportMissingImports]
import boto3  # type: ignore
# pyright: ignore[reportMissingImports]
from botocore.exceptions import ClientError  # type: ignore

from .base import NotFoundError, ObjectInfo


class S3Storage:
    """
    S3/MinIO backend.

    Atomicity notes:
    - A single PUT is atomic from the reader perspective (you either get old or new object).
    - There is no true rename; we treat overwrite PUT as "atomic write" for our use cases.
    """

    def __init__(self, *, endpoint_url: str, bucket: str, region: str = "us-east-1") -> None:
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.region = region
        self._client = boto3.client("s3", endpoint_url=self.endpoint_url, region_name=self.region)

    def _k(self, key: str) -> str:
        # We treat incoming keys as full object keys inside the bucket,
        # including S3_PREFIX when used (e.g. "trendflowml/result_store/...").
        return str(key).lstrip("/")

    def exists(self, key: str) -> bool:
        k = self._k(key)
        try:
            self._client.head_object(Bucket=self.bucket, Key=k)
            return True
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str) -> Iterable[ObjectInfo]:
        pfx = self._k(prefix).rstrip("/") + "/"
        paginator = self._client.get_paginator("list_objects_v2")
        out = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=pfx):
            for obj in page.get("Contents") or []:
                k = obj.get("Key")
                if not isinstance(k, str):
                    continue
                out.append(ObjectInfo(key=k, size_bytes=obj.get("Size"), etag=obj.get("ETag")))
        return out

    def read_bytes(self, key: str) -> bytes:
        k = self._k(key)
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=k)
            return resp["Body"].read()
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                raise NotFoundError(f"S3 key not found: {key}") from e
            raise

    def write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> None:
        k = self._k(key)
        extra = {}
        if content_type:
            extra["ContentType"] = content_type
        self._client.put_object(Bucket=self.bucket, Key=k, Body=data, **extra)

    def atomic_write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> None:
        # See class docstring: we treat PUT overwrite as atomic enough here.
        self.write_bytes(key, data, content_type=content_type)
    
    def stream_lines(self, key: str) -> Iterable[str]:
        """
        Потоковое чтение файла построчно (для JSONL файлов).
        
        Не читает весь файл в память, читает построчно.
        
        Args:
            key: Ключ объекта в S3
            
        Yields:
            Строки файла
            
        Raises:
            NotFoundError: Если объект не найден
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2330-2337)
        """
        k = self._k(key)
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=k)
            body = resp["Body"]
            
            # Читать построчно
            buffer = ""
            for chunk in body.iter_chunks(chunk_size=8192):
                buffer += chunk.decode("utf-8", errors="replace")
                
                # Обработать полные строки
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        yield line
                
            # Обработать оставшуюся часть
            if buffer.strip():
                yield buffer
                
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                raise NotFoundError(f"S3 key not found: {key}") from e
            raise
    
    async def stream_jsonl(self, key: str) -> AsyncIterator[dict]:
        """
        Async streaming чтение JSONL файла построчно.
        
        Не читает весь файл в память, читает построчно и парсит JSON.
        
        Args:
            key: Ключ объекта в S3
            
        Yields:
            Словари с распарсенными JSON объектами
            
        Raises:
            NotFoundError: Если объект не найден
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2330-2337)
        """
        k = self._k(key)
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=k)
            body = resp["Body"]
            
            # Читать построчно
            buffer = ""
            for chunk in body.iter_chunks(chunk_size=8192):
                buffer += chunk.decode("utf-8", errors="replace")
                
                # Обработать полные строки
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Пропустить невалидные строки
                        continue
                
            # Обработать оставшуюся часть
            if buffer.strip():
                try:
                    yield json.loads(buffer.strip())
                except json.JSONDecodeError:
                    pass
                    
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                raise NotFoundError(f"S3 key not found: {key}") from e
            raise
    
    def generate_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        http_method: str = "GET"
    ) -> str:
        """
        Генерация presigned URL для прямого доступа к объекту.
        
        Args:
            key: Ключ объекта в S3
            expiration: Время жизни URL в секундах (по умолчанию 1 час)
            http_method: HTTP метод для URL (GET или PUT)
            
        Returns:
            Presigned URL для доступа к объекту
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2340-2346)
        """
        k = self._k(key)
        try:
            url = self._client.generate_presigned_url(
                http_method.lower(),
                Params={
                    "Bucket": self.bucket,
                    "Key": k
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise RuntimeError(f"Failed to generate presigned URL for {key}: {e}") from e


