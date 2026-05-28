"""
Unit тесты для S3Storage операций

Тестирует:
- Async streaming чтение JSONL
- Presigned URL генерация
- Streaming методы
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import AsyncIterator

from storage.s3 import S3Storage
from storage.base import NotFoundError


class TestS3StorageStreaming:
    """Тесты для streaming операций S3Storage."""
    
    @pytest.fixture
    def s3_storage(self):
        """Фикстура для S3Storage с мок boto3 клиентом."""
        storage = S3Storage(
            endpoint_url="http://localhost:9000",
            bucket="test-bucket",
            region="us-east-1"
        )
        
        # Мок boto3 клиента
        mock_client = MagicMock()
        storage._client = mock_client
        
        return storage
    
    def test_stream_lines(self, s3_storage):
        """Тест потокового чтения строк."""
        # Настроить мок ответа
        mock_body = MagicMock()
        mock_body.iter_chunks.return_value = [
            b"line1\nline2\n",
            b"line3\nline4"
        ]
        
        mock_response = {"Body": mock_body}
        s3_storage._client.get_object.return_value = mock_response
        
        # Вызвать stream_lines
        lines = list(s3_storage.stream_lines("test/key.jsonl"))
        
        # Проверить результат
        assert len(lines) == 4
        assert lines[0] == "line1"
        assert lines[1] == "line2"
        assert lines[2] == "line3"
        assert lines[3] == "line4"
        
        # Проверить вызов boto3
        s3_storage._client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/key.jsonl"
        )
    
    @pytest.mark.asyncio
    async def test_stream_jsonl(self, s3_storage):
        """Тест async streaming чтения JSONL."""
        # Настроить мок ответа
        mock_body = MagicMock()
        mock_body.iter_chunks.return_value = [
            b'{"key1": "value1"}\n',
            b'{"key2": "value2"}\n',
            b'{"key3": "value3"}'
        ]
        
        mock_response = {"Body": mock_body}
        s3_storage._client.get_object.return_value = mock_response
        
        # Вызвать stream_jsonl
        events = []
        async for event in s3_storage.stream_jsonl("test/key.jsonl"):
            events.append(event)
        
        # Проверить результат
        assert len(events) == 3
        assert events[0] == {"key1": "value1"}
        assert events[1] == {"key2": "value2"}
        assert events[2] == {"key3": "value3"}
        
        # Проверить вызов boto3
        s3_storage._client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/key.jsonl"
        )
    
    @pytest.mark.asyncio
    async def test_stream_jsonl_invalid_json(self, s3_storage):
        """Тест обработки невалидного JSON в stream_jsonl."""
        # Настроить мок ответа с невалидным JSON
        mock_body = MagicMock()
        mock_body.iter_chunks.return_value = [
            b'{"valid": "json"}\n',
            b'invalid json\n',
            b'{"another": "valid"}'
        ]
        
        mock_response = {"Body": mock_body}
        s3_storage._client.get_object.return_value = mock_response
        
        # Вызвать stream_jsonl
        events = []
        async for event in s3_storage.stream_jsonl("test/key.jsonl"):
            events.append(event)
        
        # Проверить, что невалидные строки пропущены
        assert len(events) == 2
        assert events[0] == {"valid": "json"}
        assert events[1] == {"another": "valid"}
    
    @pytest.mark.asyncio
    async def test_stream_jsonl_not_found(self, s3_storage):
        """Тест обработки NotFoundError в stream_jsonl."""
        from botocore.exceptions import ClientError
        
        # Настроить мок для 404 ошибки
        error_response = {"Error": {"Code": "404"}}
        s3_storage._client.get_object.side_effect = ClientError(
            error_response, "GetObject"
        )
        
        # Проверить, что выбрасывается NotFoundError
        with pytest.raises(NotFoundError):
            async for _ in s3_storage.stream_jsonl("test/nonexistent.jsonl"):
                pass


class TestS3StoragePresignedURL:
    """Тесты для presigned URL генерации."""
    
    @pytest.fixture
    def s3_storage(self):
        """Фикстура для S3Storage с мок boto3 клиентом."""
        storage = S3Storage(
            endpoint_url="http://localhost:9000",
            bucket="test-bucket",
            region="us-east-1"
        )
        
        # Мок boto3 клиента
        mock_client = MagicMock()
        storage._client = mock_client
        
        return storage
    
    def test_generate_presigned_url(self, s3_storage):
        """Тест генерации presigned URL."""
        # Настроить мок для generate_presigned_url
        expected_url = "https://s3.amazonaws.com/test-bucket/test/key?X-Amz-Algorithm=..."
        s3_storage._client.generate_presigned_url.return_value = expected_url
        
        # Вызвать generate_presigned_url
        url = s3_storage.generate_presigned_url(
            key="test/key",
            expiration=3600,
            http_method="GET"
        )
        
        # Проверить результат
        assert url == expected_url
        
        # Проверить вызов boto3
        s3_storage._client.generate_presigned_url.assert_called_once_with(
            "get",
            Params={
                "Bucket": "test-bucket",
                "Key": "test/key"
            },
            ExpiresIn=3600
        )
    
    def test_generate_presigned_url_default_expiration(self, s3_storage):
        """Тест генерации presigned URL с дефолтным expiration."""
        expected_url = "https://s3.amazonaws.com/test-bucket/test/key?X-Amz-Algorithm=..."
        s3_storage._client.generate_presigned_url.return_value = expected_url
        
        # Вызвать без указания expiration (должен быть 3600 по умолчанию)
        url = s3_storage.generate_presigned_url("test/key")
        
        # Проверить, что expiration=3600 использован
        s3_storage._client.generate_presigned_url.assert_called_once_with(
            "get",
            Params={
                "Bucket": "test-bucket",
                "Key": "test/key"
            },
            ExpiresIn=3600
        )
    
    def test_generate_presigned_url_put(self, s3_storage):
        """Тест генерации presigned URL для PUT."""
        expected_url = "https://s3.amazonaws.com/test-bucket/test/key?X-Amz-Algorithm=..."
        s3_storage._client.generate_presigned_url.return_value = expected_url
        
        # Вызвать с PUT методом
        url = s3_storage.generate_presigned_url(
            key="test/key",
            expiration=7200,
            http_method="PUT"
        )
        
        # Проверить вызов boto3
        s3_storage._client.generate_presigned_url.assert_called_once_with(
            "put",
            Params={
                "Bucket": "test-bucket",
                "Key": "test/key"
            },
            ExpiresIn=7200
        )
    
    def test_generate_presigned_url_error(self, s3_storage):
        """Тест обработки ошибки при генерации presigned URL."""
        from botocore.exceptions import ClientError
        
        # Настроить мок для ошибки
        error_response = {"Error": {"Code": "AccessDenied"}}
        s3_storage._client.generate_presigned_url.side_effect = ClientError(
            error_response, "GeneratePresignedUrl"
        )
        
        # Проверить, что выбрасывается RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            s3_storage.generate_presigned_url("test/key")
        
        assert "Failed to generate presigned URL" in str(exc_info.value)


class TestFileSystemStorageStreaming:
    """Тесты для streaming операций FileSystemStorage."""
    
    @pytest.fixture
    def fs_storage(self, tmp_path):
        """Фикстура для FileSystemStorage с временной директорией."""
        from storage.fs import FileSystemStorage
        return FileSystemStorage(root_dir=str(tmp_path))
    
    def test_stream_lines(self, fs_storage):
        """Тест потокового чтения строк."""
        # Создать тестовый файл
        test_key = "test/file.jsonl"
        test_content = "line1\nline2\nline3\n"
        fs_storage.write_bytes(test_key, test_content.encode("utf-8"))
        
        # Вызвать stream_lines
        lines = list(fs_storage.stream_lines(test_key))
        
        # Проверить результат
        assert len(lines) == 3
        assert lines[0] == "line1"
        assert lines[1] == "line2"
        assert lines[2] == "line3"
    
    @pytest.mark.asyncio
    async def test_stream_jsonl(self, fs_storage):
        """Тест async streaming чтения JSONL."""
        # Создать тестовый JSONL файл
        test_key = "test/file.jsonl"
        test_content = '{"key1": "value1"}\n{"key2": "value2"}\n{"key3": "value3"}\n'
        fs_storage.write_bytes(test_key, test_content.encode("utf-8"))
        
        # Вызвать stream_jsonl
        events = []
        async for event in fs_storage.stream_jsonl(test_key):
            events.append(event)
        
        # Проверить результат
        assert len(events) == 3
        assert events[0] == {"key1": "value1"}
        assert events[1] == {"key2": "value2"}
        assert events[2] == {"key3": "value3"}
    
    @pytest.mark.asyncio
    async def test_stream_jsonl_not_found(self, fs_storage):
        """Тест обработки NotFoundError в stream_jsonl."""
        # Проверить, что выбрасывается NotFoundError
        with pytest.raises(NotFoundError):
            async for _ in fs_storage.stream_jsonl("nonexistent.jsonl"):
                pass
    
    def test_generate_presigned_url(self, fs_storage):
        """Тест генерации presigned URL для FileSystemStorage."""
        # FileSystemStorage возвращает относительный путь
        url = fs_storage.generate_presigned_url("test/key")
        
        # Проверить результат
        assert url == "/storage/test/key"

