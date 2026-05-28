"""Общие фикстуры и конфигурация для тестов Fetcher."""

import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Generator

# Устанавливаем переменные окружения для тестов (config читает FETCHER_POSTGRES_DSN)
_test_dsn = "postgresql+psycopg2://fetcher:fetcher@localhost:5432/fetcher_test"
os.environ.setdefault("POSTGRES_DSN", _test_dsn)
os.environ.setdefault("FETCHER_POSTGRES_DSN", os.environ.get("POSTGRES_DSN", _test_dsn))


def _postgres_available() -> bool:
    """Проверить доступность PostgreSQL (для пропуска тестов с маркером database)."""
    try:
        import psycopg2
        from urllib.parse import urlparse
        dsn = os.environ.get("FETCHER_POSTGRES_DSN") or os.environ.get("POSTGRES_DSN", "")
        if dsn.startswith("postgresql+psycopg2://"):
            dsn = dsn.replace("postgresql+psycopg2://", "postgresql://", 1)
        parsed = urlparse(dsn)
        conn = psycopg2.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            user=parsed.username or "fetcher",
            password=parsed.password or "fetcher",
            dbname=(parsed.path or "/fetcher_test").lstrip("/") or "fetcher_test",
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


def pytest_configure(config):
    """Регистрируем маркер database для тестов, требующих живую БД."""
    config.addinivalue_line(
        "markers",
        "database: тест требует запущенный PostgreSQL (например, docker-compose up -d postgres)",
    )


def pytest_collection_modifyitems(config, items):
    """Пропускать integration/chaos тесты, если PostgreSQL недоступен (без docker-compose)."""
    if _postgres_available():
        return
    skip_no_db = pytest.mark.skip(reason="PostgreSQL недоступен (запустите: docker-compose up -d postgres)")
    for item in items:
        if "integration" in item.keywords or "chaos" in item.keywords or "e2e" in item.keywords:
            item.add_marker(skip_no_db)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("FETCHER_KAFKA_ENABLED", "false")
os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("S3_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("S3_SECRET_ACCESS_KEY", "minioadmin123")
os.environ.setdefault("S3_BUCKET_RAW", "video-analytics-raw-test")


@pytest.fixture
def mock_storage() -> Generator[Mock, None, None]:
    """Фикстура для мока Storage клиента."""
    with patch("fetcher.storage.storage_client") as mock:
        mock_storage_client = MagicMock()
        mock_storage_client.upload_file.return_value = "s3://bucket/path/to/file"
        mock_storage_client.download_file.return_value = None
        mock_storage_client.object_exists.return_value = True
        mock_storage_client.delete_object.return_value = None
        mock.return_value = mock_storage_client
        yield mock_storage_client


@pytest.fixture
def mock_redis() -> Generator[Mock, None, None]:
    """Фикстура для мока Redis клиента."""
    with patch("fetcher.rate_limiter.get_redis_client") as mock:
        mock_redis_client = MagicMock()
        mock_redis_client.incr.return_value = 1
        mock_redis_client.expire.return_value = True
        mock_redis_client.set.return_value = True
        mock_redis_client.get.return_value = None
        mock_redis_client.delete.return_value = 1
        mock.return_value = mock_redis_client
        yield mock_redis_client


@pytest.fixture
def mock_db_session() -> Generator[Mock, None, None]:
    """Фикстура для мока DB сессии (unit-тесты). Не коммитит данные — оркестратор/workers не увидят run."""
    with patch("fetcher.db.session_scope") as mock:
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None
        mock.return_value = mock_session
        yield mock_session


@pytest.fixture
def integration_test_run():
    """Run + VideoSource в реальной БД для integration-тестов. Возвращает объект с .id (UUID), без detached Run."""
    from fetcher.db import session_scope
    from fetcher.models import Run, VideoSource
    import uuid
    from datetime import datetime, timezone
    with session_scope() as db:
        run_id = uuid.uuid4()
        run = Run(
            id=run_id,
            source_type="youtube",
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            status="PENDING",
            started_at=datetime.now(timezone.utc),
        )
        db.add(run)
        db.flush()
        vs = VideoSource(
            run_id=run_id,
            platform="youtube",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            normalized_video_id="dQw4w9WgXcQ",
        )
        db.add(vs)
        db.commit()
    return type("RunRef", (), {"id": run_id})()


@pytest.fixture(scope="session")
def postgres_available() -> bool:
    """Доступность PostgreSQL в текущем окружении."""
    return _postgres_available()


@pytest.fixture
def sample_video_url() -> str:
    """Пример URL видео для тестирования."""
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def sample_run_id() -> str:
    """Пример UUID run'а для тестирования."""
    return "123e4567-e89b-12d3-a456-426614174000"


@pytest.fixture
def sample_platform_video_id() -> str:
    """Пример platform_video_id для тестирования."""
    return "dQw4w9WgXcQ"

