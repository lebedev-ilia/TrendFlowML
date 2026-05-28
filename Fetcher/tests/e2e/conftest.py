"""Фикстуры для E2E тестов Fetcher."""

from datetime import datetime
from pathlib import Path

import pytest


class InMemoryStorage:
    """In-memory storage для E2E: сохраняет объекты по (bucket, key) для проверки manifest."""

    def __init__(self):
        self._store: dict[tuple[str, str], bytes] = {}

    def upload_file(self, local_path: str | Path, bucket: str, key: str) -> None:
        path = Path(local_path)
        if path.exists():
            self._store[(bucket, key)] = path.read_bytes()
        else:
            self._store[(bucket, key)] = b""

    def download_file(self, bucket: str, key: str, local_path: str | Path) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(self._store.get((bucket, key), b""))

    def object_exists(self, bucket: str, key: str) -> bool:
        return (bucket, key) in self._store

    def delete_object(self, bucket: str, key: str) -> None:
        self._store.pop((bucket, key), None)

    def list_objects(
        self, bucket: str, prefix: str = "", max_keys: int = 1000
    ) -> list[tuple[str, datetime]]:
        return [
            (k, datetime.utcnow())
            for (b, k) in self._store
            if b == bucket and (not prefix or k.startswith(prefix))
        ][:max_keys]

    def generate_presigned_url(
        self, bucket: str, key: str, expires_in: int = 3600
    ) -> str:
        return f"https://mock-storage/{bucket}/{key}?expires={expires_in}"


@pytest.fixture
def e2e_storage():
    """In-memory storage для E2E (manifest и артефакты)."""
    return InMemoryStorage()
