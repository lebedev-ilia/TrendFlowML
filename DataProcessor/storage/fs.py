from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Iterable, Optional, AsyncIterator

from .base import NotFoundError, ObjectInfo


class FileSystemStorage:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = os.fspath(root_dir)

    def _abs(self, key: str) -> str:
        key = str(key).lstrip("/")
        return os.path.join(self.root_dir, key)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._abs(key))

    def list(self, prefix: str) -> Iterable[ObjectInfo]:
        base = self._abs(prefix)
        if not os.path.exists(base):
            return []
        out = []
        for root, _, files in os.walk(base):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, self.root_dir).replace(os.sep, "/")
                try:
                    st = os.stat(p)
                    out.append(ObjectInfo(key=rel, size_bytes=int(st.st_size)))
                except Exception:
                    out.append(ObjectInfo(key=rel))
        return out

    def read_bytes(self, key: str) -> bytes:
        p = self._abs(key)
        if not os.path.exists(p):
            raise NotFoundError(f"FS key not found: {key}")
        return Path(p).read_bytes()

    def write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> None:
        p = Path(self._abs(key))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def atomic_write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> None:
        p = Path(self._abs(key))
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=p.name, dir=str(p.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, str(p))
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
    
    def stream_lines(self, key: str) -> Iterable[str]:
        """
        Потоковое чтение файла построчно (для JSONL файлов).
        
        Не читает весь файл в память, читает построчно.
        
        Args:
            key: Ключ файла
            
        Yields:
            Строки файла
            
        Raises:
            NotFoundError: Если файл не найден
        """
        p = self._abs(key)
        if not os.path.exists(p):
            raise NotFoundError(f"FS key not found: {key}")
        
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line.strip()
    
    async def stream_jsonl(self, key: str) -> AsyncIterator[dict]:
        """
        Async streaming чтение JSONL файла построчно.
        
        Не читает весь файл в память, читает построчно и парсит JSON.
        
        Args:
            key: Ключ файла
            
        Yields:
            Словари с распарсенными JSON объектами
            
        Raises:
            NotFoundError: Если файл не найден
        """
        p = self._abs(key)
        if not os.path.exists(p):
            raise NotFoundError(f"FS key not found: {key}")
        
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Пропустить невалидные строки
                    continue
    
    def generate_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        http_method: str = "GET"
    ) -> str:
        """
        Генерация presigned URL для прямого доступа к файлу.
        
        Для FileSystemStorage возвращает относительный путь (не настоящий presigned URL).
        В production должен использоваться S3Storage с настоящими presigned URL.
        
        Args:
            key: Ключ файла
            expiration: Время жизни URL в секундах (игнорируется для FS)
            http_method: HTTP метод для URL (игнорируется для FS)
            
        Returns:
            Относительный путь к файлу
        """
        # Для FileSystemStorage просто возвращаем путь
        # В production это должно быть заменено на настоящий presigned URL через S3
        return f"/storage/{key}"


