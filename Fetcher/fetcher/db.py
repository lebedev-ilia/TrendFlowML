from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import settings


def _get_ssl_args() -> dict:
    """Получить SSL аргументы для PostgreSQL подключения."""
    ssl_args = {}

    # Если ssl_mode указан в настройках, используем его
    if settings.postgres_ssl_mode:
        ssl_args["sslmode"] = settings.postgres_ssl_mode
    # Иначе пытаемся извлечь из DSN
    elif "sslmode=" in settings.postgres_dsn:
        # sslmode уже в DSN, не добавляем
        pass
    else:
        # По умолчанию не используем SSL (для локальной разработки)
        # В production рекомендуется явно указать sslmode=require
        pass

    return ssl_args


# Создаём engine с поддержкой SSL
connect_args = _get_ssl_args()
engine = create_engine(
    settings.postgres_dsn,
    pool_pre_ping=True,
    future=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def session_scope():
    """Контекстный менеджер для работы с сессией БД Fetcher.

    Пример:

    ```python
    from fetcher.db import session_scope

    with session_scope() as db:
        db.add(obj)
    ```
    """

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


__all__ = ["engine", "SessionLocal", "session_scope"]


