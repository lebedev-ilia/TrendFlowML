from __future__ import annotations

"""Простой proxy-пул для Fetcher (MVP).

На этом этапе реализуем наивный round-robin поверх списка proxies из настроек
и базовые метрики отказов на proxy.
"""

from itertools import cycle
from threading import Lock
from typing import Dict, Iterator, Optional, Tuple

from .config import settings
from .db import session_scope
from .metrics import proxy_failure_rate
from .models import Proxy, ProxyUsage

_lock = Lock()
_proxy_cycle: Optional[Iterator[str]] = None
# В памяти считаем (successes, failures) по каждому proxy_id (строка URL)
_proxy_stats: Dict[str, Tuple[int, int]] = {}
# Порог failure rate для автоматического исключения прокси (по умолчанию 50%)
_proxy_failure_threshold: float = 0.5
# Минимальное количество запросов для оценки (чтобы не исключать прокси после 1 ошибки)
_proxy_min_requests: int = 10


def _ensure_cycle() -> Optional[Iterator[str]]:
    global _proxy_cycle
    with _lock:
        if not settings.enable_proxies or not settings.proxies:
            _proxy_cycle = None
            return None
        if _proxy_cycle is None:
            _proxy_cycle = cycle(settings.proxies)
        return _proxy_cycle


def get_proxy_health_score(proxy_url: str) -> float:
    """Получить health score для proxy (0.0 = плохой, 1.0 = отличный).

    Health score = 1.0 - failure_rate

    Args:
        proxy_url: URL прокси

    Returns:
        Health score от 0.0 до 1.0
    """
    with _lock:
        success_count, failure_count = _proxy_stats.get(proxy_url, (0, 0))
        total = success_count + failure_count
        if total == 0:
            return 1.0  # Нет данных = считаем хорошим
        failure_rate = failure_count / total
        return 1.0 - failure_rate


def is_proxy_healthy(proxy_url: str) -> bool:
    """Проверить, здоров ли proxy (не превышен ли порог failure rate).

    Args:
        proxy_url: URL прокси

    Returns:
        True если proxy здоров (failure_rate < threshold), False иначе
    """
    with _lock:
        success_count, failure_count = _proxy_stats.get(proxy_url, (0, 0))
        total = success_count + failure_count
        if total < _proxy_min_requests:
            # Недостаточно данных для оценки
            return True
        failure_rate = failure_count / total
        return failure_rate < _proxy_failure_threshold


def get_next_proxy(country: Optional[str] = None) -> Optional[str]:
    """Вернуть следующий здоровый proxy URL из пула или None.

    Пропускает прокси с высоким failure rate (автоматическое исключение плохих прокси).
    Поддерживает geographic rotation (выбор прокси по стране).

    Args:
        country: ISO 3166-1 alpha-2 код страны для geographic rotation (опционально)

    Returns:
        URL прокси или None, если пул пуст или все прокси нездоровы
    """
    # Если указана страна, пытаемся найти прокси из этой страны
    if country:
        try:
            with session_scope() as db:
                proxies: list[Proxy] = (
                    db.query(Proxy)
                    .filter(Proxy.enabled == True)
                    .filter(Proxy.country == country.upper())
                    .all()
                )
                if proxies:
                    # Выбираем здоровый прокси из указанной страны
                    for proxy in proxies:
                        if is_proxy_healthy(proxy.url):
                            return proxy.url
        except Exception:
            # При ошибке БД fallback на обычный round-robin
            pass

    # Fallback: обычный round-robin из настроек
    it = _ensure_cycle()
    if it is None:
        return None

    # Пробуем найти здоровый прокси (максимум N попыток, чтобы не зависнуть)
    max_attempts = len(settings.proxies) * 2 if settings.proxies else 1
    for _ in range(max_attempts):
        try:
            proxy = next(it)
            if is_proxy_healthy(proxy):
                return proxy
            # Прокси нездоров, пробуем следующий
        except Exception:
            return None

    # Все прокси нездоровы или пул пуст
    return None


def record_proxy_result(
    proxy_url: Optional[str],
    success: bool,
    operation: str = "unknown",
    latency_ms: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """Записать результат запроса через proxy и обновить proxy_failure_rate.

    Обновляет in-memory статистику, метрики Prometheus и логирует в БД (proxy_usage).

    Args:
        proxy_url: URL прокси
        success: Успешность запроса
        operation: Тип операции (metadata, download, comments)
        latency_ms: Время ответа в миллисекундах (опционально)
        error_message: Сообщение об ошибке (опционально)
    """
    if not proxy_url:
        return

    # Обновляем in-memory статистику
    with _lock:
        success_count, failure_count = _proxy_stats.get(proxy_url, (0, 0))
        if success:
            success_count += 1
        else:
            failure_count += 1
        _proxy_stats[proxy_url] = (success_count, failure_count)
        total = success_count + failure_count
        if total > 0:
            rate = failure_count / total
            # Получаем country из БД, если прокси там есть
            country = "unknown"
            try:
                with session_scope() as db:
                    proxy: Optional[Proxy] = (
                        db.query(Proxy).filter(Proxy.url == proxy_url).first()
                    )
                    if proxy and proxy.country:
                        country = proxy.country
            except Exception:
                # Игнорируем ошибки БД для метрик
                pass

            proxy_failure_rate.labels(proxy_id=proxy_url, country=country).set(rate)

    # Логируем в БД (best-effort, не блокируем основной поток)
    try:
        with session_scope() as db:
            proxy: Optional[Proxy] = db.query(Proxy).filter(Proxy.url == proxy_url).first()
            if proxy:
                usage_log = ProxyUsage(
                    proxy_id=proxy.id,
                    operation=operation,
                    success=success,
                    latency_ms=latency_ms,
                    error_message=error_message,
                )
                db.add(usage_log)
                # commit происходит автоматически при выходе из session_scope
    except Exception as e:
        # Логируем ошибку, но не прерываем основной поток
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to log proxy usage to DB: {e}", exc_info=False)


__all__ = [
    "get_next_proxy",
    "record_proxy_result",
    "get_proxy_health_score",
    "is_proxy_healthy",
]


