"""
Утилиты для ретраев с экспоненциальной задержкой.
"""
import time
import logging
from typing import Any, Callable, Optional, List

logger = logging.getLogger(__name__)


def retry_with_backoff(
    func: Callable[[], Any],
    max_attempts: int = 2,
    backoff_base: float = 1.0,
    retry_on: Optional[List] = None,
) -> Any:
    """
    Повторяет функцию с экспоненциальной задержкой при транзиентных ошибках.
    
    Args:
        func: Функция для повторения (callable без аргументов)
        max_attempts: Максимальное количество попыток (по умолчанию: 2)
        backoff_base: Базовая задержка в секундах для экспоненциальной задержки (по умолчанию: 1.0)
        retry_on: Список типов исключений/строк для повторения (None = повторять на всех)
    
    Returns:
        Результат func()
    
    Raises:
        Последнее исключение если все попытки провалились
    """
    if retry_on is None:
        retry_on = []
    
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Проверяем нужно ли повторять
            should_retry = False
            if not retry_on:
                # Повторяем на всех исключениях
                should_retry = True
            else:
                # Проверяем соответствует ли исключение критериям повторения
                for retry_pattern in retry_on:
                    if isinstance(retry_pattern, type) and isinstance(e, retry_pattern):
                        should_retry = True
                        break
                    elif isinstance(retry_pattern, str) and retry_pattern.lower() in error_str:
                        should_retry = True
                        break
            
            if not should_retry or attempt == max_attempts - 1:
                # Не повторяем или последняя попытка
                raise
            
            # Экспоненциальная задержка
            delay = backoff_base * (2 ** attempt)
            logger.warning(f"Retry attempt {attempt + 1}/{max_attempts} after {delay}s: {e}")
            time.sleep(delay)
    
    # Не должно достичь сюда, но на всякий случай
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry failed without exception")


def run_clap_with_oom_fallback(
    extractor,
    audio_path: str,
    tmp_dir: str,
    clap_segments: list,
    initial_batch_size: int,
    segment_parallelism: int = 1,
) -> Any:
    """
    Запускает CLAP extractor с автоматическим OOM fallback (уменьшает batch_size, максимум 2 попытки).
    
    Args:
        extractor: Экземпляр CLAPExtractor
        audio_path: Путь к аудио файлу
        tmp_dir: Временная директория
        clap_segments: Список сегментов
        initial_batch_size: Начальный размер батча
        segment_parallelism: Количество воркеров для параллельной предобработки (по умолчанию: 1)
    
    Returns:
        ExtractorResult
    """
    batch_size = initial_batch_size
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            return extractor.run_segments(  # type: ignore[attr-defined]
                audio_path, tmp_dir, clap_segments, segment_parallelism=segment_parallelism, max_inflight=1, model_batch_size=batch_size
            )
        except (RuntimeError, Exception) as e:
            error_str = str(e).lower()
            # Проверяем это ли OOM
            is_oom = (
                "out of memory" in error_str or
                "cuda out of memory" in error_str or
                "oom" in error_str or
                ("memory" in error_str and ("error" in error_str or "failed" in error_str))
            )
            
            if is_oom and attempt < max_attempts - 1 and batch_size > 1:
                # Уменьшаем batch_size и повторяем
                batch_size = max(1, batch_size // 2)
                logger.warning(f"OOM detected for clap_extractor, reducing batch_size to {batch_size} and retrying (attempt {attempt + 1}/{max_attempts})")
                continue
            else:
                # Не OOM или последняя попытка, пробрасываем дальше
                raise
    
    # Не должно достичь сюда
    raise RuntimeError(f"OOM fallback failed for clap_extractor after {max_attempts} attempts")

