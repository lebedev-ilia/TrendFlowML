"""
OpenTelemetry Tracing Service

Этот модуль предоставляет функции для создания и управления трейсами OpenTelemetry.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2546-2562)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Глобальная переменная для tracer
_tracer = None


def get_tracer():
    """
    Получить tracer для создания spans.
    
    Returns:
        Tracer instance или None если tracing не настроен
    """
    global _tracer
    
    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer(__name__)
        except ImportError:
            logger.debug("OpenTelemetry not available")
            return None
    
    return _tracer


def create_span(name: str, **attributes):
    """
    Создать span для трейсинга.
    
    Args:
        name: Имя span'а
        **attributes: Атрибуты span'а
        
    Returns:
        Context manager для span'а или None если tracing не настроен
        
    Пример:
        ```python
        with create_span("process_video", run_id=run_id, video_id=video_id):
            # код обработки
        ```
    """
    tracer = get_tracer()
    if not tracer:
        # Возвращаем dummy context manager если tracing не настроен
        from contextlib import nullcontext
        return nullcontext()
    
    span = tracer.start_as_current_span(name)
    
    # Установить атрибуты
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, str(value))
    
    return span

