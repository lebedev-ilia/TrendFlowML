"""
Utils для DataProcessor API

Этот пакет содержит утилиты и вспомогательные функции:
- errors.py: Кастомные исключения
- validators.py: Дополнительная валидация payload
- retry.py: Retry логика с exponential backoff
- logging.py: Структурированное логирование
- config_validator.py: Валидация конфигурации при старте
- error_handling.py: Утилиты для обработки ошибок в фоновых задачах

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1385, 2241-2244, 2524-2544, 2231-2416)
"""
