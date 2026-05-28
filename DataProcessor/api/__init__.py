"""
DataProcessor API Module

Этот модуль содержит FastAPI приложение для управления обработкой видео через HTTP API.

Структура:
- main.py: FastAPI приложение и точка входа
- config.py: Настройки API сервера
- dependencies.py: FastAPI dependencies
- endpoints/: API endpoints
- schemas/: Pydantic models для запросов и ответов
- services/: Бизнес-логика
- utils/: Утилиты и вспомогательные функции

Документация:
- См. DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md
- См. DataProcessor/docs/API_DEVELOPMENT_CHECKLIST.md
"""

__version__ = "0.1.0"

