#!/usr/bin/env python3
"""Скрипт запуска Embedding Service"""

import sys
from pathlib import Path

# Добавить корень DataProcessor в путь
dp_root = Path(__file__).parent.parent
if str(dp_root) not in sys.path:
    sys.path.insert(0, str(dp_root))

import uvicorn
from embedding_service.api.main import create_app
from embedding_service.config.settings import EmbeddingServiceConfig


def main():
    """Запустить Embedding Service"""
    print("🚀 Запуск Embedding Service...")
    
    # Загрузить конфигурацию
    config = EmbeddingServiceConfig()
    
    print(f"📊 База данных: {config.postgres_host}:{config.postgres_port}/{config.postgres_db}")
    print(f"🔧 Triton: {config.triton_base_url}")
    print(f"💾 FAISS индексы: {config.faiss_index_path}")
    
    # Создать приложение (с обработкой ошибок)
    try:
        app = create_app(config)
        print("\n✅ Сервис готов!")
        print(f"📖 API документация: http://localhost:{config.server_port}/docs")
        print(f"❤️  Health check: http://localhost:{config.server_port}/health")
        print("\nНажмите Ctrl+C для остановки\n")
    except Exception as e:
        import traceback
        print(f"\n❌ Ошибка при инициализации сервиса:")
        print(f"   {type(e).__name__}: {e}")
        print("\nВозможные причины:")
        print("  1. PostgreSQL не запущен или недоступен")
        print(f"     Проверьте: psql -h {config.postgres_host} -p {config.postgres_port} -U {config.postgres_user} -d {config.postgres_db}")
        print("  2. База данных не создана")
        print(f"     Создайте: CREATE DATABASE {config.postgres_db};")
        print("  3. Расширение pgvector не установлено")
        print("     Установите: CREATE EXTENSION IF NOT EXISTS vector;")
        print("  4. Неверные настройки подключения в .env или переменных окружения")
        print("     Проверьте POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
        print("\nДетали ошибки:")
        traceback.print_exc()
        raise
    
    # Запустить сервер
    # Для reload нужно использовать строку импорта, но тогда нужно создать app в модуле
    # Альтернатива: использовать объект приложения без reload (или с reload=False)
    # Используем объект приложения - это работает, но reload будет показывать предупреждение
    import logging
    
    # Настроить логирование для детальных ошибок
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        app,
        host=config.server_host,
        port=config.server_port,
        reload=False,  # Отключаем reload для избежания предупреждения (можно включить, если нужно)
        log_level="info",
    )


if __name__ == "__main__":
    main()

