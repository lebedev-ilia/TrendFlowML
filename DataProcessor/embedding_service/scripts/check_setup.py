#!/usr/bin/env python3
"""Скрипт проверки настройки Embedding Service"""

import sys
from pathlib import Path

# Добавить корень DataProcessor в путь
dp_root = Path(__file__).parent.parent.parent
if str(dp_root) not in sys.path:
    sys.path.insert(0, str(dp_root))


def check_dependencies():
    """Проверить установленные зависимости"""
    print("📦 Проверка зависимостей...")
    
    required = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("psycopg2", "psycopg2-binary"),
        ("faiss", "faiss-gpu или faiss-cpu"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
        ("insightface", "insightface"),
    ]
    
    missing = []
    for module, package in required:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - НЕ УСТАНОВЛЕН")
            missing.append(package)
    
    return len(missing) == 0


def check_postgres():
    """Проверить подключение к PostgreSQL"""
    print("\n🗄️  Проверка PostgreSQL...")
    
    try:
        from embedding_service.config.settings import EmbeddingServiceConfig
        from embedding_service.core.database.postgres import PostgresEmbeddingStore
        
        config = EmbeddingServiceConfig()
        
        print(f"  Хост: {config.postgres_host}:{config.postgres_port}")
        print(f"  База: {config.postgres_db}")
        print(f"  Пользователь: {config.postgres_user}")
        
        store = PostgresEmbeddingStore(
            host=config.postgres_host,
            port=config.postgres_port,
            database=config.postgres_db,
            user=config.postgres_user,
            password=config.postgres_password,
        )
        
        print("  ✅ Подключение успешно!")
        
        # Проверить pgvector
        try:
            store._init_schema()
            print("  ✅ pgvector доступен!")
        except Exception as e:
            print(f"  ⚠️  Ошибка pgvector: {e}")
            print("     Убедитесь, что установлено расширение: CREATE EXTENSION vector;")
        
        store.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка подключения: {e}")
        return False


def check_triton():
    """Проверить доступность Triton"""
    print("\n🔧 Проверка Triton Inference Server...")
    
    try:
        from embedding_service.config.settings import EmbeddingServiceConfig
        from dp_triton.http_client import TritonHttpClient
        
        config = EmbeddingServiceConfig()
        
        print(f"  URL: {config.triton_base_url}")
        
        client = TritonHttpClient(
            base_url=config.triton_base_url,
            timeout_sec=5.0,
        )
        
        if client.ready():
            print("  ✅ Triton сервер готов!")
            return True
        else:
            print("  ⚠️  Triton сервер не отвечает")
            print("     Это нормально, если вы используете только ArcFace")
            return False
            
    except ImportError:
        print("  ⚠️  dp_triton не найден (это нормально)")
        return False
    except Exception as e:
        print(f"  ⚠️  Triton недоступен: {e}")
        print("     Это нормально, если вы используете только ArcFace")
        return False


def check_directories():
    """Проверить существование директорий"""
    print("\n📁 Проверка директорий...")
    
    try:
        from embedding_service.config.settings import EmbeddingServiceConfig
        
        config = EmbeddingServiceConfig()
        
        # Проверить FAISS индекс
        faiss_dir = Path(config.faiss_index_path)
        if faiss_dir.exists():
            print(f"  ✅ {faiss_dir} существует")
        else:
            print(f"  ⚠️  {faiss_dir} не существует (будет создан автоматически)")
            faiss_dir.mkdir(parents=True, exist_ok=True)
            print(f"     Создана директория")
        
        # Проверить storage
        if config.storage_type == "local":
            storage_dir = Path(config.storage_local_path)
            if storage_dir.exists():
                print(f"  ✅ {storage_dir} существует")
            else:
                print(f"  ⚠️  {storage_dir} не существует (будет создан автоматически)")
                storage_dir.mkdir(parents=True, exist_ok=True)
                print(f"     Создана директория")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False


def main():
    """Главная функция"""
    print("=" * 60)
    print("🔍 Проверка настройки Embedding Service")
    print("=" * 60)
    
    results = []
    
    results.append(("Зависимости", check_dependencies()))
    results.append(("PostgreSQL", check_postgres()))
    results.append(("Triton", check_triton()))
    results.append(("Директории", check_directories()))
    
    print("\n" + "=" * 60)
    print("📊 Итоги:")
    print("=" * 60)
    
    all_ok = True
    for name, ok in results:
        status = "✅ OK" if ok else "❌ ОШИБКА"
        print(f"  {name}: {status}")
        if not ok and name != "Triton":  # Triton опционален
            all_ok = False
    
    if all_ok:
        print("\n✅ Все проверки пройдены! Сервис готов к запуску.")
        print("\n💡 Запуск сервиса:")
        print("   python embedding_service/run_server.py")
    else:
        print("\n❌ Некоторые проверки не пройдены. Исправьте ошибки и повторите.")
        print("\n📖 См. SETUP.md для детальных инструкций")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

