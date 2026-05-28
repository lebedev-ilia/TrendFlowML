# DataProcessor API Dockerfile
#
# Этот Dockerfile создает образ для запуска DataProcessor API сервера.
# Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1698-1732)
#
# Сборка:
#   docker build -f docker/api/Dockerfile -t dataprocessor-api:latest .
#
# Запуск:
#   docker run -p 8000:8000 dataprocessor-api:latest

FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
# ffmpeg нужен для работы с видео файлами
# libsm6, libxext6, libxrender-dev нужны для некоторых ML библиотек
# libgomp1 нужен для OpenMP (используется в некоторых ML библиотеках)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements для API
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Копирование кода DataProcessor
# Копируем весь проект, так как API зависит от модулей storage, state и т.д.
COPY . .

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STORAGE_TYPE=fs
ENV STORAGE_ROOT=/data
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_LEVEL=INFO
ENV LOG_FORMAT=json
ENV MAX_CONCURRENT_RUNS=4

# Создание директорий для данных
RUN mkdir -p /data/result_store /data/frames_dir /data/profiles_cache

# Порт API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Запуск API сервера
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

