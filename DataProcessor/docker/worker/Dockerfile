# DataProcessor Worker Dockerfile
#
# Этот Dockerfile создает образ для запуска DataProcessor Worker процесса.
# Worker читает задачи из Redis Streams и обрабатывает их через subprocess isolation.
#
# Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1803-1822)
#
# Сборка:
#   docker build -f docker/worker/Dockerfile -t dataprocessor-worker:latest .
#
# Запуск:
#   docker run dataprocessor-worker:latest

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
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements для API (worker использует те же зависимости)
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Копирование кода DataProcessor
# Копируем весь проект, так как worker зависит от модулей storage, state и т.д.
COPY . .

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STORAGE_TYPE=fs
ENV STORAGE_ROOT=/data
ENV LOG_LEVEL=INFO
ENV LOG_FORMAT=json
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_DB=0
ENV REDIS_URL=redis://redis:6379/0

# Создание директорий для данных
RUN mkdir -p /data/result_store /data/frames_dir /data/profiles_cache

# Worker не открывает порты (работает только с Redis)

# Запуск worker процесса
CMD ["python", "-m", "api.worker"]
