FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . /app/

# Переменные окружения по умолчанию
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose порт для API
EXPOSE 8000

# По умолчанию запускаем API (можно переопределить в docker-compose)
CMD ["uvicorn", "fetcher.api:app", "--host", "0.0.0.0", "--port", "8000"]

