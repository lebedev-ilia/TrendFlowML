# syntax=docker/dockerfile:1.4
#
# Контекст = каталог backend/ (~несколько МБ), профили — отдельный build-context `profiles`
# (Compose: additional_contexts; вручную из backend/: docker build -f Dockerfile --build-context profiles=../DataProcessor/profiles .)
#
# В образе: /srv/backend — приложение, /srv/DataProcessor/profiles — yaml для seed на старте.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /srv/DataProcessor
COPY --from=profiles . /srv/DataProcessor/profiles/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
