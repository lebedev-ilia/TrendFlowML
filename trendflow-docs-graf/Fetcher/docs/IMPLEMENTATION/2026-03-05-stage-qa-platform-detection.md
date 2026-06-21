# Улучшение определения платформы по URL

**Дата**: 2026-03-05  
**Стадия**: Дополнительные задачи разработки  
**Статус**: ✅ Частично реализовано

## Обзор

Улучшена функция `normalize_source()` для определения платформы по URL и обновлены workers для использования платформы из `run.source_type` или `video_source.platform` вместо hardcoded "youtube".

## Требования

Из оставшихся задач:
- Улучшить `normalize_source()` для определения платформы (youtube, tiktok, instagram, etc.)
- Определение платформы из `run.source_type` в workers (вместо hardcoded "youtube")

## Реализация

### 1. Улучшение `normalize_source()` в `fetcher/orchestrator.py`

**Изменения**:

- Определение платформы по домену URL:
  - `youtube.com` или `youtu.be` → `"youtube"`
  - `tiktok.com` → `"tiktok"` (TODO: реализовать нормализацию)
  - `instagram.com` → `"instagram"` (TODO: реализовать нормализацию)

- Нормализация для YouTube:
  - Использует `yt-dlp` для извлечения `video_id`
  - Обработка ошибок и валидация

- Обработка неподдерживаемых платформ:
  - Вызывает `ValueError` с понятным сообщением

**Пример**:
```python
platform, video_id = normalize_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
# platform = "youtube", video_id = "dQw4w9WgXcQ"
```

### 2. Обновление workers для определения платформы

**Изменения в `fetcher/workers/metadata.py`, `video.py`, `comments.py`**:

- Определение платформы из `video_source.platform` (приоритет 1)
- Fallback на `run.source_type` (приоритет 2)
- Fallback на `"youtube"` (приоритет 3)

**Логика**:
```python
# Определяем платформу из run или из video_source
video_source = db.query(VideoSource).filter(...).first()
if video_source and video_source.platform:
    platform = video_source.platform
elif run.source_type:
    platform = run.source_type.lower()
else:
    platform = "youtube"  # Fallback
```

## Известные ограничения

- **TikTok и Instagram**: Определение платформы работает, но нормализация не реализована (вызывает `ValueError`)
- **Валидация URL**: Базовая валидация есть, но можно добавить более строгую проверку формата URL

## Следующие шаги

- Реализовать нормализацию для TikTok (когда будет TikTokAdapter)
- Реализовать нормализацию для Instagram (когда будет InstagramAdapter)
- Добавить более строгую валидацию URL перед обработкой
- Добавить unit тесты для `normalize_source()` с разными платформами

## Связанные файлы

- `fetcher/orchestrator.py` - функция `normalize_source()`
- `fetcher/workers/metadata.py` - определение платформы
- `fetcher/workers/video.py` - определение платформы
- `fetcher/workers/comments.py` - определение платформы
---

## Навигация

[README](README.md) · [Fetcher](../INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
