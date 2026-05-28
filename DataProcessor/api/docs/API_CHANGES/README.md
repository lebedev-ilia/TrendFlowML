# Изменения в API

Эта директория содержит документы об изменениях в API: endpoints, schemas, responses и т.д.

## Структура

Каждый документ именуется по формату:
```
YYYY-MM-DD-change-description.md
```

Где:
- `YYYY-MM-DD` - дата изменения
- `change` - тип изменения (endpoint, schema, response, etc.)
- `description` - краткое описание

## Примеры

- `2024-01-XX-endpoint-process-added.md` - Добавлен endpoint POST /process
- `2024-01-XX-schema-processrequest-updated.md` - Обновлена схема ProcessRequest
- `2024-01-XX-response-runstatus-added.md` - Добавлен RunStatusResponse

## Формат документа

Каждый документ должен содержать:

1. **Метаданные**
   - Дата изменения
   - Версия API
   - Тип изменения (добавление/изменение/удаление)

2. **Описание изменения**
   - Что изменилось
   - Почему изменилось
   - Какие endpoints/schemas затронуты

3. **Детали**
   - Старое поведение (если изменение)
   - Новое поведение
   - Примеры запросов/ответов

4. **Миграция**
   - Как мигрировать существующий код
   - Breaking changes (если есть)
   - Deprecation timeline (если есть)

5. **Связанные изменения**
   - Ссылки на другие документы
   - Связанные задачи/PR

