# Миграция Legacy → V2 API (ЗАВЕРШЕНА)

Этот документ описывает завершённую миграцию существующих `/api/*` endpoints на новую доменную модель (`core.*` schema).

**Статус**: ✅ Миграция завершена. V2 API теперь является единственным API.

---

## 1) Результат миграции

### Текущее состояние (после миграции)

**Единственный API (`/api/*`)**:
- **Схема БД**: `core.*` (основные таблицы)
- **Модели**: `backend/app/dbv2/models.py` (User, Workspace, Channel, Video, AnalysisJob, Prediction, ...)
- **Endpoints**: `/api/auth`, `/api/workspaces`, `/api/channels`, `/api/videos`, `/api/analysis`
- **Особенности**:
  - Мультитенантность через workspaces
  - Billing через subscriptions
  - Каналы (channels) как отдельная сущность
  - AnalysisJob вместо Run
  - Интеграция с DataProcessor через адаптер (`dataprocessor_adapter.py`)

**Legacy таблицы (`public.*`)**:
- Используются только для обратной совместимости с DataProcessor
- Таблицы `artifacts`, `run_logs` для хранения результатов анализа
- Не используются основным API

---

## 2) Выполненные этапы миграции

### ✅ Фаза 1: Параллельная работа
- Legacy API и V2 API работали параллельно
- Документация обновлена

### ✅ Фаза 2: Адаптеры для DataProcessor
- Создан `backend/app/services/dataprocessor_adapter.py`
- Маппинг `AnalysisJob` → legacy формат для DataProcessor
- Celery task `process_analysis_job` использует адаптер

### ✅ Фаза 3: Перенос endpoints
- Все endpoints перенесены на v2 API
- Префиксы изменены с `/api/v2/*` на `/api/*`
- Роутеры переименованы (убраны префиксы `v2_`)

### ✅ Фаза 4: Завершение миграции
- Legacy endpoints удалены
- Legacy роутеры удалены
- `deps.py` обновлён для использования v2 моделей
- `tasks_v2.py` переименован в `tasks.py`
- `schemas_v2.py` переименован в `schemas.py`
- Все импорты обновлены
- Документация обновлена

---

## 3) Маппинг сущностей Legacy → V2

### Users
- `public.users` → `core.users`
- **Изменения**: добавлены `email_verified`, `oauth_accounts`, `security`, `memberships`
- **Миграция**: прямая (UUID совпадают)

### Videos
- `public.videos` → `core.videos`
- **Изменения**: добавлены `channel_id`, `video_type`, `storage_path`, `file_size_mb`, `checksum`
- **Миграция**: требует создания workspace и channel для каждого пользователя

### Runs → AnalysisJobs
- `public.runs` → `core.analysis_jobs`
- **Изменения**: добавлены `workspace_id`, `processing_config_id`, `model_version_id`
- **Миграция**: требует маппинга `profile_id` → `processing_config_id`

### Profiles → ProcessingConfigs (будущее)
- `public.analysis_profiles` → (пока нет прямой замены в v2)
- **Текущее решение**: временно используется legacy `analysis_profiles` через `processing_config_id`
- **План**: создать `core.processing_configs` (см. `DATABASE_ARCH.md`)

---

## 3) Чеклист миграции (выполнен)

- [x] Создать v2 API endpoints
- [x] Документировать v2 API
- [x] Создать адаптер для DataProcessor
- [x] Обновить Celery tasks для работы с v2
- [x] Удалить legacy endpoints
- [x] Переименовать `tasks_v2.py` → `tasks.py`
- [x] Переименовать `schemas_v2.py` → `schemas.py`
- [x] Обновить все импорты
- [x] Обновить документацию

---

## 4) Архитектура после миграции

```
API Request (/api/*)
    ↓
V2 Router (workspaces/channels/videos/analysis)
    ↓
V2 Models (core.* schema)
    ↓
Celery Task (process_analysis_job)
    ↓
DataProcessor Adapter (преобразует v2 → legacy формат)
    ↓
DataProcessor (работает как раньше)
    ↓
Результаты → AnalysisJob + Prediction (v2) + legacy artifacts (для совместимости)
```

---

## 5) Следующие шаги (опционально)

1. **Создать data migration** для переноса существующих данных из legacy в v2 (если нужно)
2. **Создать `core.processing_configs`** вместо использования legacy `analysis_profiles`
3. **Обновить клиенты (UI)** для использования нового API (если ещё не обновлены)

---

**Последнее обновление**: 2025-01-XX  
**Статус**: ✅ Миграция завершена
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
