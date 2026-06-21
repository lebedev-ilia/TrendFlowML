# Database schema

Backend использует **PostgreSQL** и **SQLAlchemy 2.0**.

Основная схема доменной модели — **`core.*`** (модели в `backend/app/dbv2/models.py`). В том же проекте Alembic может управлять **legacy**-метаданными в `public` (см. `alembic/env.py`, `app/models.py`); новый API ориентирован на `core.*`.

> **Для полной спецификации** с дополнительными таблицами (subscription_usage, billing_transactions, video_snapshots, processing_configs, explainability_summary, recommendations, api_keys, audit_logs и др.) см. [DATABASE_ARCH.md](DATABASE_ARCH.md).

## 1) Таблицы (core.*)

### Users & Auth

#### core.users
- `id` (UUID, primary key)
- `email` (unique, not null)
- `email_verified` (boolean, default false)
- `password_hash` (string, nullable)
- `created_at`, `updated_at` (timestamps)
- `deleted_at` (timestamp, nullable, soft delete)

#### core.user_oauth_accounts
- `id` (UUID, primary key)
- `user_id` (FK → users)
- `provider` (string, e.g. "google", "github")
- `provider_user_id` (string)
- `access_token`, `refresh_token` (text, nullable)
- `token_expires_at` (timestamp, nullable)
- Unique constraint: `(provider, provider_user_id)`

#### core.user_security
- `user_id` (UUID, primary key, FK → users)
- `two_factor_enabled` (boolean, default false)
- `two_factor_secret` (string, nullable)
- `password_reset_token` (string, nullable)
- `password_reset_expires_at` (timestamp, nullable)

### Workspaces

#### core.workspaces
- `id` (UUID, primary key)
- `name` (string, not null)
- `slug` (string, unique, not null)
- `owner_user_id` (UUID, FK → users)
- `archived_at` (timestamp, nullable)
- `created_at`, `updated_at` (timestamps)

#### core.workspace_members
- `id` (UUID, primary key)
- `workspace_id` (UUID, FK → workspaces)
- `user_id` (UUID, FK → users)
- `role` (enum: `owner|admin|editor|viewer`)
- `invited_by` (UUID, FK → users, nullable)
- `joined_at` (timestamp)
- `archived_at` (timestamp, nullable)
- Unique constraint: `(workspace_id, user_id)`

### Billing

#### core.subscription_plans
- `id` (integer, primary key)
- `name` (string, not null)
- `max_videos_per_month` (integer)
- `max_analyses_per_month` (integer)
- `max_channels` (integer)
- `max_storage_gb` (integer)
- `has_api_access` (boolean)
- `has_advanced_explainability` (boolean)
- `price` (float)

#### core.subscriptions
- `id` (UUID, primary key)
- `workspace_id` (UUID, FK → workspaces)
- `plan_id` (integer, FK → subscription_plans)
- `status` (enum: `active|canceled|expired`)
- `current_period_start` (timestamp)
- `current_period_end` (timestamp)
- `cancel_at_period_end` (boolean, default false)
- `created_at`, `updated_at` (timestamps)

### Channels

#### core.channels
- `id` (UUID, primary key)
- `workspace_id` (UUID, FK → workspaces)
- `platform` (string, e.g. "youtube", "tiktok")
- `external_channel_id` (string, nullable)
- `channel_name` (string)
- `connected_oauth_id` (UUID, FK → user_oauth_accounts, nullable)
- `archived_at` (timestamp, nullable)
- `created_at`, `updated_at` (timestamps)
- Index: `(platform, external_channel_id)`

### Videos

#### core.videos
- `id` (UUID, primary key)
- `channel_id` (UUID, FK → channels)
- `external_video_id` (string, nullable)
- `title` (string)
- `description` (text, nullable)
- `duration_seconds` (integer)
- `video_type` (enum: `shorts|video`)
- `source_type` (enum: `upload|link`)
- `source_url` (string, nullable)
- `storage_path` (string, nullable)
- `file_size_mb` (float, nullable)
- `checksum` (string, nullable)
- `archived_at` (timestamp, nullable)
- `created_at`, `updated_at` (timestamps)
- Unique constraint: `(channel_id, external_video_id)`

### Analysis

#### core.analysis_jobs
- `id` (UUID, primary key)
- `workspace_id` (UUID, FK → workspaces)
- `video_id` (UUID, FK → videos)
- `triggered_by_user_id` (UUID, FK → users)
- `processing_config_id` (UUID)  # Временная ссылка на legacy analysis_profiles
- `model_version_id` (string)
- `status` (enum: `queued|processing|completed|failed|canceled`)
- `retry_count` (integer, default 0)
- `error_message` (text, nullable)
- `started_at` (timestamp, nullable)
- `completed_at` (timestamp, nullable)
- `created_at`, `updated_at` (timestamps)
- Indexes: `(workspace_id)`, `(video_id)`

#### core.ingestion_runs
Ингестия по URL (YouTube и др.); связь с Fetcher. PK — `run_id` (тот же UUID, что уходит в Fetcher).

- `run_id` (UUID, primary key)
- `user_id` (UUID, FK → users)
- `source_url` (string)
- `workspace_id` (UUID, FK → workspaces, nullable)
- `ingestion_status` (string, например `pending|running|completed|failed`)
- `idempotency_key` (string, nullable, unique)
- `fetcher_stage`, `fetcher_error_code`, `fetcher_error_message` — синхронизация из Fetcher (polling)
- `created_at`, `updated_at` (timestamps)
- Indexes: `(user_id)`, `(workspace_id)`

#### core.predictions
- `id` (UUID, primary key)
- `analysis_job_id` (UUID, FK → analysis_jobs)
- `horizon_days` (integer)
- `predicted_views` (float)
- `predicted_likes` (float)
- `percentile_score` (float)
- `confidence_lower` (float)
- `confidence_upper` (float)
- `model_version_id` (string)
- `created_at`, `updated_at` (timestamps)

## 2) ENUM типы (core.*)

Все ENUM-типы находятся в schema `core`:

- `core.workspace_role`: `owner`, `admin`, `editor`, `viewer`
- `core.subscription_status`: `active`, `canceled`, `expired`
- `core.video_type`: `shorts`, `video`
- `core.source_type`: `upload`, `link`
- `core.analysis_status`: `queued`, `processing`, `completed`, `failed`, `canceled`

## 3) Миграции (Alembic)

Конфиг:
- `backend/alembic.ini`
- `backend/alembic/env.py`
- миграции: `backend/alembic/versions/*`

Запуск:

```bash
cd backend
export TF_BACKEND_DB_DSN="postgresql+psycopg://trendflow:trendflow@localhost:5432/trendflow"
alembic -c alembic.ini upgrade head
```

Миграции:
- `0001_core_init`: создаёт schema `core`, ENUM-типы и все таблицы
- `0002_legacy_init`: создаёт legacy таблицы в schema `public` (для обратной совместимости с DataProcessor)

## 4) Auto-create (dev режим)

Если `TF_BACKEND_DB_AUTO_CREATE=true`, то на старте приложения выполняется:
- `CREATE SCHEMA IF NOT EXISTS core`
- `BaseV2.metadata.create_all(bind=engine)`

**Рекомендация**: в production использовать `TF_BACKEND_DB_AUTO_CREATE=false` и полагаться на Alembic миграции.

## 5) Legacy таблицы (для обратной совместимости)

Legacy таблицы в schema `public.*` используются только для:
- Обратной совместимости с DataProcessor (через адаптер)
- Хранения артефактов и логов (legacy таблицы `artifacts`, `run_logs`)

Эти таблицы создаются через миграцию `0002_legacy_init` и не используются основным API.

## 6) Что **не** реализовано (gap относительно целевой архитектуры)

Сравнение с каноничными контрактами:

- нет `billing_ledger`, `user_balances`, `render_cache` в core.*
- нет `processing_configs` в core.* (временно используется legacy `analysis_profiles`)
- нет аналитических индексов

Список расхождений фиксируется в `backend/docs/GAPS_AND_ALIGNMENT.md`.
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
