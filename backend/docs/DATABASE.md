# Database schema (as implemented)

Модели определены в `backend/app/models.py`. Это **фактическая** схема,
которая создаётся на старте через `Base.metadata.create_all`.

## 1) Таблицы

### users

- `id` (uuid string)
- `email` (unique)
- `password_hash`
- `role` (`user`/`admin`)
- `created_at`, `updated_at`

### videos

- `id`
- `platform_id` (`upload` сейчас)
- `video_id` (каноничный id)
- `source_type` (`upload`)
- `title/description/language/category`
- `created_at`

### video_files

- `id`
- `sha256_hex` (unique)
- `size_bytes`
- `mime_type`
- `object_key` (путь к raw файлу)
- `created_at`
- `retention_until` (не используется логикой)

### video_sources

Связь видео с источником и метаданными.

- `video_id` (FK → videos)
- `youtube_url` (не используется)
- `uploaded_file_id` (FK → video_files)
- `fetched_at`, `duration_sec`, `width`, `height`

### user_video_links

ACL таблица: кто имеет доступ к видео.

- `user_id` (FK → users)
- `video_id` (FK → videos)
- `created_at`

### analysis_profiles

- `id`
- `user_id` (nullable: public profiles)
- `name`, `description`
- `is_public`
- `config_json` (jsonb)
- `config_hash`
- `created_at`, `updated_at`

### profile_components

В коде пока не используется, но создана для будущей детализации.

- `profile_id` (FK → analysis_profiles)
- `component_name`
- `enabled`, `required`
- `component_params`
- `cost_units`

### runs

- `id`
- `user_id` (FK → users)
- `video_id` (FK → videos)
- `profile_id` (nullable)
- `config_hash`
- `status` (`queued|running|succeeded|failed|cancelled`)
- `stage` (`segmenter|audio|text|visual|render`)
- `created_at`, `started_at`, `finished_at`
- `cancel_requested_at`
- `error_code`, `error_message`
- `estimated_cost_units`, `actual_cost_units` (пока не заполняются)

### run_components

- `run_id`
- `component_name`
- `status`
- `schema_version`, `producer_version`
- `started_at`, `finished_at`, `duration_ms`
- `device_used`
- `empty_reason`, `error_code`, `error_message`
- `cost_units`

### artifacts

- `id`
- `run_id`
- `component_name`
- `kind` (`npz|json|html|...`)
- `object_key` (путь)
- `size_bytes`, `sha256_hex`
- `created_at`

### run_logs

- `id` (bigserial)
- `run_id`
- `ts`
- `level` (`info|warning|error|debug`)
- `message`

### uploads

Служебная таблица для upload‑flow.

- `id`
- `user_id`
- `video_id`
- `status` (`init|uploaded|completed`)
- `temp_path`, `filename`
- `created_at`, `updated_at`

## 2) Индексы и миграции

Сейчас индексы не создаются вручную (кроме `unique`/`primary key`),
а миграции отсутствуют: используется `create_all` на старте приложения.

## 3) Что **не** реализовано в схеме backend

Сравнение с каноничными контрактами:

- нет `billing_ledger`, `user_balances`, `render_cache`
- нет `profile_model_mapping`
- нет аналитических индексов

Список расхождений фиксируется в `backend/docs/GAPS_AND_ALIGNMENT.md`.

