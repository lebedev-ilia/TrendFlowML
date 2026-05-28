# Audit v3 — Templates (per-component)

Этот файл содержит короткие шаблоны, чтобы все компоненты документировались одинаково.

## Template A: README компонента (обязательно, single source-of-truth)

Расположение: **в папке компонента**, рядом с кодом.

### 0) TL;DR

- Что делает компонент (1–2 предложения)
- Какие фичи даёт (1–2 предложения)

### 1) Ownership / Versions

- `component_name`
- `producer`: (например `VisualProcessor.core`)
- `producer_version`
- `schema_version`
- `audit_v3_status`: `draft | in_progress | passed`

### 2) Inputs

- **Primary input**:
  - Visual: `frames_dir` + `metadata.json` (Segmenter)
  - Audio: `audio/audio.wav` + `audio/segments.json`
  - Text: источник текста (comments/title/description/transcript/mock) + privacy policy
- **Hard dependencies (no-fallback)**:
  - upstream NPZ / другие компоненты / обязательные файлы
- **Soft dependencies**:
  - что влияет на качество, но не блокирует запуск

### 3) Outputs (NPZ = source-of-truth)

Для каждого output блока:

- **Group name**: (например `sequence_embeddings`, `tabular_summary`, `events`)
  - **Keys**: список ключей NPZ
  - **dtype/shape**: канонично
  - **units / ranges**: где применимо
  - **Downstream class**: `model_facing | analytics | debug-only`
  - **Usefulness (0–10)**: кратко почему
  - **Risk/noise (0–10)**: кратко почему

Дополнение (Audit v3, обязательно):
- рядом с компонентом должен быть `SCHEMA.md` (human schema),
- для VisualProcessor (и далее для других процессоров) должна существовать machine schema в общем реестре схем (keys/dtype/shape/tiers/required).

### 4) Empty vs Error semantics

- **Valid empty cases**:
  - `empty_reason` (из словаря + расширения)
  - какие ключи/маски должны быть выставлены
- **Error cases**:
  - что считается ошибкой (missing deps, invalid sampling, модель не загрузилась, и т.д.)

### 5) Sampling requirements (обязательные)

Компонент **требует** от Segmenter:

- **Visual**:
  - min/target/max кадров
  - coverage (равномерно / начало-середина-конец / shot-stratified / по событиям)
  - min/target/max resolution, запрет/допуск up/downscale
  - требования к `union_timestamps_sec` и к стабильности индексов
- **Audio**:
  - семейство сегментов (`families.*` из `audio/segments.json`)
  - окна/stride, min/target/max сегментов
  - требования к sample_rate/mono, и error если не соблюдены
- **Text**:
  - единица обработки
  - минимальные требования к источнику (что должно быть, что может быть пусто)

### 6) Reproducibility / Model system

- `models_used[]` / `model_signature` (если есть ML)
- seeds / детерминизм / device policy

### 7) Backend contract (что отдаём наружу)

- Какие агрегаты/поля должен получить backend (без HTML)
- Имена ключей, типы, версии

### 8) Testing / Validation

- smoke тест (1–2 видео)
- validation NPZ (обязательно):
  - meta baseline contract
  - schema validation (keys/dtype/shape) через общую систему схем для audited компонентов
- ссылки на рендеры (dev-only)

## Template B: Решение по фиче (Decision Record)

Для каждой спорной/важной фичи фиксируем:

- **Feature**: `component.key` или группа
- **Decision**: `keep | modify | drop`
- **Downstream**: `model_facing | analytics | debug-only`
- **Reason**: почему
- **Expected impact**: на baseline / transformer / UX
- **Schema impact**: нужен ли bump `schema_version`


