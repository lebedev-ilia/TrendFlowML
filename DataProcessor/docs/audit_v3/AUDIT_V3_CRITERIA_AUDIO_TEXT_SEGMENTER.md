# Audit v3 — критерии для AudioProcessor + TextProcessor (+ Segmenter в конце) под финальные контракты моделей (v2)
Дата: 2026-02-19  
Связанные документы (source-of-truth):
- Audit rules: `DataProcessor/docs/audit_v3/DECISIONS_AND_RULES.md`
- AudioProcessor preflight rules (source-of-truth): `DataProcessor/docs/audit_v3/AUDIOPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md`
- TextProcessor preflight rules (source-of-truth): `DataProcessor/docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md`
- Артефакты/NPZ/meta/empty_reason: `DataProcessor/docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- Общая система схем: `DataProcessor/docs/contracts/SCHEMAS_SYSTEM.md`
- Segmenter contract: `DataProcessor/docs/contracts/SEGMENTER_CONTRACT.md`
- Privacy/retention: `DataProcessor/docs/contracts/PRIVACY_AND_RETENTION.md`
- Финальный интерфейс моделей v2: `Models/docs/contracts/MODEL_INTERFACE_V2.md`
- Шаблоны аудита (README/Decision Record): `DataProcessor/docs/audit_v3/TEMPLATES.md`
- Run log (как фиксируем прогоны): `DataProcessor/docs/audit_v3/RUN_LOG.md`

Этот документ — “боевой” чеклист и процедура, с которой мы можем **просто идти по компонентам Audio/Text** и доводить их до состояния “audited (v3)”.

---

## 0) Позиционирование: что именно делаем в Audit v3 для Audio/Text

Мы доводим компоненты до **production-ready по логике/контрактам/полезности фич**:

- **NPZ = source-of-truth** (рендеры/JSON = dev-only, presentation/debug).
- После аудита алгоритмы считаются “стабильными” (меняем только флаги/пресеты и версии).
- Цель — согласовать выходы процессоров с **двумя путями моделей** (см. `MODEL_INTERFACE_V2.md`):
  - **Tabular path**: стабильные скаляры по `feature_names/feature_values` (baseline/QA/fallback).
  - **Token path (vNext)**: readiness публиковать/собирать `TokenStreams` (embeddings/sequences/events) без хранения raw текста.

---

## 1) Процедура аудита одного компонента (шаги, которые повторяем всегда)

### 1.1 Быстрый старт (что нужно собрать перед изменениями)

- **Определить компонент**:
  - имя (`component_name`)
  - владелец (AudioProcessor/TextProcessor)
  - версия артефакта (`schema_version`) и версия продюсера (`producer_version`)
- **Определить входы**:
  - upstream зависимости (какой компонент даёт данные)
  - контрактные файлы (например `frames_dir/audio/segments.json`, `VideoDocument` поля)
- **Выбрать validation set (обязательная дисциплина Audit v3)**:
  - **Fixed smoke set** (проверка empty semantics / “не падать без аудио”): `example/example_videos/video*_fixed.mp4`
  - **Audio-present set** (проверка реальной аудио‑экстракции): `example/example_videos/video*.mp4`
  - Источник истины по наборам + правилам использования: `DataProcessor/docs/audit_v3/AUDIOPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md`
- **Открыть реальные артефакты** (лучше по 3–5 видео/примеров из соответствующего набора):
  - `DataProcessor/dp_results/youtube/.../<component>/*.npz`
  - рендеры `.../<component>/_render/render.html`

### 1.2 Декомпозиция выхода на tiers (обязательный этап)

Для каждого ключа/группы фичей решаем и фиксируем:

- `model_facing`: пойдёт в baseline/vNext.
- `analytics`: полезно для анализа, но не хотим нести в модель без доказательств.
- `debug-only`: только QA/рендер/диагностика.

Правило: шумные/эвристические — либо удаляем, либо переводим в `analytics/debug-only` (см. `DECISIONS_AND_RULES.md`).

### 1.3 Empty semantics (обязательный этап)

Для компонента фиксируем:

- `status ∈ {ok, empty, error}`
- каноничный `empty_reason` (используем словарь из `ARTIFACTS_AND_SCHEMAS.md`, расширяем только явно)
- NaN + masks policy (никаких “нулей-заглушек” вместо missing)

### 1.4 Sampling requirements (обязательный этап)

Компонент **сам формулирует требования к sampling** (даже если sampling исполняет Segmenter):

- Audio: какие families в `audio/segments.json` обязательны, какие параметры (окна/кол-во сегментов/кривая).
- Text: лимиты и политика отбора (например max comments, chunking, token budgets).

Segmenter аудируется в конце, но требования компонентов фиксируются сейчас.

### 1.5 Схемы (обязательный этап)

Каждый audited компонент обязан иметь:

- **Human schema**: `SCHEMA.md` рядом с кодом компонента
- **Machine schema**: JSON-схема (vp_schema_v1 формат) + runtime validation

Ключевое правило optional: **если фича отключена — ключ должен отсутствовать**, а не быть `None`.

### 1.6 Render/QA (обязательный этап)

Рендер — мини-дашборд уровня аудита (см. `DECISIONS_AND_RULES.md`):

- Key facts (версии, модели, размеры, timings)
- Top / Anti-top
- Таблицы, фильтры, примеры
- Offline (без CDN)
- Privacy баннер, если есть текст/PII

### 1.7 Acceptance критерии компонента (фиксируем и закрываем)

Компонент считается “audited v3”, когда:

- контракт артефакта стабилен (schema + tiers + empty semantics + versioning),
- есть доказательство полезности/корректности (рендер + sanity distributions),
- входы/зависимости и sampling requirements документированы,
- no-network соблюдён (`dp_models`).

### 1.8 Правило “стабильность фичей” (baseline-friendly)

Если компонент публикует табличные фичи (`feature_names/feature_values`):

- **Запрещено** менять порядок `feature_names` в рамках одного `schema_version`.
- **Разрешено** добавлять новые фичи **только**:
  - через bump `schema_version` (если audited/known schema), либо
  - через явный “движущийся” режим, но тогда компонент не считается audited.
- Для удаления/переименования — тоже bump `schema_version`.

Это ключевой анти-риск для baseline dataset/training и для сравнимости прогонов.

### 1.9 Run-log дисциплина (обязательный этап аудита)

После любых изменений компонента (или существенного уточнения контрактов) мы обязаны добавить запись в:

- `DataProcessor/docs/audit_v3/RUN_LOG.md`

**Зачем**: чтобы воспроизводимо подтверждать, что правки реально работают, и чтобы фиксировать **фактический sampling** (пока Segmenter не аудирован финально).

Минимальный формат записи (ориентируемся на стиль Visual audit):

- Run id: `<platform>/<video_id>/<run_id>` + список компонентов
- **Команда запуска (факт)** (entrypoint + args, включая profile/config)
- **Конфиг (важный фрагмент)**: только те knobs, которые влияют на качество/стоимость/политику (batching, retain flags, model specs)
- **Результаты / артефакты**:
  - `manifest.json`
  - NPZ paths + `schema_version`/`producer_version`
  - render paths (dev-only)
- **Быстрая валидация артефакта**:
  - keys присутствуют
  - базовые shape/dtype sanity
  - time alignment (если применимо)
- **Sampling (факт)**:
  - Audio: число сегментов, распределение длительностей, фактический “шаг” по времени
  - Text: сколько comments/чанков реально использовано, какие лимиты сработали

Если запись в run-log отсутствует — компонент не считаем закрытым как audited.

---

## 2) Общие требования (для AudioProcessor и TextProcessor)

### 2.1 Артефакты и meta

- NPZ обязателен, и в нём обязателен `meta` со стандартными полями (см. `ARTIFACTS_AND_SCHEMAS.md`):
  - `producer`, `producer_version`, `schema_version`, `created_at`
  - `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
  - `status`, `empty_reason` (если empty)
- Если компонент использует модели:
  - `models_used[]` + `model_signature` (через общие meta_builder)

### 2.2 Версионирование

- Любое изменение смысла/формата ключей/dtype/shape/empty semantics → bump `schema_version`.
- Любое изменение алгоритма при сохранении контракта → bump `producer_version`.

### 2.3 Runtime validation

- audited компонент должен иметь “known schema” и валидироваться fail-fast (см. `SCHEMAS_SYSTEM.md`).

### 2.4 Privacy / retention / logging

- raw ASR/OCR/comments/text **не сохраняем** по умолчанию (см. `PRIVACY_AND_RETENTION.md`).
- Если есть retain флаг — он должен:
  - быть явно указан в README/SCHEMA,
  - переводить raw поля в `debug-only`,
  - предупреждать в render.
- В логах по умолчанию: только длины/хэши/статусы, без raw текста.

### 2.5 Связь с моделями v2

Компонент должен явно описывать, как он участвует:

- **Tabular path**: какие `feature_names/feature_values` являются model_facing, каков frozen subset.
- **Token path**: какие sequences/embeddings/events могут быть собраны в `TokenStreams`:
  - `tokens`, `times_s`/`spans_s`, `mask`, optional `meta_json`.

### 2.6 Manifest.json (run-level truth) — обязательные ожидания аудита

Компонент обязан корректно отражаться в `manifest.json` (см. `ARTIFACTS_AND_SCHEMAS.md`):

- `components[].name` = имя компонента/экстрактора
- `components[].status` = ok|empty|error
- `components[].producer_version` и `components[].schema_version`
- `components[].artifacts[]` содержит пути к NPZ и (если есть) sub-artifacts (`.npy`, assets)
- `duration_ms`, `started_at`, `finished_at`, `device_used` (если применимо)

Аудит считается незакрытым, если артефакты не найдены в manifest или статусы расходятся с meta.

### 2.7 Документация компонента (обязательный формат)

Каждый audited компонент обязан иметь README в формате шаблона:

- `DataProcessor/docs/audit_v3/TEMPLATES.md` → **Template A**

Критично:

- `audit_v3_status: draft|in_progress|passed` (обновлять при прохождении)
- раздел “Sampling requirements” обязателен
- раздел “Render (dev-only)” обязателен

Для спорных/шумных фичей — фиксируем решение через **Decision Record** (Template B).

---

## 3) Rollout системы схем для Audio/Text (куда класть machine schema)

До внедрения общего реестра (как у Visual) используем практический стандарт:

- Human schema:
  - `DataProcessor/AudioProcessor/src/extractors/<name>/SCHEMA.md`
  - `DataProcessor/TextProcessor/src/extractors/<name>/SCHEMA.md`
- Machine schema (vp_schema_v1):
  - `DataProcessor/AudioProcessor/schemas/<schema_version>.json`
  - `DataProcessor/TextProcessor/schemas/<schema_version>.json`

Если в процессе окажется удобнее сделать единый реестр `DataProcessor/schemas/` — фиксируем решением и мигрируем.

### 3.1 Machine schema минимальный шаблон (vp_schema_v1)

Для каждого `schema_version` машинная схема должна фиксировать:

- `schema_system_version="vp_schema_v1"`
- `schema_version` (совпадает с `meta.schema_version`)
- `producer` (совпадает с `meta.producer`)
- `artifact_kind="npz"`
- `allow_extra_keys=false` для audited компонентов
- `meta.required_keys`: минимум
  - `producer`, `producer_version`, `schema_version`, `created_at`
  - `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
  - `status`, `empty_reason`
- `fields`: минимум должны быть описаны:
  - `feature_names`, `feature_values` (если табличный компонент)
  - все model_facing arrays/sequences + их masks/times
  - `meta`

---

## 4) Segmenter (audit v3 — выполняем в конце, но критерии фиксируем заранее)

### 4.1 Time-axis contract (DoD)

- Visual: `union_timestamps_sec` остаётся source-of-truth.
- Audio: `audio/segments.json` остаётся source-of-truth.

### 4.2 SamplingPlan (v2 planning) — DoD

Если включено модельным профилем (vNext), Segmenter/Orchestrator должен уметь исполнять `SamplingPlan`:

- multi-pass: coarse → refine around events
- отчёт “plan vs fact” в per-run reports

### 4.3 Acceptance tests (минимум)

- воспроизводимость: одинаковый SamplingPlan при одинаковом `config_hash` + versions
- корректная union-domain индексация
- отсутствие “тихих” отклонений от требований компонентов (иначе fail-fast)

---

## 5) AudioProcessor — критерии аудита по классам экстракторов

AudioProcessor живёт на `frames_dir/audio/segments.json` и обязан быть строгим по families/сегментам (no-fallback).

### 5.0 Контракт “нет аудио дорожки” (обязательный критерий Audit v3)

Реальный кейс: часть видео может не иметь audio stream (особенно audit pack `*_fixed.mp4`).

**Acceptance (FINAL)**:

- Segmenter обязан писать `frames_dir/audio/segments.json` даже при отсутствии аудио:
  - `audio_present=false`
  - `empty_reason` установлен (каноничный)
  - `families={}` (пусто)
- AudioProcessor обязан трактовать это как **валидный empty**, а не error:
  - не требовать `frames_dir/audio/audio.wav`
  - не валидировать required families
  - **не запускать** extractors
  - записать NPZ для каждого requested компонента со `status="empty"` и `empty_reason` (из segments.json или каноничный аналог)
- Render (dev-only) не должен падать на empty артефактах (включая значения `NaN` в tabular фичах).

### 5.1 Общие требования ко всем audio extractors

- `feature_names/feature_values`:
  - стабильный порядок
  - документированный frozen subset (model_facing)
- Если есть sequences:
  - `segment_centers_sec` монотонен
  - длины массивов согласованы
  - missing → NaN + masks
- `meta.stage_timings_ms` и `timings_by_extractor` обязательны (наблюдаемость).
- **Feature gating policy (обязательная ясность)**:
  - если extractor требует enable‑флаги для включения групп фичей (например `spectral_enable_basic_features`),
    то в audited состоянии обязателен **presets/дефолт**, который гарантирует, что “включённый extractor” не падает с ошибкой вида “no features enabled”.
  - любые presets/флаги, влияющие на контракт ключей, должны быть зафиксированы в README/SCHEMA и отражаться в `config_hash`.

### 5.1b Naming rules (табличные фичи)

- Все `feature_names` должны быть:
  - **детерминированны** (без случайных suffix’ов)
  - **стабильны** по порядку
  - без “случайных” единиц измерения в названии (единицы фиксируем в SCHEMA.md)
- Если есть семейства/подгруппы — кодируем в имени префиксом (`tempo__`, `clap__`, …) или через понятные категории (как сейчас делает baseline extractor).

### 5.2 Tier-0 baseline extractors (обязательный минимум)

#### `clap_extractor`

- **Model-facing**:
  - скаляры (например norms/quality)
  - embeddings: либо агрегаты, либо sequence (для vNext)
- **TokenStreams readiness**:
  - `tokens = embedding_sequence` (N,D)
  - `times_s = segment_centers_sec` (N,)
  - `mask` по наличию валидных эмбеддингов
- **Render must-have**:
  - распределение norms
  - top/anti-top сегменты по norm/confidence

#### `tempo_extractor`

- **Quality gates** (пример):
  - BPM разумный диапазон (например 40–220)
  - warnings не доминируют
- **Model-facing**:
  - robust скаляры + (опционально) время-серия `windowed_bpm`
- **Render**:
  - timeline bpm, распределение, пики/аномалии

#### `loudness_extractor`

- **Quality gates**:
  - доля NaN/Inf = 0 для обязательных фичей
  - диапазоны LUFS/dBFS разумны
- **Render**:
  - timeline loudness + распределения

### 5.3 Speech/ASR/diarization (privacy-sensitive)

Правило v2: raw ASR текст по умолчанию не сохраняем как source-of-truth.

- Если extractor публикует текстовые артефакты:
  - они `debug-only` и под retain flag
  - модели используют только derived embeddings/proxy features

**TokenStreams readiness**:

- diarization: события/спаны как `spans_s` (start/end), optional speaker_id в `meta_json` (без PII).

### 5.4 Sampling requirements (Audio)

В `SCHEMA.md` каждого audio extractor обязательно фиксируем:

- required families в `audio/segments.json` (например `clap`, `tempo`, `primary`, `asr`)
- минимальные параметры:
  - max_windows / cap / min_video_duration
  - required window sizes (sec)
- поведение при нарушении:
  - missing family → error (если required)
  - пустой список сегментов → empty с корректным `empty_reason`

### 5.5 Run-log: минимальные “факт” метрики для Audio

В `RUN_LOG.md` для каждого smoke/validation run по аудио-экстракторам фиксируем минимум:

- `audio_duration_sec`, `sample_rate`
- `N_segments` по каждой required family
- базовые stats длительностей сегментов: min/p50/p90/max
- если есть embedding sequence: `N_tokens`, доля masked/empty

---

## 6) TextProcessor — критерии аудита по классам экстракторов

TextProcessor работает с `VideoDocument` и downstream от Audio (ASR/transcript) и/или внешних полей (title/description/comments).

### 6.1 Общие требования ко всем text extractors

- По умолчанию: **без raw текста** в model_facing артефактах.
- Если нужно хранить “что-то про текст”:
  - длины, счётчики, хэши, embeddings, агрегаты.
- `feature_names/feature_values` должны быть стабильны и документированы (особенно любые cosine/entropy/cluster метрики).

### 6.1b Privacy-safe идентификаторы (обязательно для text)

Если extractor делает дедупликацию/отбор/кеширование по тексту:

- в артефактах разрешены только:
  - `content_hash` (sha256/другой крипто-хэш), длины, счётчики
  - embeddings/агрегаты
- запрещено сохранять “кусочки” raw текста в `payload`/meta по умолчанию.

### 6.1c Мульти-источники текста (ASR как source-of-truth)

- Если extractor использует transcript, он обязан явно указать:
  - источник (например “whisper ASR из AudioProcessor”),
  - поведение, если transcript отсутствует (empty_reason vs error).

### 6.2 Tier-0 (лексические/детерминированные)

Примеры: `lexico_static_features`, `tags_extractor`.

- Детерминизм: одинаковый вход → одинаковые фичи.
- Любые in-memory мутации документа (например tags removal) должны быть:
  - задокументированы
  - тестируемы (smoke)

### 6.3 Embedding extractors (dp_models, offline)

Примеры: title/description/comments/transcript embedders.

- **No-network**: строго через `dp_models` (weights_digest обязателен в meta).
- **Кеширование** (если есть) должно быть:
  - детерминированным
  - versioned по weights_digest/model_version
- **TokenStreams readiness (vNext)**:
  - собрать `tokens (Kc,D)` и `mask (Kc,)` без raw текста
  - `meta_json` с provenance (weights_digest/model_version + политика отбора комментариев)

### 6.4 Aggregation / similarity / corpus-based (leakage risk)

Любые reference/corpus similarity функции (FAISS/corpus titles) должны иметь:

- digest/версию корпуса
- anti-leakage политику (time-frozen относительно prediction time)
- одинаковый pack на train/infer

До этого — держать как `analytics` (не model_facing).

**Дополнение**: любые corpus packs (topics DB, FAISS indices, title corpus) должны иметь:

- `pack_version`
- `pack_digest`
- описанное окно формирования (time-frozen policy)

Иначе — запрещено в model_facing.

### 6.5 Sampling requirements (Text)

В `SCHEMA.md` каждого text extractor обязательно фиксируем:

- требования к входным полям документа (title/description/comments/transcript)
- лимиты:
  - max comments (например 100)
  - политика дедупликации/отбора
  - chunking параметров (для transcript chunks)
- поведение при missing:
  - отсутствуют комментарии → empty (`comments_missing_or_disabled`)
  - отсутствует transcript → empty (`no_text_available`) или специфичный reason

### 6.6 Run-log: минимальные “факт” метрики для Text

В `RUN_LOG.md` для каждого smoke/validation run по text-экстракторам фиксируем минимум:

- какие источники текста использованы (title/description/comments/transcript)
- сколько объектов реально обработано после лимитов/дедупа:
  - `comments_selected_count` (и лимит)
  - `transcript_chunks_count` (и параметры chunking)
- если есть embeddings/tokens:
  - `D` и `Kc` (если фиксированные text tokens)
  - доля пустых/замаскированных

---

## 7) Шаблон “карточки аудита компонента” (копировать в README/issue)

### 7.1 Component Audit Card (template)

- **component_name**:
- **owner_processor**: audio|text
- **schema_version**:
- **producer_version**:
- **inputs**:
  - upstream artifacts:
  - required fields/files:
- **sampling requirements**:
- **outputs (model_facing)**:
- **outputs (analytics)**:
- **outputs (debug-only)**:
- **empty semantics**:
  - status=empty when:
  - empty_reason:
- **hard deps**:
- **soft deps**:
- **privacy**:
- **render QA**:
  - expected distributions:
  - anti-patterns:
- **tokenstreams readiness** (v2):
- **acceptance**: pass/fail checklist
