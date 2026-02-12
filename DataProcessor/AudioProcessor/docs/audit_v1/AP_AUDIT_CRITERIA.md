# Критерии аудита AudioProcessor и его Extractors (AP audit criteria)

**Версия**: v1.0  
**Дата**: 2026-01-XX  
**Применимо к**: `DataProcessor/AudioProcessor` (процессор) и всем `AudioProcessor/src/extractors/*` (экстракторам)

---

## 0. Обзор: что мы аудируем и зачем

Этот документ фиксирует **обязательные критерии** для приведения AudioProcessor к baseline‑контрактам TrendFlow/DataProcessor и дальнейшего доведения каждого extractor до полуфинального "production‑grade" уровня.

Аудит должен подтвердить, что:
- AudioProcessor и каждый extractor **соответствуют контрактам входа/выхода**, storage и meta.
- Соблюдается **no‑fallback policy** (ошибки — явные, empty — валидная пустота).
- Соблюдается **Segmenter contract** (audio/audio.wav, audio/segments.json).
- Любые ML‑модели загружаются **только через `dp_models` (ModelManager)**, без сетевых загрузок.
- Есть воспроизводимость, измерения производительности, quality‑валидация и UI‑рендер (presentation layer) из NPZ.

---

## 0.1 Термины и уровни аудита

### A) Уровень "AudioProcessor компонент"

Код‑входные точки и orchestration:
- `AudioProcessor/run_cli.py`
- `AudioProcessor/src/core/main_processor.py` (оркестратор экстракторов)
- `AudioProcessor/src/core/base_extractor.py` (базовый класс)
- интеграция с `manifest.json` + `artifact_validator`

### B) Уровень "Extractor компонент"

Подкомпонент AudioProcessor, реализующий `BaseExtractor.run()` или `BaseExtractor.run_segments()` и возвращающий:
- `ExtractorResult` с полями:
  - `success` (bool)
  - `payload` (Dict[str, Any])
  - `error` (Optional[str])
  - `processing_time` (Optional[float])
  - `device_used` (str)

**Важно**: в baseline аудит делается **по каждому extractor**: он считается "мини‑компонентом" со своими входами/зависимостями/моделями/сегментами/фичами/производительностью.

---

## 0.2 Source-of-truth документы (обязательные ссылки)

- Базовые критерии: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- Контракты:  
  - `docs/contracts/CONTRACTS_OVERVIEW.md`  
  - `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`  
  - `docs/contracts/SEGMENTER_CONTRACT.md` (audio/segments.json, families)
  - `docs/contracts/ERROR_HANDLING_AND_EDGE_CASES.md`
- Модельная система: `dp_models/*` + `docs/models_docs/*` (ModelManager правила)

---

## 1. Архитектурное соответствие (AudioProcessor / Extractor)

### 1.1 Интерфейсы и границы ответственности

**AudioProcessor как компонент**:
- [ ] Имеет CLI entrypoint (`run_cli.py` с argparse) и может быть вызван оркестратором DataProcessor.
- [ ] Работает строго в рамках **per-run result_store** (см. 1.4).
- [ ] Обновляет `manifest.json` через общий механизм (`RunManifest.upsert_component()`) и **не пишет произвольные JSON артефакты** в `result_store` (кроме `manifest.json`).
- [ ] **Строгая загрузка экстракторов (обязательное правило)**: если extractor указан в `--extractors`, но не смог импортироваться/создаться → run должен завершаться `status="error"` (fail-fast), а не "тихий skip".
- [ ] Поддерживает **Segmenter contract**: читает `audio/audio.wav` и `audio/segments.json` из `frames_dir` (**`--frames-dir` обязателен; legacy mode запрещён**).

**Extractor как компонент**:
- [ ] Реализует `BaseExtractor.run(input_uri, tmp_path)` или `BaseExtractor.run_segments(input_uri, tmp_path, segments, ...)`.
- [ ] Не делает "скрытых" глобальных сайд‑эффектов (сетевые вызовы, скачивание моделей, запись raw audio в логи/файлы) без явной политики/флага.
- [ ] Если extractor требует специфичный вход (например, `audio/segments.json` family) — это требование **декларировано** и **проверяется fail-fast** (см. 1.3 / 1.6).

**Evidence**:
- путь к файлам + номера строк (интерфейс, проверки, место вызова в pipeline);
- пример запуска `AudioProcessor/run_cli.py` с минимальными флагами.

### 1.2 Контракты входа (Segmenter contract)

- [ ] Входная единица обработки: **`audio/audio.wav`** (Segmenter) + **`audio/segments.json`** (contract `audio_segments_v1`).
- [ ] Источники сегментов (families в `segments.json`) явно интерпретируются как:
  - **required** для конкретного extractor'а (например, `clap` family для `clap_extractor`),
  - **optional** (валидный empty, если family отсутствует, но это не критично).
- [ ] Для extractors с `run_segments()`:
  - читают `families.<name>.segments[]` из `audio/segments.json`,
  - используют `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`,
  - не генерируют сегменты сами (Segmenter — единственный владелец sampling).
- [ ] Любые эвристики выбора family/сегментов считаются **policy** и должны быть:
  - задокументированы,
  - стабильны,
  - не маскировать отсутствие required входа.

**Стандартные families (Segmenter contract)**:
- `primary`: короткие окна вокруг time‑anchors → `loudness_extractor`
- `clap`: короткие окна на нелинейной кривой → `clap_extractor`
- `tempo`: длинные sliding windows → `tempo_extractor`
- `asr`: длинные sliding windows → `asr_extractor`
- `diarization`: фиксированные окна → `speaker_diarization_extractor`
- `emotion`: перекрывающиеся окна → `emotion_diarization_extractor`
- `source_separation`: длинные окна → `source_separation_extractor`

**Evidence**:
- схема `audio/segments.json` (contract `audio_segments_v1`);
- пример `audio/segments.json` и его разбор;
- тест/скрипт, который валидирует вход.

### 1.3 No-fallback policy (fail-fast vs valid empty)

**Правило**: если extractor/профиль требует вход, а вход отсутствует/невалиден → **`raise RuntimeError` / status=error**, а не "молчаливый default".

- [ ] Отсутствие обязательного `audio/audio.wav` → fail-fast на уровне CLI (`raise RuntimeError`).
- [ ] Отсутствие обязательного `audio/segments.json` → fail-fast на уровне CLI (`raise RuntimeError`).
- [ ] Отсутствие обязательного family в `segments.json` для required extractor'а → fail-fast (`raise RuntimeError`).
- [ ] Пустой список segments в family → fail-fast (`raise ValueError("segments is empty (no-fallback)")`).
- [ ] Любая ошибка модели/инференса/валидации → `status="error"` + `error_code` + понятное сообщение (см. `ERROR_HANDLING_AND_EDGE_CASES.md`).
- [ ] **Observability (обязательное правило)**: любые ошибки и любые "empty" статусы **должны логгироваться** (без raw audio данных) и отражаться в `manifest.json` (component status/error_code/notes).

**Исключение**: "валидная пустота" (см. 1.6) — когда данных действительно нет и это нормальный кейс.

**Evidence**:
- фрагменты кода с проверками и явными ошибками;
- отсутствие "тихих" except/pass, которые превращают error в ok.

### 1.4 Per-run storage (ResultStore)

**AudioProcessor должен писать результаты только в**:

`result_store/<platform_id>/<video_id>/<run_id>/`

Обязательные файлы:
- `manifest.json`
- `<component_name>/<component_name>_features.npz` (канонический путь, фиксированное имя)

- [ ] Имя NPZ файла стабильное (fixed name: `{component_name}_features.npz`). Перезапуск run не оставляет "старых" артефактов.
- [ ] Запись NPZ и manifest атомарная (tmp → `os.replace()`).
- [ ] Любые дополнительные файлы в `result_store` разрешены только если они:
  - являются **sub-artifacts** (например, `.npy` больших матриц) и
  - явно перечислены в `manifest.json.artifacts[]` и/или в NPZ как ссылки,
  - не нарушают запрет на "произвольные JSON" (кроме `manifest.json`).

**AudioProcessor-specific правило sub-artifacts (`.npy`) — обязательный стандарт**:
- [ ] Любые промежуточные/вспомогательные векторные артефакты `*.npy` (embeddings/sequences/aggregates) для AudioProcessor должны сохраняться **per-run** внутри:
  - `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/_artifacts/*.npy` (опционально, если нужны большие массивы)
- [ ] Эти `*.npy` файлы обязаны быть перечислены в `manifest.json.components[].artifacts[]` (type=`"npy"`).
- [ ] Extractors, которые пишут `*.npy`, должны принимать `tmp_path` через параметры и **не** использовать глобальные директории как source-of-truth.
- [ ] Запрещено полагаться на `glob + mtime` по глобальным директориям для выбора "последнего артефакта" (недетерминировано, cross-run leakage).
- [ ] В `payload`/NPZ нельзя возвращать **абсолютные** пути к `*.npy`; допускаются только:
  - числовые признаки (`feature_values`),
  - relpath внутри `_artifacts/` (если это нужно для intra-run), либо хранение relpath только in-memory.

**Критически важно (AudioProcessor‑специфика)**:
- "кеши" и "артефакты" — разные сущности.
  - **Cache**: content-addressed, не обязателен для корректности, может жить вне `result_store` и иметь TTL.
  - **Artifact**: часть результата run; должен быть внутри `run_rs_path` и отражён в `manifest.json`.

**Evidence**:
- фактический путь артефактов;
- содержимое `manifest.json` для компонента `audio_processor` или конкретного extractor'а.

### 1.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Обязательные ключи:
- `feature_names: object[str]` — имена фичей (scalars)
- `feature_values: float32[]` — значения фичей (scalars), выровнены с `feature_names`
- Дополнительные ключи по extractor'у (например, `embedding`, `embedding_sequence`, `tempo_estimates`, `segment_centers_sec`)
- `meta: object(dict)` — обязательные поля meta

Обязательные поля `meta` (минимум):
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status` ∈ {`ok`, `empty`, `error`}
- `empty_reason` (только если `status="empty"`, иначе null)
- `models_used[]`, `model_signature` (если используются модели)
- `device_used` (если применимо)
- `scheduler_knobs` (если применимо: `segment_parallelism`, `max_inflight`, `model_batch_size`)

**Extractor-level требования**:
- [ ] Каждая фича имеет стабильное имя (без случайных suffix/prefix).
- [ ] Единицы измерения и диапазоны (где применимо) зафиксированы в README extractor'а.
- [ ] Missing values: NaN + маски (`*_present`), а не "0 по умолчанию", если 0 — валидное значение.
- [ ] Для per-segment sequences: `segment_centers_sec` (float32[N]) строго монотонно возрастает.

**Evidence**:
- пример NPZ `{component}_features.npz` + вывод валидатора `validate_npz(...)`.

### 1.6 Valid empty outputs (AudioProcessor)

Валидная пустота означает: данные отсутствуют по причине отсутствия аудио/тишины/политики, но pipeline продолжает работать.

Стандартные значения `empty_reason` (канонические):
- `audio_missing_or_extract_failed` — аудио отсутствует или не удалось извлечь
- `video_too_short` — видео слишком короткое для анализа
- `video_too_long` — видео слишком длинное (превышен cap)
- `dependency_missing` (только если это реально "опциональная зависимость отключена профилем", иначе это error)

Требования:
- [ ] При `status="empty"`: `empty_reason` обязательно.
- [ ] Фичи при empty должны быть:
  - либо NaN (и соответствующие `*_present=false`),
  - либо явно документированный "empty-safe" набор (например, длины=0), если это семантически корректно.
- [ ] Empty не должен скрывать ошибки парсинга/модели/валидации.
- [ ] **Observability**: empty-case должен быть явно зафиксирован (лог/manifest notes), чтобы downstream и UI могли объяснить пользователю "почему данных нет".

**Evidence**:
- пример отсутствия аудио → NPZ `status=empty`.

---

## 2. Model system: no-network + ModelManager (обязательное правило)

### 2.1 Запрещённые практики (fail audit)

В AudioProcessor/Extractors запрещено:
- сетевые загрузки моделей/весов/данных во время run:
  - `from_pretrained(...)` без `local_files_only`,
  - `torch.hub.load(...)`,
  - `requests.get(...)` для весов,
  - любые "pretrained=True" с implicit download,
  - runtime `pip install` / `nltk.download(...)` и т.п.

### 2.2 Обязательное правило: `dp_models` (ModelManager)

- [ ] Любая ML‑модель (CLAP, Whisper, speaker diarization, emotion, source separation) резолвится через `dp_models`.
- [ ] Спеки моделей заданы в `dp_models/spec_catalog/audio/*.yaml` (или другом каноническом каталоге).
- [ ] Артефакты лежат в `DP_MODELS_ROOT` и валидируются fail-fast (missing weights → error).
- [ ] В `meta.models_used[]` фиксируются: `model_name`, `model_version`, `weights_digest`, `runtime`, `engine`, `precision`, `device`.

**Особые требования для моделей**:
- [ ] **CLAP**: spec `laion_clap`, загрузка через `ModelManager.resolve()`, checkpoint путь из artifacts.
- [ ] **Whisper (ASR)**: spec `whisper_{size}_inprocess`, runtime `inprocess` (PyTorch), shared tokenizer `shared_tokenizer_v1`.
- [ ] **Speaker diarization**: spec `speaker_diarization_{size}_triton`, runtime `triton`.
- [ ] **Emotion diarization**: spec `emotion_diarization_{size}_triton`, runtime `triton`.
- [ ] **Source separation**: spec `source_separation_{size}_triton`, runtime `triton`.

**Evidence**:
- ссылка на код резолва через `dp_models`;
- пример `models_used[]` и `model_signature` в NPZ.

---

## 3. Segmenter contract: audio/segments.json (AudioProcessor‑специфика)

Источник правил: `docs/contracts/SEGMENTER_CONTRACT.md` (раздел 9.3).

### 3.1 Обязательные поля segments.json

- [ ] `schema_version="audio_segments_v1"` (проверяется fail-fast).
- [ ] `sample_rate`, `total_samples`, `audio_duration_sec`, `video_duration_sec`.
- [ ] `families.<name>.segments[]` для каждого используемого extractor'а.

### 3.2 Структура сегмента

Каждый сегмент содержит:
- `start_sec`, `end_sec`, `center_sec` (float)
- `start_sample`, `end_sample` (int, индексы в `audio/audio.wav`)

- [ ] Extractors используют `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`.
- [ ] Extractors не генерируют сегменты сами (Segmenter — единственный владелец sampling).

### 3.3 Families и их назначение

- [ ] Каждый extractor документирует, какую family он использует:
  - `clap_extractor` → `families.clap.segments[]`
  - `tempo_extractor` → `families.tempo.segments[]`
  - `loudness_extractor` → `families.primary.segments[]`
  - `asr_extractor` → `families.asr.segments[]`
  - `speaker_diarization_extractor` → `families.diarization.segments[]`
  - `emotion_diarization_extractor` → `families.emotion.segments[]`
  - `source_separation_extractor` → `families.source_separation.segments[]`
- [ ] Отсутствие required family → fail-fast (`raise RuntimeError`).

**Evidence**:
- код чтения `audio/segments.json` в `run_cli.py`;
- примеры использования families в extractors.

---

## 4. Зависимости между extractors и между процессорами

### 4.1 Внутри AudioProcessor (extractor graph)

- [ ] Если extractor зависит от результатов предыдущих (например, `speech_analysis_extractor` использует ASR + diarization), зависимость:
  - описана в README,
  - отражена в "execution plan" (порядок запуска),
  - проверяется fail-fast, если required вход отсутствует.
- [ ] "Передача данных" между extractors делается через `payload` результатов, без скрытых глобальных синглтонов.
- [ ] Для передачи **путей/идентификаторов sub-artifacts** между extractor'ами в рамках одного run допускается in-memory реестр, при этом:
  - он не является persisted contract,
  - не должен попадать в NPZ payload как raw пути,
  - нужен для детерминизма (без glob/mtime).

### 4.2 Между процессорами (cross-processor dependencies)

- [ ] Любые cross‑modal фичи (Audio ↔ Visual / Text) считаются зависимостями от других компонентов DataProcessor и должны:
  - опираться на NPZ source-of-truth других компонентов (а не на JSON),
  - иметь чётко описанный контракт входа/выхода,
  - иметь понятную empty/error семантику.

**Evidence**:
- список зависимостей и пример, как они резолвятся в run_rs_path.

---

## 5. Наблюдаемость: progress + stage timings

### 5.1 Промежуточный прогресс (обязательный критерий)

AudioProcessor должен уметь репортить прогресс выполнения для UI/backend.

Требования:
- [ ] Есть stage-based прогресс минимум по стадиям:
  - `load_input` (чтение audio/segments.json)
  - `run_extractors` (запуск extractors)
  - `save_npz` (сохранение артефактов)
  - `validate_artifact` (валидация NPZ)
  - `update_manifest` (обновление manifest.json)
- [ ] Во время `run_extractors` прогресс обновляется **по мере завершения экстракторов** (не по таймеру), минимум 10 обновлений на run (если экстракторов ≥10; иначе — по каждому).
- [ ] Для extractors с `run_segments()` прогресс обновляется **по мере обработки сегментов** (не по таймеру), минимум 10 обновлений за run (если сегментов ≥10; иначе — по каждому).
- [ ] Формат прогресса **машиночитаем** и безопасен (без raw audio данных).

Рекомендованный формат события (stdout JSON‑line, проксируется оркестратором):
- `platform_id`, `video_id`, `run_id`
- `component="audio_processor"` или `component="<extractor_name>"`
- `stage_id`, `stage_name`
- `extractor` (если применимо)
- `progress_pct` ∈ [0..100]
- `ts`

**Evidence**:
- пример реального output (лог/JSON‑lines) с ≥10 обновлениями.

### 5.2 Stage timings (обязательный критерий)

- [ ] В NPZ (в `meta` или отдельной секции `summary`) сохранены тайминги стадий:
  - минимум: `load_input_ms`, `run_extractors_ms`, `save_npz_ms`, `validate_npz_ms`, `update_manifest_ms`.
- [ ] Также сохраняются per-extractor timings (агрегировано) — например `timings_by_extractor` в `meta` (privacy-safe).
- [ ] Для extractors с `run_segments()` сохраняются тайминги обработки сегментов (mean/std/min/max).

**Evidence**:
- NPZ содержит `stage_timings_ms` (dict) + ссылка на код измерений.

---

## 6. Feature contract: управление выходными фичами (feature gating)

Цель: управляемо включать/выключать группы фичей (стоимость/качество/IO/NPZ size).

Требования:
- [ ] Есть механизм выбора фич через CLI/конфиг:
  - `--extractors <csv>` (выбор extractors, что эквивалентно выбору групп фичей),
  - `--features <csv>` и/или `--feature-set <name>` (если extractor поддерживает feature gating внутри).
- [ ] Есть "default extractor set" и он документирован (baseline Tier-0: `clap,tempo,loudness`).
- [ ] В `meta` фиксируются:
  - `extractors_enabled[]` (список запущенных extractors)
  - `features_enabled[]` (если extractor поддерживает feature gating)
  - `features_produced[]` (фактически произведённые фичи)
- [ ] Нет "скрытых" фич: если фича может появиться в output, она перечислена и gated.

**Правило feature gating (обязательное для всех extractors)**:
- [ ] **Персональный флаг для каждой фичи**: каждая фича должна иметь отдельный флаг включения/выключения (например, `--enable-token-counts`, `--enable-token-sequences`).
- [ ] **Группы фичей допустимы только если**:
  - Алгоритмы для извлечения фичей одинаковые (одна функция генерирует несколько фичей), или
  - Выходных фичей у компонента очень много (≥20), и группировка упрощает управление.
- [ ] **Зависимости фичей**: если фича зависит от другой фичи или от других extractors, это должно быть явно прописано в README extractor'а (раздел "Feature dependencies").
- [ ] Все фичи должны быть контролируемы через аргументы (вкл/выкл), и зависимости должны быть документированы.

**Evidence**:
- примеры запуска с разными extractor sets;
- NPZ meta отражает включённые extractors/фичи;
- README содержит раздел "Feature dependencies" с явным описанием зависимостей.

---

## 7. Производительность и ресурсы (AudioProcessor + Extractors)

### 7.1 Обязательные измерения

Для AudioProcessor и для каждого "тяжёлого" extractor'а (CLAP, ASR, diarization, emotion, source separation):
- [ ] latency per unit (unit = `audio_window` или `audio_segment`) — явно задокументировано.
- [ ] CPU RSS peak.
- [ ] GPU VRAM peak (если используется GPU).
- [ ] Spikes (p95/p99) + правило детектора spikes.

### 7.2 Где хранить результаты измерений

Источник правды: `docs/models_docs/resource_costs/<component>_costs_v1.json`  
Дополнительно допускается детализация по экстракторам: `audio_processor_<extractor>_costs_v1.json`.

### 7.3 Требования к batching / OOM

- [ ] Для GPU‑extractors есть конфигурируемый `batch_size` (например, `--clap-batch-size`).
- [ ] Для extractors с `run_segments()` есть конфигурируемый `segment_parallelism` и `max_inflight`.
- [ ] При OOM поддерживается управляемый rollback batch_size (см. `ERROR_HANDLING_AND_EDGE_CASES.md`), либо fail-fast с понятным error_code.

**Evidence**:
- JSON resource_costs;
- команды запуска бенчмарка и воспроизводимость результатов.

---

## 8. Проверка качества выхода (quality validation)

### 8.1 Минимальные sanity-checks (обязательны)

Для каждого extractor'а:
- [ ] диапазоны значений разумны (no NaN там, где обязаны быть числа; нет inf; нет отрицательных там, где нельзя);
- [ ] консистентность связных фичей (например, `*_present` ↔ значение);
- [ ] статистические инварианты (пример: tempo BPM ∈ [40, 220], loudness dBFS ∈ [-∞, 0], embedding norms > 0).
- [ ] Для per-segment sequences: монотонность `segment_centers_sec`, согласованность размеров массивов.

### 8.2 Human-friendly визуализация / UI render (must-have для AP)

AudioProcessor должен иметь функционал, который превращает NPZ output в **UI‑дружелюбный** формат (distribution, top features, flags, timings), без использования LLM как источника истины.

Требования:
- [ ] Есть deterministic "renderer" (python модуль/скрипт) который читает `{component}_features.npz` и строит render-context JSON (без raw audio данных).
- [ ] Этот render-context может быть использован LLM (см. `LLM_RENDERING.md`) только как вход для генерации текста.
- [ ] **HTML renderer для дебага**: опциональный HTML renderer с raw данными для локального дебага и ручной проверки качества (только для локального использования, не в production артефактах).
- [ ] **README визуализация**: в README каждого extractor'а должен быть раздел "Visualization" с описанием лучших практик визуализации данных компонента для UI/сайта (timeline, графики, распределения, интерактивные элементы).

**Правило визуализации (обязательное для всех extractors)**:
- [ ] README extractor'а содержит раздел "Visualization" с описанием:
  - Какие данные лучше визуализировать (timeline, distributions, aggregations)
  - Рекомендуемые типы графиков (line charts, bar charts, heatmaps, etc.)
  - Интерактивные элементы (tooltips, zoom, filters)
  - Примеры визуализаций (опционально, скриншоты или ссылки)
- [ ] HTML renderer для дебага должен быть явно помечен как debug-only и не попадать в production артефакты.
- [ ] **Отдельный файл render.py (обязательное правило)**: каждый extractor должен иметь свой файл `src/extractors/<extractor_name>/render.py`, содержащий:
  - Функцию `render_<extractor_name>(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]` для генерации render-context JSON
  - Функцию `render_<extractor_name>_html(npz_path: str, output_path: str) -> str` для генерации HTML debug страницы (опционально)
  - Все функции должны быть экспортированы через `__all__` или явно импортированы в основном `renderer.py`
- [ ] Основной `src/core/renderer.py` должен содержать только:
  - Утилиты (`load_npz`, `extract_meta`)
  - Функцию `render_component()` для динамической загрузки и вызова renderer'ов из модулей extractors
  - Функцию `render_all_components()` для batch рендеринга
  - Регистр `RENDERERS` для динамической загрузки renderer'ов (опционально, может быть заменён на динамический импорт)

**Что визуализировать** (зависит от типа extractor'а):

**Для clap_extractor**:
- Timeline с `segment_centers_sec` и embedding norms
- t-SNE визуализация `embedding_sequence` (опционально)
- Статистики: `clap_norm`, `clap_magnitude_mean/std`

**Для tempo_extractor**:
- Timeline с `windowed_bpm` по времени
- Распределение `tempo_estimates`
- Warnings (например, `low_confidence`, `tempo_out_of_range`)

**Для loudness_extractor**:
- Timeline с `segment_rms/segment_dbfs/segment_lufs` по времени
- Распределение RMS/dBFS/LUFS
- Наличие/отсутствие LUFS через `lufs_present`

**Для asr_extractor**:
- Timeline с сегментами ASR (без raw текста, только token counts)
- Статистики: `segments_count`, `token_total`, `token_density_per_sec`

**Для speaker_diarization_extractor**:
- Timeline с speaker segments
- Количество спикеров, доли времени по спикерам

**Рекомендуемый формат**:
- HTML страница с интерактивными элементами (timeline, графики)
- Или JSON с метаданными для внешней визуализации

**Evidence**:
- скрипт/модуль рендера + пример JSON;
- пример HTML/графиков (опционально), либо структуры данных для фронта.

---

## 9. Документация (обязательные разделы)

### 9.1 README AudioProcessor

В `AudioProcessor/README.md` должны быть разделы:
- **Input contract** (Segmenter contract: audio/audio.wav, audio/segments.json, families)
- **Output contract** (NPZ schema, пути, meta)
- **Models** (через dp_models, no-network)
- **Sampling requirements** (как Segmenter строит families, ссылка на SEGMENTER_CONTRACT.md)
- **Parallelization** (внутренний/внешний параллелизм, thread-safety, GPU sharing, segment_parallelism)
- **Performance characteristics** (ссылка на resource_costs)
- **Quality validation & human-friendly inspection** (sanity + render)
- **Features** (extractor gating, default set)

### 9.2 README каждого extractor'а

В `AudioProcessor/src/extractors/<name>/README.md` обязаны быть:
- входы: какие поля `audio/segments.json` нужны (family, segments) и какие зависимости от других extractor'ов
- выходы: список фичей + единицы/диапазоны + masks/empty semantics
- **Feature dependencies**: явное описание зависимостей между фичами (если фича A зависит от фичи B, или от другого extractor'а)
- модели (если есть): spec name / runtime / engine / precision / device
- параметры: допустимые значения + дефолты + влияние на стоимость (Δ latency / Δ cost — best-effort)
- **Feature gating**: описание всех флагов включения/выключения фичей (персональные флаги для каждой фичи)
- параллелизм/батчинг/лимиты и поведение при OOM
- качество: sanity checks + как визуально посмотреть результат
- **Visualization**: описание лучших практик визуализации данных компонента для UI/сайта (timeline, графики, распределения, интерактивные элементы)

**Evidence**:
- ссылки на README, отсутствие дублирования "контента" в других местах.

---

## 10. Шаблон "карточки аудита" для extractor'а (заполняется на каждый)

Рекомендуемый файл:

`AudioProcessor/docs/audit_v1/components/<EXTRACTOR>_AUDIT.md`

Секции:
1. Summary (ok/needs fixes)
2. Contract compliance checklist (по разделам 1–9)
3. Models used (dp_models specs, weights_digest)
4. Features list + gating status
5. Performance (resource_costs ссылки)
6. Quality validation (sanity + render snapshots)
7. Open issues + fix plan (PR list)

---

## 11. Чек-лист аудитора (коротко)

### Архитектура / контракты
- [ ] per-run storage + manifest upsert
- [ ] NPZ meta обязательные поля + validate_npz
- [ ] no-fallback policy соблюдён
- [ ] empty semantics корректны (canonical empty_reason)
- [ ] Segmenter contract соблюдён (audio/segments.json, families)

### Модели / воспроизводимость
- [ ] dp_models only, no downloads
- [ ] models_used/model_signature корректны
- [ ] scheduler_knobs зафиксированы в meta

### Наблюдаемость / качество / ресурсы
- [ ] progress events есть и безопасны
- [ ] stage timings сохранены
- [ ] resource_costs измерены
- [ ] есть sanity checks + UI render

---

## 12. Специфика AudioProcessor (отличия от TextProcessor)

### 12.1 Segmenter contract (обязательно)

- [ ] AudioProcessor **не извлекает** аудио из видео, а берёт `audio/audio.wav`, подготовленный Segmenter.
- [ ] Extractors работают с `audio/segments.json` families, не генерируют сегменты сами.
- [ ] Универсальная нелинейная кривая (sampling curve) для families документирована в Segmenter contract.

### 12.2 Параллелизм на уровне segments

- [ ] Extractors с `run_segments()` поддерживают `segment_parallelism` и `max_inflight`.
- [ ] Параллелизм контролируется scheduler через CLI аргументы (`--segment-parallelism`, `--max-inflight`).
- [ ] Thread-safety extractors документирована.

### 12.3 Privacy (менее критично, чем для TextProcessor)

- [ ] По умолчанию AudioProcessor **не сохраняет raw audio** в артефактах (только фичи/статистики).
- [ ] Raw audio может быть в `_tmp_*` для дебага, но не считается source-of-truth.
- [ ] В логах запрещено raw audio данные: только длины/счётчики/статусы.

---

## Ссылки

- **Baseline component audit criteria**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **Artifacts and schemas**: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- **Model system rules**: `docs/models_docs/MODEL_SYSTEM_RULES.md`
- **TextProcessor audit criteria** (reference): `TextProcessor/docs/audit_v1/TP_AUDIT_CRITERIA.md`

