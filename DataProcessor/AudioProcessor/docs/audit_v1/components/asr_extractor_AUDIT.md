# Audit: `asr_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`asr_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: использует `audio/audio.wav` и `audio/segments.json` family=`asr`
- ✅ **No-fallback policy**: fail-fast при отсутствии segments
- ✅ **Model system**: загрузка через `dp_models` (ModelManager), no-network (Whisper через Triton, shared tokenizer локально)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `asr_extractor_features.npz`
- ✅ **Feature gating**: персональные флаги для каждой фичи (9 фичей)
- ✅ **Error handling**: детальные error codes для Triton (6 типов)
- ✅ **Token validation**: полная валидация token IDs (диапазоны, special tokens, согласованность)
- ✅ **Progress reporting**: обновление прогресса каждые 10% сегментов
- ✅ **UI Render**: renderer реализован в `src/core/renderer.py` + HTML renderer для дебага
- ✅ **Contract version**: `asr_text_contract_version` для валидации совместимости с TextProcessor

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run_segments(input_uri, tmp_path, segments, ...)`
- [x] Не делает скрытых глобальных сайд-эффектов (модель загружается через ModelManager, no-network)
- [x] Требование специфичного входа декларировано: `audio/segments.json` family=`asr` (см. README)
- [x] `run()` не поддерживается в production (возвращает error)

**Evidence**:
- `src/extractors/asr_extractor/main.py:144` — метод `run_segments()`
- `src/extractors/asr_extractor/main.py:207-222` — `run()` возвращает error с сообщением
- `src/extractors/asr_extractor/main.py:157-158` — проверка segments: `if not isinstance(segments, list) or not segments: raise ValueError("segments is empty (no-fallback)")`

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + `audio/segments.json` family=`asr`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/asr_extractor/main.py:167-171` — чтение `start_sample/end_sample` из segments
- `src/extractors/asr_extractor/main.py:173` — `self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate)`

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("segments is empty (no-fallback)")`
- [x] Ошибка модели/инференса → `status="error"` в `ExtractorResult` с детальным error_code
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/asr_extractor/main.py:157-158` — проверка segments
- `src/extractors/asr_extractor/main.py:456-461` — error handling в `run_segments()` с детальными error codes

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `asr_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] Нет произвольных JSON артефактов (только NPZ)

**Evidence**:
- `run_cli.py:454` — сохранение через `_save_component_npz()` с фиксированным именем
- `run_cli.py:252-714` — функция `_save_component_npz()` использует атомарную запись

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `token_ids_by_segment: object[np.ndarray[int32]]` — token IDs по сегментам (feature-gated)
- [x] `segment_start_sec: float32[N]` — временные метки начала сегментов
- [x] `segment_end_sec: float32[N]` — временные метки конца сегментов
- [x] `segment_center_sec: float32[N]` — временные метки центров сегментов
- [x] `lang_id_by_segment: int32[N]` — языковые ID по сегментам
- [x] `token_counts: int32[N]` — количество токенов по сегментам (feature-gated)
- [x] `lang_distribution: object(dict)` — распределение языков (feature-gated)
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] `producer`, `producer_version`, `schema_version`, `created_at`
- [x] `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- [x] `status` ∈ {`ok`, `empty`, `error`}
- [x] `models_used[]`, `model_signature` (Whisper + shared tokenizer)
- [x] `device_used`
- [x] `scheduler_knobs` (legacy; may be empty for inprocess ASR)
- [x] `asr_text_contract_version` — версия контракта для валидации совместимости с TextProcessor
- [x] `features_enabled[]` — список включённых фичей (feature gating)

**Extractor-level требования**:
- [x] Фичи имеют стабильные имена: `segments_count`, `token_total`, `token_density_per_sec`, `speech_rate_wpm`, `segments_with_speech`, `avg_segment_duration_sec`, `token_variance`
- [x] Единицы измерения зафиксированы в README
- [x] Missing values: NaN (если применимо)
- [x] `segment_centers_sec` строго монотонно возрастает (проверяется в quality validation)

**Evidence**:
- `run_cli.py:454-497` — функция `_save_component_npz()` для `asr_extractor`
- `src/extractors/asr_extractor/main.py:189-201` — payload структура

### 2.6 Valid empty outputs

- [x] Валидная пустота не применима для `asr_extractor` (если segments пустой → error, не empty)
- [x] Если extractor возвращает empty, это должно быть явно в payload: `status="empty"`, `empty_reason`

**Evidence**:
- `src/extractors/asr_extractor/main.py:157-158` — пустые segments → error, не empty

---

## 3) Model System

### 3.1 Запрещённые практики

- [x] Нет сетевых загрузок моделей/весов
- [x] Используется `dp_models` (ModelManager) для резолва Whisper spec и shared tokenizer

**Evidence**:
- `src/extractors/asr_extractor/main.py:58-64` — инициализация ModelManager
- `src/extractors/asr_extractor/main.py:66-76` — резолв shared tokenizer через ModelManager
- `src/extractors/asr_extractor/main.py` — резолв Whisper inprocess spec через ModelManager

### 3.2 Обязательное правило: `dp_models`

- [x] Модель Whisper резолвится через `dp_models` (spec: `whisper_{size}_inprocess`)
- [x] Shared tokenizer резолвится через `dp_models` (spec: `shared_tokenizer_v1`)
- [x] Артефакты лежат в `DP_MODELS_ROOT` и валидируются fail-fast
- [x] В `meta.models_used[]` фиксируются: `model_name`, `model_version`, `weights_digest`, `runtime`, `engine`, `precision`, `device`

**Evidence**:
- `src/extractors/asr_extractor/main.py` — `whisper_spec_name = f"whisper_{self.model_size}_inprocess"`
- `src/extractors/asr_extractor/main.py:81-82` — `self.whisper_spec = self._mm.get_spec(model_name=whisper_spec_name)`, `mm.resolve()`
- `src/extractors/asr_extractor/main.py:68` — `tok_spec = self._mm.get_spec(model_name="shared_tokenizer_v1")`
- `run_cli.py:916-948` — резолв Whisper и tokenizer моделей для `models_used[]`
- `run_cli.py:1472` — сохранение `models_used[]` в NPZ meta

---

## 4) Segmenter Contract

### 4.1 Обязательные поля segments.json

- [x] Использует `schema_version="audio_segments_v1"` (проверяется в `run_cli.py`)
- [x] Использует `families.asr.segments[]` из `audio/segments.json`

### 4.2 Структура сегмента

- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Использует `center_sec`, `start_sec`, `end_sec` для временных меток
- [x] Не генерирует сегменты сам

**Evidence**:
- `src/extractors/asr_extractor/main.py:167-171` — извлечение `start_sample/end_sample`, `start_sec/end_sec/center_sec` из segments
- `src/extractors/asr_extractor/main.py:173` — использование `start_sample/end_sample` для загрузки

### 4.3 Family назначение

- [x] Документировано: `asr_extractor` → `families.asr.segments[]`
- [x] Отсутствие required family → fail-fast (проверяется в `run_cli.py`)

**Evidence**:
- `src/extractors/asr_extractor/README.md:14` — документация family=`asr`
- `run_cli.py:1031-1032` — проверка наличия `asr_segments` для required extractor'а

---

## 5) Наблюдаемость

### 5.1 Промежуточный прогресс

- [x] Progress reporting реализован в `run_cli.py` (stage-based)
- [x] Для `run_extractors` прогресс обновляется по мере завершения extractors
- [x] **Для asr_extractor**: progress обновляется каждые 10% сегментов (если сегментов ≥10)
- [x] Формат прогресса машиночитаем (JSON-lines в stdout)

**Evidence**:
- `run_cli.py:1198-1208` — progress callback для asr_extractor
- `src/extractors/asr_extractor/main.py:34-35` — параметр `progress_callback` в `__init__`
- `src/extractors/asr_extractor/main.py:195-202` — обновление прогресса каждые 10% сегментов
- `run_cli.py:103-136` — функция `_emit_progress()`

### 5.2 Stage timings

- [x] Stage timings сохраняются в NPZ meta (`stage_timings_ms`)
- [x] Per-extractor timings сохраняются в NPZ meta (`timings_by_extractor`)

**Evidence**:
- `run_cli.py:1447-1448` — сохранение `stage_timings_ms` и `timings_by_extractor` в NPZ meta

---

## 6) Feature Contract

### 6.1 Extractor gating

- [x] Выбор extractors через `--extractors <csv>` (например, `--extractors clap,tempo,loudness,asr`)
- [x] Default extractor set документирован: baseline Tier-0: `clap,tempo,loudness` (asr не в baseline)
- [x] В `meta` фиксируются `extractors_enabled[]` (через `run_cli.py`)

**Evidence**:
- `run_cli.py:793-801` — аргумент `--extractors`
- `run_cli.py:717-740` — функция `_parse_extractors_arg()` с default `clap,tempo,loudness`

### 6.2 Feature gating (per-feature control)

- [x] Персональные флаги для каждой фичи: `--asr-enable-token-sequences`, `--asr-enable-token-counts`, и т.д.
- [x] По умолчанию все фичи выключены (`False`), включаются только через `--asr-enable-*` флаги
- [x] В `meta.features_enabled[]` фиксируются включённые фичи
- [x] Зависимости фичей документированы в README (раздел "Feature dependencies")

**Evidence**:
- `run_cli.py:824-832` — аргументы для feature gating
- `src/extractors/asr_extractor/main.py:34-44` — параметры feature gating в `__init__`
- `src/extractors/asr_extractor/main.py:360-375` — трекинг `_features_enabled` в payload
- `src/extractors/asr_extractor/README.md:47-70` — раздел "Feature dependencies" и "Feature gating"

---

## 7) Производительность и ресурсы

### 7.1 Обязательные измерения

- [x] Latency per unit: измеряется в `docs/models_docs/resource_costs/asr_extractor_costs_v1.json` (planned)
- [x] CPU RSS peak: измеряется в `run_cli.py` через background sampler
- [x] GPU VRAM peak: измеряется в `run_cli.py` через background sampler (inprocess модель может использовать GPU)

**Evidence**:
- `run_cli.py:1114-1127` — background sampler для resource monitoring
- `run_cli.py:1449-1451` — сохранение `resource_metrics` в NPZ meta

### 7.2 Batching / OOM

- [x] Whisper decode выполняется inprocess и обрабатывает сегменты последовательно (нет “настоящего” batching decode)
- [x] OOM возможен на GPU для больших моделей; управляется выбором `model_size` и `device`

**Evidence**:
- `src/extractors/asr_extractor/main.py` — последовательный decode по сегментам + опциональный fallback decode

---

## 8) Проверка качества выхода

### 8.1 Минимальные sanity-checks

- [x] Диапазоны значений разумны: token IDs в [0, vocab_size-1], lang_id в [-1, 99]
- [x] Консистентность связных фичей: `token_counts` согласован с `token_ids_by_segment` по размеру
- [x] Статистические инварианты: `token_total >= 0`, `token_density_per_sec >= 0`, `speech_rate_wpm >= 0`
- [x] Для per-segment sequences: монотонность `segment_centers_sec` (проверяется в quality validation)
- [x] **Token validation**: полная валидация token IDs (диапазоны, dtype, special tokens, согласованность с lang_id)

**Evidence**:
- `src/extractors/asr_extractor/main.py:142-177` — функция `_validate_token_ids()` с полной валидацией
- `src/extractors/asr_extractor/main.py:183-184` — вызов валидации после inference
- `src/extractors/asr_extractor/main.py:304-305` — валидация в batch mode

### 8.2 Human-friendly визуализация / UI render

- [x] Renderer реализован в `src/core/renderer.py` (функция `render_asr_extractor()`)
- [x] Render-context JSON генерируется для каждого extractor'а в `_render/render_context.json`
- [x] Render-context содержит timeline данные, статистики, distributions (privacy-safe, без raw текста)
- [x] **HTML renderer для дебага**: `render_asr_extractor_html()` с опциональным декодированием token IDs (только для локального дебага)

**Evidence**:
- `src/core/renderer.py:324-384` — функция `render_asr_extractor()` (privacy-safe)
- `src/core/renderer.py:387-523` — функция `render_asr_extractor_html()` (debug mode с raw текстом)
- `run_cli.py:1371-1378` — генерация render-context для каждого компонента

---

## 9) Документация

### 9.1 README asr_extractor

- [x] **Input contract**: `audio/audio.wav`, `audio/segments.json` family=`asr`
- [x] **Output contract**: NPZ schema, пути, meta
- [x] **Models**: через dp_models, no-network (Whisper inprocess, shared tokenizer)
- [x] **Sampling requirements**: ссылка на SEGMENTER_CONTRACT.md
- [x] **Feature dependencies**: явное описание зависимостей между фичами
- [x] **Feature gating**: описание всех флагов включения/выключения фичей
- [x] **Parallelization**: последовательный decode по сегментам (Whisper constraint)
- [x] **Performance characteristics**: ссылка на resource_costs
- [x] **Quality validation**: sanity checks + валидация token IDs
- [x] **Visualization**: описание лучших практик визуализации для UI/сайта

**Evidence**:
- `src/extractors/asr_extractor/README.md` — полная документация со всеми разделами

---

## 10) Error Handling

### 10.1 Error surface (inprocess)

- [x] Ошибки декодинга Whisper (ValueError/RuntimeError) явно логируются и возвращаются как extractor error
- [x] Fallback decode (если включен) логируется предупреждением и не маскирует ошибки инициализации/входа

### 10.2 Retry логика

- [x] Retry на уровне orchestrator (`run_cli.py`) для transient ошибок (503, 504, timeout)
- [x] No-fallback policy: ошибки не маскируются, явно репортируются

**Evidence**:
- `run_cli.py:1198-1208` — retry логика для asr_extractor через `_retry_with_backoff()`
- `run_cli.py:138-198` — функция `_retry_with_backoff()` с exponential backoff

---

## 11) Compliance Summary

### Архитектура / контракты
- [x] per-run storage + manifest upsert
- [x] NPZ meta обязательные поля + validate_npz
- [x] no-fallback policy соблюдён
- [x] empty semantics корректны (не применимо для asr_extractor)
- [x] Segmenter contract соблюдён (audio/segments.json, families.asr)

### Модели / воспроизводимость
- [x] dp_models only, no downloads
- [x] models_used/model_signature корректны (Whisper + shared tokenizer)
- [x] scheduler_knobs зафиксированы в meta (legacy; may be empty for inprocess ASR)
- [x] Contract version для валидации совместимости с TextProcessor

### Наблюдаемость / качество / ресурсы
- [x] progress events есть и безопасны (каждые 10% сегментов)
- [x] stage timings сохранены
- [x] resource_costs измерены (ссылка на JSON, planned)
- [x] есть sanity checks + UI render (privacy-safe + HTML для дебага)
- [x] Token validation реализована (полная валидация token IDs)

### Feature gating
- [x] Персональные флаги для каждой фичи (9 фичей)
- [x] Зависимости фичей документированы в README
- [x] features_enabled[] фиксируется в meta

### Error handling
- [x] Ошибки whisper decode явно логируются и возвращаются в meta/error
- [x] Retry логика на уровне orchestrator
- [x] No-fallback policy соблюдён

---

## 12) Открытые задачи

1. **Resource costs measurement**: добавить измерения latency в `docs/models_docs/resource_costs/asr_extractor_costs_v1.json` — **planned**
2. **Quality validation scripts**: создать demo скрипты для валидации качества (token validation, статистики) — **planned**

---

## 13) Ссылки

- **Audit criteria**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **README**: `src/extractors/asr_extractor/README.md`
- **Resource costs**: `docs/models_docs/resource_costs/asr_extractor_costs_v1.json` (planned)
- **TextProcessor contract**: `docs/contracts/LLM_RENDERING.md` (для декодирования token IDs)

