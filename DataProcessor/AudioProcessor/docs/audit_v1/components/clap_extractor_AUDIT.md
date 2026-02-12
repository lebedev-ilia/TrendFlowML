# Audit: `clap_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`clap_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: использует `audio/audio.wav` и `audio/segments.json` family=`clap`
- ✅ **No-fallback policy**: fail-fast при отсутствии segments
- ✅ **Model system**: загрузка через `dp_models` (ModelManager), no-network
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `clap_extractor_features.npz`
- ✅ **Batching/OOM**: поддержка `model_batch_size`, OOM fallback в orchestrator
- ✅ **UI Render**: renderer реализован в `src/core/renderer.py`

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run_segments(input_uri, tmp_path, segments, ...)`
- [x] Не делает скрытых глобальных сайд-эффектов (модель загружается через ModelManager, no-network)
- [x] Требование специфичного входа декларировано: `audio/segments.json` family=`clap` (см. README)

**Evidence**:
- `src/extractors/clap_extractor/__init__.py:179` — метод `run_segments()`
- `src/extractors/clap_extractor/__init__.py:206-207` — проверка segments: `if not isinstance(segments, list) or not segments: raise ValueError("segments is empty (no-fallback)")`

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + `audio/segments.json` family=`clap`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/clap_extractor/__init__.py:217-219` — чтение `start_sample/end_sample` из segments
- `src/extractors/clap_extractor/__init__.py:219` — `self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=None)`

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("segments is empty (no-fallback)")`
- [x] Ошибка модели/инференса → `status="error"` в `ExtractorResult`
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/clap_extractor/__init__.py:206-207` — проверка segments
- `src/extractors/clap_extractor/__init__.py:310-311` — error handling в `run_segments()`

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `clap_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] Нет произвольных JSON артефактов (только NPZ)

**Evidence**:
- `run_cli.py:1220` — сохранение через `_save_component_npz()` с фиксированным именем
- `run_cli.py:100-598` — функция `_save_component_npz()` использует атомарную запись

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `embedding: float32[D]` — агрегат (mean) по сегментам
- [x] `embedding_sequence: float32[N,D]` — эмбеддинги по сегментам
- [x] `segment_centers_sec: float32[N]` — временные метки сегментов
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] `producer`, `producer_version`, `schema_version`, `created_at`
- [x] `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- [x] `status` ∈ {`ok`, `empty`, `error`}
- [x] `models_used[]`, `model_signature` (если используются модели)
- [x] `device_used`
- [x] `scheduler_knobs` (segment_parallelism, max_inflight, model_batch_size)

**Extractor-level требования**:
- [x] Фичи имеют стабильные имена: `clap_norm`, `clap_magnitude_mean`, `clap_magnitude_std`, `clap_non_zero_count`, `segments_count`, `embedding_dim`
- [x] Единицы измерения зафиксированы в README
- [x] Missing values: NaN (если применимо)
- [x] `segment_centers_sec` строго монотонно возрастает (проверяется в quality validation)

**Evidence**:
- `run_cli.py:139-598` — функция `_save_component_npz()` для `clap_extractor` (строки 139-195)
- `src/extractors/clap_extractor/__init__.py:290-308` — payload структура

### 2.6 Valid empty outputs

- [x] Валидная пустота не применима для `clap_extractor` (если segments пустой → error, не empty)
- [x] Если extractor возвращает empty, это должно быть явно в payload: `status="empty"`, `empty_reason`

**Evidence**:
- `src/extractors/clap_extractor/__init__.py:206-207` — пустые segments → error, не empty

---

## 3) Model System

### 3.1 Запрещённые практики

- [x] Нет сетевых загрузок моделей/весов
- [x] Используется `dp_models` (ModelManager) для резолва checkpoint

**Evidence**:
- `src/extractors/clap_extractor/__init__.py:95-105` — установка offline флагов
- `src/extractors/clap_extractor/__init__.py:114-125` — резолв через ModelManager

### 3.2 Обязательное правило: `dp_models`

- [x] Модель CLAP резолвится через `dp_models`
- [x] Spec: `laion_clap` (ModelManager)
- [x] Артефакты лежат в `DP_MODELS_ROOT` и валидируются fail-fast
- [x] В `meta.models_used[]` фиксируются: `model_name`, `model_version`, `weights_digest`, `runtime`, `engine`, `precision`, `device`

**Evidence**:
- `src/extractors/clap_extractor/__init__.py:117-118` — `mm.get_spec(model_name="laion_clap")`
- `src/extractors/clap_extractor/__init__.py:119` — `mm.resolve(spec)` возвращает weights_digest и artifacts
- `run_cli.py:1245-1273` — сохранение `models_used[]` в NPZ meta

---

## 4) Segmenter Contract

### 4.1 Обязательные поля segments.json

- [x] Использует `schema_version="audio_segments_v1"` (проверяется в `run_cli.py`)
- [x] Использует `families.clap.segments[]` из `audio/segments.json`

### 4.2 Структура сегмента

- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Использует `center_sec` для временных меток
- [x] Не генерирует сегменты сам

**Evidence**:
- `src/extractors/clap_extractor/__init__.py:212` — извлечение `center_sec` из segments
- `src/extractors/clap_extractor/__init__.py:217-219` — использование `start_sample/end_sample`

### 4.3 Family назначение

- [x] Документировано: `clap_extractor` → `families.clap.segments[]`
- [x] Отсутствие required family → fail-fast (проверяется в `run_cli.py`)

**Evidence**:
- `src/extractors/clap_extractor/README.md:12` — документация family=`clap`
- `run_cli.py:905-906` — проверка наличия `clap_segments` для required extractor'а

---

## 5) Наблюдаемость

### 5.1 Промежуточный прогресс

- [x] Progress reporting реализован в `run_cli.py` (stage-based)
- [x] Для `run_extractors` прогресс обновляется по мере завершения extractors
- [x] Формат прогресса машиночитаем (JSON-lines в stdout)

**Evidence**:
- `run_cli.py:1124-1130` — progress reporting для каждого extractor'а
- `run_cli.py:99-120` — функция `_emit_progress()`

### 5.2 Stage timings

- [x] Stage timings сохраняются в NPZ meta (`stage_timings_ms`)
- [x] Per-extractor timings сохраняются в NPZ meta (`timings_by_extractor`)

**Evidence**:
- `run_cli.py:1447-1448` — сохранение `stage_timings_ms` и `timings_by_extractor` в NPZ meta

---

## 6) Feature Contract

### 6.1 Extractor gating

- [x] Выбор extractors через `--extractors <csv>` (например, `--extractors clap,tempo,loudness`)
- [x] Default extractor set документирован: baseline Tier-0: `clap,tempo,loudness`
- [x] В `meta` фиксируются `extractors_enabled[]` (через `run_cli.py`)

**Evidence**:
- `run_cli.py:679-687` — аргумент `--extractors`
- `run_cli.py:603-623` — функция `_parse_extractors_arg()` с default `clap,tempo,loudness`

### 6.2 Feature sets (future)

- [ ] Feature sets (baseline/standard/full) для выбора подмножества фичей внутри extractor'а — **planned**

---

## 7) Производительность и ресурсы

### 7.1 Обязательные измерения

- [x] Latency per unit: измеряется в `docs/models_docs/resource_costs/clap_extractor_costs_v1.json`
- [x] CPU RSS peak: измеряется в `run_cli.py` через background sampler
- [x] GPU VRAM peak: измеряется в `run_cli.py` через background sampler

**Evidence**:
- `run_cli.py:1136-1146` — background sampler для resource monitoring
- `run_cli.py:1449-1451` — сохранение `resource_metrics` в NPZ meta

### 7.2 Batching / OOM

- [x] Конфигурируемый `batch_size` через `--clap-batch-size` (scheduler-controlled)
- [x] OOM fallback реализован в `run_cli.py` (автоматическое уменьшение batch_size, максимум 2 попытки)

**Evidence**:
- `run_cli.py:678` — аргумент `--clap-batch-size`
- `run_cli.py:1167-1175` — использование `_run_clap_with_oom_fallback()` для OOM fallback
- `run_cli.py:110-178` — функция `_run_clap_with_oom_fallback()`

---

## 8) Проверка качества выхода

### 8.1 Минимальные sanity-checks

- [x] Диапазоны значений разумны: embedding norms > 0, finite values
- [x] Консистентность связных фичей: `embedding_sequence` согласован с `segment_centers_sec` по размеру
- [x] Статистические инварианты: embedding norms > 0, finite values
- [x] Для per-segment sequences: монотонность `segment_centers_sec` (проверяется в quality validation)

**Evidence**:
- `src/extractors/clap_extractor/__init__.py:285-288` — вычисление статистик (norms, magnitude)
- Quality validation скрипты: `scripts/baseline/demo_clap_extractor_quality.py`

### 8.2 Human-friendly визуализация / UI render

- [x] Renderer реализован в `src/core/renderer.py` (функция `render_clap_extractor()`)
- [x] Render-context JSON генерируется для каждого extractor'а в `_render/render_context.json`
- [x] Render-context содержит timeline данные, статистики, distributions

**Evidence**:
- `src/core/renderer.py:45-120` — функция `render_clap_extractor()`
- `run_cli.py:1371-1378` — генерация render-context для каждого компонента

---

## 9) Документация

### 9.1 README clap_extractor

- [x] **Input contract**: `audio/audio.wav`, `audio/segments.json` family=`clap`
- [x] **Output contract**: NPZ schema, пути, meta
- [x] **Models**: через dp_models, no-network
- [x] **Sampling requirements**: ссылка на SEGMENTER_CONTRACT.md
- [x] **Parallelization**: batching через `model_batch_size`, segment_parallelism=1 (unsafe)
- [x] **Performance characteristics**: ссылка на resource_costs
- [x] **Quality validation**: ссылка на demo скрипты

**Evidence**:
- `src/extractors/clap_extractor/README.md` — полная документация

---

## 10) Compliance Summary

### Архитектура / контракты
- [x] per-run storage + manifest upsert
- [x] NPZ meta обязательные поля + validate_npz
- [x] no-fallback policy соблюдён
- [x] empty semantics корректны (не применимо для clap_extractor)
- [x] Segmenter contract соблюдён (audio/segments.json, families.clap)

### Модели / воспроизводимость
- [x] dp_models only, no downloads
- [x] models_used/model_signature корректны
- [x] scheduler_knobs зафиксированы в meta

### Наблюдаемость / качество / ресурсы
- [x] progress events есть и безопасны
- [x] stage timings сохранены
- [x] resource_costs измерены (ссылка на JSON)
- [x] есть sanity checks + UI render

---

## 11) Открытые задачи

1. **Feature gating**: поддержка feature sets (baseline/standard/full) для выбора подмножества фичей — **planned**
2. **Progress reporting внутри extractor**: обновление прогресса по мере обработки сегментов (если сегментов ≥10) — **planned**

---

## 12) Ссылки

- **Baseline audit**: `docs/baseline/components/audio/CLAP_EXTRACTOR_BASELINE_AUDIT.md`
- **Audit criteria**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **README**: `src/extractors/clap_extractor/README.md`
- **Resource costs**: `docs/models_docs/resource_costs/clap_extractor_costs_v1.json`

