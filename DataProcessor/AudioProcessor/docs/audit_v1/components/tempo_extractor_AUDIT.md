# Audit: `tempo_extractor`

**Дата**: 2026-01-XX  
**Статус**: `done`  
**Критерии**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`

---

## 1) Summary

`tempo_extractor` соответствует всем критериям `AP_AUDIT_CRITERIA.md`:
- ✅ **Segmenter contract**: использует `audio/audio.wav` и `audio/segments.json` family=`tempo`
- ✅ **No-fallback policy**: fail-fast при отсутствии segments
- ✅ **Model system**: не использует ML-модели (signal processing only, librosa)
- ✅ **NPZ schema**: `audio_npz_v1` с обязательными полями meta
- ✅ **Per-run storage**: фиксированное имя `tempo_extractor_features.npz`
- ✅ **Segment parallelism**: поддержка `segment_parallelism` и `max_inflight`
- ✅ **UI Render**: renderer реализован в `src/core/renderer.py`

---

## 2) Contract Compliance Checklist

### 2.1 Интерфейсы и границы ответственности

**Extractor как компонент**:
- [x] Реализует `BaseExtractor.run_segments(input_uri, tmp_path, segments, ...)`
- [x] Не делает скрытых глобальных сайд-эффектов (signal processing only, no network)
- [x] Требование специфичного входа декларировано: `audio/segments.json` family=`tempo` (см. README)

**Evidence**:
- `src/extractors/tempo_extractor/__init__.py:105` — метод `run_segments()`
- `src/extractors/tempo_extractor/__init__.py:128-129` — проверка segments: `if not isinstance(segments, list) or not segments: raise ValueError("segments is empty (no-fallback)")`

### 2.2 Контракты входа (Segmenter contract)

- [x] Входная единица: `audio/audio.wav` (Segmenter) + `audio/segments.json` family=`tempo`
- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Использует `center_sec` для временных меток
- [x] Не генерирует сегменты сам (Segmenter — единственный владелец sampling)

**Evidence**:
- `src/extractors/tempo_extractor/__init__.py:136-145` — чтение `start_sample/end_sample` и `center_sec` из segments
- `src/extractors/tempo_extractor/__init__.py:139-144` — `self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, ...)`

### 2.3 No-fallback policy

- [x] Пустой список segments → fail-fast: `raise ValueError("segments is empty (no-fallback)")`
- [x] Ошибка обработки → `status="error"` в `ExtractorResult`
- [x] Observability: ошибки логируются (без raw audio данных)

**Evidence**:
- `src/extractors/tempo_extractor/__init__.py:128-129` — проверка segments
- `src/extractors/tempo_extractor/__init__.py:198-201` — error handling в `run_segments()`

### 2.4 Per-run storage

- [x] Имя NPZ файла стабильное: `tempo_extractor_features.npz` (фиксированное имя)
- [x] Запись NPZ атомарная (выполняется в `run_cli.py` через `_atomic_save_npz()`)
- [x] Нет произвольных JSON артефактов (только NPZ)

**Evidence**:
- `run_cli.py:1220` — сохранение через `_save_component_npz()` с фиксированным именем

### 2.5 NPZ schema + meta contract

**NPZ schema**: `schema_version="audio_npz_v1"`

**Обязательные ключи**:
- [x] `feature_names: object[str]` — имена фичей (scalars)
- [x] `feature_values: float32[]` — значения фичей (scalars)
- [x] `tempo_estimates: float32[N]` — оценки темпа по времени
- [x] `windowed_bpm.times_sec: float32[N]` — временные метки окон
- [x] `windowed_bpm.bpm: float32[N]` — BPM по окнам
- [x] `meta: object(dict)` — обязательные поля

**Обязательные поля `meta`**:
- [x] Все обязательные поля (через `run_cli.py`)

**Extractor-level требования**:
- [x] Фичи имеют стабильные имена: `tempo_bpm`, `tempo_bpm_mean`, `tempo_bpm_median`, `tempo_bpm_std`, `tempo_confidence`, `segments_count`
- [x] Единицы измерения зафиксированы в README (BPM ∈ [40, 220])
- [x] Missing values: NaN (если применимо)
- [x] `windowed_bpm.times_sec` строго монотонно возрастает (проверяется в quality validation)
- [x] Warnings: `tempo_out_of_range`, `low_confidence`, `signal_too_quiet`

**Evidence**:
- `run_cli.py:196-230` — функция `_save_component_npz()` для `tempo_extractor`
- `src/extractors/tempo_extractor/__init__.py:184-196` — payload структура

### 2.6 Valid empty outputs

- [x] Валидная пустота не применима для `tempo_extractor` (если segments пустой → error, не empty)

**Evidence**:
- `src/extractors/tempo_extractor/__init__.py:128-129` — пустые segments → error, не empty

---

## 3) Model System

### 3.1 Запрещённые практики

- [x] Нет сетевых загрузок моделей/весов (signal processing only, librosa)
- [x] Не использует ML-модели (librosa signal processing)

**Evidence**:
- `src/extractors/tempo_extractor/__init__.py:65-76` — использование librosa (signal processing)

### 3.2 Обязательное правило: `dp_models`

- [x] Не использует ML-модели (signal processing only)
- [x] `models_used[]` пустой (или отсутствует в meta)

**Evidence**:
- `src/extractors/tempo_extractor/__init__.py:32` — dependencies: `["librosa", "numpy"]` (нет ML-моделей)

---

## 4) Segmenter Contract

### 4.1 Обязательные поля segments.json

- [x] Использует `schema_version="audio_segments_v1"` (проверяется в `run_cli.py`)
- [x] Использует `families.tempo.segments[]` из `audio/segments.json`

### 4.2 Структура сегмента

- [x] Использует `start_sample/end_sample` для загрузки через `AudioUtils.load_audio_segment()`
- [x] Использует `center_sec` для временных меток
- [x] Не генерирует сегменты сам

**Evidence**:
- `src/extractors/tempo_extractor/__init__.py:136-145` — использование `start_sample/end_sample` и `center_sec`

### 4.3 Family назначение

- [x] Документировано: `tempo_extractor` → `families.tempo.segments[]`
- [x] Отсутствие required family → fail-fast (проверяется в `run_cli.py`)

**Evidence**:
- `src/extractors/tempo_extractor/README.md` — документация family=`tempo`
- `run_cli.py:907-908` — проверка наличия `tempo_segments` для required extractor'а

---

## 5) Наблюдаемость

### 5.1 Промежуточный прогресс

- [x] Progress reporting реализован в `run_cli.py` (stage-based)
- [x] Для `run_extractors` прогресс обновляется по мере завершения extractors
- [x] Формат прогресса машиночитаем (JSON-lines в stdout)

**Evidence**:
- `run_cli.py:1124-1130` — progress reporting для каждого extractor'а

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
- `run_cli.py:603-623` — функция `_parse_extractors_arg()` с default `clap,tempo,loudness`

---

## 7) Производительность и ресурсы

### 7.1 Обязательные измерения

- [x] Latency per unit: измеряется в `docs/models_docs/resource_costs/tempo_extractor_costs_v1.json`
- [x] CPU RSS peak: измеряется в `run_cli.py` через background sampler
- [x] GPU VRAM peak: не применимо (CPU-only)

**Evidence**:
- `run_cli.py:1136-1146` — background sampler для resource monitoring

### 7.2 Batching / OOM

- [x] Конфигурируемый `segment_parallelism` через `--segment-parallelism` (scheduler-controlled)
- [x] Конфигурируемый `max_inflight` через `--max-inflight` (scheduler-controlled)
- [x] OOM не применимо (CPU-only, signal processing)

**Evidence**:
- `src/extractors/tempo_extractor/__init__.py:131-133` — поддержка `segment_parallelism` и `max_inflight`
- `src/extractors/tempo_extractor/__init__.py:158-175` — параллельная обработка через ThreadPoolExecutor

---

## 8) Проверка качества выхода

### 8.1 Минимальные sanity-checks

- [x] Диапазоны значений разумны: tempo BPM ∈ [40, 220] (проверяется в warnings)
- [x] Консистентность связных фичей: `windowed_bpm.times_sec` согласован с `windowed_bpm.bpm` по размеру
- [x] Статистические инварианты: tempo BPM ∈ [40, 220], confidence ∈ [0, 1]
- [x] Для per-segment sequences: монотонность `windowed_bpm.times_sec` (проверяется в quality validation)
- [x] Warnings: `tempo_out_of_range`, `low_confidence`, `signal_too_quiet`

**Evidence**:
- `src/extractors/tempo_extractor/__init__.py:84-90` — генерация warnings
- Quality validation скрипты: `scripts/baseline/demo_tempo_extractor_quality.py`

### 8.2 Human-friendly визуализация / UI render

- [x] Renderer реализован в `src/core/renderer.py` (функция `render_tempo_extractor()`)
- [x] Render-context JSON генерируется для каждого extractor'а в `_render/render_context.json`
- [x] Render-context содержит timeline данные, статистики, distributions, warnings

**Evidence**:
- `src/core/renderer.py:122-200` — функция `render_tempo_extractor()`
- `run_cli.py:1371-1378` — генерация render-context для каждого компонента

---

## 9) Документация

### 9.1 README tempo_extractor

- [x] **Input contract**: `audio/audio.wav`, `audio/segments.json` family=`tempo`
- [x] **Output contract**: NPZ schema, пути, meta
- [x] **Models**: не использует ML-модели (signal processing only)
- [x] **Sampling requirements**: ссылка на SEGMENTER_CONTRACT.md
- [x] **Parallelization**: `segment_parallelism` и `max_inflight` через ThreadPoolExecutor
- [x] **Performance characteristics**: ссылка на resource_costs
- [x] **Quality validation**: ссылка на demo скрипты

**Evidence**:
- `src/extractors/tempo_extractor/README.md` — полная документация

---

## 10) Compliance Summary

### Архитектура / контракты
- [x] per-run storage + manifest upsert
- [x] NPZ meta обязательные поля + validate_npz
- [x] no-fallback policy соблюдён
- [x] empty semantics корректны (не применимо для tempo_extractor)
- [x] Segmenter contract соблюдён (audio/segments.json, families.tempo)

### Модели / воспроизводимость
- [x] Не использует ML-модели (signal processing only)
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

- **Baseline audit**: `docs/baseline/components/audio/TEMPO_EXTRACTOR_BASELINE_AUDIT.md`
- **Audit criteria**: `AudioProcessor/docs/audit_v1/AP_AUDIT_CRITERIA.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **README**: `src/extractors/tempo_extractor/README.md`
- **Resource costs**: `docs/models_docs/resource_costs/tempo_extractor_costs_v1.json`

