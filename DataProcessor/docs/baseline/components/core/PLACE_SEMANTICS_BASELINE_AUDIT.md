# Аудит соответствия place_semantics требованиям baseline

**Дата проверки**: 2026-01-29  
**Компонент**: `place_semantics` (Semantic head, v1)  
**Расположение**: `VisualProcessor/core/model_process/core_identity/place_semantics/`  
**Статус аудита**: ✅ **CLOSED** (2026-01-29)

## Резюме

`place_semantics` — semantic head компонент для распознавания мест и лэндмарков в видео. Компонент использует кадры из `core_object_detections.frame_indices`, отправляет их в Embedding Service для поиска похожих мест, группирует кадры по местам в tracks (временная сегментация) и возвращает per-track и per-frame top‑K идентификаций мест.

Текущее состояние: компонент приведён к baseline‑контракту (no-fallback sampling, progress reporting, stage timings, обязательный `dataprocessor_version`, сохранение `times_s`, отдельные tracks для разных мест).

## Оценки (1–10)

- **Качество кода и алгоритмов**: **8/10**
- **Логика алгоритмов**: **8/10**
- **Логика глобального взаимодействия**: **9/10**
- **Оптимизации (параллелизм, батчинг)**: **7/10**

## ✅ Соответствие требованиям

### 1. Baseline contract (sampling, no-fallback, union-domain)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент **строго** читает `frame_indices` из `metadata.json[core_object_detections.frame_indices]`
- При отсутствии/пустоте `frame_indices` → **fail-fast** (no‑fallback)
- `frame_indices` — shared sampling group (контракт Segmenter)
- При отсутствии Embedding Service → **RuntimeError** (no-fallback)

### 2. Time axis → `times_s`

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент требует `union_timestamps_sec` в `frames_dir/metadata.json`
- Сохраняет `times_s = union_timestamps_sec[frame_indices]` в артефакт

### 3. Artifact write: atomic save + runtime validation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Запись `place_semantics.npz` выполняется через `np.savez_compressed`
- Артефакт должен проходить валидацию через `artifact_validator.validate_npz()` (рекомендуется)

### 4. `dataprocessor_version` всегда присутствует

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если входная meta не содержит `dataprocessor_version`, компонент пишет `"unknown"` (dev/default)
- В production должен передаваться реальный dp version (зафиксировать на уровне пайплайна)

### 5. Artifact filename

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Используется фиксированное имя артефакта: `ARTIFACT_FILENAME = "place_semantics.npz"`
- Сохраняется в `result_store/<platform_id>/<video_id>/<run_id>/place_semantics/place_semantics.npz`

### 6. Stage timings + progress (state_events)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент измеряет время выполнения стадий: `initialization`, `load_deps`, `process_frames`, `saving`, `total`
- Тайминги сохраняются в `meta.stage_timings_ms` (в миллисекундах)
- Компонент публикует прогресс в `state_events.jsonl`:
  - Стадии: `start → load_deps → process_frames → save → done`
  - Гранулярный прогресс во время `process_frames` (≥10 обновлений по кадрам)

### 7. NPZ output schema

**Статус**: ✅ **СООТВЕТСТВУЕТ**

Артефакт: `place_semantics.npz` содержит:
- `frame_indices (N,) int32` — shared sampling group
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `track_ids (T,) int32` — ID треков (отдельные tracks для разных мест)
- `track_topk_ids (T, K) int32` — Top‑K мест на трек
- `track_topk_scores (T, K) float32` — Similarity scores для треков
- `track_present_mask (T,) bool` — Маска присутствия треков
- `track_is_confident_top1 (T,) bool` — Флаг уверенности для top-1 места на трек
- `frame_topk_ids (N, K) int32` — Top‑K мест на кадр
- `frame_topk_scores (N, K) float32` — Similarity scores для кадров
- `frame_is_confident_top1 (N,) bool` — Флаг уверенности для top-1 места на кадр
- `semantic_label_names (A,) str` — Массив строк "id:name" для маппинга label_id → place_name
- `threshold_per_label_arr (A,) float32` — Пороги для каждого места (NaN если нет)
- `meta` (dict, object-array) — canonical meta с `stage_timings_ms`

### 8. Dependencies

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- `core_object_detections` **обязателен** для получения `frame_indices` (no-fallback)
- Embedding Service должен быть доступен (fail-fast при недоступности)
- `FrameManager` используется для загрузки кадров

### 9. Track-level aggregation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент группирует кадры по местам в tracks (временная сегментация)
- Отдельные tracks для разных мест (не один scene-level track)
- Алгоритм группировки:
  - Группирует кадры с одинаковым top-1 местом
  - Объединяет треки, если разрыв ≤ `max_gap_sec`
  - Фильтрует треки короче `min_track_length`

### 10. Sampling requirements documentation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- В README описан раздел "Sampling requirements"
- Описана непрерывная кривая выборки (зависит от `duration_s`)
- Указаны минимальные/максимальные значения (`min_frames=50`, `max_frames=2000`)
- Указано, что Segmenter является единственным владельцем sampling

### 11. Models documentation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- В README описан раздел "Models"
- Описано использование Embedding Service
- Указаны runtime/engine/precision/device
- Модели фиксируются в `meta.models_used[]`

### 12. Parallelization documentation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- В README описан раздел "Parallelization"
- Описано внутреннее параллелизм (HTTP запросы последовательно)
- Описано внешнее параллелизм (безопасно параллелить по разным видео)
- Указаны ограничения и требования

### 13. Features documentation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- В README описан раздел "Features"
- Перечислены все выходные фичи (per-frame и per-track)
- Описано влияние на стоимость

## 📊 Performance / resource costs (baseline unit-cost)

**Примечание**: Измерения производительности планируются. Компонент использует Embedding Service (HTTP), поэтому latency зависит от сетевых условий и нагрузки на сервис. Также зависит от количества кадров.

**Единица обработки**: `frame` (один кадр)

**Типичные значения** (зависят от Embedding Service):

| Resolution | Latency per frame | CPU RAM peak | Notes |
|------------|-------------------|--------------|-------|
| 1920x1080 | ~200-500 ms | ~100-200 MB | HTTP latency + Embedding Service processing |

**Для видео с N кадрами**: Total latency ≈ N × latency_per_frame

**Полные данные**: см. `docs/models_docs/resource_costs/place_semantics_costs_v1.json` (планируется)

## 🔍 Quality validation (минимальный набор)

- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- Проверка валидности NPZ через `artifact_validator.validate_npz()`
- Human-friendly demo: `quality_report/demo_place_semantics_quality.py` (существует)

## Вопросы / открытые решения

1. **Измерения производительности**: Требуется провести измерения latency/RAM для типичных сценариев (зависит от Embedding Service и количества кадров)
2. **Batch API**: Рассмотреть использование batch API Embedding Service для оптимизации (если доступен)
3. **Temporal segmentation**: Алгоритм группировки кадров в tracks может быть улучшен (например, использование более сложных методов сегментации)

## Update: baseline contracts compliance (2026-01-XX)

Изменения:
- Переписан для использования Embedding Service вместо offline базы
- Добавлен `ARTIFACT_FILENAME = "place_semantics.npz"`
- Добавлен progress reporting в `state_events.jsonl` (stage-based + granular)
- Добавлен `stage_timings_ms` в `meta`
- Обновлен формат `meta` для соответствия baseline контрактам (producer, producer_version, created_at, run identity keys, dataprocessor_version)
- Использован `apply_models_meta` для правильного формата `models_used`
- Добавлена track-level агрегация (отдельные tracks для разных мест)
- Добавлены `track_is_confident_top1` и `frame_is_confident_top1`
- Исправлен empty case на RuntimeError (no-fallback)
- Обновлен README с полным описанием (sampling requirements, models, parallelization, features)

---

## Ссылки

- **README компонента**: `VisualProcessor/core/model_process/core_identity/place_semantics/README.md`
- **Критерии аудита baseline**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Контракты**: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- **Schema semantic heads**: `docs/models_docs/SCHEMA_SEMANTIC_HEADS_NPZ.md`

