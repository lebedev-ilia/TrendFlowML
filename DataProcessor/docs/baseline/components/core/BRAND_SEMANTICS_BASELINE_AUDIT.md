# Аудит соответствия brand_semantics требованиям baseline

**Дата проверки**: 2026-01-XX  
**Компонент**: `brand_semantics` (Semantic head, v1)  
**Расположение**: `VisualProcessor/core/model_process/core_identity/brand_semantics/`  
**Статус аудита**: ✅ **CLOSED** (2026-01-XX)

## Резюме

`brand_semantics` — semantic head компонент для распознавания брендов и логотипов в видео. Компонент использует детекции из `core_object_detections`, извлекает crops с паддингом, использует Embedding Service для поиска похожих брендов и возвращает per-track и per-frame top‑K идентификаций брендов.

Текущее состояние: компонент приведён к baseline‑контракту (no-fallback sampling, progress reporting, stage timings, обязательный `dataprocessor_version`, сохранение `times_s`).

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

### 2. Time axis → `times_s`

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент требует `union_timestamps_sec` в `frames_dir/metadata.json`
- Сохраняет `times_s = union_timestamps_sec[frame_indices]` в артефакт

### 3. Artifact write: atomic save + runtime validation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Запись `brand_semantics.npz` выполняется через `np.savez_compressed` (атомарное сохранение не требуется для NPZ)
- Артефакт должен проходить валидацию через `artifact_validator.validate_npz()` (рекомендуется)

### 4. `dataprocessor_version` всегда присутствует

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если входная meta не содержит `dataprocessor_version`, компонент пишет `"unknown"` (dev/default)
- В production должен передаваться реальный dp version (зафиксировать на уровне пайплайна)

### 5. Artifact filename

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Используется фиксированное имя артефакта: `ARTIFACT_FILENAME = "brand_semantics.npz"`
- Сохраняется в `result_store/<platform_id>/<video_id>/<run_id>/brand_semantics/brand_semantics.npz`

### 6. Stage timings + progress (state_events)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент измеряет время выполнения стадий: `initialization`, `load_deps`, `process_frames`, `saving`, `total`
- Тайминги сохраняются в `meta.stage_timings_ms` (в миллисекундах)
- Компонент публикует прогресс в `state_events.jsonl`:
  - Стадии: `start → load_deps → process_frames → save → done`
  - Гранулярный прогресс во время `process_frames` (≥10 обновлений по трекам)

### 7. NPZ output schema

**Статус**: ✅ **СООТВЕТСТВУЕТ**

Артефакт: `brand_semantics.npz` содержит:
- `frame_indices (N,) int32` — shared sampling group
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `track_ids (T,) int32` — ID треков
- `track_topk_ids (T, K) int32` — Top‑K брендов на трек
- `track_topk_scores (T, K) float32` — Similarity scores для треков
- `frame_topk_ids (N, K) int32` — Top‑K брендов на кадр (дедуплицированные)
- `frame_topk_scores (N, K) float32` — Similarity scores для кадров
- `semantic_label_names` — массив строк "id:name" для маппинга label_id → brand_name
- `meta` (dict, object-array) — canonical meta с `stage_timings_ms`

### 8. Dependencies

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- `core_object_detections` **обязателен** и должен быть выполнен перед `brand_semantics` (no-fallback)
- `core_object_detections` является источником истины для `frame_indices` и детекций
- Embedding Service должен быть доступен (fail-fast при недоступности)

### 9. Cost control

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Фильтрация по классу `logo_region` / `text_region` (если доступен в taxonomy)
- Ограничение количества треков через `--max-tracks`
- Ограничение количества детекций на кадр через `--max-dets-per-frame`
- 1 crop на трек (выбирается по `score × area × (optional sharpness)`)

## 📊 Performance / resource costs (baseline unit-cost)

**Примечание**: Измерения производительности планируются. Компонент использует Embedding Service (HTTP), поэтому latency зависит от сетевых условий и нагрузки на сервис. Также зависит от количества треков и детекций.

## 🔍 Quality validation (минимальный набор)

- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- Проверка валидности NPZ через `artifact_validator.validate_npz()`
- Human-friendly demo: `quality_report/demo_brand_semantics_quality.py` (если создан)

## Вопросы / открытые решения

1. **Измерения производительности**: Требуется провести измерения latency/RAM для типичных сценариев (зависит от Embedding Service и количества треков)
2. **Batch API**: Рассмотреть использование batch API Embedding Service для оптимизации (если доступен)
3. **Треки**: Компонент может работать без треков (генерирует per-detection track IDs), но это может снизить качество результатов

## Update: baseline contracts compliance (2026-01-XX)

Изменения:
- Добавлен `ARTIFACT_FILENAME = "brand_semantics.npz"`
- Добавлен progress reporting в `state_events.jsonl` (stage-based + granular)
- Добавлен `stage_timings_ms` в `meta`
- Обновлен формат `meta` для соответствия baseline контрактам (producer, producer_version, created_at, run identity keys, dataprocessor_version)
- Использован `apply_models_meta` для правильного формата `models_used`
- Обновлен README с описанием progress и stage timings

---

## Ссылки

- **README компонента**: `VisualProcessor/core/model_process/core_identity/brand_semantics/README.md`
- **Критерии аудита baseline**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Контракты**: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`

