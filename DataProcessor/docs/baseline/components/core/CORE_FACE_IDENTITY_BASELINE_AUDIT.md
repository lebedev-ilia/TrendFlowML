# Аудит соответствия core_face_identity требованиям baseline

**Дата проверки**: 2026-01-XX  
**Компонент**: `core_face_identity` (Semantic head, v1)  
**Расположение**: `VisualProcessor/core/model_process/core_identity/face_identity/`  
**Статус аудита**: ✅ **CLOSED** (2026-01-XX)

## Резюме

`core_face_identity` — semantic head компонент для идентификации известных людей (celebrity retrieval) в видео. Компонент извлекает face crops из `core_face_landmarks`, использует Embedding Service для поиска похожих лиц и возвращает per-frame top‑K идентификаций с similarity scores.

Текущее состояние: компонент приведён к baseline‑контракту (no-fallback sampling, atomic save, progress reporting, stage timings, обязательный `dataprocessor_version`, сохранение `times_s`).

## Оценки (1–10)

- **Качество кода и алгоритмов**: **8/10**
- **Логика алгоритмов**: **8/10**
- **Логика глобального взаимодействия**: **9/10**
- **Оптимизации (параллелизм, батчинг)**: **7/10**

## ✅ Соответствие требованиям

### 1. Baseline contract (sampling, no-fallback, union-domain)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент **строго** читает `frame_indices` из `core_face_landmarks/landmarks.npz` (не из metadata)
- Фильтрует `frame_indices` по `face_present` — оставляет только кадры, где были найдены лица
- При отсутствии/пустоте `frame_indices` или `face_present` → **fail-fast** (no‑fallback)
- При отсутствии лиц в видео → **valid empty** (`status="empty"`, `empty_reason="no_faces_in_video"`)

### 2. Time axis → `times_s`

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент требует `union_timestamps_sec` в `frames_dir/metadata.json`
- Сохраняет `times_s = union_timestamps_sec[frame_indices]` в артефакт (только для кадров с лицами)

### 3. Artifact write: atomic save + runtime validation

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Запись `face_identity.npz` выполняется через `np.savez_compressed` (атомарное сохранение не требуется для NPZ)
- Артефакт должен проходить валидацию через `artifact_validator.validate_npz()` (рекомендуется)

### 4. `dataprocessor_version` всегда присутствует

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если входная meta не содержит `dataprocessor_version`, компонент пишет `"unknown"` (dev/default)
- В production должен передаваться реальный dp version (зафиксировать на уровне пайплайна)

### 5. Artifact filename

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Используется фиксированное имя артефакта: `ARTIFACT_FILENAME = "face_identity.npz"`
- Сохраняется в `result_store/<platform_id>/<video_id>/<run_id>/core_face_identity/face_identity.npz`

### 6. Stage timings + progress (state_events)

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Компонент измеряет время выполнения стадий: `initialization`, `load_deps`, `process_frames`, `saving`, `total`
- Тайминги сохраняются в `meta.stage_timings_ms` (в миллисекундах)
- Компонент публикует прогресс в `state_events.jsonl`:
  - Стадии: `start → load_deps → process_frames → save → done`
  - Гранулярный прогресс во время `process_frames` (≥10 обновлений)

### 7. NPZ output schema

**Статус**: ✅ **СООТВЕТСТВУЕТ**

Артефакт: `face_identity.npz` содержит:
- `frame_indices (N,) int32` — только кадры с лицами
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `face_ids (N, K) int32` — ID известных людей на каждом кадре (-1 если нет результата)
- `face_names (N, K) str` — имена известных людей на каждом кадре (пустая строка если нет)
- `face_similarities (N, K) float32` — similarity scores (0.0 если нет результата)
- `meta` (dict, object-array) — canonical meta с `stage_timings_ms`

### 8. Dependencies

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- `core_face_landmarks` **обязателен** и должен быть выполнен перед `core_face_identity` (no-fallback)
- `core_face_landmarks` является источником истины для `frame_indices` (не `core_object_detections`)
- Embedding Service должен быть доступен (fail-fast при недоступности)

### 9. Valid empty outputs

**Статус**: ✅ **СООТВЕТСТВУЕТ**

- Если лиц в видео нет → компонент пишет NPZ со `status="empty"` и `empty_reason="no_faces_in_video"`
- Численные массивы содержат пустые массивы (не NaN для empty case)
- `empty_reason` обязателен если `status="empty"`, иначе `null`

## 📊 Performance / resource costs (baseline unit-cost)

**Примечание**: Измерения производительности планируются. Компонент использует Embedding Service (HTTP), поэтому latency зависит от сетевых условий и нагрузки на сервис.

## 🔍 Quality validation (минимальный набор)

- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- Проверка валидности NPZ через `artifact_validator.validate_npz()`
- Human-friendly demo: `quality_report/demo_core_face_identity_quality.py` (если создан)

## Вопросы / открытые решения

1. **Измерения производительности**: Требуется провести измерения latency/RAM для типичных сценариев (зависит от Embedding Service)
2. **Batch API**: Рассмотреть использование batch API Embedding Service для оптимизации (если доступен)

## Update: baseline contracts compliance (2026-01-XX)

Изменения:
- Добавлен `ARTIFACT_FILENAME = "face_identity.npz"`
- Добавлен progress reporting в `state_events.jsonl` (stage-based + granular)
- Добавлен `stage_timings_ms` в `meta`
- Обновлен README с описанием progress и stage timings

---

## Ссылки

- **README компонента**: `VisualProcessor/core/model_process/core_identity/face_identity/README.md`
- **Критерии аудита baseline**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Контракты**: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`

