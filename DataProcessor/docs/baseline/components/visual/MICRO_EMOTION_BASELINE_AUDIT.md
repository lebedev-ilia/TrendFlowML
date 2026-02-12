# Аудит соответствия micro_emotion требованиям baseline

**Дата проверки**: 2026-01-27  
**Компонент**: `micro_emotion` (Visual module)  
**Расположение**: `VisualProcessor/modules/micro_emotion/`

## Резюме

Компонент `micro_emotion` приведён к baseline-архитектуре и контрактам:
- наследуется от `BaseModule`,
- строго использует `frame_indices` от Segmenter (no-fallback) и `union_timestamps_sec` как source-of-truth time-axis,
- имеет обязательную зависимость от `core_face_landmarks` (работает только с лицами),
- реализует валидный `empty` при полном отсутствии лиц (`empty_reason="no_faces_in_video"`),
- запускает OpenFace через Docker (**GPU-only** по политике) и падает на partial failures (policy: error),
- пишет один NPZ-артефакт фиксированного имени (`micro_emotion.npz`) и валидируется через `validate_npz()`,
- пишет progress-events в `state_events.jsonl` (unit=`frame`),
- фиксирует `stage_timings_ms` в output (profiling by stages).

## ✅ Соответствие требованиям

### 1) Наследование и интерфейсы
- ✅ `MicroEmotionModule(BaseModule)` реализует `process(frame_manager, frame_indices, config)` и `required_dependencies()`.

### 2) Контракты входа/выхода (no-fallback)
- ✅ `frame_indices` строго из `metadata["micro_emotion"]["frame_indices"]`
- ✅ `times_s = union_timestamps_sec[frame_indices]`
- ✅ отсутствие обязательных входов/артефактов ⇒ `RuntimeError`

### 3) NPZ source-of-truth + фиксированное имя
- ✅ `result_store/.../micro_emotion/micro_emotion.npz`
- ✅ `schema_version="micro_emotion_npz_v1"` (зарегистрирован в `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`)

### 4) Empty semantics
- ✅ если лиц нет вообще ⇒ `status="empty"`, `empty_reason="no_faces_in_video"`

### 5) Прогресс + profiling
- ✅ `state_events.jsonl` прогресс по кадрам
- ✅ `summary.stage_timings_ms` содержит время стадий

## ⚠️ Требует evidence / измерений

### Производительность (measured resource_costs)
- ⚠️ Нужен файл `docs/models_docs/resource_costs/micro_emotion_costs_v1.json`:
  - unit: `frame`
  - latency per frame (OpenFace docker), CPU/GPU RAM, spikes

## Ссылки

- `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- `docs/contracts/SEGMENTER_CONTRACT.md`
- `Models/docs/contracts/ENCODER_CONTRACT.md`
- `VisualProcessor/modules/micro_emotion/README.md`


