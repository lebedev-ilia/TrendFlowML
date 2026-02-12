# Аудит соответствия high_level_semantic требованиям baseline

**Дата проверки**: 2026-01-27  
**Компонент**: `high_level_semantic` (Visual module)  
**Расположение**: `VisualProcessor/modules/high_level_semantic/`

## Резюме

Компонент `high_level_semantic` приведён к baseline-архитектуре и контрактам:
- наследуется от `BaseModule`,
- строго использует `frame_indices` от Segmenter (no-fallback),
- использует `union_timestamps_sec` как source-of-truth time-axis,
- не грузит CLIP/LLM веса внутри модуля (читает `core_clip`),
- пишет **один** NPZ-артефакт фиксированного имени + валидируется через `validate_npz()`,
- репортит промежуточный прогресс через `state_events.jsonl`,
- имеет feature gating через `feature_groups` и require-флаги для upstream.

Оставшиеся пункты baseline (в основном “performance evidence”) требуют измерений и фиксации в `docs/models_docs/resource_costs/`.

---

## ✅ Соответствие требованиям

### 1) Наследование и интерфейсы
- **Статус**: ✅ **СООТВЕТСТВУЕТ**
- `HighLevelSemanticModule(BaseModule)` реализует `process(frame_manager, frame_indices, config)` и `required_dependencies()`.

### 2) Контракты входа/выхода (no-fallback)
- **Статус**: ✅ **СООТВЕТСТВУЕТ**
- `frame_indices` берётся только из `metadata["high_level_semantic"]["frame_indices"]` (Segmenter-owned).
- `union_timestamps_sec` обязателен; `times_s = union_timestamps_sec[frame_indices]`.
- Отсутствие required upstream артефактов → `RuntimeError`.

### 3) NPZ как source-of-truth + фиксированное имя
- **Статус**: ✅ **СООТВЕТСТВУЕТ**
- Артефакт: `result_store/.../high_level_semantic/high_level_semantic.npz`
- `schema_version = "high_level_semantic_npz_v1"` (зарегистрирован в `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`)
- `meta` содержит baseline identity keys + versions + models_used/model_signature (best-effort from BaseModule).

### 4) Валидация артефакта
- **Статус**: ✅ **СООТВЕТСТВУЕТ**
- NPZ валидируется через `artifact_validator.validate_npz()` внутри `BaseModule.save_results()` (fail-fast).

### 5) Features contract / gating
- **Статус**: ✅ **СООТВЕТСТВУЕТ**
- `feature_groups` управляет набором выходных сигналов.
- В `ui.feature_groups` фиксируется набор включенных групп.

### 6) Промежуточный прогресс
- **Статус**: ✅ **СООТВЕТСТВУЕТ**
- Пишет события в `state/<platform>/<video>/<run>/state_events.jsonl`, backend пушит `component.progress`.
- Unit прогресса: `frame`.

### 7) Human-friendly визуализация
- **Статус**: ✅ **СООТВЕТСТВУЕТ**
- Скрипт: `VisualProcessor/modules/high_level_semantic/quality_report/demo_high_level_semantic_quality.py`

---

## ⚠️ Требует доработки / evidence

### 1) Измерения производительности (baseline requirement)
- **Статус**: ⚠️ **НЕТ EVIDENCE**
- Нужно добавить `docs/models_docs/resource_costs/high_level_semantic_costs_v1.json`:
  - unit: `frame`
  - latency/RAM peak (ориентир: лёгкий модуль, без heavy inference)

---

## Ссылки

- `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- `docs/contracts/SEGMENTER_CONTRACT.md`
- `Models/docs/contracts/ENCODER_CONTRACT.md`
- `VisualProcessor/modules/high_level_semantic/README.md`


