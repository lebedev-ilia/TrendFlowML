# FRAMES_COMPOSITION — Baseline audit

## 1. Резюме

- **Компонент**: `frames_composition` (Visual module)
- **Статус**: **ready (baseline‑compliant)**, но включение в Tier‑0 ещё не решено
- **Артефакт**: `frames_composition/frames_composition.npz`
- **schema_version**: `frames_composition_npz_v1`

## 2. Соответствие требованиям (чек-лист)

### 2.1 Архитектура (Visual modules)
- [x] Наследуется от `BaseModule` (`VisualProcessor/modules/base_module.py`)
- [x] Реализован `process(frame_manager, frame_indices, config)`
- [x] Реализован `required_dependencies()`
- [x] **No-fallback** по `frame_indices` (берутся только из metadata; если нет → error)

### 2.2 Контракты входа/выхода
- [x] `frame_indices` читает из `frames_dir/metadata.json["frames_composition"]["frame_indices"]`
- [x] `times_s` берёт строго из `union_timestamps_sec[frame_indices]`
- [x] NPZ meta содержит обязательные поля (producer/run identity/status/models_used/model_signature)
- [x] Атомарная запись NPZ + runtime validation через `artifact_validator.validate_npz()`
- [x] Fixed filename в per-run storage (уникальность за счёт `run_id` в пути)

### 2.3 Empty semantics
- [x] Если “нет лиц” → `status="empty"`, `empty_reason="no_faces_in_video"`
- [x] Численные значения не подменяются нулями (NaN / корректные флаги)

### 2.4 Запрет JSON артефактов
- [x] В `result_store` не пишет JSON (кроме `manifest.json`, который пишет оркестратор)

### 2.5 Progress / observability
- [x] Пишет progress-events в `state_events.jsonl` (PR‑5), backend пушит `component.progress` через WS

### 2.6 Feature gating
- [x] Управляемые группы фич через `--feature-set` / `--features`

## 3. Производительность компонента

**Unit**: `frame`

Измерений latency/RAM пока нет (нужно добавить в `docs/models_docs/resource_costs/frames_composition_costs_v1.json` после утверждения Tier‑0).

## 4. Проверка качества выхода

- Human-friendly визуализация: `VisualProcessor/modules/frames_composition/quality_report/demo_frames_composition_quality.py`

## 5. Доп. замечания / plan

- Если компонент будет включён в Tier‑0, нужно:
  - добавить бенчмарк + `resource_costs/frames_composition_costs_v1.json`,
  - определить, какие агрегаты нужны training schema (возможно сократить/зафиксировать набор).


