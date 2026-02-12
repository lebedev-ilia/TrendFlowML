# Behavioral — Baseline Audit

## 1. Резюме

Компонент `behavioral` приведён к baseline‑контрактам: BaseModule, per‑run storage, time‑axis из `union_timestamps_sec`, фиксация `schema_version`, UI payload в meta. Нужны измерения производительности и дальнейшая валидация качества.

## 2. Соответствие требованиям (чек‑лист)

### Архитектура
- [x] Наследование от `BaseModule`
- [x] `process(frame_manager, frame_indices, config)` реализован
- [x] `required_dependencies()` → `["core_face_landmarks"]`
- [x] Читает `frame_indices` из metadata (Segmenter)
- [x] `times_s = union_timestamps_sec[frame_indices]`
- [x] NPZ meta включает обязательные поля (run identity, status, dataprocessor_version)
- [x] No‑fallback: при отсутствии `core_face_landmarks` → `RuntimeError`
- [x] Per‑run storage: `result_store/<platform>/<video>/<run>/behavioral/behavioral_features.npz`
- [x] Артефакт валидируется через `artifact_validator.validate_npz()`
- [x] README: Sampling requirements / Models / Parallelization

### Производительность
- [ ] Файл измерений: `docs/models_docs/resource_costs/behavioral_costs_v1.json` (TBD)
- [ ] README: раздел "Performance characteristics" заполнен измеренными значениями

### Качество
- [x] README: раздел "Quality validation & human-friendly inspection"
- [x] Скрипт отчёта: `VisualProcessor/modules/behavioral/quality_report/demo_behavioral_quality.py`
- [ ] Проведён прогон на реальных видео (TBD)

## 3. Производительность компонента

Пока нет измерений. Нужно собрать latency/RAM:
- unit: `frame`
- p50/p95/p99 для CPU
- peak RSS

## 4. Проверка качества выхода компонента

### Скрипт качества
`VisualProcessor/modules/behavioral/quality_report/demo_behavioral_quality.py`

### Что проверять
- корректность `times_s`, монотонность
- отсутствие NaN при `landmarks_present=true`
- адекватные диапазоны (0..1) для нормированных фич

## 5. Дополнительные замечания

UI payload формируется в `meta.ui_payload`, предусмотрен экспорт JSON (`--ui-json-path`).

## 6. Итоговая оценка

Архитектурно — соответствует baseline. Требуются измерения и подтверждение качества на тестовых данных.

## 7. План действий

1) Выполнить прогон на тестовых видео  
2) Зафиксировать ресурсные метрики  
3) Утвердить полезность фичей для ML/аналитики  

## 8. Ссылки

- `VisualProcessor/modules/behavioral/README.md`
- `VisualProcessor/modules/behavioral/FEATURES_DESCRIPTION.md`
- `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`

