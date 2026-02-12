# ✅ Baseline Audit — `detalize_face`

Компонент: `DataProcessor/VisualProcessor/modules/detalize_face/`  
Тип: Visual module (face-derived features; **disabled by default**)  
Статус: **✅ CLOSED (baseline)** (2026‑01‑28)

---

## Резюме

`detalize_face` вычисляет **детальные face-derived признаки** (взгляд/внимание/моргание/качество/речь‑прокси) на основе `core_face_landmarks` и сохраняет **NPZ-only** артефакт, совместимый с baseline контрактами.

Ключевые решения baseline:
- **Sampling owner**: Segmenter. Модуль использует `metadata["detalize_face"]["frame_indices"]` и обрабатывает **строго пересечение** с кадрами, где `core_face_landmarks` обнаружил лица.
- **No JSON artifacts**: UI данные живут в `meta.ui_payload` внутри NPZ (в `result_store` не пишется `detalize_face.json`).
- **Model-facing output**: хранит только **строгие time-series + агрегаты**, без “сырых per-frame dict’ов”.
- **Orchestrator gating**: при `core_face_landmarks.status="empty"` / `no_faces_in_video` VisualProcessor **не запускает** `detalize_face` и пишет статус `empty` в manifest.

Hard deps (no‑fallback):
- `core_face_landmarks` (`landmarks.npz`)

---

## Соответствие `BASELINE_COMPONENT_AUDIT_CRITERIA.md`

### 1) Наследование / интерфейсы
- ✅ `DetalizeFaceModule` наследуется от `BaseModule`
- ✅ реализует `process(frame_manager, frame_indices, config)`
- ✅ `required_dependencies()` → `["core_face_landmarks"]`

### 2) Контракты входа/выхода
- ✅ `frame_indices` берутся только из Segmenter metadata (no self‑sampling)
- ✅ обработка кадров **только с лицами**: `frame_indices ∩ core_face_landmarks.frames_with_face`
- ✅ `times_s` строго из `union_timestamps_sec[frame_indices]` (no‑fallback)
- ✅ valid empty: `status="empty"`, `empty_reason="no_faces_in_video"` (если пересечение пустое / faces отсутствуют)

### 3) Per‑run storage + atomic save + validation
- ✅ Артефакт: `result_store/<platform>/<video>/<run_id>/detalize_face/detalize_face.npz` (фиксированное имя)
- ✅ запись атомарная через `BaseModule.save_results()`
- ✅ runtime validation через `artifact_validator.validate_npz()`

### 4) Observability (progress + timings)
- ✅ progress events в `state_events.jsonl` (stage‑based)
- ✅ `summary.stage_timings_ms`

### 5) UI payload
- ✅ `meta.ui_payload` содержит pointers на NPZ keys для графиков (face_count + primary_* серии)

---

## Артефакт (NPZ)

Путь: `.../detalize_face/detalize_face.npz`

Ключи (baseline v1):
- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `face_count (N,) float32`
- `primary_gaze_at_camera_prob (N,) float32`
- `primary_blink_rate (N,) float32`
- `primary_attention_score (N,) float32`
- `primary_quality_proxy_score (N,) float32`
- `primary_face_sharpness (N,) float32`
- `primary_occlusion_proxy (N,) float32`
- `primary_speech_activity_prob (N,) float32`
- `faces_agg` (dict, object-array) — per-track агрегаты
- `summary` (dict, object-array) — агрегаты и `stage_timings_ms`
- `meta` (dict, object-array) — canonical meta + `ui_payload`

Schema:
- `detalize_face_npz_v1`

---

## Известные ограничения / next steps

- Некоторые “структурные” фичи (например симметрия/уникальность) должны оставаться **gated/off-by-default** из‑за риска bias/privacy и потенциальной шумности.
- Для дальнейшего улучшения качества можно добавить более строгий трекинг (не только IoU) и явную нормализацию на неравномерный sampling (использовать `times_s` вместо fps‑оценки там, где возможно).


