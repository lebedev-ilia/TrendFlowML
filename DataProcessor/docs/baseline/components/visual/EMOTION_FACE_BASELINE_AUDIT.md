# EMOTION_FACE — Baseline Audit

## Резюме

Компонент `emotion_face` приведён к baseline-контрактам: NPZ-only, строгая time-axis из `union_timestamps_sec` (no-fallback), отсутствие fallback по зависимостям, granular progress и регистрация `schema_version`.

## Соответствие требованиям (чек-лист)

### Архитектура

- ✅ Visual module (`BaseModule`): `DataProcessor/VisualProcessor/modules/emotion_face/core/video_processor.py`
- ✅ Fixed artifact: `emotion_face/emotion_face.npz`
- ✅ `schema_version`: `emotion_face_npz_v1` (зарегистрирован в `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`)
- ✅ `status` / `empty_reason`: empty при `no_faces_in_video`
- ✅ No-fallback:
  - отсутствует `core_face_landmarks/landmarks.npz` ⇒ error
  - отсутствует `union_timestamps_sec` ⇒ error
- ✅ UI payload: хранится в `meta.ui_payload` (без отдельного JSON)

### Контракты времени/выборки

- ✅ `times_s` строится строго как `union_timestamps_sec[frame_indices]`
- ✅ Sampling policy (decision):
  - сначала берём кадры, где `core_face_landmarks.face_present` true
  - затем применяем `face_frame_stride` и `max_frames`

### Models policy

- ✅ EmoNet подключён через `dp_models.ModelManager`:
  - spec: `emonet_8_inprocess`
  - веса: `DP_MODELS_ROOT/bundled_models/visual/emonet/emonet_8.pth`

### Progress / stage timings

- ✅ Progress: granular updates в `state_events.jsonl` (>=10 апдейтов) на стадии `process_frames`
- ✅ `summary.stage_timings_ms` записывается в NPZ

## Проверка качества выхода

- ✅ Human-friendly отчёт: `DataProcessor/VisualProcessor/modules/emotion_face/quality_report/demo_emotion_face_quality.py`

## Производительность

Замеры resource costs пока не добавлялись (помечено как follow-up).

## Дополнительные замечания / follow-ups

- Advanced/noisy features (`microexpressions`, `emotional_individuality`, `face_asymmetry`) остаются выключенными по умолчанию и должны включаться явно через конфиг.


