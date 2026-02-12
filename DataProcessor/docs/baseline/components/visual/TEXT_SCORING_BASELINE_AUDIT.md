# ✅ Baseline Audit — `text_scoring`

Компонент: `DataProcessor/VisualProcessor/modules/text_scoring/`  
Тип: Visual module (Tier‑0 baseline)  
Статус: **✅ CLOSED (baseline)** (2026‑01‑28)  

---

## Резюме

`text_scoring` — consumer OCR‑артефакта и вычисляет:
- **CTA** (presence + timestamps + strength)
- **text continuity** (сколько/как быстро меняется/когда появляется)
- **multimodal alignment** в baseline режиме **C (face‑only)**: face signal опционален (`use_face_data=true`), motion/audio выключены по умолчанию.

OCR — optional: при отсутствии/пустоте OCR модуль пишет валидный NPZ с `meta.status="empty"`.

---

## Соответствие `BASELINE_COMPONENT_AUDIT_CRITERIA.md`

### 1) BaseModule интерфейс
- ✅ Inherits from `BaseModule`
- ✅ Implements `process(frame_manager, frame_indices, config)`
- ✅ Fixed artifact name via `ARTIFACT_FILENAME="text_scoring.npz"`

### 2) I/O contracts
- ✅ `frame_indices` из Segmenter metadata (no self‑sampling)
- ✅ `times_s` = `union_timestamps_sec[frame_indices]` (no‑fallback, без fps‑fallback)
- ✅ Schema registry: `text_scoring_npz_v1` добавлен в `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`

### 3) Empty/Error policy
- ✅ OCR missing/empty/outside_sampling → `meta.status="empty"` с `meta.empty_reason`
- ✅ `use_face_data=true` требует `core_face_landmarks` (иначе error)

### 4) Privacy
- ✅ Raw OCR text **не сохраняется** по умолчанию (`retain_raw_ocr_text=false`)
- ✅ При редактировании сохраняются только `text_len` + `text_hash_sha256`

### 5) Observability & UI
- ✅ Progress events (`state_events.jsonl`) + stage timings
- ✅ `meta.ui_payload` (privacy‑safe) для визуализации:
  - `text_presence`, `text_count_per_frame`
  - CTA markers (first/mean/last)

---

## Артефакт (NPZ)

Путь: `.../text_scoring/text_scoring.npz`  
Schema: `text_scoring_npz_v1`

Ключи (основное):
- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `text_present () bool`
- `text_presence (N,) bool`
- `text_count_per_frame (N,) int32`
- `features` (dict, object-array)
- `ocr_raw` / `ocr_unique_elements` (privacy‑aware)
- `meta` (dict, object-array)

Human demo:
- `VisualProcessor/modules/text_scoring/quality_report/demo_text_scoring_quality.py`

---

## Известные ограничения / roadmap

- Motion/audio alignment оставлен как future improvement (можно добавить `core_optical_flow` и AudioProcessor signals, но только через `times_s`/time‑windows).
- `text_emphasis_peak_*`, `ocr_language_entropy`, `text_movement_speed` отключены по умолчанию как “noisy”; доступны как opt‑in flags.


