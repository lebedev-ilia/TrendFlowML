# ⏳ Baseline Audit — `tempo_extractor`

Компонент: `DataProcessor/AudioProcessor/src/extractors/tempo_extractor/`  
Тип: Audio extractor (Tier‑0 baseline)  
Статус: **✅ CLOSED (baseline)** (2026‑01‑16)  

---

## Резюме

`tempo_extractor` оценивает темп (BPM) и устойчивость оценки на базе `librosa`:
- global BPM (median/mean/std)
- windowed BPM по Segmenter‑окнам (family=`tempo`) для устойчивости и анализа изменений.

В baseline этот компонент **не зависит** от `core_object_detections` и может аудититься/запускаться отдельно (только аудио).

---

## Соответствие `BASELINE_COMPONENT_AUDIT_CRITERIA.md` (черновик)

### 1) Интерфейс / storage

- CLI: `DataProcessor/AudioProcessor/run_cli.py` сохраняет артефакты в per‑run storage:
  - `result_store/<platform>/<video>/<run_id>/tempo_extractor/tempo_extractor_features.npz`
- Сохранение атомарное (tmp → `os.replace`)
- Есть runtime‑валидация: `validate_npz()` (meta‑контракт)

### 2) Контракты входа/выхода

- Вход:
  - `audio/audio.wav` от Segmenter
  - `audio/segments.json` (contract `audio_segments_v1`), family=`tempo`
- No‑fallback:
  - нет аудио → `status="empty"`, `empty_reason="audio_missing_or_extract_failed"` (допустимая пустота)
  - segments family пустой/невалидный → error (fail-fast)

### 3) NPZ schema

- `schema_version`: `audio_npz_v1`
- ключи (основные, tempo):
  - `feature_names` (object[str])
  - `feature_values` (float32)
  - `tempo_estimates` (float32[T])
  - `windowed_times_sec` (float32[W])
  - `windowed_bpm` (float32[W])
  - `warnings` (object[str])
  - `meta` (object[dict]) с обязательными run identity keys

---

## Performance (TODO)

Источник правды:
- `docs/models_docs/resource_costs/tempo_extractor_costs_v1.json`

Evidence:
- micro‑bench: `scripts/baseline/run_tempo_extractor_micro.py` (family=`tempo`)

---

## Quality validation (TODO)

Human‑friendly demo:
- `scripts/baseline/demo_tempo_extractor_quality.py`

Evidence (реальный прогон `NSumhkOwSg`):
- NPZ: `storage/reports/out/tempo_extractor_real/result_store/youtube/NSumhkOwSg/tempo_real/tempo_extractor/tempo_extractor_features.npz`
- HTML: `storage/reports/out/tempo_extractor_real/demo_tempo_extractor_quality_20260116-041812-653758.html`


