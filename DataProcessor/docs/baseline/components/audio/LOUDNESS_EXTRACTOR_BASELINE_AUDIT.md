# ⏳ Baseline Audit — `loudness_extractor`

Компонент: `DataProcessor/AudioProcessor/src/extractors/loudness_extractor/`  
Тип: Audio extractor (Tier‑0 baseline)  
Статус: **✅ CLOSED (baseline)** (2026‑01‑16)  

---

## Резюме

`loudness_extractor` считает базовые метрики громкости/динамики:
- RMS, peak, dBFS
- LUFS (опционально, если доступен `pyloudnorm`)
- short‑term RMS статистики (mean/std/median/p10/p90)
- последовательности по Segmenter‑окнам (segment RMS/peak/dbfs/lufs)

Окна берутся из Segmenter `audio/segments.json`, family=`primary` (окна вокруг time‑anchors).

---

## Соответствие `BASELINE_COMPONENT_AUDIT_CRITERIA.md`

### 1) Интерфейс / storage

- CLI: `DataProcessor/AudioProcessor/run_cli.py`
- Per‑run storage:
  - `result_store/<platform>/<video>/<run_id>/loudness_extractor/loudness_extractor_features.npz`
- Сохранение атомарное (tmp → `os.replace`)
- Есть runtime‑валидация: `validate_npz()` (meta‑контракт)

### 2) Контракты входа/выхода

- Вход:
  - `audio/audio.wav` от Segmenter
  - `audio/segments.json` (contract `audio_segments_v1`), family=`primary`
- No‑fallback:
  - family пустой/невалидный → error (fail-fast)

### 3) NPZ schema (`audio_npz_v1`)

Основные ключи:
- `feature_names` (object[str])
- `feature_values` (float32)
- `segment_centers_sec` (float32[N])
- `segment_rms/segment_peak/segment_dbfs/segment_lufs` (float32[N])
- `lufs_present` (bool)
- `meta` (object[dict]) с обязательными run identity keys

---

## Performance (measured)

Источник правды:
- `docs/models_docs/resource_costs/loudness_extractor_costs_v1.json`

Evidence:
- micro‑bench: `scripts/baseline/run_loudness_extractor_micro.py` (family=`primary`)

---

## Quality validation (human-friendly)

Human‑friendly demo:
- `scripts/baseline/demo_loudness_extractor_quality.py`

Что проверяем:
- формы массивов, монотонность `segment_centers_sec`
- finite значения RMS/DBFS
- наличие/отсутствие LUFS через `lufs_present`
- графики `segment_dbfs/segment_rms` по времени

Evidence (реальный прогон `NSumhkOwSg`):
- NPZ: `storage/reports/out/audio_tier0_real/result_store/youtube/NSumhkOwSg/audio_tier0_real/loudness_extractor/loudness_extractor_features.npz`
- HTML: `storage/reports/out/audio_tier0_real/demo_loudness_extractor_quality_20260116-051623-495987.html`

---

## Ссылки

- Контракт Segmenter: `docs/contracts/SEGMENTER_CONTRACT.md`
- Критерии аудита: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- README компонента: `AudioProcessor/src/extractors/loudness_extractor/README.md`


