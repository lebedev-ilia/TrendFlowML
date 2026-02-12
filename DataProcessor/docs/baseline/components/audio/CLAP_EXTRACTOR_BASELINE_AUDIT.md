# ⏳ Baseline Audit — `clap_extractor`

Компонент: `DataProcessor/AudioProcessor/src/extractors/clap_extractor/`  
Тип: Audio extractor (Tier‑0 baseline)  
Статус: **✅ CLOSED (baseline)** (2026‑01‑16)  

---

## Резюме

`clap_extractor` считает **семантические аудио‑эмбеддинги CLAP** по Segmenter‑окнам и сохраняет:
- `embedding_sequence` (по окнам),
- `embedding` (агрегат по видео).

Окна берутся из Segmenter `audio/segments.json`, family=`clap`, построенной по **универсальной нелинейной кривой** (параметры `k/min/max/...` лежат в самом `segments.json`).

---

## Соответствие `BASELINE_COMPONENT_AUDIT_CRITERIA.md`

### 1) Интерфейс / storage

- CLI: `DataProcessor/AudioProcessor/run_cli.py`
- Per‑run storage:
  - `result_store/<platform>/<video>/<run_id>/clap_extractor/clap_extractor_features.npz`
- Сохранение атомарное (tmp → `os.replace`)
- Есть runtime‑валидация: `validate_npz()` (meta‑контракт)

### 2) Контракты входа/выхода

- Вход:
  - `audio/audio.wav` от Segmenter
  - `audio/segments.json` (contract `audio_segments_v1`), family=`clap`
- No‑fallback:
  - family пустой/невалидный → error (fail-fast)

### 3) NPZ schema (`audio_npz_v1`)

Основные ключи:
- `feature_names` (object[str])
- `feature_values` (float32)
- `embedding` (float32[D])
- `embedding_sequence` (float32[N, D])
- `segment_centers_sec` (float32[N])
- `meta` (object[dict]) с обязательными run identity keys

---

## Performance (measured)

Источник правды:
- `docs/models_docs/resource_costs/clap_extractor_costs_v1.json`

Evidence:
- micro‑bench: `scripts/baseline/run_clap_extractor_micro.py` (family=`clap`)

---

## Quality validation (human-friendly)

Human‑friendly demo:
- `scripts/baseline/demo_clap_extractor_quality.py`

Что проверяем:
- формы массивов (`embedding_sequence`, `segment_centers_sec`)
- монотонность `segment_centers_sec`
- finite значения
- динамика norms/cosine similarity по времени

Evidence (реальный прогон `NSumhkOwSg`):
- NPZ: `storage/reports/out/audio_tier0_real/result_store/youtube/NSumhkOwSg/audio_tier0_real/clap_extractor/clap_extractor_features.npz`
- HTML: `storage/reports/out/audio_tier0_real/demo_clap_extractor_quality_20260116-051623-288785.html`

---

## Ссылки

- Контракт Segmenter: `docs/contracts/SEGMENTER_CONTRACT.md`
- Критерии аудита: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- README компонента: `AudioProcessor/src/extractors/clap_extractor/README.md`


