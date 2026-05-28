## hpss_extractor — AUDIT v3 report

### TL;DR

Компонент переведён на Audit v3 контракт: **per-extractor schema** (`hpss_extractor_npz_v1`), **строгая сегментация** от Segmenter (`families.hpss`), **strict alignment** для per-segment outputs через `segment_mask` + NaN policy, **waveform paths в meta.extra** (без массивов в NPZ), **offline HTML renderer** (без Chart.js CDN), **enable_energy_metrics=True** по умолчанию.

---

### 1) Ownership / Scope

- **Component**: `DataProcessor/AudioProcessor/src/extractors/hpss_extractor`
- **Audit focus**: логика извлечения, sampling/контракты, связи и семантика empty/error.

---

### 2) Key decisions (по ответам)

1. **Per-extractor schema**: да → `hpss_extractor_npz_v1` (machine schema + SCHEMA.md).
2. **Family**: `hpss` (без изменений).
3. **run() disabled**: да → в audited mode `run()` возвращает error; только `run_segments()`.
4. **enable_energy_metrics**: default `True`.
5. **Model-facing output**:
   - `hpss_harmonic_share`, `hpss_percussive_share`, `hpss_balance_score`, `hpss_dominance`
   - per-segment: `hpss_harmonic_share_by_segment`, `hpss_percussive_share_by_segment` (NaN для failed).
6. **Strict alignment**: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`, `hpss_harmonic_share_by_segment`, `hpss_percussive_share_by_segment`; NaN для failed segments.
7. **External .npy**: NPZ — source-of-truth; waveform paths в `meta.extra` (`hpss_harmonic_npy_path`, `hpss_percussive_npy_path`); массивы waveform **не** встраиваются в NPZ.
8. **Renderer**: offline HTML (vanilla canvas, без Chart.js CDN).
9. **Optional keys**: при выключенных фичах ключи опускаются.
10. **Segment arrays**: `segment_start_sec`, `segment_end_sec`, `segment_center_sec` (вместо legacy `segment_centers_sec`/`segment_durations_sec`).

---

### 3) Inputs / Sampling

- **Required**: Segmenter `audio/segments.json` family `hpss` (`families.hpss.segments[]`).
- **No-fallback**: если family отсутствует/пустая при включённом extractor → `error`.
- `run()` **disabled** для audited контракта.

---

### 4) Outputs / Contract

#### 4.1 Model-facing (frozen subset)

- Tabular: `hpss_harmonic_share`, `hpss_percussive_share`, `hpss_balance_score`, `hpss_dominance` (string).
- Per-segment sequences (strict-aligned):
  - `hpss_harmonic_share_by_segment: float32[N]` (masked → NaN)
  - `hpss_percussive_share_by_segment: float32[N]` (masked → NaN)

#### 4.2 Analytics

- Time axis + mask: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`.
- Optional (feature-gated): `hpss_harmonic_share_series`, `hpss_percussive_share_series`, energy/stability/spectral features.

#### 4.3 Waveforms

- Пути в `meta.extra`: `hpss_harmonic_npy_path`, `hpss_percussive_npy_path` (относительные пути к .npy).
- Массивы waveform **не** сохраняются в NPZ.

#### 4.4 Empty vs Error semantics

- **empty**: upstream `audio_present=false`.
- **error**: missing/empty `families.hpss.segments` при `audio_present=true`, HPSS/STFT failure.

---

### 5) Implementation summary

| Item | Status |
|------|--------|
| Schema `hpss_extractor_npz_v1` | ✅ |
| SCHEMA.md | ✅ |
| run_cli schema_version mapping | ✅ |
| run() disabled | ✅ |
| enable_energy_metrics=True default | ✅ |
| run_segments strict alignment | ✅ |
| segment_mask + NaN for failed | ✅ |
| NPZ: waveform paths in meta.extra only | ✅ |
| NPZ: segment arrays (start/end/center/mask) | ✅ |
| Renderer: offline HTML (no CDN) | ✅ |
| CLI: --hpss-disable-energy-metrics | ✅ |

---

### 6) Version

- **hpss_extractor**: `2.1.0`
