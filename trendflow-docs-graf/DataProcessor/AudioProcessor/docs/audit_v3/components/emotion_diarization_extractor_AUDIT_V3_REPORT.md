## emotion_diarization_extractor — AUDIT v3 report

### TL;DR

Компонент переведён на Audit v3 контракт: **per-extractor schema** (`emotion_diarization_extractor_npz_v1`), **строгий offline/no-network** через ModelManager (без HuggingFace runtime downloads), **строгая сегментация от Segmenter** (`families.emotion`, без full-audio fallback), **strict alignment** для per-segment outputs через `segment_mask` + NaN/-1 policy, и **offline HTML renderer** (без CDN).

Тесты по просьбе пользователя не запускались.

---

### 1) Ownership / Scope

- **Component**: `DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor`
- **Audit focus**: логика извлечения, sampling/контракты, связи и семантика empty/error (не performance).

---

### 2) Key decisions (по ответам)

1. **Per-extractor schema rollout**: да → `emotion_diarization_extractor_npz_v1` (machine + human schema).
2. **No-network**: да → исправлен `dp_models/providers/speechbrain.py`, чтобы грузить SpeechBrain модель **только из local_artifacts**, не снимая offline env.
3. **Isolated venv/subprocess**: удалено из orchestration (runner) → direct-mode only.
4. **Sampling ownership**: Segmenter-only → `process_full_audio` отключён, только `run_segments()` по `families.emotion`.
5. **Model-facing output**: минимальный полезный набор:
   - табличные агрегаты (entropy/dominant/transitions/stability/diversity)
   - per-segment sequences `emotion_id` + `emotion_confidence` (strict-aligned).
6. **Strict alignment**: добавлены `segment_mask` + NaN/-1 policy.
7. **<5s audio**: valid empty → `status="empty"`, `empty_reason="audio_too_short"`.
8. **Логика lengths**: исправлено вычисление `wav_lens` — теперь берём реальные длины сегментов, а не эвристику “non-zero samples”.
9. **device/meta hygiene**: `device_used` теперь фактический, `model_name` не пишется в feature vector.
10. **Renderer**: offline-only (без Plotly CDN).
11. **Optional keys**: при выключенных фичах ключи **опускаются**, а не сохраняются как пустые массивы.

---

### 3) Inputs / Sampling

- **Required**: Segmenter `audio/segments.json` family `emotion` (`families.emotion.segments[]`).
- **No-fallback**: если family отсутствует/пустая при включённом extractor → `error`.
- Full-audio режим (`run()` / `process_full_audio`) **disabled** для audited контракта.

---

### 4) Outputs / Contract

#### 4.1 Model-facing (frozen subset)

- `feature_names`, `feature_values` (минимальный фиксированный набор скаляров):
  - `segments_count`
  - `emotion_entropy`
  - `dominant_emotion_id`
  - `dominant_emotion_prob`
  - `emotion_transitions_count`
  - `emotion_stability_score`
  - `emotion_diversity_score`
- Sequences (strict-aligned):
  - `emotion_id: int32[N]` (masked → `-1`)
  - `emotion_confidence: float32[N]` (masked → `NaN`)

#### 4.2 Analytics

- time-axis + mask:
  - `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`
- `emotion_labels`
- optional (feature-gated): `emotion_probs[N,C]`, `emotion_mean_probs[C]`, distribution/quality dicts.

#### 4.3 Empty vs Error semantics

- **empty**:
  - `audio_silent` (silence detection)
  - `audio_too_short` (<5s)
  - upstream `audio_present=false` (общее правило AudioProcessor)
- **error**:
  - missing required `families.emotion`
  - missing local weights / model load failed offline
  - invalid numeric outputs

---

### 5) ModelManager / Reproducibility (no-network)

Критический фикс: ранее SpeechBrain provider **временно отключал offline env** и загружал модель по HF id. Теперь:

- используется **только** `spec.local_artifacts` (dir) через `models_root`
- offline env **не снимается**
- если зависимости (например WavLM) отсутствуют в оффлайн кэше → корректный fail-fast (`weights_missing`/`model_load_failed`).

---

### 6) Renderer (offline)

- Убрана зависимость от Plotly CDN.
- HTML рисует timeline на `<canvas>`; работает полностью оффлайн.

---

### 7) Files changed / added

#### Added

- `DataProcessor/AudioProcessor/schemas/emotion_diarization_extractor_npz_v1.json`
- `DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor/SCHEMA.md`
- `DataProcessor/AudioProcessor/docs/audit_v3/components/emotion_diarization_extractor_AUDIT_V3_REPORT.md`

#### Updated

- `DataProcessor/dp_models/providers/speechbrain.py` — local-only загрузка, без снятия offline env
- `DataProcessor/AudioProcessor/src/core/extractor_runner.py` — удалён subprocess/venv path, direct-mode
- `DataProcessor/AudioProcessor/run_cli.py` — schema lookup: `emotion_diarization_extractor_npz_v1`
- `DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor/main.py` — strict alignment + mask, empty semantics, lengths fix, disable run()
- `DataProcessor/AudioProcessor/src/core/npz_savers/emotion_diarization.py` — audited NPZ contract + optional keys omission
- `DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor/render.py` — offline HTML (no CDN)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` — audited defaults (ids/conf enabled), disable flags
- `DataProcessor/AudioProcessor/src/core/main_processor.py` — audited defaults + manifest flattening for emotion diarization
- `DataProcessor/AudioProcessor/docs/MAIN_INDEX.md`
- `DataProcessor/AudioProcessor/README.md`
- `DataProcessor/docs/COMPONENTS_DESC.md`

---

### 8) Open items / follow-ups

- Убедиться, что оффлайн кэш HF (если нужен для WavLM) корректно заполняется при сборке/деплое `dp_models` (это вне логики extractor).
- При желании можно вернуть cross-video batching в `extract_batch_segments()` при сохранении strict mask контракта (сейчас correctness-first).
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/emotion_diarization_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
