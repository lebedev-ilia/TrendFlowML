# `speaker_turn_embeddings_aggregator` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/speaker_turn_embeddings_aggregator/main.py`

## Резюме

- A-policy refactor:
  - strict model loading через `dp_models` + `weights_digest` (no-network)
  - privacy: нет raw speaker names/texts в `result`, нет raw-derived hashes в filenames
  - deterministic artifacts: per-run fixed имена `speaker_spkNNN_{mean,max}.npy`
  - valid empty semantics: всегда `features_flat`, при отсутствии входа → `tp_spkemb_present=0`
  - feature-gating/limits: `compute_mean/max`, `write_artifacts`, `max_speakers`, `max_turns_per_speaker`, `min/max_chars_per_turn`
  - in-memory registry: canonical `doc.tp_artifacts["speakers"]["embeddings"]` + legacy alias `doc.tp_artifacts["speaker_embeddings"]`

## Upstream contract (ожидаемо)

Предпочтительный вход (prod‑режим): `doc.speaker_diarization["speaker_segments"]` + `doc.asr["segments"]`.
Legacy вход поддержан: `doc.speakers` (dict с `{name, description}`), но не считается рекомендуемым для production из-за PII риска.


