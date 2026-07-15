# Критерии приёмки: high_level_semantic

Согласованы: 2026-07-16 (брифинг → авто-штамп 100% PASS).

## Универсальные хард-гейты (U1–U6)

| Гейт | Критерий |
|---|---|
| U1 | Валидатор `validate_high_level_semantic.py --struct --ranges` → rc=0 |
| U2 | `frame_indices` строго возрастает; `times_s` неубывает |
| U3 | Нет неожиданных NaN/Inf: audio (loudness/tempo) — NaN by design без upstream; emo_* — NaN by design без emotion_face или при нет лиц |
| U4 | expected-empty: при emotion_face.npz отсутствует или status=empty → hls status=ok, emo_* = NaN (не error) |
| U5 | Golden-детерминизм: 2 прогона на одном входе → max\|Δ\|=0.0 |
| U6 | Видео 3 разных длин (N=43/65/119) → schema+struct+ranges OK, rc=0 |

## Компонентные критерии

| Критерий | Порог |
|---|---|
| C1 | `clip_sim_prev` и `clip_novelty_prev` present_ratio ≥ 0.97 (1-й кадр NaN by design — нет пред. кадра) |
| C2 | `scene_embeddings` L2-норма строк = 1.0 ± 0.01 (L2-нормализованы после mean-pooling) |
| C3 | `event_type_id` содержит только допустимые значения ∈ {1, 200, 210} |
| C4 | NaN by design (задекларированные исключения — НЕ баги): `loudness_dbfs`/`tempo_bpm` при отсутствии audio-deps; `emo_valence/arousal/intensity` при отсутствии emotion_face или нет лиц |

## Примечания
- emotion_face сделана **опциональной dep** (graceful fallback): если npz отсутствует или status=empty → emo_* = NaN, status=ok. Флаг `ui.upstream.emotion_face_present` фиксирует факт.
- audio-deps (loudness/tempo/clap) — мягкие, NaN по умолчанию (require_audio_* флаги не включаются в VisualProcessor standalone).
