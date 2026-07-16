# Критерии приёмки: asr_extractor

**Согласовано:** 2026-07-16  
**Версия компонента:** ASRExtractor v2.3.2 (Whisper small inprocess, schema asr_extractor_npz_v2)

---

## Универсальные гейты (U1–U6)

| Гейт | Критерий | Применимость |
|------|----------|-------------|
| U1 | validate_asr.py rc=0 (без --struct, не поддерживается) | ✅ |
| U2 | segment_start_sec≤segment_end_sec, монотонны | ✅ |
| U3 | status=ok → feature_values ≥50% finite, lang_code осмысленный | ✅ |
| U4 | status=empty → feature_values NaN×F, seg_n=0, empty_reason заполнен | ✅ |
| U5 | Golden: 9 ok-runs одного видео → max\|Δfv\|=0.0, token_counts идентичны | ✅ |
| U6 | audio 3s–30s без падений (реальные видео) | ✅ |

## Специфические критерии (C1–C3)

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | feature_names = 20 штук (frozen subset) + fv finite ≥50% при ok | 20 фич |
| C2 | lang_distribution непустой, lang_code ∈ {ru,en,...} при ok | ≥1 lang |
| C3 | asr_quality__avg_logprob_mean > -2.0, no_speech_prob_mean < 0.8 | не строго (зависит от видео) |

## Исключения

- **status=empty** (2 NPZ): feature_values NaN×8, seg_n=0 by design (audio_missing)
- **validate_asr.py: нет --struct** (только schema + --qa) — менее строгий чем clap/text_scoring
- **no_speech_prob max=0.621** на одном видео — допустимо (тихое видео)
- **token_variance=0.0** часто (1 сегмент) — by design (всё аудио в 1 окно)
