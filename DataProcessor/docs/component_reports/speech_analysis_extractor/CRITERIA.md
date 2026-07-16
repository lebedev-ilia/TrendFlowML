# Критерии приёмки: speech_analysis_extractor

Одобрено: 2026-07-17

## Универсальные хард-гейты (U1–U6)

| № | Гейт | Ожидание |
|---|------|---------|
| U1 | validate_speech_analysis.py rc=0 | rc=0 на всех NPZ |
| U2 | Ось времени | asr_lang_id_by_segment[N] совпадает с segments_count из asr_payload |
| U3 | Не константа на корпусе | duration_sec, asr_token_density различимы; sample_rate=16000 — const by design |
| U4 | Expected-empty работает | audio_too_short → empty; audio_silent → empty; audio_missing → empty (оркестратор) |
| U5 | Golden-детерминизм | max|Δ|=0.0 (pure numpy, нет стохастики) |
| U6 | Разные длины видео | short/medium/long: success=True без крашей |

## Компонентные критерии (C1–C5)

| № | Критерий | Порог |
|---|----------|-------|
| C1 | status=ok → feature_values finite | NaN=0, Inf=0 |
| C2 | sample_rate по дизайну константа | 16000 Гц, не является U3-дефектом |
| C3 | asr_lang_distribution = raw counts | По контракту с asr_extractor (int counts, не доли); зависимость подтверждена кодом asr_extractor:1011 |
| C4 | Golden max\|Δ\|=0.0 | Pure numpy aggregation без стохастики → побайтово идентично |
| C5 | empty_reason различает причины | audio_too_short / audio_silent / audio_missing_or_extract_failed — три разных сценария |

## Примечания

- Компонент CPU-only, нет нейросетей. Всё валидируется локально.
- При enable_asr_metrics=False, enable_diarization_metrics=False, enable_pitch_metrics=False: F=2 (duration_sec, sample_rate) — не дефект.
- При enable_pitch_metrics=True но pitch_payload=None: pitch-поля в NPZ НЕ пишутся — фикс audit_v4 уже применён.
- Тяжёлые модели (Whisper, pyannote, pitch) — в зависимых компонентах, не в этом.
