# Критерии приёмки: speaker_diarization_extractor

**Согласовано:** 2026-07-16

---

## Универсальные гейты (U1–U6)

| # | Критерий | Порог | Тип |
|---|----------|-------|-----|
| U1 | `validate_speaker_diarization.py` rc=0 | rc==0 | HARD |
| U2 | turns_start_end монотонны (start[i] < end[i], end[i] <= start[i+1]) | 100% | HARD |
| U3 | finite/health: nan_rate=0, inf_rate=0 в ok-сегментах | 0 | HARD |
| U4 | expected-empty: sc=0 → K=0, S=0, feature_values=NaN (by design) | 100% соблюдение | HARD |
| U5 | golden: sc=0 — идентичный вывод при повторе; sc>0 — структурно стабилен | sc=0 детерминирован | HARD |
| U6 | разные длины видео (short/med/long) → все rc=0 | 100% | HARD |

## Специфичные критерии (S1–S4)

| # | Критерий | Порог | Тип |
|---|----------|-------|-----|
| S1 | speaker_count >= 0, целое | >=0 | HARD |
| S2 | num_turns (K) разумен: K <= speaker_count * 200 | не взрывается | HARD |
| S3 | segment_speaker_embeddings shape (S, 192) при S>0 | dim=192 | HARD |
| S4 | feature_values[0]=speaker_count верно | совпадает с K/S | HARD |

## Примечания

- **NaN by design**: при sc=0 (audio_silent/audio_missing) feature_values=NaN — это норма; при status=ok, sc=0 — feature_values валидны (все значения заполнены)
- **Стохастика**: pyannote нейросеть; при sc>0 структура стабильна, точные границы могут плавать ±0.1с
- **F=10 реальные имена**: speaker_count, duration_sec, sample_rate, rms, peak, speaker_balance_score, dominant_speaker_id, speaker_turns_count, speaker_turns_density, speaker_transitions_count
