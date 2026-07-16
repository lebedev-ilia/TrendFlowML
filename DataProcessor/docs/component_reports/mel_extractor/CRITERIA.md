# CRITERIA.md — mel_extractor

**Дата согласования:** 2026-07-16  
**Компонент:** mel_extractor v2.1.1, schema mel_extractor_npz_v2  
**Одобрено:** Второй агент (от имени владельца)

---

## Универсальные хард-гейты (U1–U6)

| ID | Критерий | Порог |
|----|----------|-------|
| U1 | validate_mel.py rc=0 (--struct --qa) на всех NPZ | rc=0 |
| U2 | Ось времени: segment_start/end/center_sec + segment_mask присутствуют, N-согласованы | miss=0, mismatch=0 |
| U3 | feature_values конечны (nan=0, inf=0) на ok NPZ; config NaN в empty — явное исключение (audio_missing by design) | nan=0, inf=0 на status=ok |
| U4 | Expected-empty: ≥1 NPZ с status=empty и empty_reason=audio_missing_or_extract_failed | ≥1 |
| U5 | Golden: max\|Δ\| feature_values между runs одного видео ≈ 0 (torchaudio детерминирован) | ≤ 1e-6 |
| U6 | Различимость: mel_energy CV > 30% или mel_centroid_mean CV > 10% между разными видео | CV > 10% |

## Специфичные критерии (C1–C4)

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | mel_mean_by_segment shape = (N, 128), N == len(segment_mask) | shape OK |
| C2 | mel_stability ∈ [0, 1] на всех ok NPZ | ∈ [0,1] |
| C3 | mel_stats_vector size = 4×128 = 512, если присутствует | size=512 |
| C4 | mel_flatness ∈ (0, 1] на всех ok NPZ | > 0 и ≤ 1 |

## Примечания

- **NaN by design**: в empty NPZ (audio_missing_or_extract_failed) поля sample_rate, n_fft, hop_length, n_mels, fmax, duration_sec = NaN — orchestrator передаёт пустой payload, config не сохраняется. Это не баг, но можно улучшить в future (передать config в empty payload).
- **feature-gated**: mel_mean_by_segment, mel_stats_vector, mel_mean/std/min/max присутствуют когда включены соответствующие флаги (time_series/stats_vector/statistics). В storage всегда включены.
- **GPU не требуется**: mel_extractor работает на CPU (torchaudio). GPU ускоряет, но не обязателен для валидации.
