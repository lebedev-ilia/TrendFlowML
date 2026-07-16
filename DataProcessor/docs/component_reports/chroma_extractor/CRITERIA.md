# Критерии приёмки: chroma_extractor

Версия: v2.1.1 | Дата: 2026-07-16

## Универсальные гейты (U1–U6)

| # | Критерий | Порог |
|---|----------|-------|
| U1 | validate_chroma.py --schema --struct --qa rc=0 для всех NPZ | rc=0 всегда |
| U2 | segment_centers_sec, segment_durations_sec, segment_mask одинаковой длины N; chroma_mean_by_segment shape=(N,12) | exact |
| U3 | nan_rate = 0 (status=ok); NaN/sentinel by design (status=empty) | nan=0 для ok |
| U4 | empty NPZ: валидатор rc=0 (is_empty guard в validate_structure); chroma_mean=[nan] и dominant_class=-1 допустимы | rc=0 |
| U5 | Golden: max\|Δ\|=0.0 при одном входном файле (librosa.chroma_cqt детерминирован); разница между разными скачиваниями — by design | 0.0 (same audio) |
| U6 | Дискриминативность: chroma_entropy CV ≥ 10% и/или chroma_contrast CV ≥ 10% | ≥10% |

## Специфичные критерии (C1–C4)

| # | Критерий | Порог |
|---|----------|-------|
| C1 | chroma_mean_by_segment: shape=(N, 12), dtype float32 | exact |
| C2 | chroma_dominant_class ∈ [0, 11] для ok NPZ; -1 sentinel для empty — допустим | [0,11] или -1 |
| C3 | tuning_estimate: finite float (cents отклонение от A=440 Hz строя); обычно ∈ [-0.5, 0.5] | finite |
| C4 | chroma_entropy > 0 для ok NPZ (log-энтропия профиля) | >0 |

## Примечания по дизайну

- **chroma_mean=[nan] при empty**: npz_saver пишет `chroma_mean=[nan]` shape=(1,) при пустом payload. Это by design.
- **dominant_class=-1**: sentinel "нет доминирующего класса" при empty. By design.
- **chroma_harmonic_stability ≈ 0.987**: почти константа (CV=0.6%) — все видео имеют высокую гармоничность. Не баг.
- **Схема**: chroma_extractor_npz_v1 (segment axis: centers+durations, не start/end как у mel/mfcc).
- **Golden**: librosa.chroma_cqt детерминирован при однотредовом запуске с одним аудио файлом.
