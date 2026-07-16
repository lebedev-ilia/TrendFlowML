# Критерии приёмки: clap_extractor

**Согласовано:** 2026-07-16  
**Версия компонента:** CLAPExtractor v1.1.1 (LAION CLAP, schema clap_extractor_npz_v1)

---

## Универсальные гейты (U1–U6)

| Гейт | Критерий | Применимость |
|------|----------|-------------|
| U1 | validate_clap.py rc=0 (--struct) на всех 16 NPZ | ✅ |
| U2 | segment_start_sec монотонен, start<end, center∈[start,end] | ✅ |
| U3 | Различимость: status=ok → embedding dim=512, norms 0.8-1.0, NaN=0 | ✅ |
| U4 | status=empty/error → embedding_present=False, fv NaN×5, seg_n=0 | ✅ |
| U5 | Golden: 2 прогона одного видео → diff=0 (CUDA inference_mode+float32) | GPU local |
| U6 | N=5/30 сегментов без падений | GPU local |

## Специфические критерии (C1–C3)

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | embedding dim=512, L2 norm ∈ [0.5, 1.5], NaN=0 | dim=512, norm∈[0.5,1.5] |
| C2 | embedding_sequence shape (N,512) = строго aligned с segments | shapes match |
| C3 | segment_mask: ≥1 True при status=ok; все False при error/empty | seg_count≥1 при ok |

## Исключения

- **status=error** (2 NPZ): embedding пустой by design (ошибка загрузки аудио)
- **status=empty** (2 NPZ): audio_missing_or_extract_failed by design
- **NaN в embedding_sequence** при masked сегментах (by design: `np.full(..., np.nan)`)
- **U5 golden**: CLAP с autocast(float32) детерминирован при одинаковом device/seed
