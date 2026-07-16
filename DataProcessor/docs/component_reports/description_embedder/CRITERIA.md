# CRITERIA: description_embedder

**Версия**: v1.x | **Дата**: 2026-07-16

---

## Универсальные гейты (U1–U6)

| Гейт | Критерий |
|------|----------|
| U1 | Валидатор rc=0 на всех NPZ (batch) |
| U2 | feature_names.size == feature_values.size; tp_descemb_dim∈{384,768,1024} |
| U3 | tp_descemb_l2_norm≈1.0 при present=1; 0 NaN в полях descemb |
| U4 | Absent description: graceful (rc=0, present=0) |
| U5 | Golden: max\|Δ\|=0.0 (19+ runs одного видео) |
| U6 | Дискриминативность: encode_ms или n_chunks CV>10% по видео |

## Специфичные критерии (C1–C4)

| Код | Критерий |
|-----|----------|
| C1 | tp_descemb_dim=1024, shape=(1024,) |
| C2 | tp_descemb_l2_norm=1.0000 (±0.001) |
| C3 | sim=1.0 при одинаковых описаниях — by design (5/6 видео с mock desc) |
| C4 | artifact_written=1 ↔ _artifacts/description_embedding.npy существует |

## Примечания

- Аналог title_embedder: sim=1.0 у 5/6 — by design (одинаковый mock description из autogen.json).
- Chunked encoding: длинные описания чанкуются и усредняются length-weighted.
