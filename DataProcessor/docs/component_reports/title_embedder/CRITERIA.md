# CRITERIA: title_embedder

**Версия**: v1.2.0 | **Дата**: 2026-07-16

---

## Универсальные гейты (U1–U6)

| Гейт | Критерий |
|------|----------|
| U1 | Валидатор rc=0 на всех NPZ (batch) |
| U2 | feature_names.size == feature_values.size; tp_titleemb_dim∈{384,768,1024} |
| U3 | tp_titleemb_present=1 и tp_titleemb_l2_norm≈1.0 при title_present=1; tp_titleemb_present=0 при absent title |
| U4 | empty/absent случай: graceful (rc=0, present=0) — не крэш |
| U5 | Golden: max\|Δ\|=0.0 (10 runs одного видео — sentence-transformers детерминирован) |
| U6 | Дискриминативность: `tp_titleemb_norm_raw` или `tp_titleemb_encode_ms` CV>10% между видео с разными заголовками |

---

## Специфичные критерии (C1–C4)

| Код | Критерий |
|-----|----------|
| C1 | `tp_titleemb_dim=1024` (all-MiniLM-L6-v2 даёт 384; проверить актуальную модель); embedding shape=(dim,) |
| C2 | `tp_titleemb_l2_norm≈1.0` (±0.01) при present=1 — нормализован |
| C3 | sim=1.0 при идентичном тексте заголовка — **by design** (5/6 E2E видео имеют одинаковый mock title → ожидаемо) |
| C4 | artifact_written=1 ↔ _artifacts/title_embedding.npy существует |

---

## Примечания

- `sim=1.0` у 5 из 6 видео — НЕ баг: все 5 используют `_tmp/text_input_autogen.json` с title='E2E sample video 1: creator tips for short-form content'. Шестое видео (-Q6fnPIybEI) имеет другой входной текст → другой эмбеддинг.
- Одинаковые title_len_chars=55 у 5 видео подтверждает одинаковый текст.
- Батч-валидатор проверяет 16 полей `tp_titleemb_*` из text_features.npz.
