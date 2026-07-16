# CRITERIA — embedding_pair_topk_extractor

Компонент: CPU-only cosine similarity + top-K (numpy/FAISS), без нейросети.

## Универсальные хард-гейты

| Гейт | Критерий |
|------|----------|
| U1 | validate rc=0 на всех NPZ (batch + --struct + --ranges) |
| U2 | N/A — нет seq/time axis (только features_flat, скаляры) |
| U3 | Различимость: topk_max std > 0 на ≥5 видео с разными chunks |
| U4 | Empty-path (нет артефактов): present=0, 69 ключей, все cosine/topk = NaN; disabled: disabled_by_policy=1 |
| U5 | Golden max|Δ|=0.0 (numpy CPU детерминирован) |
| U6 | Работает на видео с n_chunks=1..20+, dim=512..1024 без падений |

## Критерии компонента

| Критерий | Порог |
|----------|-------|
| C1 | validate rc=0 × 28 NPZ (batch, все статусы) |
| C2 | golden max|Δ|=0.0 (5+ прогонов) |
| C3 | NaN by design: extra block (6 полей) при emit_extra_metrics=False → NaN; unfilled top-K slots → NaN (не дефект) |
| C4 | dim mismatch → dim_mismatch_flag=1, topk=NaN (graceful, без исключения); top_k_slots > SCHEMA_MAX_TOP_K_SLOTS → clamped=1 (корректно) |

## Фиксированные исключения

- `title_desc_cosine` одинакова на тестовом датасете (одно видео) — by design (одинаковые title.npy/desc.npy), не дефект компонента
- NaN rate ~0.47 по ok-NPZ — by design из extra block + unfilled slots при n_chunks < top_k_slots
- emit_extra_metrics=False (production default): n_chunks/source/faiss_mode = NaN
