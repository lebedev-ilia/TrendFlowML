# `band_energy_extractor` — выходы и фичи

**Реализация:** [`../main.py`](../main.py) · **NPZ:** [`../../../../schemas/band_energy_extractor_npz_v1.json`](../../../../schemas/band_energy_extractor_npz_v1.json) · **Валидатор:** [`../utils/validate_band_energy.py`](../utils/validate_band_energy.py).

## Табличный срез

`feature_names` / `feature_values` — плоские скаляры (энергии/баланс по договорённости), см. NPZ.

## Ключевые массивы

| Ключ | Смысл |
|------|--------|
| `band_edges_hz` | Границы полос **[B, 2]**, Гц |
| `band_energy_shares` | Доли энергии по полосам, сумма **≈1** (нормировка) |
| `band_shares_by_segment` | Опц. **[N, B]** по сегментам |
| `segment_centers_sec`, `segment_durations_sec`, `segment_mask` | Сегментация |

**Нормальные диапазоны:** `band_energy_shares` и по-сегментные доли ∈ [0,1]; `band_edges` монотонны, ≥ 0. Подробности — валидатор и `view_csv_feature_qa` → `band_energy_extractor`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
