# `hpss_extractor` — выходы и фичи

**Реализация:** [`../main.py`](../main.py) · **NPZ:** [`../../../../schemas/hpss_extractor_npz_v1.json`](../../../../schemas/hpss_extractor_npz_v1.json) · **Валидатор:** [`../utils/validate_hpss.py`](../utils/validate_hpss.py).

## Табличный срез

`feature_names` / `feature_values` — плоские скаляры; полный контракт в JSON-схеме.

## Массивы

| Ключ | Смысл |
|------|--------|
| `hpss_harmonic_share_by_segment` | Доля гармоники по сегментам [0,1] |
| `hpss_percussive_share_by_segment` | Доля перкуссии [0,1] |
| `hpss_*_series` | Опц. ряды по кадрам |
| `segment_*_sec`, `segment_mask` | Временная ось |

**Нормальные диапазоны:** на сегменте `harmonic_share + percussive_share ≈ 1` (с допуском); доли ≥ 0. Детали — `validate_hpss` и `view_csv_feature_qa` → `hpss_extractor`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
