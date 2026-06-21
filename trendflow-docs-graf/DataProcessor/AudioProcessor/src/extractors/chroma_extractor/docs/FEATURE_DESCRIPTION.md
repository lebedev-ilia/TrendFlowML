# `chroma_extractor` — выходы и фичи

**Реализация:** [`../main.py`](../main.py) · **NPZ:** [`../../../../schemas/chroma_extractor_npz_v1.json`](../../../../schemas/chroma_extractor_npz_v1.json) · **Валидатор:** [`../utils/validate_chroma.py`](../utils/validate_chroma.py).

## Табличный срез

`feature_names` / `feature_values` + агрегаты по 12 bin chroma (см. схему).

## Скаляры/вектора (смысл)

| Ключ | Смысл |
|------|--------|
| `chroma_mean` | [12] — средний профиль |
| `chroma_entropy` | энтропия распределения |
| `chroma_harmonic_stability` | устойчивость/гармоничность (логика в коде) |
| `chroma_contrast` | контраст |
| `chroma_dominant_class`, `chroma_dominant_energy` | доминанта pitch class / энергия |
| `tuning_estimate` | оценка строя, полутона |
| Сегментация | `segment_*`, `chroma_mean_by_segment` [N,12] |

**Нормальные диапазоны:** `chroma_mean` обычно ≥ 0; `entropy` ≥ 0; `dominant_class` 0..11; `tuning_estimate` в разумных полутонах. Подсветка в HTML: `view_csv_feature_qa` → `chroma_extractor`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
