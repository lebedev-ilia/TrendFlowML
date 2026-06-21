# `clap_extractor` — описание фич (референс)

Полный текст с трассировкой артефактов и табличным набором: **[`../FEATURE_DESCRIPTION.md`](../FEATURE_DESCRIPTION.md)** (источник в корне компонента — исторически).

Кратко:
- **NPZ:** `clap_extractor/clap_extractor_features.npz`, схема `clap_extractor_npz_v1`
- **Таблично:** `feature_names` / `feature_values` (norm, magnitudes, `segments_count`, …)
- **Валидатор:** [`../utils/validate_clap.py`](../utils/validate_clap.py)
- **Melt/QA:** `view_csv_feature_qa.json` → компонент `clap_extractor`

Этот файл в `docs/` дублирует входную точку для единообразия с остальными экстракторами (`docs/FEATURE_DESCRIPTION.md` обязателен в аудите).
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
