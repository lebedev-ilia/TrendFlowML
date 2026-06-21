# `emotion_diarization_extractor` — выходы и фичи

**Реализация:** [`../main.py`](../main.py) · **NPZ:** [`../../../../schemas/emotion_diarization_extractor_npz_v1.json`](../../../../schemas/emotion_diarization_extractor_npz_v1.json) · **Валидатор:** [`../utils/validate_emotion_diarization.py`](../utils/validate_emotion_diarization.py).

## Табличный срез

`feature_names` / `feature_values` + сегментная классификация эмоций.

## Ключевые поля

| Ключ | Смысл |
|------|--------|
| `emotion_id` | Индекс класса [N] |
| `emotion_confidence` | Уверенность [0,1] по сегменту |
| `emotion_labels` | Список меток [C] |
| `emotion_probs` / `emotion_mean_probs` | Распределения по классам |
| `segment_*_sec`, `segment_mask` | Временная ось |

**Нормальные диапазоны:** `emotion_confidence` ∈ [0,1]; `emotion_id` в 0..C-1. QA/HTML: `view_csv_feature_qa` → `emotion_diarization_extractor`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
