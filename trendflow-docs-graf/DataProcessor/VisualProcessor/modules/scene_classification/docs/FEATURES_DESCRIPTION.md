# Описание фичей модуля scene_classification

## 1. Базовые фичи сцены (baseline)

### indices, start_frame, end_frame, length_frames, length_seconds
- **indices**: Список индексов кадров, принадлежащих данной сцене.
- **start_frame / end_frame**: Границы сцены в индексах кадров.
- **length_frames**: Длина сцены в кадрах.
- **length_seconds**: Длина сцены в секундах с учётом реального `fps` из `FrameManager`.

Policy update:
- **Heuristics are forbidden**: keyword ontologies (indoor/outdoor, nature/urban) removed; heuristic fusion removed.
- Segmentation uses **hard shot boundaries from `cut_detection`** and then groups consecutive shots until duration ≥ `min_scene_seconds` (default **2.0s**).

### mean_score, class_entropy_mean, top1_prob_mean, top1_vs_top2_gap_mean, fraction_high_confidence_frames
- **mean_score**: средняя вероятность top‑1 предсказания сцены по кадрам.
- **class_entropy_mean**: средняя энтропия распределения по 365 классам Places365 (Shannon entropy) по кадрам — мера «неопределённости» модели.
- **top1_prob_mean**: средняя уверенность top‑1 класса по кадрам.
- **top1_vs_top2_gap_mean**: средний разрыв между top‑1 и top‑2 вероятностями; большие значения соответствуют более «чётким» сценам.
- **fraction_high_confidence_frames**: доля кадров в сцене, где top‑1 вероятность > 0.7.

Все эти признаки вычисляются на основе полных распределений вероятностей Places365 и агрегируются как средние по кадрам.

## 2. Удалённые эвристические фичи (policy)

Удалены как запрещённые эвристики:
- indoor/outdoor
- nature/urban
- fused label mixing (Places+CLIP)

## 5. Aesthetic Score

### mean_aesthetic_score, aesthetic_std, aesthetic_frac_high
Оценка эстетической привлекательности сцены (0.0‑1.0).

- На уровне кадра: **CLIP‑based zero‑shot** (позитивные против негативных промптов) на базе `core_clip` (без локального CLIP и без эвристик).
- На уровне сцены:
  - **mean_aesthetic_score** — среднее по кадрам;
  - **aesthetic_std** — стандартное отклонение по кадрам (стабильность эстетики);
  - **aesthetic_frac_high** — доля кадров с эстетическим скором > 0.8.

## 6. Luxury Score

### mean_luxury_score
Оценка "роскошности" сцены (0.0‑1.0).

- На уровне кадра: **CLIP‑based zero‑shot** (через `core_clip`) — без эвристик.
- На уровне сцены: **mean_luxury_score** — среднее по кадрам.

Рекомендуется использовать luxury‑фичи только в комбинации с другими признаками и с дополнительным контролем bias.

## 7. Atmosphere Sentiment

### mean_cozy, mean_scary, mean_epic, mean_neutral, atmosphere_entropy
Вероятности атмосферы сцены (уютная, страшная, эпическая, нейтральная).

- На уровне кадра: **CLIP zero‑shot (4 класса)** на базе `core_clip` (без эвристик).
- На уровне сцены:
  - **mean_cozy / mean_scary / mean_epic / mean_neutral** — средние вероятности по кадрам;
  - **atmosphere_entropy** — энтропия усреднённого распределения `[cozy, scary, epic, neutral]`, показывающая «определённость» атмосферы.

### scene_change_score, label_stability, dominant_places_topk_ids, dominant_places_topk_probs
- **scene_change_score**: мера вариативности сцены, основанная на стандартном отклонении confidence top‑1 по кадрам (чем выше, тем менее стабильна сцена).
- **label_stability**: доля кадров в сцене, у которых top‑1 лейбл совпадает с доминирующим лейблом сцены.
- **dominant_places_topk_ids**: top‑K (до 5) индексы классов Places365, доминирующих в сцене (по суммарному весу вероятностей по кадрам).
- **dominant_places_topk_probs**: соответствующие суммарные веса (агрегированные вероятности) для этих классов.

## Примечания

Все фичи с префиксом "mean_" вычисляются как среднее значение по всем кадрам, принадлежащим одной сцене. Сцены агрегируются из последовательных кадров с одинаковым предсказанным лейблом Places365, после чего:

- длина сцены нормируется по `fps` (`length_seconds`);
- применяется fps‑aware порог `min_scene_seconds` (или эквивалент через `min_scene_length_frames / fps`);
- дополнительно считаются робастные агрегаты: энтропии, gaps, доли high‑confidence кадров и стабильность лейблов.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
