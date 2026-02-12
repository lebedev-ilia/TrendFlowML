## Targets / Splits / Metrics (baseline + v1 + v2)

### Prediction time

- Предсказываем **в любой момент времени**.
- Вход: `snapshot_0` (состояние на момент анализа).

### Snapshot_0 input fields (v1.0)

`snapshot_0` содержит (вход в модели):
- `views_0`
- `likes_0`
- `comments_0` (как фича, **не** таргет)
- `channel_subscribers_0`
- `channel_total_views_0`
- `channel_total_videos_0`
- `comments_text_list_0` (≤100) — raw не храним, используем только для извлечения embeddings/агрегатов.

### Targets (v1.0)

- Предсказываем только: `views`, `likes`.
- Горизонты: 7d (masked), 14d, 21d.

Функция таргета:
- \(\Delta x_h = x_h - x_0\)
- \(y_h = \log(1 + \Delta x_h)\)

### Loss weights

- Базовые веса по горизонтам: 7d=0.5, 14d=1.0, 21d=1.0.
- В v1 используем **обучаемые веса горизонтов** (uncertainty weighting) с safety cap [0.2..5.0].

### Splits (offline evaluation)

Split = hybrid:
- time-split по `publishedAt`
- channel-group split по `channel_id`

### Metrics

- **North star**: Spearman на \(y=\log1p(\Delta)\).
- Secondary:
  - MAE на \(y\)
  - Spearman по 8 age buckets (для устойчивости качества)

### Golden sets

- Holdout: 2000 видео (фиксированные снапшоты 0/7/14/21).
- Regression mini: 200 видео.


