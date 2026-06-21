## v1 predictor (Transformers)

### Роль

v1 — основная multimodal модель предсказания (supervised), обучаемая end-to-end вместе с trainable encoder.

### High-level architecture (v1.0)

- VisualEncoder → visual token sequence
- AudioEncoder → audio token sequence
- Text/comments → несколько text tokens (Kc=4..8)
- FusionTransformer: cross-attention fusion между модальностями + meta/text tokens
- Multi-head outputs: 6 значений (views/likes × 7/14/21), 7d masked

### Fusion

Используем **cross-attention** (качественнее и устойчивее, чем “concat → 1 transformer” при тех же бюджетах).

### Time encoding

Каждый token получает time embedding:
- `time_pos_emb = MLP(t_center / duration_sec)`
где `t_center` берётся из `summary_times_s`.

### Comments/text (качество без raw текста)

- raw текст не сохраняем
- для ≤100 комментариев строим embeddings per-comment
- агрегируем в **несколько tokens** (Kc=4..8) (например attention pooling/top-K информативных)
- эти tokens участвуют в fusion transformer

### Outputs

6 выходов:
- `views_7d`, `views_14d`, `views_21d`
- `likes_7d`, `likes_14d`, `likes_21d`

Masked loss для 7d.

### Uncertainty

Используем **quantile heads**:
- минимум p10/p50/p90 для каждого из 6 выходов
- point estimate = p50

### Loss balancing

Горизонты балансируем через **обучаемые веса** (uncertainty weighting) с safety cap [0.2..5.0].

### Compute budget (v1.0)

- 30–50M params
- inference latency после готовых encoder токенов: 2–5s
---

## Навигация

[Models](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
