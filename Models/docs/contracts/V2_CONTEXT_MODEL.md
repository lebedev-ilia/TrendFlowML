## v2 predictor (v1 + external context)

### Роль

v2 = v1 prediction + внешний контекст (trends/news/etc) в **воспроизводимом** виде.

### Архитектура (v1.0)

- `ContextBuilder` строит `context_features`
- `ContextAdjustmentModel` корректирует v1 prediction:
  - вход: (v1_pred, snapshot_0 meta, context_features)
  - выход: скорректированные 6 значений (views/likes × 7/14/21)

### context_features contract

- формат: **набор именованных фичей** (таблично) + `context_schema_version`
- `context_features` сохраняется как артефакт run (иначе воспроизводимость невозможна)

### TTL и деградация

- TTL контекста по умолчанию: **48 часов**
- если контекст недоступен/просрочен → fallback на v1, `prediction_status="degraded"`


