## TrendFlow — Docs for our trainable models

Этот каталог содержит **канонические документы** по нашим моделям, которые мы **сами создаём и обучаем**:
- **Encoder** (VisualEncoder/AudioEncoder) — после DataProcessor, приводит variable-length последовательности к fixed-budget токенам.
- **Baseline** (Boosting) — baseline predictor.
- **v1** (Transformers) — multimodal transformer predictor (end-to-end, включая trainable encoder).
- **v2** — v1 + внешний контекст (ContextBuilder + ContextAdjustmentModel).

### Индекс

- `contracts/MODEL_CONTRACTS_V1.md` — **финальные контракты v1.0** (source-of-truth).
- `contracts/ENCODER_CONTRACT.md` — контракт Encoder (входы/выходы/тайм-ось/бюджеты).
- `contracts/TARGETS_SPLITS_METRICS.md` — таргеты, сплиты, метрики, golden sets.
- `contracts/BASELINE_MODEL.md` — baseline: входы/выходы/freeze policy/feature_schema_version.
- `contracts/V1_TRANSFORMER_MODEL.md` — v1: архитектура, fusion, text tokens, uncertainty.
- `contracts/V2_CONTEXT_MODEL.md` — v2: context_features, TTL, деградация.
- `contracts/MODEL_SYSTEM_RULES.md` — версионирование/кэш/воспроизводимость/`model_signature` (общие правила).
- `contracts/PREDICTION_REPORT_CONTRACT.md` — формат `prediction_report.json` для UI: этапы прогона, артефакты, головы, интервалы.
- `roadmaps/BASELINE_TO_TRAINING_ROADMAP.md` — план: подготовка пайплайна → датасет → обучение (перенесён из DataProcessor docs).
---

## Навигация

[Models](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
