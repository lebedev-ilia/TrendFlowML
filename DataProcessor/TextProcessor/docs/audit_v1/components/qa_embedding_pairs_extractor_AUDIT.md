# `qa_embedding_pairs_extractor` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/qa_embedding_pairs_extractor/main.py`

## Резюме

- Переход на `doc.asr` как preferred transcript source (legacy `doc.transcripts` только при `allow_legacy_transcripts=True`).
- Убран per-run JSON sidecar `qa_question_embeddings_meta.json`.
- Выход: только `features_flat` (`tp_qa_*`), а relpath матрицы вопросов живёт in-memory в `doc.tp_artifacts["qa"]["question_embeddings"]`.
- Fixed bug: извлечение вопросов сохраняет `?` и больше не ломается от sentence split.
- Добавлены лимиты/дедуп/мультиязык и feature-gating по источникам.
- Empty semantics (A-policy): при `num_questions=0` не пишем `.npy` и не заполняем `tp_artifacts`.
- dp_models compliance: модель грузится через `get_model_with_meta` и фиксирует `weights_digest`/`model_version`.
- feature-gating уровня компонента: `enabled` + `tp_qa_disabled_by_policy`.
- стабильная схема `tp_qa_*` + privacy-safe индикаторы записи sub-artifacts.

## TODO

- `resource_costs` замеры и лимиты по количеству извлекаемых вопросов (для UX/стоимости).


