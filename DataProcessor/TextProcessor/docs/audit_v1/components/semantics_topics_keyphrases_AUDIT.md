# `semantics_topics_keyphrases` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `DataProcessor/TextProcessor/src/extractors/semantics_topics_keyphrases/main.py`  
**Критерии**: `DataProcessor/TextProcessor/docs/audit_v1/TP_AUDIT_CRITERIA.md`

## 1) Итоговое решение (после Q&A)

- ✅ Убрали per-video BERTopic/KMeans (несопоставимо между видео).
- ✅ Перешли на **global taxonomy retrieval**:
  - `dp_models/bundled_models/text/topics_v1/topics.jsonl`
  - prompt embeddings строятся в cache (не result_store), модель — через `dp_models` (`get_model`).
- ✅ Transcript source-of-truth: `doc.asr` (legacy `doc.transcripts` только через `allow_legacy_transcripts`).
- ✅ Privacy: raw keyphrases не сохраняются по умолчанию; `export_raw_keyphrases=True` — gated.
- ✅ Placeholder метрики удалены (документированы как отдельные качественные компоненты).
- ✅ Sub-artifacts: keyphrase embeddings `*.npy` пишутся per-run в `text_processor/_artifacts/` и relpath живёт только in-memory (`doc.tp_artifacts.topics.keyphrase_embeddings`).

## 2) Контракты входа

- `VideoDocument.asr.segments[].text` (preferred)
- `VideoDocument.title`, `VideoDocument.description` (optional, но обычно присутствуют)
- Legacy `VideoDocument.transcripts` только при `allow_legacy_transcripts=True`

## 3) Контракты выхода

- `result.features_flat` (числовые скаляры `tp_topics_*`)
- optional `tp_topics_keyphrases_raw` только при `export_raw_keyphrases=True`
- per-run `.npy` sub-artifact для keyphrase embeddings:
  - `tp_topics_keyphrase_embeddings_<id>.npy` (в `text_processor/_artifacts/`)

## 4) Улучшения “качественнее, но сложнее” (зафиксированы)

- Keyphrases: YAKE / KeyBERT/MMR (как отдельный более качественный режим/компонент).
- Language selection: отдельный LanguageDetector (через `dp_models`) вместо смешанных prompts.
- Topics DB: расширить taxonomy до 200–500 тем, стабилизировать ids и добавить quality набор.

## 5) Remaining TODO

- `DataProcessor/docs/models_docs/resource_costs/text_processor_semantics_topics_keyphrases_costs_v1.json` (best-effort placeholder added; требуется заполнить бенчмарком)
- fixtures + smoke-run для `SemanticTopicExtractor` на нескольких текстах (RU/EN/mixed)


