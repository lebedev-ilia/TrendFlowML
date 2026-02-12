# `embedding_source_id_extractor` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/embedding_source_id_extractor/main.py`

## Резюме

Экстрактор:
- выбирает primary embedding детерминированно из `doc.tp_artifacts` (без glob/mtime)
- генерирует переносимый `vector_id` **без зависимости от abs_path** (sha256 по значениям float32, 24 hex)
- `embedding_relpath` трактуется как путь относительно `text_processor/_artifacts/`
- missing primary:
  - `strict_missing_primary=True` → fail-fast
  - иначе valid empty (`tp_embid_present=0` + `embedding_source_id.error="no_embedding_found"`)
- safe-join: relpath не может “вылезти” из `artifacts_dir`

## Выход

`result` содержит:

- `features_flat` (scalars):
  - `tp_embid_present`
  - one-hot policy flags (`tp_embid_policy_*`)
  - one-hot primary kind flags (`tp_embid_primary_is_*`)
- `embedding_source_id` (privacy-safe dict with strings):
  - `vector_id`, `vector_store_uri`, `model_version`, `weights_digest`, `embedding_relpath`, `primary_source`

## TODO

- Зафиксировать `resource_costs` (очень дешёвый CPU).


