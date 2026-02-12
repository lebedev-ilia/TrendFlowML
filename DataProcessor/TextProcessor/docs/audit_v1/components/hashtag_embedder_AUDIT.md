# `hashtag_embedder` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/hashtag_embedder/main.py`

## Резюме

Сохраняет агрегированный эмбеддинг хештегов как per-run `*.npy` (без `.meta.json`), не возвращает абсолютные пути, регистрирует relpath в `doc.tp_artifacts`. Введены фиксированные имена артефактов, split compute/write, стабильная схема `features_flat`, и cache default off.

Prod hardening (A-policy):
- `doc.hashtags` optional by default; fail-fast включается `require_hashtags=true`
- determinism: cache-key включает `model_name|weights_digest` + canonicalized tags + params
- canonicalization: casefold/strip/dedup/sort + лимиты `max_tags/max_tag_len`
- feature-gating: `compute_embedding`, `write_artifact`
- optional disk cache (default off): TTL/лимиты + best-effort cleanup

## Контракт

- **Вход**: `VideoDocument.hashtags` (заполняется `tags_extractor`, порядок: `TagsExtractor` должен быть раньше)
- **Выход**:
  - `.npy`: `hashtag_embedding.npy` (fixed per-run)
  - `result.features_flat`: `tp_hashemb_*`
  - `doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]` (только если `write_artifact=true`)

## Privacy

- ✅ Хештеги как raw список не сохраняются в NPZ этим компонентом; используются только для inference в рантайме.

## Resource costs

- CPU/GPU зависит от модели sentence-transformers.
- Время ~ \(O(n)\) по числу уникальных тегов (батчинг), память ~ \(O(n \cdot d)\) внутри батча.


