# `description_embedder` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/description_embedder/main.py`

## Резюме

Переведён на per-run sub‑artifacts (`*.npy`), удалены `.meta.json` sidecar’ы и абсолютные пути из `result`. Для downstream связывания используется `doc.tp_artifacts`.

Дополнительно (prod hardening, A-policy):
- валидная пустота: если `description` отсутствует → **не создаём** фейк-вектор, `tp_descemb_present=0`
- model/tokenizer резолвятся через `dp_models` (offline, weights_digest фиксируется)
- hash/cache включают `weights_digest` + параметры chunking/pooling
- disk cache управляем: TTL/лимиты + best-effort уборка
- token-aware chunking через `shared_tokenizer_v1`

## Контракт

- **Вход**: `VideoDocument.description` (может быть пустым)
- **Выход**:
  - `.npy`: `description_embedding.npy` (per-run, fixed name)
  - `result.features_flat`: `tp_descemb_*`
  - `doc.tp_artifacts["embeddings"]["description"]["relpath"]` (только если `write_artifact=true`)

## Соответствие критериям

- ✅ Per-run storage (`text_processor/_artifacts`)
- ✅ No `.meta.json` sidecars
- ✅ No abs paths в `result`
- ✅ Observability: `system` + `timings_s`

## TODO

- `resource_costs` замеры.


