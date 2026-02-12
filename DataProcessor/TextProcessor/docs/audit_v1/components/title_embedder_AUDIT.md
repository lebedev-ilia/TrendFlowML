# `title_embedder` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/title_embedder/main.py`

## Резюме

Компонент приведён к production-стандарту per-run артефактов: сохраняет `*.npy` в `text_processor/_artifacts/`, **не пишет `.meta.json`**, **не возвращает абсолютные пути** и регистрирует relpath в `doc.tp_artifacts`.

Дополнительно (prod hardening):
- model loading строго через `dp_models` (no-network, fail-fast)
- `model_registry` кэширует модельный handle с ключом, включающим `weights_digest` (обновление весов → новый handle)
- дисковый cache эмбеддингов управляем: TTL + лимиты + best-effort уборка (конфиг)
- hash артефакта/кеша включает `weights_digest` (детерминизм при смене весов)
- фиксированное per-run имя артефакта (без hash в названии)
- split compute/write + stable `features_flat` schema

## Контракт

- **Вход**: `VideoDocument.title` (optional by default, valid empty); fail-fast при `require_title=true`.
- **Выход**:
  - `.npy`: `title_embedding.npy` (per-run, fixed name)
  - `result.features_flat`: `tp_titleemb_*`
  - `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (только если `write_artifact=true`)

## Соответствие критериям

- ✅ Per-run storage (`text_processor/_artifacts`)
- ✅ No `.meta.json` sidecars (model meta → `manifest.json.models_used`)
- ✅ No abs paths в `result`
- ✅ Observability: `system` + `timings_s`

## TODO

- `resource_costs` замеры для CPU/GPU пресетов.


