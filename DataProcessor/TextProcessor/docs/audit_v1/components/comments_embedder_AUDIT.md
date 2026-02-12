# `comments_embedder` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/comments_embedder/main.py`

## Резюме

Embedder комментариев приведён к стандарту per-run sub‑artifacts: сохраняет матрицу эмбеддингов в `text_processor/_artifacts/*.npy`, не создаёт `.meta.json`, не возвращает абсолютные пути и регистрирует relpath в `doc.tp_artifacts`.

Prod hardening (A-policy):
- valid empty: комментариев может не быть → `tp_commentsemb_present=0`, без фейк-матрицы и без `tp_artifacts["embeddings"]["comments"]`
- фиксированное per-run имя артефакта: `comments_embeddings.npy`
- dp_models: фиксируется `weights_digest` (и кладётся в `tp_artifacts`)
- детерминированный отбор/лимиты комментариев (max_comments, truncation, dedup, selection_policy)
- optional cache (TTL/лимиты), по умолчанию выключен

## Контракт

- **Вход**: `VideoDocument.comments[].text`
- **Выход**:
  - `.npy`: `comments_embeddings.npy` (per-run)
  - `result.features_flat`: `tp_commentsemb_*`
  - `doc.tp_artifacts["embeddings"]["comments"]["relpath"]`

## Соответствие критериям

- ✅ Per-run storage
- ✅ No sidecar JSON
- ✅ No abs paths в `result`
- ✅ Empty semantics: при отсутствии комментариев возвращает `present=0` (через features_flat)

## TODO

- `resource_costs` замеры для разных N комментариев / batch_size.


