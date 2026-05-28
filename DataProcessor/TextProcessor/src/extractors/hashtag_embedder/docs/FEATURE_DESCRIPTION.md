# `hashtag_embedder` — описание фич и артефактов

**Компонент:** `HashtagEmbedder` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **23** скаляров `tp_hashemb_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/hashtag_embedder_output_v1.json`](../../../../schemas/hashtag_embedder_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Плотный вектор хранится в **`hashtag_embedding.npy`** (per-run artifacts), в `features_flat` — только скаляры.

**Версия:** 1.2.0 (`HashtagEmbedder.VERSION`).

---

## 1. Назначение

- Закодировать уникальные канонические хештеги sentence-transformers, агрегировать (**mean** / **max** / **logsumexp**) в один вектор, **L2-нормализовать** (`l2_norm` ≈ **1** при `present=1`).  
- Опционально: кеш по детерминированному ключу, запись **`hashtag_embedding.npy`**.

---

## 2. Группы полей

| Группа | Заметки |
|--------|---------|
| Сводка | `tp_hashemb_present` — **1**, если эмбеддинг посчитан в памяти (не «только артефакт») |
| Размерность / норма | `dim`, `tag_count`, `l2_norm` — **NaN** в empty / `compute_embedding=False` |
| Политика | `require_hashtags_enabled`, `disabled_by_policy_hint` (0/1 из `tp_artifacts.tags`) |
| Счётчики | `n_input_tags`, `n_unique_tags`, `n_tags_truncated` — **≥ 0** |
| Gating | `compute_enabled`, `write_artifact_enabled`, `artifact_written` |
| Кеш | `cache_enabled`; `cache_hit` — **0/1** при выставлении, **NaN** в шаблоне до подмены; при **`cache_enabled=False`** на успешном пути **0**; в **batch** путь не кеширует per-doc → **0** |
| Модель | `model_digest_u24` — `int(weights_digest[:6], 16)` (скаляр float) |
| Устройство | `fp16` — фактически **1** только при **CUDA** и `fp16=True` в конфиге |
| Тайминги | `encode_ms`, `agg_ms` — **NaN**, если нет соответствующей фазы (пусто, `compute_disabled`, кеш hit без encode) |
| Агрегация (one-hot) | `agg_mean` / `agg_max` / `agg_logsumexp` — ровно **один** **1** |

Параметр **`emit_extra_metrics`** в конструкторе есть, в **текущей** логике **не** зануляет поля `features_flat` (тайминги выставляются по факту фаз).

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (finite) | `present`, gating, `cache_enabled`, `artifact_written` (кроме `cache_hit` — см. схему), `fp16`, `device_cuda`, `use_frequencies`, `agg_*`, `require_*` |
| `tp_hashemb_disabled_by_policy_hint` | **0/1** |
| `tp_hashemb_l2_norm` (finite) | **≈ 1.0** (после L2-нормировки; допуск **0.9–1.1** в валидаторе) |
| `tp_hashemb_model_digest_u24` (finite) | **≥ 0** |
| `tp_hashemb_encode_ms`, `tp_hashemb_agg_ms` (finite) | **≥ 0**, типично **< 1e7** мс |
| one-hot `agg_mean`+`max`+`logsumexp` | **сумма = 1** |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_hashtag_embedder_text_npz.py`](../utils/validate_hashtag_embedder_text_npz.py)

---

## 5. Чеклист

1. **23** имён в срезе = JSON.  
2. При `present=1`: `l2_norm` в разумном коридоре вокруг **1**, `dim` ≥ **1**.  
3. `vector` смотреть в артефакте, не в NPZ-таблице.
