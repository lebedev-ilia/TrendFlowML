# `title_embedder` — описание фич и артефактов

**Компонент:** `TitleEmbedder` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **16** скаляров `tp_titleemb_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/title_embedder_output_v1.json`](../../../../schemas/title_embedder_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · **Вектор** — `title_embedding.npy`, не в `features_flat`. · [`../README.md`](../README.md).

**`emit_extra_metrics`:** в 1.2.0 **не** меняет набор/NaN-правила `features_flat` (см. README).

**Версия:** 1.2.0 (`TitleEmbedder.VERSION`).

---

## 1. Назначение

- L2-нормированный эмбеддинг `doc.title` (sentence-transformers / `get_model_with_meta`).  
- Опционально: **L2 норма raw-вектора** до нормировки (`compute_raw_norm`), кеш, запись `title_embedding.npy`.

---

## 2. Группы полей

| Группа | Заметки |
|--------|---------|
| Gating | `tp_titleemb_title_present` — **1** при непустом title после `normalize_whitespace` |
| | `tp_titleemb_present` — **1** если эмбеддинг **считан** (не путать с артефактом) |
| Конфиг (зеркала) | `require_title_enabled`, `compute_enabled`, `write_artifact_enabled`, `cache_enabled`, `fp16`, `device_cuda`, `compute_raw_norm` — **0/1** |
| Артефакт | `artifact_written` — **1** если файл записан; при `write_artifact_enabled=0` → **0** |
| Кеш | `cache_hit` — **0/1** при `cache_enabled=1` и ветках с фиксированным значением; **NaN** при пустом title; при `cache_enabled=0` на путях без энкода → **0.0** |
| Метрики | `dim` — размерность; `norm_raw` — L2 raw; `l2_norm` — L2 **после** нормировки (≈ **1**); `model_digest_u24` — `int(weights_digest[:6], 16)`; `encode_ms` — мс **encode** |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (finite) | зеркала, `artifact_written`, `present` (где применимо) |
| `tp_titleemb_model_digest_u24` | **0 … 0xFFFFFF** |
| `tp_titleemb_encode_ms` (finite) | **≥ 0**, **&lt; 1e7** мс |
| при `present=1` | `dim` **≥ 1**; `l2_norm` **≈ 1**; `norm_raw` **&gt; 0** при finite и `compute_raw_norm=1` |
| при `present=0` / нет title | `dim`, `norm_raw`, `l2_norm`, `encode_ms` — часто **NaN**; `cache_hit` может быть **NaN** |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_title_embedder_text_npz.py`](../utils/validate_title_embedder_text_npz.py)

---

## 5. Чеклист

1. **16** имён = `title_embedder_output_v1` (`allow_extra_keys: false`).  
2. `l2_norm` ~ **1** при успешной нормировке.  
3. `norm_raw` **NaN** при `compute_raw_norm=0` (и логике `compute`).
