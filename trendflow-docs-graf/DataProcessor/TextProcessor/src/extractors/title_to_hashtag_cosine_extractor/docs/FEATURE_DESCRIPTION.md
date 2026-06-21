# `title_to_hashtag_cosine_extractor` — описание фич и артефактов

**Компонент:** `TitleToHashtagCosineExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **11** скаляров `tp_titlehashcos_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/title_to_hashtag_cosine_extractor_output_v1.json`](../../../../schemas/title_to_hashtag_cosine_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Векторы берутся из **`doc.tp_artifacts["embeddings"]["title|hashtag"]["relpath"]`** (файлы в `artifacts_dir`), не сериализуются в `features_flat`.

**Версия:** 1.2.0 (`TitleToHashtagCosineExtractor.VERSION`).

---

## 1. Назначение

- Загрузить **title** и **hashtag** эмбеддинги (`.npy`), **L2-нормировать**, **cosine** = **dot(a,b)** ∈ **[-1, 1]**.  
- Флаги: безопасный путь (`unsafe`), отсутствие/битый файл (`*_embed_missing`), **dim mismatch**, **нулевая норма**.

---

## 2. Ключи (смысл)

| Ключ | Заметки |
|------|---------|
| `present` | **1** только если оба вектора загружены, **одинаковая dim**, **нормы > 0** после `_l2n` |
| `cosine` | **NaN** если `present=0`; иначе **[-1, 1]** |
| `require_*_enabled` | зеркала opt-in **RuntimeError** при отсутствии соответствующего эмбеддинга |
| `title_present` / `hashtag_present` | **1** если вектор успешно получен (`ok` в `_try_load_embedding`) |
| `unsafe_relpath_flag` | **1** при path traversal / исключении join |
| `*_embed_missing_flag` | **1** при `missing` / `bad_file` (не `unsafe`) |
| `dim_mismatch_flag` | **1** если оба `ok`, но `size` различается |
| `zero_norm_flag` | **1** если после load оба ненулевые по размеру, но L2-норма **0** до/при нормировке |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Все, кроме `cosine` (finite) | **0/1** |
| `tp_titlehashcos_cosine` (finite) | **[-1, 1]** |
| при `present=1` | `cosine` finite в **[-1, 1]** |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_title_to_hashtag_cosine_extractor_text_npz.py`](../utils/validate_title_to_hashtag_cosine_extractor_text_npz.py)

---

## 5. Чеклист

1. **11** имён = `title_to_hashtag_cosine_extractor_output_v1` (`allow_extra_keys: false`).  
2. `present=1` ⇒ **cosine** finite, флаги ошибок **0** (кроме согласованных edge-кейсов в доке upstream).
---

## Навигация

[README (root)](../README.md) · [SCHEMA (root)](../SCHEMA.md) · [TextProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
