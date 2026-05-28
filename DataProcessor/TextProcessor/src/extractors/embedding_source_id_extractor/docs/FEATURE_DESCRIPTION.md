# `embedding_source_id_extractor` — описание фич и артефактов

**Компонент:** `EmbeddingSourceIdExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **13** скаляров `tp_embid_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/embedding_source_id_extractor_output_v1.json`](../../../../schemas/embedding_source_id_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Строковые поля (`vector_id`, `primary_source`, пути) живут в **`result.embedding_source_id`**, в machine JSON **не** входят.

**Версия:** 1.3.0 (`EmbeddingSourceIdExtractor.VERSION`).

---

## 1. Назначение

- Детерминированно выбрать **primary** эмбеддинг из `doc.tp_artifacts` по `primary_source_policy`.  
- При успехе: вычислить `vector_id` (SHA-256 over LE float32 bytes, первые 24 hex), **`tp_embid_present=1`**.  
- Все поля `features_flat` — **0/1** (скаляры float32); вложенный **`embedding_source_id`** — в отчётах/JSON, не в табличном срезе схемы `*_output_v1`.

---

## 2. Группы

| Группа | Ключи | Заметки |
|--------|--------|--------|
| Сводка | `tp_embid_present` | **1** только если вектор загружен, непустой, без NaN/Inf |
| Политика (one-hot) | `tp_embid_policy_transcript_first` … `transcript_only` | Ровно **один** = **1.0** (зеркало `primary_source_policy`) |
| Тип primary | `tp_embid_primary_is_transcript`, `…_title`, `…_description` | `transcript` = любой `primary_source` вида `transcript_*` (`startswith("transcript_")`); иначе title/description; при soft-fail до загрузки возможны **0/0/0** |
| Флаги | `tp_embid_strict_missing_primary_enabled`, `unsafe_relpath`, `primary_embed_missing`, `nan_inf` | 0/1 |
| | | При **`strict_missing_primary=True`** ошибки → **исключение**, не ветка с `error` в `embedding_source_id` |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Все 13 полей (finite) | **0.0** или **1.0** |
| One-hot политики (5) | **Сумма = 1.0** |
| One-hot типа primary (3) | **Сумма ∈ {0, 1}** (не две «единицы» сразу) |
| `tp_embid_present = 1` | Обычно **сумма primary one-hot = 1** (успешный путь) |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_embedding_source_id_extractor_text_npz.py`](../utils/validate_embedding_source_id_extractor_text_npz.py)

---

## 5. Чеклист

1. Срез `tp_embid_*` в `text_features.npz` = **13** имён, совпадающих с JSON.  
2. `policy_*` — строгий one-hot.  
3. `vector_id` / тексты смотреть в **`embedding_source_id`** (не в NPZ-скалярах).
