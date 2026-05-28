# `speaker_turn_embeddings_aggregator` — описание фич и артефактов

**Компонент:** `SpeakerTurnEmbeddingsAggregatorExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **17** скаляров `tp_spkemb_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/speaker_turn_embeddings_aggregator_output_v1.json`](../../../../schemas/speaker_turn_embeddings_aggregator_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Векторы per-speaker — в **`speaker_<id>_mean.npy` / `_max.npy`** (не в `features_flat`); реестр — `doc.tp_artifacts["speakers"]["embeddings"]`.

**Версия:** 1.3.0 (`SpeakerTurnEmbeddingsAggregatorExtractor.VERSION`).

---

## 1. Назначение

- Собрать **тексты реплик по спикеру** (diarization + ASR по overlap, либо legacy `doc.speakers` по `name`/`description`).  
- Закодировать **sentence-transformer**, агрегировать **L2-нормированный** mean и/или max по осям.  
- Пять полей **`tp_spkemb_batch_size` … `tp_spkemb_max_chars_per_turn`** — **NaN** при **`emit_extra_metrics=False`** (см. `_apply_extra_metrics_spkemb`).

---

## 2. Группы

| Группа | Заметки |
|--------|---------|
| Gating | `tp_spkemb_present` — **1** только если **записан** хотя бы один артефакт (`n_saved>0`); иначе **0** (в т.ч. при «есть спикеры, но эмбеддинги пусты») |
| Счётчики | `speakers_total` — спикеров с непустым turn list после `max_speakers`; `speakers_embedded` ≤ `speakers_total`; `turns_total` — сумма длин списков реплик |
| Зеркала пайплайна | `write_artifacts`, `compute_mean`, `compute_max` — **0/1** |
| Режим | `input_present`, `input_mode_diar_asr`, `input_mode_legacy_doc_speakers`, `asr_present`, `diar_present` — в prod-пути diar+ASR оба **1**; legacy — только `input_present`+legacy **1** |
| **Extra** | 5 целочисленных тюнинг-полей либо **все NaN** |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (finite) | `present`, `write_*`, `compute_*`, 5 input-флагов |
| `speakers_total`, `speakers_embedded`, `turns_total` (finite) | **≥ 0**; `speakers_embedded` ≤ `speakers_total` |
| Режим | не оба `input_mode_*` = **1** одновременно |
| Extra (все finite, `emit_extra=True`) | `batch_size` ≥ **1**; `max_speakers` / `max_turns_per_speaker` ≥ **1**; `min_chars` ∈ **[0, max_chars]**; `max_chars` > **0** |
| Без `emit` | **все пять** extra — **NaN** (смешанные finite/NaN — неконсистентны) |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_speaker_turn_embeddings_aggregator_text_npz.py`](../utils/validate_speaker_turn_embeddings_aggregator_text_npz.py)

---

## 5. Чеклист

1. **17** имён в срезе = `speaker_turn_embeddings_aggregator_output_v1`.  
2. Extra: либо **5× NaN**, либо **5×** целевые config-значения.  
3. `present=1` ⇒ ожидается **speakers_embedded ≥ 1** (и `speakers_total ≥ 1`).
