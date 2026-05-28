# `transcript_aggregator` — описание фич и артефактов

**Компонент:** `TranscriptAggregatorExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **19** скаляров `tp_tragg_*` в `text_processor/text_features.npz` (merged, **все** ключи из схемы; **9 extra** при `emit_extra_metrics=false` — **NaN**, это норма).  
**Контракт:** [`../../../../schemas/transcript_aggregator_output_v1.json`](../../../../schemas/transcript_aggregator_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Per-run агрегаты: `transcript_{source}_agg_mean.npy` / `agg_max.npy`, `transcript_combined_agg_*.npy` в `artifacts_dir` (не в `features_flat`).

**Версия:** 1.3.0 (`TranscriptAggregatorExtractor.VERSION`).

---

## 1. Назначение

- Читает матрицы чанков (`chunk_embeddings_relpath`) из `doc.tp_artifacts`, пишет **mean** (веса `exp(-decay·i)` × optional ASR confidence) и **max-pool** по измерениям, L2 на выходе.
- Источники: `whisper`, `youtube_auto`; **combined** — `vstack` в порядке `sources`.
- `tp_tragg_present=1` если `results` непустой (хотя бы один успешный агрегат).

---

## 2. Ключи (смысл)

| Группа | Ключи |
|--------|--------|
| Присутствие | `present`, `present_whisper`, `present_youtube` (**youtube_auto**), `present_combined` — **0/1** |
| Конфиг (зеркала) | `decay_rate` (float ≥ 0), `compute_std/mean/max/combined`, `write_artifacts` — **0/1** кроме `decay_rate` |
| Extra (`emit_extra_metrics`) | `*_n_chunks` — число чанков (float); **NaN** если extra выключены |
| Std (`compute_std` + extra) | `*_mean_std`, `*_max_std` — streaming std по **элементам** векторов агрегации; **NaN** если `compute_std=false` или мало элементов |

Имя **`tp_tragg_present_youtube`** в коде = источник **`youtube_auto`** (историческое имя поля).

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Флаги `present_*`, `compute_*`, `write_artifacts` | **0/1** (finite) |
| `tp_tragg_decay_rate` | finite, **≥ 0** (типично малый, напр. 0.01) |
| `*_n_chunks` (finite) | **≥ 0** |
| `*_mean_std`, `*_max_std` (finite) | **≥ 0** |
| при `compute_std=0` | все `*_mean_std` / `*_max_std` — **NaN** (если extra включены и поля материализованы) |

---

## 4. Тайминги (не в merged `features_flat`)

В сыром результате экстрактора: `timings_s`: **`load`**, **`aggregate`**, **`total`** (секунды, ≥ 0).  
В `text_features.npz` они попадают в **`payload.timings_by_extractor["TranscriptAggregatorExtractor"]`** (privacy-safe summary). Проверка: флаг **`--timings`** в [`../utils/validate_transcript_aggregator_text_npz.py`](../utils/validate_transcript_aggregator_text_npz.py).

Ожидания: все компоненты finite, **≥ 0**; **`total`** ≥ **`load`** и ≥ **`aggregate`** (с небольшим допуском на округление).

---

## 5. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_transcript_aggregator_text_npz.py`](../utils/validate_transcript_aggregator_text_npz.py)

---

## 6. Чеклист

1. **19** имён = `transcript_aggregator_output_v1` (`allow_extra_keys: false`).  
2. Пустой прогон по чанкам: `present=0`, extra **NaN** при `emit_extra_metrics=false`.  
3. Таблица/CSV: колонки с **NaN** по extra — ожидаемы при дефолтном `emit_extra_metrics` в прогоне.
