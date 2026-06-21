# `content_domain` — описание фич и трассировка артефактов (Audit v4)

**Компонент:** `content_domain` (semantic head поверх `core_clip`)  
**NPZ:** `content_domain/content_domain.npz`  
**Schema:** `content_domain_npz_v2` (`SCHEMA.md`, `VisualProcessor/schemas/content_domain_npz_v2.json`)  
**Код:** `main.py` (запись), `render.py` (render-context / HTML)

Сверка с пользовательским обзором: `README.md`.

---

## 1. Назначение

По **сэмплированным кадрам** из `core_clip.frame_indices` и эмбеддингам `core_clip/embeddings.npz` строится **косинусное сходство** эмбеддинга кадра с CLIP text-эмбеддингами промптов доменов (офлайн БД `domains.jsonl`). Пишется per-frame **top‑K** (контракт **K=5**) и один **video-level** псевдотрек (`track_ids=[0]`) с max-over-time по доменам.

Пороги **`confidence_threshold_top1`** / per-label **не отрезают** top‑K; они только заполняют флаги `*_is_confident_top1`.

---

## 2. Источник данных в коде

| Этап | Что происходит |
|------|----------------|
| `metadata.json` | `frame_indices` (как у `core_clip`), `union_timestamps_sec` → `times_s = uts[frame_indices]` |
| `core_clip/embeddings.npz` | строки эмбеддингов по индексам кадров; L2-норма перед матмулом |
| Domain DB | `label_ids`, имена, промпты; `threshold_per_label`, `db_*` в meta |
| Triton | text encoder по spec `clip_text_model_spec` → матрица текстовых эмбеддингов **(A, D)** |
| Similarity | `sims = frame_emb @ label_emb.T` → **cosine** в [-1, 1] при нормированных векторах |

---

## 3. Тензоры NPZ (tabular / model_facing)

| Ключ | dtype / shape | Смысл |
|------|----------------|-------|
| `frame_indices` | int32 `(N)` | Копия запрошенных индексов кадров (= группа сэмплирования `core_clip`) |
| `times_s` | float32 `(N)` | Время в секундах по `union_timestamps_sec` |
| `semantic_label_names` | str `(A)` | Строки вида `"id:name"`; id стабильны в рамках `db_digest` |
| `threshold_per_label_arr` | float32 `(A)` | Порог уверенности по метке; **NaN**, если для id нет явного порога (тогда в логике top1 — `threshold_global`) |
| `frame_topk_ids` | int32 `(N, 5)` | id доменов top‑K (**-1** в «хвосте», если **A < 5**) |
| `frame_topk_scores` | float32 `(N, 5)` | Косинусы; **NaN** в неиспользуемых слотах K |
| `frame_is_confident_top1` | bool `(N)` | top1_score ≥ порога для top1 id |
| `track_ids` | int32 `(1)` | Всегда `[0]` |
| `track_present_mask` | bool `(1)` | `[True]` при успешном прогоне |
| `track_topk_ids` | int32 `(1, 5)` | Видео-агрегат: top‑K по **max за кадром** по каждому домену |
| `track_topk_scores` | float32 `(1, 5)` | Соответствующие max-cosine |
| `track_is_confident_top1` | bool `(1)` | Уверенность для top1 трека |
| `meta` | object | См. §4 |
| `meta_json` | str | Тот же meta, JSON string (кросс-venv) |

---

## 4. Meta (плоское в CSV: префикс `meta_`)

Пишется в `main.py` (`meta_out`). В **wide batch/melt** попадает только то, что даёт `flatten_meta` в `qa/component_feature_qa.py`: скаляры, короткие строки; `stage_timings_ms` → **`meta_timing_<stage>`** в **секундах** в JSON meta, но при флаттене тайминги в мс — **`meta_timing_*`** (см. ниже — в коде `stage_timings_ms` значения в **мс**).

Уточнение: в `main.py` `stage_timings_ms` задаётся как `float(value) * 1000.0` от секунд wall — итоговые ключи: `initialization`, `load_deps`, `process_frames`, `saving`, `total` — **в миллисекундах**.

После `flatten_meta` с префиксом `meta_` и вложенного `stage_timings_ms` имена колонок: **`meta_timing_initialization`**, **`meta_timing_load_deps`**, … — значения **уже в мс** (как float в CSV).

Частые поля:

| Поле в meta | Колонка CSV (типично) |
|-------------|------------------------|
| `status` | `meta_status` |
| `schema_version` | `meta_schema_version` |
| `producer_version` | `meta_producer_version` |
| `top_k` | `meta_top_k` (всегда **5**) |
| `confidence_threshold_top1` | `meta_confidence_threshold_top1` |
| `threshold_global` | `meta_threshold_global` |
| `clip_text_model_spec` | `meta_clip_text_model_spec` |
| `domain_db_dir` | `meta_domain_db_dir` |
| `db_name`, `db_version`, `db_digest` | `meta_db_name`, … |
| `core_clip_model_signature` | `meta_core_clip_model_signature` |
| `model_signature` | `meta_model_signature` |
| `stage_timings_ms` | `meta_timing_*` (мс) |

`threshold_per_label` (dict) и длинные строки **могут не попасть** в плоский CSV — смотрите `meta` в NPZ / `meta_json`.

---

## 5. Сводка для melt / `--melt-interesting`

Конфиг: `view_csv_melt_interesting.json` → блок `content_domain` (сейчас: `meta_confidence_threshold_top1`, `meta_status`, `meta_threshold_global`, `meta_top_k` + все `meta_timing_*`).

Рекомендуемые **QA-диапазоны** (подсветка `--melt-qa`): `storage/result_store/view_csv_feature_qa.json` → секция `content_domain`.

---

## 6. Проверка артефакта

```bash
# из корня DataProcessor, с установленным numpy
python3 VisualProcessor/core/model_process/core_identity/content_domain/utils/validate_content_domain.py /path/to/content_domain.npz --struct
python3 .../validate_content_domain.py /path/to/content_domain.npz --qa
```

`--struct`: ключи, согласованность N/K/A, косинусы в [-1,1], topk ids в допустимом множестве label id.  
`--qa`: правила из `view_csv_feature_qa.json` для компонента `content_domain`.

---

## 7. Согласованность с README

`README.md` описывает интеграцию, Triton, batch — табличные поля и meta должны **совпадать** с перечислением выше; при расхождении **источник истины** — `main.py` + `SCHEMA.md`.
---

## Навигация

[SCHEMA](SCHEMA.md) · [Module README](../README.md) · [VisualProcessor](../../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
