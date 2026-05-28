# Audit v4 — `comments_embedder` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store` (см. §2.1).  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **18** ключей `tp_commentsemb_*` — контракт [`comments_embedder_output_v1`](../../../../TextProcessor/schemas/comments_embedder_output_v1.json). Артефакт: `text_processor/_artifacts/comments_embeddings.npy`.  
**Статистика L2 (инструмент):** `storage/audit_v4/comments_embedder_l2/comments_embedder_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/comments_embedder/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/comments_embedder_engineering_log_v4_2.md`](../audit_4_2/text_processor/comments_embedder_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/comments_embedder/SCHEMA.md); downstream: [`comments_aggregator`](comments_aggregator_audit_v4.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + `.npy` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **5** комментариев, **D=1024** |
| **B** | ✗ | Другие N, лимиты chars, CUDA vs CPU |
| **C** | ✗ | Нет текста, `compute_embeddings=false`, cache path |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | 18 имён, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN осознанно | ✓ | При **`emit_extra_metrics=False`** (дефолт в ctor) `_finalize_commentsemb_features_flat` принудительно ставит **NaN** во всех ключах из `_COMMENTSEMB_EXTRA_KEYS` (10 шт.), включая **`tp_commentsemb_artifact_written`**, тайминги, флаги cache/device/fp16/digest — см. [`main.py`](../../../../TextProcessor/src/extractors/comments_embedder/main.py) |
| Ядро без NaN на **A** | ✓ | `present`, `count`, `dim`, `n_*`, `total_chars_used`, `truncated_*` — конечны |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Наблюдения → выводы | ✓ | §5–6 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура | ✗ | TODO |

#### §5.3 — Сверка с Models

| Вопрос | Ответ |
|--------|--------|
| Модель | **intfloat/multilingual-e5-large** через `dp_models` / `get_model_with_meta` (как в коде) |
| Baseline tabular v1 | Нет прямого перечисления `tp_commentsemb_*` в [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация **`comments_embedder`:** **1.3.0** ([`main.py`](../../../../TextProcessor/src/extractors/comments_embedder/main.py)). На уровне `text_features.npz` — как у других TextProcessor шагов: `meta.schema_version` = `text_npz_v1`.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/comments_embedder/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/comments_embedder_l2/comments_embedder_audit_v4_stats.json`) берёт 5 путей A+B (как у Visual L2) и выделяет только `tp_commentsemb_*`, плюс проверяет наличие `comments_embeddings.npy`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный срез (**18** ключей); **3** файла `meta.status=error` и не содержат табличного слоя (пустой `feature_names`), артефакт `.npy` отсутствует.

Причина блокировки — сбой всего `text_processor` (часто OOM в эмбеддерах до выполнения `comments_embedder`), а не логика `comments_embedder`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Табличный срез

| Ключ | Значение |
|------|----------|
| `tp_commentsemb_present` | **1** |
| `tp_commentsemb_count` | **5** |
| `tp_commentsemb_dim` | **1024** |
| `tp_commentsemb_n_input` / `n_deduped` / `n_selected` | **5** / **5** / **5** |
| `tp_commentsemb_total_chars_used` | **113** |
| `tp_commentsemb_truncated_by_total_chars_flag` | **0** |
| `_COMMENTSEMB_EXTRA_KEYS` (10 шт.) | **все NaN** при дефолтном **`emit_extra_metrics=False`** |

Сверка множеств имён JSON ↔ NPZ: **расхождений нет**.

### 2.2 Матрица `comments_embeddings.npy`

| Проверка | Результат |
|----------|-----------|
| Форма | **(5, 1024)** `float32` |
| Конечность | да |
| L2 по строкам | **≈ 1** (L2-normalized) |

### 2.3 Согласованность «NPZ vs диск»

Файл **`comments_embeddings.npy`** на диске **есть**, но **`tp_commentsemb_artifact_written`** в NPZ = **NaN** из‑за gating extras. Потребителю нельзя опираться только на этот скаляр при дефолтном конфиге — смотреть **`manifest`**, наличие `.npy`, или флаги **`comments_aggregator`** (`tp_commentsagg_artifact_*`).

### 2.4 Отладка

Отдельного `*_report.html` для `comments_embedder` в этом run **нет**; скалярные поля попадают в агрегированный **`render_context.json`** под общим контекстом TextProcessor.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **18** ключей со схемой.
- Эмбеддинги численно аккуратны (L2, конечность).
- Отбор/дедуп отражён в `n_input` / `n_deduped` / `n_selected`.

**Минусы / внимание**

- **Gating `emit_extra_metrics`:** обнуляет в NaN не только тайминги/cache, но и **`artifact_written`**, **device/fp16/digest** — для аналитики без включения флага **потеря информации** относительно факта записи артефакта.
- Machine schema описывает NaN прежде всего для `cache_hit`; фактическое поведение шире — стоит держать в **[`SCHEMA.md`](../../../../TextProcessor/src/extractors/comments_embedder/SCHEMA.md)** / README в синхроне (вне объёма L1-задачи, но зафиксировано здесь).

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ артефакт | **9** | Ключи и матрица согласованы |
| Понятность для downstream | **7** | NaN-гейт на `artifact_written` и диагностике |
| Численное качество | **9** | L2, формы |
| Edge coverage | **6** | Один успешный путь на **A** |

**Итог L1: ~7.8 / 10** (до докрутки документации gating и прогонов **B/C**).
