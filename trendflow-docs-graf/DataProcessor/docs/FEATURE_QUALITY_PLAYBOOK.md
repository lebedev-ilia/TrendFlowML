# Руководство: оценка качества фич и подготовка к обучению на 100k видео

Дата: 2026-04-23  
Цель: дать практичный, воспроизводимый процесс проверки качества алгоритмов (качества фич) на сотнях видео и подготовки стабильного датасета для модели прогнозирования популярности.

---

## 1) Что считаем «качеством фич»

Качество оцениваем по 4 осям:

1. **Корректность** — фича вычислена по контракту (shape/dtype/range), без неожиданных NaN/inf.
2. **Стабильность** — повторный прогон на том же видео дает близкие значения.
3. **Различимость** — фича не константна на корпусе, имеет полезную вариативность.
4. **Предсказательная ценность** — фича улучшает качество модели на offline-валидации.

Если фича проваливается по любой оси — это кандидат на доработку/отключение.

---

## 2) Стандартный pipeline проверки (до обучения)

### Шаг A. Собрать batch-таблицу по run-ам

Использовать:

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/batch_runs_feature_report.py \
  --run-glob "/abs/path/to/storage/result_store/youtube/*/*" \
  --max-runs 20 \
  --output-csv "/abs/path/to/storage/result_store/batch_features_report_20runs.csv"
```

Если нужно по заранее выбранным run:

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/batch_runs_feature_report.py \
  --run-dir "/.../youtube/<video_id>/<run_id>" \
  --run-dir "/.../youtube/<video_id>/<run_id>" \
  --output-csv "/.../batch_features_report_selected.csv"
```

### Шаг B. Построить melt HTML для анализа

```bash
DataProcessor/.data_venv/bin/python storage/result_store/view_csv.py \
  --csv "/abs/path/to/storage/result_store/batch_features_report_20runs.csv" \
  --melt --melt-interesting --melt-qa \
  --out "/abs/path/to/storage/result_store/batch_features_report_20runs.melt.interesting.qa.view.html" \
  --no-open
```

Результат:
- колонка `component`
- колонка `feature`
- колонка `пояснение`
- колонка `норма` (если есть QA-правило)
- значения по видео/run
- QA-подсветка ячеек вне `min/max/enum`.

### Шаг C. Прогнать per-component валидаторы

Для каждого процессора:
- TextProcessor: `validate_*_text_npz.py`
- AudioProcessor / VisualProcessor: `utils/validate_*.py` / `validate_*_npz.py`

Минимум:
- `--struct` (контракт)
- `--ranges` (диапазоны)
- `--timings` (если поддерживается)

Это источник истины по контракту фич.

---

## 3) Чек-лист «готово / не готово» для компонента

Компонент считается готовым к масштабному обучению, если:

1. Есть `docs/FEATURE_DESCRIPTION.md` с описанием всех выходных полей.
2. Есть валидатор с диапазонами и проверкой структуры.
3. Есть русские пояснения в `storage/result_store/view_csv_feature_descriptions_ru.json`.
4. Для ключевых полей есть QA-правила в `storage/result_store/view_csv_feature_qa.json` (рекомендуется обязательно для критичных фич).
5. На батче не наблюдается аномально высокий NaN-rate/константность.

---

## 4) Какие метрики считать автоматически по каждой фиче

Для каждого `component + feature`:

- `coverage`: доля непустых значений
- `nan_rate`, `inf_rate`
- `n_unique`
- `mean`, `std`, `p01`, `p50`, `p99`
- `out_of_range_rate` (по QA rules)
- `drift` между батчами (KS + эвристический score; см. `DataProcessor/tools/feature_batch_drift.py`)

Рекомендуемая сводка:

```text
feature_health_score = 100
  - penalty_nan
  - penalty_out_of_range
  - penalty_constant
  - penalty_drift
```

Смотреть вручную только топ проблемных фич.

---

## 5) Как валидировать «стабильность» (reproducibility)

Для 50-200 «golden» видео:

1. Прогнать пайплайн 2 раза на одном и том же коде/конфиге.
2. Сравнить:
   - точное совпадение для deterministic фич,
   - допуск (`abs diff` / `rel diff`) для плавающих.
3. Ввести пороги fail:
   - >1% фич отличаются выше допуска,
   - или отличие в критичных полях.

Это защищает от скрытых регрессий перед 100k.

### Инструмент: `golden_batch_compare.py`

`DataProcessor/tools/golden_batch_compare.py` сопоставляет строки по `(platform_id, video_id, run_id, component)` и сравнивает все прочие колонки с допусками `--abs-eps` / `--rel-eps`. Выход: CSV расхождений + Markdown; код выхода **2**, если есть несовпадения (удобно для CI).

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/golden_batch_compare.py \
  --csv-a "/abs/path/batch_run_A.csv" \
  --csv-b "/abs/path/batch_run_B.csv" \
  --out-csv "/abs/path/golden_mismatches.csv" \
  --out-md "/abs/path/golden_mismatches.md"
```

---

## 6) Как оценивать полезность фич для задачи популярности

### 6.1 Baseline + Ablation

Построить минимум 3 режима:

1. `baseline`: только простые metadata (без тяжелых сигналов).
2. `baseline + all_features`.
3. `all_features - component_X` (ablation по компонентам).

Сравнивать по основной offline-метрике (зависит от постановки):
- regression: MAE / RMSE / Spearman
- ranking: NDCG / pairwise AUC
- classification: PR-AUC / ROC-AUC + calibration

### 6.2 Leakage-safe split

Для популярности обязательно:
- split по времени публикации (time-based),
- запрет фич, зависящих от будущих данных.

**Подготовка списка колонок:** `DataProcessor/tools/wide_batch_feature_manifest.py` — один проход по wide `batch_features_report*.csv`, для каждой колонки: `nonempty_rate`, `numeric_rate`, роль (`id` / `meta` / `feature`), эвристический `leakage_hint` по имени (не замена аудита; ручной denylist таргета и post-hoc полей обязателен).

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/wide_batch_feature_manifest.py \
  --csv "/abs/path/batch_features_report_20runs.csv" \
  --out-json "/abs/path/wide_batch_feature_manifest.json" \
  --out-csv "/abs/path/wide_batch_feature_manifest.csv"
```

**Сборка матрицы для обучения (join с target):** `DataProcessor/tools/build_training_matrix.py`.

- схлопывает wide-таблицу до одной строки на `(platform_id, video_id, run_id)` (берёт first non-empty по колонке),
- присоединяет target-таблицу по ключу (`video_id` по умолчанию),
- оставляет числовые признаки с `numeric_rate >= --min-numeric-rate`,
- применяет denylist из `wide_batch_feature_manifest.json` (`leakage_hint`) и пользовательские regex,
- опционально делает time-based split по дате из target CSV.

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/build_training_matrix.py \
  --batch-csv "/abs/path/batch_features_report_20runs.csv" \
  --target-csv "/abs/path/targets.csv" \
  --target-value-col "target_popularity_14d" \
  --target-time-col "published_at" \
  --manifest-json "/abs/path/wide_batch_feature_manifest.json" \
  --out-csv "/abs/path/training_matrix.csv" \
  --out-metadata-json "/abs/path/training_matrix.meta.json" \
  --out-train-csv "/abs/path/training_matrix.train.csv" \
  --out-val-csv "/abs/path/training_matrix.val.csv" \
  --out-test-csv "/abs/path/training_matrix.test.csv"
```

---

## 7) Рекомендуемый план запуска (этапы)

### Этап 1: QA-пилот (1k видео)
- цель: качество фич, не качество модели.
- выход: список проблемных компонентов и фикс-план.

### Этап 2: Model-пилот (10k видео)
- цель: оценить прирост от мультимодальных фич.
- выход: shortlist полезных компонент и фич.

### Этап 3: Pre-production (30k+ видео)
- цель: проверить drift, стоимость, стабильность.
- выход: freeze контрактов, финальный feature-set.

### Этап 4: Full run (100k)
- мониторинг по батчам:
  - NaN-rate
  - out_of_range_rate
  - missing component rate
  - inference cost per component

---

## 8) Практика ведения «реестра проблем»

Для каждой аномалии фиксировать:

- `component`
- `feature`
- тип (`nan_spike`, `out_of_range`, `drift`, `constant`)
- affected runs/videos
- причина
- статус (`open`, `fixed`, `accepted`)

Рекомендуется хранить в отдельном CSV/JSON и обновлять автоматически после каждого batch-прогона.

**Автоматизация:** `DataProcessor/tools/feature_incident_registry.py` — сливает `feature_quality_report*.csv` и (опционально) `feature_batch_drift*.csv` в `storage/result_store/feature_incidents.json`. Поля `reason` и `status` (`fixed` / `accepted`) правятся вручную в JSON; повторные прогоны обновляют `last_seen_batch` и `metrics_snapshot`. Дрейф по умолчанию берётся только при `severity=high` в отчёте дрейфа (см. `--drift-min-severity`). Константные фичи (`type=constant`) — только с флагом `--include-constant`.

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/feature_incident_registry.py \
  --batch-label "2026W17_20runs" \
  --quality-csv "/abs/path/feature_quality_report_20runs.csv" \
  --drift-csv "/abs/path/feature_batch_drift_week0_1.csv"
```

---

## 9) Минимальный еженедельный ритуал QA

0. **Опционально один вызов:** `DataProcessor/tools/feature_qa_pipeline.py` — последовательно запускает `feature_quality_audit`, опционально `feature_batch_drift` (`--baseline-csv`), опционально **`golden_batch_compare`** (`--golden-compare-csv` = эталонный второй прогон тех же run, см. §5), опционально **`run_text_extractor_validators.py`**, `view_csv.py` (melt+interesting+qa), `feature_incident_registry` и пишет `feature_shortlist_*.csv`. По умолчанию артефакты складываются в `storage/result_store/qa_runs/<label>/`, HTML — в `.../html/`, реестр инцидентов — в `storage/result_store/incidents/feature_incidents.json`. Сводка: `feature_qa_pipeline_<label>.summary.json`.

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/feature_qa_pipeline.py \
  --batch-csv "/abs/path/batch_features_report_20runs.csv" \
  --batch-label "2026W17_20runs" \
  --baseline-csv "/abs/path/batch_features_report_week0.csv" \
  --golden-compare-csv "/abs/path/batch_same_runs_rerun.csv" \
  --run-text-validators-from-batch
```

1. Сборка свежего batch CSV (`N=20..100` run).
2. Генерация melt QA HTML.
3. Авто-отчет по health-метрикам.
4. Ручной просмотр top-аномалий.
5. Обновление:
   - `FEATURE_DESCRIPTION.md`
   - `validate_*.py`
   - `view_csv_feature_descriptions_ru.json`
   - `view_csv_feature_qa.json`
6. При наличии свежих `feature_quality_report*.csv` / `feature_batch_drift*.csv` — прогон `feature_incident_registry.py` (или уже выполнено п.0 через `feature_qa_pipeline.py`).

---

## 10) Где в проекте что лежит

- Сводка run×component: `DataProcessor/tools/batch_runs_feature_report.py`
- Агрегатор валидаторов TextProcessor: `DataProcessor/tools/run_text_extractor_validators.py`
- Дрейф фич между батчами: `DataProcessor/tools/feature_batch_drift.py`
- Реестр инцидентов: `DataProcessor/tools/feature_incident_registry.py` → `storage/result_store/feature_incidents.json`
- Golden-сравнение двух batch CSV: `DataProcessor/tools/golden_batch_compare.py`
- Еженедельный пайплайн QA (всё в одном): `DataProcessor/tools/feature_qa_pipeline.py`
- Манифест колонок wide CSV перед обучением: `DataProcessor/tools/wide_batch_feature_manifest.py`
- Сборка train-матрицы и time split: `DataProcessor/tools/build_training_matrix.py`
- HTML просмотрщик: `storage/result_store/view_csv.py`
- QA-правила HTML: `storage/result_store/view_csv_feature_qa.json`
- Русские описания фич: `storage/result_store/view_csv_feature_descriptions_ru.json`
- Структура папок result_store: `storage/result_store/STRUCTURE.md`
- Локальные валидаторы: `*/utils/validate_*.py`
- Контрактные docs по компонентам: `*/docs/FEATURE_DESCRIPTION.md`

---

## 11) Критерий готовности к 100k

Перед полным прогоном должны выполняться:

1. Нет критичных (`severity=high`) проблем по структуре/диапазонам.
2. На golden set стабильность в пределах допусков.
3. Offline-модель показывает устойчивый прирост к baseline.
4. Стоимость (время/ресурсы) приемлема по SLA.

Если любой пункт не выполнен — продолжать итерации QA/feature selection.

---

## 12) AI Autopilot Checklist (RU/EN)

### RU: что можно делать прямо сейчас без участия человека

Ниже — чек-лист автоматических действий, которые агент/скрипты могут выполнять вручную по команде (без доп. решений владельца):

0. **Пайплайн одной командой:** `feature_qa_pipeline.py` (аудит → дрейф при эталоне → HTML → реестр → shortlist).
1. Собирать свежий `batch_features_report_*.csv` по run-ам (`batch_runs_feature_report.py`).
2. Строить `melt + interesting + qa` HTML (`view_csv.py`) для визуального контроля.
3. Запускать локальные валидаторы компонентов (`validate_*.py`) и сохранять агрегированный лог. Для **всех** экстракторов TextProcessor одной командой: `DataProcessor/tools/run_text_extractor_validators.py <path/to/text_features.npz> --out-json ... --out-md ...` (или `--results-base` для обхода всего `result_store`).
4. Считать health-метрики фич и выдавать top-проблемы:
   - NaN spikes
   - out-of-range
   - low coverage
   - constant-like features
5. Генерировать 3 артефакта отчёта:
   - JSON (машиночитаемый)
   - CSV (плоская таблица per feature)
   - Markdown (быстрый обзор проблем)
6. Обновлять операционный отчёт в `storage/result_store/*feature_quality*`.
7. Сравнивать текущий батч с эталонным через `feature_batch_drift.py` и сохранять `feature_batch_drift_*.md`.
8. Готовить shortlist фич к отключению/понижению приоритета (без автоприменения).
9. Обновлять `feature_incidents.json` (`feature_incident_registry.py` или шаг внутри `feature_qa_pipeline.py`).
10. Перед матрицей признаков для модели — `wide_batch_feature_manifest.py` по актуальному batch CSV.
11. Собирать финальную train-матрицу (`build_training_matrix.py`) после ручной проверки leakage denylist.

### EN: immediate no-human-in-the-loop checklist

Actions the agent can run right now in manual mode:

0. Run `feature_qa_pipeline.py` for audit + optional drift + melt HTML + registry + shortlist in one go.
1. Build a fresh wide batch CSV from selected runs.
2. Build melt+interesting+qa HTML for quick inspection.
3. Run per-component validators and aggregate issues (TextProcessor: `DataProcessor/tools/run_text_extractor_validators.py`).
4. Compute feature health metrics (coverage, NaN/inf, out-of-range, constant-like).
5. Export machine-readable and human-readable reports (JSON/CSV/MD).
6. Keep report artifacts under `storage/result_store`.
7. Run `feature_batch_drift.py` vs a frozen baseline batch and archive the Markdown summary.
8. Produce a ranked shortlist of risky features/components (no automatic disabling).
9. Merge into `feature_incidents.json` (registry script or `feature_qa_pipeline.py`).
10. Run `wide_batch_feature_manifest.py` before assembling a training feature matrix.
11. Build the final training matrix with `build_training_matrix.py` after leakage denylist review.

---

## 13a) Дрейф между двумя batch CSV

Инструмент: `DataProcessor/tools/feature_batch_drift.py`.

- Вход: два wide CSV (`--csv-a` = эталон/прошлая неделя, `--csv-b` = текущий батч).
- Выход: JSON / CSV / Markdown с KS (двухвыборочный), Δ `nan_rate`, сдвиг mean/median, эвристический `drift_score` и `severity`.
- Для корректной интерпретации лучше сравнивать батчи **одинакового размера и политики сэмплинга**; разный N допустим как «разведка», но маргинали будут смешивать эффект объёма и дрейфа.

Пример:

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/feature_batch_drift.py \
  --csv-a "/abs/path/batch_features_report_week0.csv" \
  --csv-b "/abs/path/batch_features_report_week1.csv" \
  --out-json "/abs/path/feature_batch_drift_week0_1.json" \
  --out-csv "/abs/path/feature_batch_drift_week0_1.csv" \
  --out-md "/abs/path/feature_batch_drift_week0_1.md"
```

---

## 13) Новый авто-скрипт quality audit

Добавлен инструмент:

- `DataProcessor/tools/feature_quality_audit.py`

Что делает:

- читает `batch_features_report*.csv`
- использует QA-правила из `storage/result_store/view_csv_feature_qa.json` (если есть)
- считает per `(component, feature)`:
  - `coverage`
  - `nan_rate`, `inf_rate`
  - `n_unique_nonempty`
  - `mean`, `std`, `p01`, `p50`, `p99` (для числовых)
  - `out_of_range_rate` (через QA rules)
  - `constant_like`
  - `health_score` + `severity`
- пишет:
  - JSON отчёт
  - CSV отчёт
  - Markdown summary (top issues)

Пример запуска:

```bash
DataProcessor/.data_venv/bin/python DataProcessor/tools/feature_quality_audit.py \
  --csv "/abs/path/to/storage/result_store/batch_features_report_20runs.csv" \
  --out-json "/abs/path/to/storage/result_store/feature_quality_report_20runs.json" \
  --out-csv "/abs/path/to/storage/result_store/feature_quality_report_20runs.csv" \
  --out-md "/abs/path/to/storage/result_store/feature_quality_report_20runs.md"
```

---

## 14) Текущая целевая постановка (зафиксировано)

- Главный фокус: **качество фич + отбор фич**.
- Цель ближайшего этапа: **подтвердить финальный набор фич для модели**.
- Целевые горизонты популярности: **7 / 14 / 21 день**.
- Источник target: таблица на HuggingFace.
- Пороги пока soft (без жёсткого hard fail), режим запуска пока ручной.

Дополнительный стратегический roadmap текущего этапа:

- `DataProcessor/docs/MODEL_FEATURE_SET_ROADMAP.md`
---

## Навигация

[Module README](../README.md) · [DataProcessor](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
