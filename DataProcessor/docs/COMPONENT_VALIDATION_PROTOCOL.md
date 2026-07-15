# Протокол валидации компонента (универсальный порядок шагов)

Единый повторяемый процесс для КАЖДОГО компонента DataProcessor (VisualProcessor /
AudioProcessor / TextProcessor) — от прогона на видео разных длин до штамповки в
прод. Чеклист статусов и подтверждённых фич — [`COMPONENT_VALIDATION_CHECKLIST.md`](COMPONENT_VALIDATION_CHECKLIST.md).

Опирается на уже существующий тулинг (не изобретаем):
- `tools/feature_quality_audit.py` — per-(component,feature): coverage, nan_rate, out_of_range_rate, constant_like, std, **health_score**.
- `tools/batch_runs_feature_report.py` — сводная таблица фич по нескольким run (разные длины).
- `backend/scripts/e2e_validate_output_quality.py` (§0.2) — NPZ/manifest контракт, finite ratio, expected-empty.
- `tools/golden_batch_compare.py` — повторяемость (тот же ролик → близкие числа).
- `backend/scripts/e2e_run_hf_videos11.py` — прогон реальных видео из HF.
- Реестр известных логических багов: [`LOGIC_ERRORS_FOR_CLAUDE.md`](LOGIC_ERRORS_FOR_CLAUDE.md).

---

## 0. Критерий «выход пригоден» (главное)

Каждая фича относится к одному из двух назначений:

**A. Для моделей (Models)** — компонент обязан отдавать **последовательность (seq)**,
которую читает Encoder (`Models/docs/contracts/ENCODER_CONTRACT.md`):
- тип seq: **dense time-series** (по времени/индексам), **sparse events** (список событий во времени) или **precomputed embeddings** (по кадрам/сегментам);
- **time-axis согласован** с source-of-truth: Visual → `frames_dir/metadata.json.union_timestamps_sec` (поля `frame_indices`/`times_s`); Audio → `times_sec`/`segment_centers_sec`/`events_times_sec`;
- значения **finite** (без неожиданных NaN/inf), **не константа** на корпусе, корректные **shape/dtype/range**;
- переменная длина ролика — ок (Encoder сам приводит к fixed budget по `duration_sec`).

**B. Для аналитиков** — агрегаты/скаляры per-video (Encoder их НЕ читает; идут в
meta/table view и в baseline). Должны быть интерпретируемы, стабильны, различимы.

Компонент может отдавать и seq, и агрегаты. Валидация проверяет **обе** ветки:
seq — на пригодность Encoder'у; агрегаты — на пользу аналитику/baseline.

## 0.1 Модельная пригодность выхода (детально)

Основано на `Models/docs/ARCHITECTURE_REVIEW.md` + `ENCODER_CONTRACT.md`. При
валидации КАЖДОГО компонента отдельно проверяем и, при необходимости, **подгоняем
выход под модели** (не только «работает», а «полезен модели»):

1. **Тип token-stream явно определён** для seq-фич: `dense` (по кадрам/сегментам),
   `sparse events` (список событий с временами) или `embeddings` (per-frame/segment).
   Encoder читает именно seq; агрегаты — в meta/table (аналитику/baseline).
2. **Seq лежит в стандартном NPZ**, а не в debug-`.npy`. ⚠️ Типовой баг: `pitch` держит
   f0-контур в `meta.f0_series_npy` (вне NPZ) → модель его не видит. При штамповке
   компонент обязан отдавать нужный seq в основном артефакте с осью времени.
3. **Time-axis согласован**: Visual — `frame_indices`/`times_s` ⊆ `union_timestamps_sec`;
   Audio — `times_sec`/`segment_centers_sec`/`events_times_sec`. Не убывает, конечен.
4. **dtype/shape/range** зафиксированы в `FEATURE_DESCRIPTION` и совпадают с ожиданиями
   Encoder; вероятности — softmax-нормированы; NaN только по задокументированной политике.
5. **Разреженность/пропуски осмысленны** (modality dropout на стороне модели): компонент
   должен уметь выдавать валидный `empty`/`present_ratio`, а не падать (ср. L2/L3/L4).
6. **Стабильность и различимость** (оси 2–3): один и тот же ролик → близкие числа;
   фича не константна на корпусе (иначе бесполезна модели).
7. **Вклад в модель** (ось 4, финальный критерий): фича либо повышает метрику baseline
   в ablation, либо явно помечена как **analyst-only**. Если ни то ни другое — кандидат
   на переработку/отключение.

Итог по компоненту фиксируем в ledger чеклиста: для каждой фичи — тип (seq/agg),
назначение (model/analyst), token-stream тип, «seq в NPZ?», «вклад в baseline?».
Целевой контракт «что отдавать модели» — `Models/docs/contracts/FEATURE_TO_MODEL_CONTRACT.md`
(создаётся в рамках изменений Models; при валидации компонента сверяемся с ним).

## 1. Матрица видео (обязательный минимум)

Разные **длины** × разный **контент**. Минимум для вердикта — ~12–18 роликов:

| Длина | Кол-во | Контент (примеры для покрытия) |
|---|---|---|
| ~10 s | 2 | talking-head; динамичный экшн |
| ~30 s | 2 | много сцен/склеек; музыка |
| ~1 min | 2 | screen-recording/игра; природа |
| ~2 min | 2 | влог; концерт/сцена |
| ~4 min | 2 | обзор/лекция; спорт |
| ~8 min+ | 2 | длинный многосюжетный |

Источник: расширить `Ilialebedev/videos11` до `videosNN` с нужными длинами; прогон
через `e2e_run_hf_videos11.py`. Фиксировать `video_id`, `duration_sec`, тип контента.

## 2. Шаги валидации (по кругу)

> **Разделение труда:** прогоны выполняет **Cursor** (у Claude нет стека/GPU/видео).
> Claude пишет `RUN_SPEC.md`, Cursor исполняет и возвращает артефакты + `RUN_RESULT.md`,
> Claude анализирует и пишет отчёт. Полная модель — `docs/CLAUDE_CURSOR_COLLAB.md`.

0. **Claude: static-review + RUN_SPEC** — разбор логики по коду + задание на прогон.
1. **Профиль (вся цепочка зависимостей!)**: включить целевой компонент + **все
   компоненты, от выхода которых он зависит** — они запускаются в правильном порядке,
   один работает с выходом другого. Примеры цепочек:
   - `action_recognition` → нужен `core_object_detections` (person-треки) + Segmenter;
   - `shot_quality` → `core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks`, `cut_detection`;
   - `scene_classification` → `core_clip` + `cut_detection`.
   Segmenter обязателен всегда. Порядок задаётся `dag/component_graph.py`.
2. **Cursor: прогон** матрицы видео (§1). Сохранить `run_id` каждого + `RUN_RESULT.md`.
3. **Метрики (цифры)**:
   - `feature_quality_audit.py` → health_score, coverage, nan_rate, oor_rate, constant_like, std по каждой фиче;
   - `batch_runs_feature_report.py` → таблица «фича × длина видео» (видно деградацию по длине);
   - `e2e_validate_output_quality.py` → контракт §0.2 (shape/dtype/finite/expected-empty);
   - `golden_batch_compare.py` (повтор того же ролика) → стабильность.
4. **Fit-check**:
   - seq-фичи: есть ли seq? согласован ли time-axis с SoT? shape подходит Encoder'у (dense/sparse/emb)?
   - агрегаты: осмысленны ли для аналитика, не константа ли по корпусу?
5. **Отчёт** по шаблону (§3) — по каждому видео: числа + словесная оценка +
   рекомендации (вплоть до «переписать компонент»). Сохранять в
   `DataProcessor/docs/component_reports/<component>/<date>.md`.
6. **Сверка владельцем**: пользователь смотрит реальное видео и сверяет с выходом
   компонента → даёт вердикт (совпадает ли то, что видно, с тем, что выдал компонент).
7. **Цикл доработки** (если не ок):
   - править алгоритм компонента (можно полностью переписать — приоритет качества);
   - **подгонять Segmenter** под компонент: бюджеты кадров/сегментов
     (`Segmenter/segmenter.py: _build_default_component_budgets`, dependency alignment),
     плотность на коротких/длинных роликах (учитывая L2: min_frames по длительности);
   - повтор с шага 2, пока владельца не устроит.
8. **Штамповка** (если ок) — §4.

## 3. Шаблон отчёта (per-component)

```
# Отчёт: <component> — валидация выхода (<дата>)
Профиль: <включённые компоненты> | Segmenter policy vX | модели/спеки: <...>

## Сводка (цифры)
| video_id | длина | health_score | coverage | nan% | const-фичи | seq? | time-axis ок? | вердикт |
|---|---|---|---|---|---|---|---|---|
| ... |

## По каждому видео (словами)
### <video_id> (<длина>, <контент>)
- Что выдал компонент (ключевые числа/классы/сигналы).
- Пригодность для Models (seq/time-axis/finite) — да/нет, почему.
- Пригодность для аналитика (агрегаты) — да/нет.
- Замеченные проблемы (ссылка на L-баг, если релевантно).
- **Рекомендации**: правки алгоритма / Segmenter / порогов / переписать.

## Итог по компоненту
- Ось 1 корректность / 2 стабильность / 3 различимость / 4 предсказательная ценность — оценка.
- Решение: [дорабатывать | штамповать vN].
- Что подогнать в Segmenter под этот компонент.
```

## 4. Штамповка компонента (Definition of Done)

Компонент считается прод-готовым, когда:
1. Все 4 оси качества ок на матрице §1 (health_score высокий, nan/const под контролем, стабильность подтверждена).
2. seq-фичи проходят Encoder-контракт (§0.A); агрегаты полезны аналитику (§0.B) — **подтверждено владельцем**.
3. Есть `FEATURE_DESCRIPTION.md` (все выходные поля, NaN-политика, диапазоны) + валидатор `utils/validate_*` + строки в `view_csv_feature_qa.json`.
4. Проставлена **версия** компонента (`producer_version`/schema) и записана в чеклист.
5. Документация приведена к **единому шаблону** (§5).
6. Зафиксирована **Segmenter sampling policy** под компонент (версия политики).
7. Все фичи компонента отмечены в feature-ledger чеклиста (model/analyst, confirmed).

## 5. Единый шаблон документации компонента

`<component>/README.md`:
```
## Component: <name>  (v<X>, schema <name_vN>, status: prod-ready)
### Назначение (1-2 предложения)
### Выходные фичи
| фича | тип (seq/aggregate) | назначение (model/analyst) | shape/dtype | диапазон | NaN-политика |
### Зависимости (core-провайдеры, Segmenter budget, модели/спеки)
### Segmenter sampling (политика + версия)
### Контракт NPZ (schema) + валидатор
### Известные ограничения
### Changelog (версии)
```

## 6. Быстрый старт для одного компонента

```bash
# 1) прогнать матрицу видео (пример: 5 шт из videos11, расширить по §1)
python backend/scripts/e2e_run_hf_videos11.py --count 5     # + свои длинные ролики
# 2) собрать фичи в таблицу
DataProcessor/.data_venv/bin/python DataProcessor/tools/batch_runs_feature_report.py \
  --run-glob 'storage/result_store/youtube/*/*' --output-csv /tmp/<component>_batch.csv
# 3) health по фичам
DataProcessor/.data_venv/bin/python DataProcessor/tools/feature_quality_audit.py \
  --csv /tmp/<component>_batch.csv --out-md /tmp/<component>_health.md
# 4) контракт §0.2
python backend/scripts/e2e_validate_output_quality.py --latest-e2e-artifact
# 5) свести в отчёт (§3), отдать владельцу на сверку
```

## 7. Контракты между компонентами (вход/выход + валидация)

Многие компоненты работают с выходом других (Segmenter → core_* → modules/heads).
По мере валидации фиксируем **межкомпонентный контракт** в реестре
[`COMPONENT_CONTRACTS.md`](COMPONENT_CONTRACTS.md): для каждой связи
`producer → consumer` — какой артефакт/поля читаются, shape/dtype, обязательность,
поведение при отсутствии (empty vs error) и **валидация на входе consumer'а**
(fail-fast с понятным `empty_reason`, а не падение — ср. L2/L3).

Правило штамповки: у прод-готового компонента вход из зависимостей **валидируется**,
а выход соответствует зафиксированному контракту для downstream-потребителей и для
Encoder (§0.1).
