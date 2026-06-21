# Audit v4 — `action_recognition` (VisualProcessor)

**Дата:** 2026-04-06 (обновление: 2026-04-13)  
**Уровень отчёта (план §3.1):** **L2 — product stats** (**A + B**, 5 run).  
**Артефакт (набор A, фактический в `storage/result_store`):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/action_recognition/action_recognition_features.npz`  
**Код / контракт:** `DataProcessor/VisualProcessor/modules/action_recognition/` · machine schema: [`VisualProcessor/schemas/action_recognition_npz_v2.json`](../../../../../VisualProcessor/schemas/action_recognition_npz_v2.json) · человекочитаемо: [`modules/action_recognition/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/action_recognition/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только набор **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика полей + вердикт | ✓ | Отчёт + `docs/SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты DataProcessor / Models | ◐ | VP schema + `ResultsStore`; Models — §5.3 |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| VisualProcessor (пилот модуля) | ✓ | Первый отчёт v4 для VP по той же сетке |
| Путь артефакта + `run_id` | ✓ | Шапка + [`RUN_LOG.md`](../../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | тот же e2e reference, что и волна AudioProcessor |
| **B** ≥5 видео | ✓ | 4 дополнительных run из `storage/result_store` + статистика в JSON (см. ниже); **но** все треки имеют `num_clips=1` |
| **C** edge | ✗ | `no_person_detections`, короткие клипы, мало кадров на трек |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1 draft**, не L3 | ✓ | `RUN_LOG`: `in_progress` |
| Нет заявления полного §8 | ✓ | DoD не закрыт |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `tracks`, `embeddings`, `results_json`, `meta` | ✓ | T=8 на **A**; `embeddings` — object, по одному `[num_clips, 256]` float32 на трек |
| Сверка с `action_recognition_npz_v2.json` | ✓ (после фикса) | Доп. ключи **`metric__*`** разрешены префиксом в схеме; в NPZ остаются только **скалярные** per-track метрики (остальное — в `results_json`) |
| Имя файла артефакта | ✓ (после фикса) | Единое имя **`action_recognition_features.npz`** при per-track `store_compressed`; legacy: **`action_recognition_emb.npz`** |

**Статус фикса (по текущему `manifest.json` набора A):** модуль `action_recognition` завершился `status=ok`, артефакт `action_recognition_features.npz` записан с `schema_version=action_recognition_npz_v2`, ошибок валидации схемы нет.

#### §4.1a — Семантика типов, строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Скаляры в `metric__*` | ◐ | Числовые метрики (`num_clips`, `stability`, …) на **A** согласованы с `results_json` |
| Списки/вложенные поля в `metric__*` | ✓ (после фикса) | Не-скаляры **не** сериализуются в плоские `metric__*` — остаются в `results_json` ([`results_store.py`](../../../../../VisualProcessor/utils/results_store.py)) |
| Источник правды для оси/клипов | ✓ | **`results_json[*]`**: на **A** например `clip_center_times_s: [0.0]`, `clip_center_frame_indices: [0]` — типы и значения осмысленны |

#### §4.2 — NaN, Inf, нули

| Критерий | Статус | Заметка |
|----------|--------|---------|
| В матрицах эмбеддингов | ✓ | 0% NaN/Inf на **A** (8×256 после конкатенации по клипам) |
| L2-норма строк | ✓ | **1.0** на каждом клипе (**A**) — согласуется с `embedding_normed_256d` в `SCHEMA.md` |
| «Нули» как missing в `metric__*` | ✓ (после фикса) | Столбцы `-1` из старого флэттенера для списков **убраны**; осмысленные int-метрики остаются в `metric__*` при полном наборе треков |

#### §4.3 — Распределения

| Критерий | Статус | Заметка |
|----------|--------|---------|
| p01…p99 на **B** | ✓ (JSON) | `storage/audit_v4/action_recognition_l2/action_recognition_audit_v4_stats.json` |
| На **A+B** | ◐ | Все треки: **`num_clips=1`** → `max_temporal_jump`, `mean_temporal_jump`, `stability_centroid_dist` = 0, **`stability=1`** — ожидаемо **вырождено** (нет временной оси внутри трека) |
| Эмбеддинги (конкатенация 8×256) | ✓ | `mean` по признакам в узком диапазоне; `std` по столбцам ~0.024 (качественно — не константа) |

#### §4.4 — Категориальные / object

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `results_json` | ✓ | Структурированные dict; списки центров кадров/времени — не через числовой tabular |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `clip_center_times_s` в `results_json` | ✓ | Список float **секунд** на клип (набор **A**: для каждого трека по 1 клипу) |
| `metric__clip_center_times_s` | N/A | Поля со списками/вложенными структурами **не** экспортируются в `metric__*` (живут только в `results_json`) |

#### §4.6 — Корреляции (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Избыточность scalar-метрик / столбцов эмбеддинга | ◐ | пока мало вариативности (все `num_clips=1`); базовая статистика есть в JSON |

#### §4.7 — Трактовка

| Наблюдение | Вывод |
|------------|--------|
| Падение / warning валидации схемы на `metric__*` | Контракт и фактический writer расходятся — **P0 для pipeline**, иначе «зелёный» артефакт недостоверен для CI |
| -1 в `metric__*` для списочных полей | Легко принять за реальный индекс; **не использовать** без правки — читать `results_json` |
| Все треки с одним клипом на **A** | Не проверяет алгоритмы стабильности/переходов; нужен **B** с длинными треками |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура после фикса схемы/ключей | ✗ | Сначала согласовать набор ключей NPZ и `allow_extra_keys` |

#### §4.9 — Sampling (**B**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `clip_len=32`, `stride=16`, `processed_frames=250`, `total_frames=338` | ◐ | Зафиксировано в `meta` на **A** |
| Зависимость числа клипов от stride / длины трека | ✗ | Таблица на **B** |

#### §4.10 — `empty` (**C**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `status=empty`, `empty_reason` | ✗ | На **A**: `status=ok` |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Порог | N/A | Плоских scalar в схеме мало; основной сигнал — эмбеддинг 256 |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| Только текущее видео + локальные кадры/детекции? | Да (SlowFast на клипах трека person) |
| Глобальная нормализация по датасету? | Нет на уровне модуля (нормировка L2 по клипу) |
| Онлайн API? | Нет; `models_used` локально |

#### §5 — Документация полей

| Подпункт | Статус | Заметка |
|----------|--------|---------|
| §5.1 `docs/SCHEMA.md` | ✓ | Есть таблица полей `results_json` |
| §5.2 README | ◐ | [`README.md`](../../../../../VisualProcessor/modules/action_recognition/README.md) — при необходимости явно описать **`metric__*`** как производное от `ResultsStore` и риск -1 |

##### §5.3 — Сверка с Models

| Вопрос | Ответ | Комментарий |
|--------|-------|-------------|
| Семантика для encoder | **Dense по времени:** `[num_clips, 256]` per track; агрегаты стабильности — tabular-кандидаты | См. `SCHEMA.md` (VisualTransformer / MLP) |
| Baseline Audio | N/A | Визуальный модуль |
| Риск для ingestion | Высокий, пока **machine schema ≠ фактический NPZ** | Исправить `allow_extra_keys` или состав `fields` |

#### §6 — Verdict

**Итог L1:** выход **`embeddings` + `results_json`** на reference **A** выглядит **численно здоровым** (нормы, отсутствие NaN, осмысленные времена в JSON). Основной инженерный риск из ранних прогонов (валидация `metric__*` и «-1 для списков») закрыт: в `metric__*` остаются только **скалярные** поля, а списки/массивы остаются в `results_json`.

**Оценка (условно, до C и §4.8):** **~8 / 10** — поднять до **~8.5+** после: (1) поднабора **B** с треками `num_clips > 1`; (2) набора **C** (`no_person_detections`/короткие треки); (3) `golden` / сигнатуры **§4.8** на стабильном A.

#### §8 — Definition of Done

**Не закрыт:** **B+C**, §4.6–§4.10, golden §4.8, исправление validation в manifest, commit в `RUN_LOG`.

---

## 1. Снимок **A** (числа)

| Поле | Значение на **A** |
|------|-------------------|
| `T` (tracks) | 8 |
| `num_clips` (каждый трек) | 1 (все треки на **A**) |
| Размер эмбеддинга | 256 |
| NaN/Inf в эмбеддингах | 0 |
| ‖embedding‖₂ (на клип) | 1.0 |
| `stability` | 1.0 (все треки) |
| `metric__*` (скалярные) | 8 ключей: `embedding_dim`, `max_temporal_jump`, `mean_temporal_jump`, `num_clips`, `num_switches`, `stability`, `stability_centroid_dist`, `track_frame_count` |
| Списки по клипам | Только в `results_json` (`clip_center_frame_indices`, `clip_center_times_s`, `temporal_jumps`, `clip_frame_indices`) |

## 2. Рекомендуемые направления фикса (вне scope одного отчёта)

1. **Набор B (L2):** подобрать ≥5 видео с треками, где `num_clips > 1`, чтобы проверить `temporal_jumps`, `stability_*`, `num_switches` на невырожденных последовательностях.  
2. **Edge set C:** `no_person_detections`, короткие клипы/треки, низкий confidence.  
3. **Batch-path:** `VisualProcessor/utils/action_recognition_batch.py` использует `ResultsStore.get_component_path()` (должен быть доступен в `utils/results_store.py`) — иначе batch-ветка нерабочая; держать в синхроне с общим ResultsStore API.

---

## 3. L2 stats (A+B, 5 run) — артефакт

- JSON: `storage/audit_v4/action_recognition_l2/action_recognition_audit_v4_stats.json`
- Итог по этим 5 run: **tracks_total=61**, **clips_total=61**, **tracks_with_multi_clips_total=0** (временная динамика `temporal_jumps`/`stability` не покрыта, нужен отдельный поднабор B).
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
