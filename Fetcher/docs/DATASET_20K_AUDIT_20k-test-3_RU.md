# Аудит прогона `20k-test-3`

Документ фиксирует результаты полного теста сбора на Google Colab (июнь 2026).  
Артефакты: `Fetcher/20k-test-3/` (копия с Colab: `/content/TrendFlowML/dataset_runs/20k-test-3`).

**Окно прогона:** 2026-06-02 **11:52 → 13:37 UTC** (~1 ч 45 мин).

---

## Итог (executive summary)

| Этап | Результат | Оценка |
|------|-----------|--------|
| Discover + metadata | 20 016 видео, 18/18 категорий, 89 shards на HF | **9.5/10** |
| Балансировка / качество метадаты | Age buckets, views/likes, 15.6k каналов | **8/10** |
| Download | 68 / 20 016 на HF (0.34%), bot-detection | **2/10** |
| Enrich | ~580 в `Avto_i_transport`, 577 на HF; пайплайн OK | **7/10** (частичный прогон) |
| Snapshots 7d/14d/21d | Не запускались | — |

**Вердикт:** тест **успешен** как проверка discover + HF + enrich. Как **полный video dataset 20k** — **не готов** без прокси/масштабирования download.

**Главный вывод:** метадата для popularity-модели (**views/likes в `snapshot_0`**) уже пригодна на HF; видео — отдельный этап с прокси и multi-Colab download.

---

## 1. Discover

| Метрика | Значение |
|---------|----------|
| Accepted | **20 016** |
| Rejected | **11 877** |
| Категорий | **18** × **1 112** видео (`collect_count`) |
| Цель на категорию | 1 000 → факт **111.2%** (задуманный over-collect) |
| Metadata shards | **89** |
| Shards на HF | **89/89** |
| Уникальных каналов | **15 632** (max 37 видео на канал, **0.18%** share) |

### `time_interval` (гибрид natural/balanced)

| Bucket | Доля |
|--------|------|
| less-1day | 38.0% |
| 1day-1week | 37.1% |
| 1week-1month | 8.0% |
| 1month-3month | 6.0% |
| прочие | ~11% |

Согласуется с весами кампании (~75% «свежих» видео).

### Причины reject

| Причина | Кол-во |
|---------|--------|
| `balancer_language` | 4 587 |
| `duplicate_seen` | 4 059 |
| `duration_above_max` | 3 141 |
| `duration_below_min` | 90 |

**Сложные категории:** `Igry` (1 840 reject), `Kino_i_serialy` (986).

### Метадата для popularity

- 100% записей с `snapshot_0`
- `viewCount` / `likeCount` у всех 20 016 (camelCase в `snapshot_0`)
- median views ≈ **950**, median likes ≈ **10**, median duration ≈ **47 s**
- Языки: ru ~47%, en/en-US ~29%

---

## 2. YouTube API keys

| Метрика | Значение |
|---------|----------|
| Суммарно `used_units` | ~**188 064** |
| Активных ключей с нагрузкой | ~**32** |
| Suspended к концу сессии | **17** |

Discover исчерпал пул; для следующего полного прогона нужны дополнительные ключи и ротация.

---

## 3. Download

| Метрика | Значение |
|---------|----------|
| В очереди | 20 016 |
| Успешно на HF | **68** (только `Avto_i_transport`) |
| Локальных mp4 | 0 (удалены после HF upload) |
| Ошибки в `queue_failures` | 4 830 записей, 2 584 уникальных video |
| Backend | 68 × pytubefix (~4.3 GB) |

**Хронология:** 11:55–12:35 — пачка 68 OK; далее массовый `download returned no local file` / bot-detection.  
**Успех:** **0.34%** — для production нужны прокси `download_only`, `yt_dlp_first`, отдельные Colab под download.

HF: **7** commits в `dataset_20k_colab_videos`.

---

## 4. Enrich (остановлен досрочно)

| Метрика | Значение |
|---------|----------|
| Локально enriched | ~530 (`Avto_i_transport`) |
| На HF enrich | **577** |
| Performance events enrich | 586 |
| Lag enrich (inventory) | ~19 437 |

Пайплайн enrich (yt-dlp, captions, thumbnails) **рабочий**; масштаб не догнан по всем категориям.

---

## 5. HF и lifecycle

| Репозиторий | Commits (прибл.) |
|-------------|------------------|
| `dataset_20k_colab_shards` | 28 |
| `dataset_20k_colab_videos` | 7 |
| `dataset_20k_colab_enrich` | 90 |

| Lifecycle | Значение |
|-----------|----------|
| `training_ready_snapshot0` | 20 016 |
| `training_ready_14_21` | 0 |
| Snapshots выполнено | 0 |

---

## 6. Scorecard

```
Discover/metadata  ████████████████████  95%
HF shards upload   ████████████████████ 100%
Channel diversity  ████████████████░░░░  80%
Balancer/intervals ████████████████████  90%
Download scale     █░░░░░░░░░░░░░░░░░░░   3%
Enrich pipeline    ██████████████░░░░░░  70% (partial)
Snapshots          ░░░░░░░░░░░░░░░░░░░░   0%
```

---

## 7. Что делать дальше

### A. Popularity только по metadata

Обучение/валидация на 20k + `snapshot_0` без mp4. Snapshots 14d/21d — отдельно по расписанию.

### B. Нужны видео

1. Прокси `download_only` в campaign  
2. `download_backend: "yt_dlp_first"`  
3. Multi-Colab: `workers-download` + `worker_shard_index` / `worker_shard_count`  
4. Discover и download на **разных** Colab (IP)  
5. Resume очереди download (68 уже на HF)

### C. Enrich

`workers-enrich` на 2–3 Colab или один overnight; HF coord включён (`hf_coord_enabled`).

### D. Следующий discover

Заменить suspended keys; планировать **40+** живых ключей на 20k с запасом.

### E. Snapshots

`--role snapshot --snapshot-index 1` через 7d от `session_started_at`.

---

## 8. Ссылки

- Конфиг прогона: `20k-test-3/runtime_dataset_campaign_20k.json`
- Inventory: `20k-test-3/state/inventory/summary.json`
- Multi-Colab runbook: [COLAB_20K_RUN_RU.md](./COLAB_20K_RUN_RU.md) §3a, §9
- Grafana coord: [COLAB_20K_RUN_RU.md](./COLAB_20K_RUN_RU.md) §8b, дашборд `Dataset Collector — Coord Sync`

---

*Аудит сформирован по локальной копии `Fetcher/20k-test-3` после завершения теста.*
