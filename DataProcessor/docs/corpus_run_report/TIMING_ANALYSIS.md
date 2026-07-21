# Фаза 1 — Анализ времени выполнения (300-видео прогон)

Источник: `per_video_component.csv` (2700 строк = 300×9), `per_video.csv`, `corpus300.json`.
Анализ по уже собранным данным, под не использовался.

## 1. Распределение wall-time по компонентам

| Компонент | OK | min | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| download | 300 | 3.9 | 4.8 | 5.3 | 6.2 | 7.8 | 9.1 | 12.4 | 18.7 | 5.8 |
| segmenter | 300 | 3.1 | 4.9 | 8.9 | 13.8 | 31.1 | 38.1 | 57.5 | 71.4 | 12.8 |
| core_clip | 300 | 33.7 | 57.8 | 62.9 | 67.8 | 73.2 | 76.0 | 83.7 | 98.9 | 62.8 |
| core_depth_midas | 300 | 21.9 | 41.3 | 61.1 | 71.2 | 75.8 | 77.5 | 79.6 | 80.5 | 56.7 |
| core_optical_flow | 300 | 21.0 | 40.5 | 62.6 | 72.5 | 77.0 | 78.7 | 81.5 | 84.5 | 56.3 |
| cut_detection | 300 | 65.1 | 91.4 | 104.5 | 113.5 | 122.8 | 128.8 | 141.3 | 165.5 | 102.6 |
| scene_classification | 299 | 38.8 | 68.6 | 101.0 | 107.9 | 112.6 | 114.5 | 123.0 | 135.2 | 90.7 |
| video_pacing | 300 | 10.3 | 15.8 | 20.2 | 24.8 | 27.2 | 27.9 | 29.8 | 35.7 | 20.3 |
| uniqueness | 300 | 1.7 | 2.1 | 2.2 | 2.4 | 2.7 | 3.1 | 3.8 | 4.2 | 2.3 |

**Доля в суммарном времени видео (по среднему):**

- `cut_detection`: 102.6с (25.0%)
- `scene_classification`: 90.7с (22.1%)
- `core_clip`: 62.8с (15.3%)
- `core_depth_midas`: 56.7с (13.8%)
- `core_optical_flow`: 56.3с (13.7%)
- `video_pacing`: 20.3с (4.9%)
- `segmenter`: 12.8с (3.1%)
- `download`: 5.8с (1.4%)
- `uniqueness`: 2.3с (0.6%)

## 2. Топ-10 самых медленных прогонов по компонентам (video_id, dur, страта)

**core_clip** (top-10):
- 99с — `0Uy1t5gvZtM` dur=9s cell=vshort/v_viral/ru
- 88с — `3jyHtuacAVQ` dur=63s cell=medium/v_10k/en
- 85с — `0X1trN5lpzM` dur=7s cell=vshort/v_low/en
- 84с — `1hg8Lep9A2Q` dur=60s cell=medium/v_100k/ru
- 81с — `2cfE3FI6CdE` dur=63s cell=medium/v_100k/other
- 79с — `0tJ8Sq3rphs` dur=685s cell=long/v_100k/other
- 79с — `3i6uYsd_PgY` dur=7s cell=vshort/v_viral/other
- 79с — `1X_yHZZQ3Gk` dur=7s cell=vshort/v_low/other
- 78с — `00RlGST0Ff4` dur=124s cell=medium/v_low/other
- 78с — `5H7W4iM2AXM` dur=83s cell=medium/v_10k/en

**core_depth_midas** (top-10):
- 80с — `0tJ8Sq3rphs` dur=685s cell=long/v_100k/other
- 80с — `3tSGXSIyJYc` dur=796s cell=long/v_1m/ru
- 80с — `76wAEJOdBq8` dur=386s cell=long/v_100k/en
- 80с — `72amlB2W4ZQ` dur=763s cell=long/v_10k/ru
- 79с — `05deYJMn-c8` dur=666s cell=long/v_low/en
- 79с — `-IH8kA5ItfI` dur=568s cell=long/v_1m/other
- 79с — `6LwVPp7gDbw` dur=732s cell=long/v_100k/en
- 79с — `2cfE3FI6CdE` dur=63s cell=medium/v_100k/other
- 79с — `30r8hM_nqe0` dur=426s cell=long/v_100k/en
- 78с — `1nW74HxDj-c` dur=484s cell=long/v_10k/en

**core_optical_flow** (top-10):
- 84с — `-u-devGZao0` dur=646s cell=long/v_1m/other
- 83с — `1nW74HxDj-c` dur=484s cell=long/v_10k/en
- 82с — `1hg8Lep9A2Q` dur=60s cell=medium/v_100k/ru
- 81с — `0tJ8Sq3rphs` dur=685s cell=long/v_100k/other
- 81с — `76wAEJOdBq8` dur=386s cell=long/v_100k/en
- 81с — `3tSGXSIyJYc` dur=796s cell=long/v_1m/ru
- 81с — `72amlB2W4ZQ` dur=763s cell=long/v_10k/ru
- 80с — `5vy2qCoaaDI` dur=713s cell=long/v_100k/ru
- 80с — `4fcxVfq_JfA` dur=469s cell=long/v_low/ru
- 80с — `0df0TNo6OhM` dur=262s cell=long/v_viral/other

**cut_detection** (top-10):
- 165с — `0X1trN5lpzM` dur=7s cell=vshort/v_low/en
- 159с — `1hg8Lep9A2Q` dur=60s cell=medium/v_100k/ru
- 143с — `3b9GbBHaJIg` dur=9s cell=vshort/v_1m/other
- 141с — `2cfE3FI6CdE` dur=63s cell=medium/v_100k/other
- 141с — `5vy2qCoaaDI` dur=713s cell=long/v_100k/ru
- 139с — `0tJ8Sq3rphs` dur=685s cell=long/v_100k/other
- 138с — `0CNnfqNXcgo` dur=9s cell=vshort/v_viral/en
- 135с — `72amlB2W4ZQ` dur=763s cell=long/v_10k/ru
- 135с — `3tSGXSIyJYc` dur=796s cell=long/v_1m/ru
- 132с — `3i6uYsd_PgY` dur=7s cell=vshort/v_viral/other

**scene_classification** (top-10):
- 135с — `1hg8Lep9A2Q` dur=60s cell=medium/v_100k/ru
- 126с — `7Iwsi9DZvVQ` dur=480s cell=long/v_low/ru
- 124с — `2cfE3FI6CdE` dur=63s cell=medium/v_100k/other
- 123с — `43KjgqW8nQE` dur=173s cell=medium/v_low/other
- 123с — `0XIOjxj7idg` dur=86s cell=medium/v_100k/ru
- 117с — `-AssN_bljFU` dur=112s cell=medium/v_low/ru
- 117с — `6HkENS2rS5Y` dur=567s cell=long/v_100k/en
- 117с — `19oAhqUr2YI` dur=80s cell=medium/v_low/ru
- 117с — `-YlmnPh-6rE` dur=189s cell=long/v_viral/en
- 116с — `0tJ8Sq3rphs` dur=685s cell=long/v_100k/other

**segmenter** (top-10):
- 71с — `-4ZTWkO09n4` dur=791s cell=long/v_100k/other
- 62с — `0tJ8Sq3rphs` dur=685s cell=long/v_100k/other
- 59с — `3Lrz8eRm3Zs` dur=655s cell=long/v_10k/other
- 58с — `5Nw89HJOm9E` dur=646s cell=long/v_10k/ru
- 50с — `6LwVPp7gDbw` dur=732s cell=long/v_100k/en
- 50с — `2M5QmTmBjHE` dur=732s cell=long/v_1m/ru
- 46с — `1luUofppMHE` dur=703s cell=long/v_10k/other
- 45с — `1nW74HxDj-c` dur=484s cell=long/v_10k/en
- 44с — `7mhRzG3QTGE` dur=563s cell=long/v_low/other
- 42с — `30r8hM_nqe0` dur=426s cell=long/v_100k/en

## 3. Гипотеза: длительность — драйвер времени (проверка количественно)

Spearman(wall, duration) по каждому компоненту + проверка «потолка» ~100с (сэмплинг кадров).

| Компонент | Spearman(wall,dur) | Spearman(wall, min(dur,100)) | вывод |
|---|---|---|---|
| download | 0.615 | 0.57 | умеренная |
| segmenter | 0.942 | 0.9 | умеренная |
| core_clip | 0.594 | 0.567 | умеренная |
| core_depth_midas | 0.885 | 0.849 | умеренная |
| core_optical_flow | 0.91 | 0.873 | умеренная |
| cut_detection | 0.67 | 0.628 | умеренная |
| scene_classification | 0.796 | 0.786 | умеренная |
| video_pacing | 0.725 | 0.702 | умеренная |
| uniqueness | -0.086 | -0.086 | слабая — почти константа |

**Интерпретация (важно — колонка «вывод» выше сгенерирована грубым порогом, читать по числам):**
- Длительность — **сильный** драйвер для `segmenter` (0.94), `core_optical_flow` (0.91), `core_depth_midas`
  (0.89), `scene_classification` (0.80), `video_pacing` (0.73). Гипотеза из BATCH_RUN_SUMMARY подтверждена.
- **Потолок «~100с» НЕ жёсткий**: капирование длительности на 100с *снижает* Spearman (напр. depth 0.885→0.849,
  flow 0.91→0.873) — значит длинные видео (>100с) всё ещё чуть дороже даже после сэмплинг-потолка (остаточная
  зависимость от полного числа кадров/сцен). Эффект небольшой, но потолок не абсолютный.
- `core_clip` — **аномалия**: корреляция слабее (0.59), потому что часть **очень коротких** видео (7-9с)
  оказались в топ-медленных (85-99с) — см. §2. Драйвер тут НЕ длительность. Гипотезы: Triton warmup на ранних
  видео / минимальный пол сэмплинга кадров для коротких / cold-start ensemble. Проверить в Фазе 2.
- `uniqueness` — константа ~2.3с (читает готовые эмбеддинги core_clip, не зависит от видео).

## 4. CPU%/GPU по компонентам (обогащённая подвыборка)

Строк с CPU%/GPU-данными: 1808 (из 2700). Подвыборка ~226 видео (обогащённый формат).

| Компонент | CPU% p50/p95 | user_s p50 | sys_s p50 | GPU util p50/p95 | GPU mem peak MiB p95 |
|---|---|---|---|---|---|
| segmenter | 181.0/376.2 | 12.1 | 3.0 | 0.0/0.0 | 2427.0 |
| core_clip | 19.0/24.0 | 9.8 | 2.9 | 8.0/20.5 | 2427.0 |
| core_depth_midas | 40.0/53.0 | 20.8 | 4.7 | 24.0/44.0 | 2513.0 |
| core_optical_flow | 32.5/39.8 | 17.1 | 4.2 | 47.0/54.8 | 2513.0 |
| cut_detection | 31.0/43.0 | 22.0 | 10.2 | 0.0/0.0 | 2513.0 |
| scene_classification | 229.0/264.0 | 238.7 | 3.3 | 0.0/0.0 | 2513.0 |
| video_pacing | 40.5/55.0 | 6.9 | 2.2 | 0.0/0.0 | 2427.0 |
| uniqueness | 59.0/72.0 | 1.3 | 0.1 | 0.0/0.0 | 2427.0 |

## 5. scene_classification soft-fail

Не-OK строк scene_classification: 1 — ['13pz52piBj0']

## 5b. scene_classification soft-fail — разбор

`13pz52piBj0` — единственный rc≠0 (rc=4) по scene_classification за весь прогон. Это **самое первое тест-видео**,
на котором scene_classification падал на inprocess-PyTorch-GPU (драйвер CUDA 12.4 vs torch cu13) ДО фикса
`--device cpu`; после фикса перезапущен успешно → видео `complete`. Т.е. это не рантайм-флейк, а артефакт
до-фиксовой отладки. **Код чинить не нужно** — устойчивость обеспечена уже сделанным переходом на CPU
(и будет снята переводом на GPU в Фазе 2). В новых прогонах этого сбоя не будет.

## 6. Выводы — что дорого, почему, и приоритеты Фазы 2

**Суммарно ≈ 410с/видео.** Топ-3: `cut_detection` 25%, `scene_classification` 22%, `core_clip` 15%.
Ключ — колонка CPU%/GPU (§4), она объясняет ПОЧЕМУ дорого и что оптимизировать:

1. **`cut_detection` (25% времени) — главный резерв.** wall p50=104с, но CPU лишь **31%** и user-time **22с**.
   То есть ~**80с/видео тратится НЕ на вычисления** (декод видео / I/O / однопоточное ожидание), GPU=0.
   Это не «дорогой алгоритм», это **overhead**. Приоритет №1 в Фазе 2: профилировать модуль
   (`VisualProcessor/modules/cut_detection/`) — где уходит время (повторный декод? покадровый Python-цикл?).
2. **`scene_classification` (22%) — CPU-bound из-за инфра-бага.** CPU **229-264%** (~2.4 ядра), user **238с**,
   GPU=0 — resnet50 крутится на CPU, потому что inprocess-torch несовместим с драйвером (портфолио §3.5).
   Это **инфраструктурная**, не алгоритмическая проблема. Перевод на GPU (совместимый torch/CUDA или Triton
   places365 — модель УЖЕ в бандле) → потенциально с ~101с до единиц секунд. Приоритет №2.
3. **Triton-стадии `clip`/`depth`/`flow` (44% суммарно) — GPU недогружен.** GPU util p50 всего **8/24/47%**,
   пик VRAM 2.5ГБ из 16. Узкое место — **JSON-сериализация UINT8-кадров + HTTP round-trips**, не GPU-compute.
   Две линии: (а) параллелить 3 независимые стадии (async, GPU есть запас) — потенциал ~120с/видео;
   (б) бинарный тензорный формат Triton вместо JSON. Приоритет №3.
4. **`core_clip` аномалия коротких видео** — разобрать отдельно (§3): почему 7-9с видео дают 85-99с.
5. **Длительность подтверждена драйвером** для segmenter/depth/flow/scene (Spearman 0.8-0.94); сэмплинг-потолок
   ~100с смягчает, но не убирает зависимость на длинных видео.

**Память НЕ узкое место** (пик 2.5/16 ГБ) → подтверждает гипотезу плана Фазы 4: узкое место — CPU (segmenter
ffmpeg-декод 181-376% CPU + scene CPU-torch), а не VRAM. Это влияет на выбор железа (быстрый CPU/много ядер
важнее большой карты) и на число параллельных процессов.

**Порядок Фазы 2:** (1) профилировать cut_detection → (2) scene_classification на GPU → (3) параллелить Triton →
(4) разобрать core_clip-аномалию. Каждая — с числом до/после на реальной выборке (правило плана).