# DataProcessor — прогон тест-корпуса (300 видео) — сводка

**Дата завершения:** 2026-07-21. **Результат:** ✅ **300/300 видео обработано, 0 фейлов.**
**Исполнитель:** TrendFlow Bot (автономный ночной прогон). **GPU:** RunPod RTX 2000 Ada 16GB.

Этот прогон закрывает разрыв из [`COMPONENT_DEEP_DIVE_PORTFOLIO_ASSESSMENT.md`](../COMPONENT_DEEP_DIVE_PORTFOLIO_ASSESSMENT.md)
§7.2/§9: вместо 5-26 мок-фикстур DataProcessor прогнан на **реальном разнообразном корпусе** из HF
(`Ilialebedev/videos*`). Результат — валидные NPZ для обучения моделей TF Agent M.

---

## 1. Что сделано

Прогон всех **300 отобранных видео** (см. `corpus300.json`, сбалансированы по 61 страте
duration×views×language) через VisualProcessor-пайплайн на GPU-поде. На каждое видео:
скачивание mp4 с HF → Segmenter → 7 компонентов → сохранение всех NPZ + полные метрики → очистка mp4.

**Компоненты (9 стадий):** `download`, `segmenter`, `core_clip` (Triton/CLIP), `core_depth_midas`
(Triton/MiDaS), `core_optical_flow` (Triton/RAFT), `cut_detection`, `scene_classification` (Places365,
CPU), `video_pacing`, `uniqueness`.

**Результаты (NPZ)** лежат на **Network Volume** (персистит после гашения пода):
`/workspace/corpus_out/<video_id>/rs/<component>/*.npz` — **2400 NPZ-файлов** (8 на видео, 54 GB).
Доступ для TF Agent M: примонтировать тот же Network Volume к новому поду.

---

## 2. Итоговые метрики

| Показатель | Значение |
|---|---|
| Видео обработано | **300 / 300** (0 partial/fail) |
| NPZ на видео | 8 (медиана=min=max) |
| Время на видео (wall) | p50 **431с**, p95 536с, среднее 410с |
| Пик GPU-памяти (макс по видео) | **3575 MiB** (из 16384) |
| GPU util макс | **95%** |
| Суммарное время прогона | ~40ч (с учётом длинных видео и restart'ов watchdog) |

### Per-component (p50/p95, полная таблица — `BATCH_REPORT.md` / `batch_report.json`)

| Компонент | OK/Fail | wall p50/p95 с | CPU% p50/p95 | RSS MB p95 | GPU util p95 |
|---|---|---|---|---|---|
| download | 300/0 | 5.3/9.1 | — | — | — |
| segmenter | 300/0 | 9.0/38.0 | 181/377 | 908 | 0 |
| core_clip (Triton) | 300/0 | 62.9/76.0 | 19/24 | 1407 | 21 |
| core_depth_midas (Triton) | 300/0 | 61.2/77.5 | 40/54 | 1956 | 44 |
| core_optical_flow (Triton) | 300/0 | 62.8/78.7 | 32/40 | 1526 | 55 |
| cut_detection | 300/0 | 104.6/128.8 | 31/43 | 2091 | 0 |
| scene_classification (CPU) | 299/1 | 101.0/114.5 | 229/264 | 1591 | 0 |
| video_pacing | 300/0 | 20.2/27.9 | 40/55 | 765 | 0 |
| uniqueness | 300/0 | 2.3/3.1 | 59/72 | 36 | 0 |

**Узкое место:** `cut_detection` (~105с, CPU-bound) и три Triton-стадии (~190с суммарно, JSON-инференс).
**Разрешение видео не влияет** на время — анализ на фикс. ширине 480 (кадры даунскейлятся), драйвер —
длительность → число сэмплированных кадров (с потолком сэмплинга ~100с).

### Полнота метрик (per-video × per-component)
- `per_video_component.csv` (2700 строк = 300×9): на каждую пару — rc, wall, **CPU user/sys/%**, max RSS,
  minor page faults, voluntary ctx-switches, **per-компонентная GPU util/mem** (атрибуция по тайм-окнам стадий).
- `per_video.csv` (300 строк): суммарное время, GPU пик/util/mem, кол-во NPZ, статус.
- Сырьё на volume: `corpus_out/<id>/metrics.jsonl` + `gpu_samples.csv` (посекундный GPU) + `.time_*`
  (полный вывод `/usr/bin/time -v` — всё до последнего).

> Примечание: у первых ~74 видео метрики в базовом формате (wall+RSS+GPU-пики per-video), у остальных
> ~226 — обогащённый (CPU%/GPU-по-компонентам). Агрегаты CPU%/GPU-per-component посчитаны по обогащённой
> подвыборке — репрезентативно.

---

## 3. Инфраструктура (Triton поднят с нуля)

Triton на этом поде отсутствовал; docker/podman в RunPod-контейнере не работают (overlay-mount/userns
запрещены). Решение: **skopeo** скачал образ `tritonserver:24.08-py3` без демона → слои распакованы
вручную → бандл `tritonserver`+CUDA собран на `/workspace/triton-bundle` (**персистится для будущих
подов**). ONNX-веса (CLIP/MiDaS/RAFT/Places365) скачаны из `Ilialebedev/trendflow_artifact_0_1`.
Python-backend preprocess-моделей заведён на системном python3.10+numpy. Итог: 14 моделей (5 ensembles +
preprocess + onnx) обслуживаются на `http://localhost:8000`.

**Запуск Triton на будущем поде:** `bash /workspace/start_triton.sh` (грузит все модели, бандл уже на volume).

---

## 4. Решённые блокеры (все автономно)

1. **Дисковая квота** тома была забита старыми test-выхлопами прошлых сессий → процессы не могли писать
   и падали. Освобождено ~60G; том расширен владельцем до 120G. seg-кадры чистятся per-video (оставляется
   только NPZ ~151M/видео вместо ~1G).
2. **Segmenter drift** — строгий лимит рассинхрона аудио/видео 1с ронял реальные видео (no-fallback).
   Поднят до **10с** (`Segmenter/segmenter.py`, `drift_tol_sec`). ~все реальные видео проходят.
3. **Leading-dash video_id** (`-4ZTW…`, `--jz7…`) — argparse принимал за флаги. Фикс: форма `--video-id=`
   в `run_corpus.sh`. Была причина всех фейлов до фикса; после — 0 фейлов.
4. **video_pacing config-drift** (§3.6) — задеплоенный конфиг не имел секции video_pacing → вариант A
   (13 NaN). Добавлена секция (entropy+histograms) → вариант B (5 NaN). Подтверждено на реальных данных.
5. **scene_classification** — inprocess PyTorch несовместим с драйвером (torch cu13 vs драйвер CUDA 12.4)
   → переключён на `--device cpu`.
6. **Устойчивость к обрывам** — на этом поде detached-процессы (tmux/setsid) капризны + SSH мигает;
   батч запущен под **watchdog** (авто-рестарт батча/Triton при падении, проверен рабочим) + resumable
   (skip по `_summary`). Прогон пережил несколько обрывов SSH без потери прогресса.

---

## 5. Файлы

- `BATCH_REPORT.md` — человекочитаемый отчёт.
- `batch_report.json` — агрегаты (p50/p95/mean/max по каждой метрике каждого компонента) для TF Agent M.
- `per_video_component.csv`, `per_video.csv` — полные метрики.
- `batch_progress.log` — лог прогона (per-video OK/FAIL/тайминги).
- `corpus300.json` — список видео (id, repo, страта, длительность).
- `scripts/` — все скрипты прогона для воспроизводимости (`run_corpus.sh`, `batch_corpus.py`,
  `watchdog.py`, `gen_report.py`, `start_triton.sh`, `setup_triton_bundle.sh`, `eta.py`).

---

## 6. Что дальше (для TF Agent M)

- NPZ готовы на Network Volume (`/workspace/corpus_out/*/rs/`) — реальный корпус вместо мок-данных.
- Аудио/текст-компоненты (AudioProcessor/TextProcessor) в этот прогон НЕ входили — отдельная фаза
  (свои venv/модели), при необходимости добавляются в `run_corpus.sh` тем же паттерном.
- Пере-оценка компонентов, помеченных в портфолио-оценке как «мёртвые из-за mock/stub данных», теперь
  возможна на этих реальных NPZ.
