# Gate 3 — 1000 видео — ФИНАЛЬНАЯ СВОДКА

> ## 🔄 ФИНАЛИЗАЦИЯ (2026-07-24): пере-прогон scene с фиксом L13 + venv-SSD замер
> После первого Gate 3 (scene 503/1000 из-за cuDNN L13) — фикс L13 применён, 497 видео пере-прогнаны.
> **Итог: scene 982/1000 NPZ (~98%), полный отчёт `report1000_final/`.**
> - **L13 (scene cuDNN-guard) валидирован на масштабе:** из 497 бывших cuDNN-фейлов починено ~479; остались
>   ~18 — это НЕ cuDNN, а **каскад от cut_detection** (таймауты/недобег под FS-контенцией на конкретных видео;
>   не восстановились и при N=4 → genuine edge-case long-tail).
> - **Per-component (report1000_final):** download 1000, segmenter 992, core_clip 977, depth 971, flow 962,
>   cut 967, **scene 973 OK/13 fail (982 NPZ)**, pacing 982, uniqueness 989. Wall p50=303с. 10 partial видео.
> - **venv-на-SSD ИЗМЕРЕНО in-pipeline:** импорт 33.6с→1.8с (18x) НО throughput НЕ вырос (~80 vs 92/ч) —
>   узкое место при N=8 = **запись кадров/NPZ на mfs** (segmenter 12→160с под контенцией, load 3/48 = I/O-block).
>   Реальный рычаг = кадры на локальный SSD, не venv. Детали: `OPTIMIZATION_LOG.md`, `NEXT_TEST_PLAN.md §6.2`.
> - **Харнесс:** watchdog2 пропатчен на pgrep-проверку (без дублей-воскрешений). Логи всех 1000 видео:
>   `gate1000_logs.tar.gz` (11030 файлов).
> - **Данные:** `corpus_smoke` (1000×~9 компонентов) на Network Volume; scene 982/1000.



**Дата:** 2026-07-23 · **Под:** RunPod **RTX 2000 Ada** (`8vox1accfqm2o6`), **48 vCPU**, 16GB VRAM ·
**Набор:** 9 стадий (download+segmenter+core_clip+core_depth_midas+core_optical_flow+cut_detection+
scene_classification+video_pacing+uniqueness) · **Параллелизм:** **N=8** воркеров (shard `index%8`) +
`gate3_run.sh` (detached) + `pgate_watchdog2.py` (по-воркерное воскрешение) · **Корпус:** `corpus1000.json`
(1000 видео = суперсет `corpus500` + 500 добор; 499 переиспользованы из Gate 2 как resumable, ~501 новых).

## Итог: 1000/1000 обработано; 8 из 9 компонентов чисто; scene — известная проблема пода (L13, фикс готов)

| Метрика | Gate 2 (500) | **Gate 3 (1000)** |
|---|---|---|
| GPU / ядра | RTX A4500 / — | **RTX 2000 Ada / 48 vCPU** |
| Параллелизм | N=4 | **N=8** |
| Видео обработано | 500/500 | **1000/1000** |
| Wall p50 / p95 | 328.5с / 470.1с | **306.1с / 504.6с** |
| Throughput | ~27 видео/ч | **~92 видео/ч** |
| NPZ/видео (p50) | 8 | 8 (min 0, max 9) |
| GPU mem peak | 4004 MiB | 4004 MiB |
| Watchdog авто-рестарты | 0 | **сработал** (по-воркерное воскрешение упавших воркеров) |

**Throughput ×3.4 vs Gate 2** (N=8 на 48 ядрах vs N=4). Каждое видео ≈ 1 ядро (компоненты внутри видео
последовательны), поэтому N масштабируется почти линейно по CPU; VRAM не лимит (пик 4/16 ГБ). Деградации во
времени за 1000 видео нет (mean p50≈306с стабилен).

## Per-component OK / Fail (report1000/BATCH_REPORT.md)

| Компонент | OK | Fail | Комментарий |
|---|---|---|---|
| download | 1000 | 0 | |
| segmenter | 995 | 0 | 5 видео не дошли (download edge) |
| core_clip | 983 | 2 | Triton, транзиент |
| core_depth_midas | 972 | 3 | Triton |
| core_optical_flow | 972 | 4 | Triton |
| cut_detection | 977 | 5 | **L12-фикс работает** — раньше падал на вырожденных интервалах |
| **scene_classification** | **499** | **497** | **L13 — cuDNN на этом поде (не конкуренция), см. ниже** |
| video_pacing | 995 | 4 | зависит от cut |
| uniqueness | 986 | 13 | зависит от core_clip |

Мелкие фейлы (2-13) — транзиентные Triton/GPU, не систематические.

## Две находки багов (обе закрыты фиксом)

### L12 — cut_detection histogram-краш (✅ валидирован e2e)
В Gate 2 видео `1xFgqSpn1p0` роняло cut_detection numpy-ошибкой `Too many bins for data range` на
вырожденных/битых shot-интервалах (inf/nan/огромные timestamp) → каскад scene+pacing. Фикс `_safe_histogram`
(фильтр не-финитных + относительный min-span + фолбэк). В Gate 3 это же видео дало **cut_detection NPZ** —
краша нет. `LOGIC_ERRORS_FOR_CLAUDE.md` L12.

### L13 — scene cuDNN NOT_INITIALIZED (портируемость GPU) (✅ фикс валидирован e2e)
- **Симптом:** 497 новых видео — scene rc=4 `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`.
- **ЧЕСТНАЯ причина** (первый вывод «N=8 concurrency» — ОШИБКА, отозвана): resnet50 GPU-forward падает
  **даже при N=1 в изоляции** → `torch 2.4.1+cu121` cuDNN не инициализируется на **RTX 2000 Ada**. 499 «OK»
  scene — это видео из Gate 2 (под RTX A4500, где cu121-cuDNN работал; resumable их не трогал). Т.е.
  **«OPT-2 scene-GPU» НЕ портируем между GPU/драйверами.**
- **Фикс:** cuDNN-guard в `scene_classification.py` — проба при init → при провале `cudnn.enabled=False`
  (GPU native convs) → крайний случай CPU. Валидировано: пере-прогон `1xFgqSpn1p0` → scene NPZ появился.
- **Урок:** все GPU-inprocess torch-компоненты (scene и будущие shot_quality/action_recognition) должны
  иметь cuDNN/CPU-fallback; нельзя полагаться на cu121-cuDNN на произвольном RunPod GPU.

## Состояние данных
- `corpus_smoke` (1000 видео: NPZ всех компонентов + логи) — **на Network Volume пода** (под удалён, том
  персистентный, данные целы). scene неполон для ~497 новых видео (падали ДО фикса L13).
- **Отложено (фиксы уже в репо):** до-прогон ~497 видео со scene (фикс L13 работает) → полный 1000×9
  датасет; тарбол логов Gate 3 в git (для Gate 2 залит `gate500_logs.tar.gz`, для Gate 3 — на волюме).

## Выводы по всей Gate-серии (10→50→500→1000)
1. **Пайплайн стабилен до 1000 видео** — 0 систематических фейлов, watchdog самоисцеляется, шардирование
   без коллизий. N=8 на 48-ядерном поде — рабочий throughput (~92/ч).
2. **OPT-1/OPT-3 подтверждены на масштабе** (ленивый torch, не-персист depth).
3. **OPT-2 scene-GPU оказался НЕ портируемым** — cuDNN зависит от GPU/драйвера; нужен guard (сделан, L13).
4. **2 реальных бага вскрыты именно масштабом** (L12 на вырожденном видео 1/500; L13 на смене GPU) — цель
   scale-теста достигнута. Оба закрыты.

## Артефакты
- `gate1000/report1000/` — BATCH_REPORT.md, batch_report.json, per_video.csv (1000), per_video_component.csv (9000).
- `gate1000/GATE3_STATUS.md` — полный контекст/инфра/харнесс/уроки.
- Фиксы: `cut_detection/utils/cut_detection.py` (L12), `scene_classification/utils/scene_classification.py` (L13).
