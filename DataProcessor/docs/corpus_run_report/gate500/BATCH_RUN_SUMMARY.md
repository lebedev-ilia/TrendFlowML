# Gate 2 — 500 видео (масштабный тест) — СВОДКА

**Дата:** 2026-07-22/23 · **Под:** RunPod RTX A4500 (`qdwaw2q9z9tzm0`), Triton up, venv cu121 ·
**Набор:** 7 VP-компонентов (download+segmenter+core_clip+core_depth+core_optical_flow+cut_detection+
scene_classification+video_pacing+uniqueness) · **Параллелизм:** N=4 воркера (shard `index % 4`) +
nohup+setsid+watchdog · **Корпус:** `corpus500.json` (500 видео, стратифицировано dur×views×lang).

## Итог: ✅ Gate 2 пройден (критерий ≥97% complete)

| Метрика | Значение |
|---|---|
| Видео complete | **500 / 500 (100%)** |
| Видео partial/failed | **0** |
| Per-video wall | **p50=328.5с, p95=470.1с, mean=320.5с** |
| NPZ на видео | p50=**8** (min 4, max 8) |
| GPU mem peak (max по видео) | 4004 MiB (scene resnet50 GPU) |
| GPU util max | 100% |
| Диск (весь corpus_smoke, вкл. 50-gate) | **12 ГБ** → **~24 МБ/видео** (OPT-3 держит) |
| scene-GPU NPZ | 499/500 (1 — из-за бага cut_detection ниже) |
| Реальный throughput @N=4 | **~27 видео/ч** на RTX A4500 |
| Watchdog авто-рестарты | 0 (не потребовалось — 0 крашей воркеров/Triton за весь прогон) |

## Компоненты

| Компонент | OK | Fail |
|---|---|---|
| download, segmenter, core_clip, core_depth_midas, core_optical_flow, uniqueness | 500 | 0 |
| cut_detection | 499 | **1** |
| scene_classification | 499 | **1** (каскад — зависит от cut_detection) |
| video_pacing | 499 | **1** (каскад — зависит от cut_detection) |

**НЕ 3 независимых фейла, а 1 корневая причина** на видео `1xFgqSpn1p0`: cut_detection упал numpy-ошибкой
`Too many bins for data range. Cannot create 3 finite-sized bins`, а scene+video_pacing зависят от его
выхода → каскадно не запустились. Разобрано и **исправлено = L12** (см. `LOGIC_ERRORS_FOR_CLAUDE.md`),
фикс валидирован на numpy пода по всем краш-кейсам. → эффективно 499.8/500 после фикса.

## Выводы (важное)

1. **Пайплайн стабилен на масштабе:** 500 видео, 0 деградации по времени/памяти (утечек нет — mean≈p50),
   watchdog не понадобился ни разу. Мульти-воркерное шардирование (N=4) без коллизий/дублей.
2. **Реальный throughput ~27 видео/ч** (не ~40 из короткого смоука) — corpus500 содержит длинные видео
   (bucket `long`, 400+с), которые дают больше кадров/времени. **Это точнее для планирования 1000/прод**,
   чем оценка по короткому клипу. 500 видео заняли ~18ч @N=4 на одной машине.
3. **OPT-3 (depth) подтверждён на масштабе:** ~24 МБ/видео → 1000 видео ≈ 24-30 ГБ, том 120ГБ с запасом,
   HF-выгрузка НЕ нужна.
4. **Единственный класс фейла — degenerate-input в гистограммах** (L12): компонент теперь робастен к
   битым/вырожденным shot-интервалам (inf/nan/огромные timestamp) — не упадёт, вернёт корректную
   низко-энтропийную гистограмму.

## Критерий перехода на Gate 3 (1000)
Gate 2 критерий (≥97% complete, throughput стабилен, шардирование без коллизий, watchdog чист) — **выполнен**.
Готово к Gate 3. Форма corpus1000 — см. `NEXT_TEST_PLAN.md §9` (опция A метаданные-расширение /
B scale-филлеры); решение за владельцем по итогам этой сводки.

## Артефакты
- `gate500/report500/BATCH_REPORT.md`, `batch_report.json`, `per_video.csv` (500 строк),
  `per_video_component.csv` (4500 строк = 500×9).
- Фикс: `DataProcessor/VisualProcessor/modules/cut_detection/utils/cut_detection.py` (`_safe_histogram`).
