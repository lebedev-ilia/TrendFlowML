# Gate 3 — 1000 видео — СТАТУС (live, 2026-07-23)

> Живой статус-док для устойчивости (продолжение после сброса контекста). Обновляется по ходу.
> Финальная сводка будет в `gate1000/BATCH_RUN_SUMMARY.md` после завершения.

## Что сделано до Gate 3 (закоммичено+запушено, `main`)
- **Gate 2 (500)**: ✅ 500/500 complete, 0 систем. фейлов. Отчёт: `corpus_run_report/gate500/`
  (BATCH_RUN_SUMMARY.md, report500/*.csv, gate500_logs.tar.gz — логи всех 500). Push `6762a43`.
- **L12 fix**: cut_detection `_safe_histogram` (краш на вырожденных/inf/огромных shot-интервалах,
  1 видео `1xFgqSpn1p0` в Gate 2). Валидирован на numpy пода. `LOGIC_ERRORS_FOR_CLAUDE.md` L12. Фикс
  на Network Volume (`/workspace/TrendFlowML/.../cut_detection.py`, 5× `_safe_histogram`).
- **corpus1000.json**: 1000 видео = суперсет corpus500 (500) + 500 добор, 62 ячейки (мед.16/ячейку),
  все 11 repo, dur 4-893с (p50=60). Пул расширен из HF-шардов 2-4 (`select_corpus_candidates.py`,
  метаданные-only) → 1184 кандидата / 1162 в индексе. Трекнут: `corpus_run_report/corpus1000.json`,
  push `dbba47a`. Рабочая копия: `automation/runner/state/corpus/corpus1000.json`.

## Инфраструктура Gate 3
- **Под**: `8vox1accfqm2o6`, **RTX 2000 Ada** ($0.24/ч), **48 vCPU**, 251GB RAM, 16GB VRAM.
  SSH `213.173.99.24:32624`, key `automation/runpod_ssh/id_ed25519`. Network Volume прикреплён.
- **Решение по «2 машинам»** (владелец разрешил 2 пода): RunPod Network Volume attach'ится к ОДНОМУ поду
  (`migrate` его перемещает, не шэрит) → 2 пода не пишут в общий `/workspace/corpus_smoke` без полной
  пересборки 2-го (Triton-бандл+venv, часы). Вместо этого — **N=12 на 48-ядерном поде** (пайплайн
  CPU-bound; VRAM не лимит: при N=4 пик 4/16ГБ). Это эквивалент «2 машин» CPU-параллелизма в одной.
  Fallback: если Triton/GPU насытится и throughput встанет — 2-й под с отдельным setup.

## Запуск Gate 3 (в работе)
- **Команда**: `PGATE_N=12 CORPUS_FILE=/workspace/corpus1000.json GATE_N=1000 bash /workspace/pgate.sh`
  + `pgate_watchdog.py` (авто-рестарт воркеров/Triton, exit при done≥1000).
- **Резюмируемость**: 499/1000 уже done (из Gate 2) → **skip**; реально обрабатывается **~501 новое**
  видео (500 добор + `1xFgqSpn1p0` пере-прогон с L12-фиксом, его папка удалена принудительно).
- **Triton**: up (200), 14 моделей. apt-deps доставлены (ffmpeg/time/bc/libarchive13/python3-numpy).
- **Вотчер завершения**: bg-задача `brc2it7bl` (poll 10мин, exit при `workers_done=12/12` или сталле).
- **Оценка**: ~501 новое @ N=12 → **~6-10ч** (throughput подтвердить на разгоне), **~$1.5-2.5**.
  Баланс RunPod на старте: $13.04.

## Статус на момент записи
- 12/12 воркеров живы, Triton 200, ранняя фаза (skip готовых + скачивание новых → CPU load низкий,
  GPU idle; ждём разгона на compute-фазах). Первые завершения/throughput — проверяются.

## Что дальше (по завершении)
1. `gen_report` по corpus_smoke (1000) → `gate1000/report1000/` (как gate500).
2. Собрать логи всех видео (tar) + сводка `gate1000/BATCH_RUN_SUMMARY.md` (сравнение с Gate 2: throughput,
   диск, деградация, каскад-фейлы; проверить `1xFgqSpn1p0` теперь complete = L12 e2e-валидация).
3. Коммит+пуш всего. **Гашение пода** (`pod_control stop_all`).
4. Обновить `COMPONENT_DEEP_DIVE_PORTFOLIO_ASSESSMENT.md`/`NEXT_TEST_PLAN.md` итогами Gate-серии.
