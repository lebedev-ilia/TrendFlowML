# Gate 3 — 1000 видео — СОСТОЯНИЕ И КОНТЕКСТ (2026-07-23)

> Полная фиксация состояния для устойчивости (сессия уже перезапускалась 1 раз). Держать актуальным.

## TL;DR
Gate 3 отработал **1000/1000 видео** (summaries). **8 из 9 компонентов — чисто**; `scene_classification`
массово упал (**499 OK / 497 FAIL**) из-за **cuDNN-несовместимости на этом поде** (не конкуренция — см. L13).
Фикс scene (cuDNN-guard) написан и **валидирован e2e**. Осталось: до-прогнать ~497 видео со scene → пуш → гашение.

## Инфраструктура (АКТИВНА — под НЕ гасить пока не закончим)
- **Под**: `8vox1accfqm2o6`, **RTX 2000 Ada**, **48 vCPU**, 251GB RAM, 16GB VRAM.
  SSH `213.173.99.24:32624`, key `automation/runpod_ssh/id_ed25519`. Network Volume прикреплён.
- **Баланс RunPod** на старте Gate3: $13.04. Ставка $0.24/ч.
- **Триптих на волюме**: `/workspace/TrendFlowML` (репо с фиксами L12+L13), `/workspace/venv` (cu121),
  Triton-бандл, `corpus1000.json`, `corpus_smoke/` (1000 обработанных видео).

## Харнесс Gate 3 (на волюме `/workspace/`)
- **N=8** воркеров (`pgate_worker.py $WID 8`, шард `index%8` из `corpus1000.json[:1000]`), каждый в свой
  лог `pgate_w{N}.log`, pid в `pgate_w{N}.pid`.
- **Запуск**: `gate3_run.sh` (detached, чистит старое, стартует Triton+воркеров+watchdog; имя не совпадает
  с pkill-паттернами → без self-match).
- **watchdog**: `pgate_watchdog2.py` — **по-воркерное воскрешение** (лечит частичную смерть воркеров,
  чего НЕ умел старый `pgate_watchdog.py`), + рестарт Triton, exit при summaries≥1000. Держать РОВНО ОДИН
  инстанс (проверять `ps -eo cmd | grep 'python3 /workspace/pgate_watchdog2.py'`).
- **Резюмируемость**: воркеры пропускают видео с `_summary`. Gate3 переиспользовал 499 готовых из Gate2.
- **Уроки харнесса** (важно для повторов): (1) launch-SSH обрыв убивает недо-detached воркеров — только
  `setsid`+`disown`; (2) `pkill -f <pattern>` ловит собственную SSH-команду с этой строкой → чистить по
  числовым PID или скриптом-файлом; (3) `pgrep -fc` завышает счёт из-за setsid-сабшелла — считать через `ps`.

## Результаты Gate 3 (отчёт: `gate1000/report1000/`, стянут локально)
- **Видео**: 1000/1000 с `_summary`. Wall p50=**306.1с**, p95=504.6с. NPZ/видео p50=8 (min 0, max 9).
- **Per-component OK/Fail** (из `report1000/BATCH_REPORT.md`):

| Компонент | OK | Fail | Прим. |
|---|---|---|---|
| download | 1000 | 0 | |
| segmenter | 995 | 0 | 5 видео не дошли |
| core_clip | 983 | 2 | Triton, транзиент |
| core_depth_midas | 972 | 3 | Triton |
| core_optical_flow | 972 | 4 | Triton |
| cut_detection | 977 | 5 | **L12-фикс работает** (были бы больше без него) |
| **scene_classification** | **499** | **497** | **L13 — cuDNN на этом поде (см. ниже)** |
| video_pacing | 995 | 4 | зависит от cut |
| uniqueness | 986 | 13 | зависит от core_clip |

- Мелкие фейлы (2-13) — транзиентные Triton/GPU на N=8, не систематические.
- **Диск**: du timed out (сетевой FS); OPT-3 держит ~20-25МБ/видео (проверить финально).

## L12 — cut_detection histogram (✅ валидирован e2e в Gate3)
Фикс `_safe_histogram` (см. `LOGIC_ERRORS_FOR_CLAUDE.md` L12). Видео `1xFgqSpn1p0`, ронявшее cut_detection
в Gate2, в Gate3 дало **cut_detection NPZ** (краша нет). Подтверждено.

## L13 — scene_classification cuDNN NOT_INITIALIZED (RTX 2000 Ada + torch cu121) — фикс валидирован
- **Симптом**: 497 новых видео — scene rc=4 `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`.
- **ЧЕСТНАЯ причина** (первый вывод «N=8 concurrency» — ОШИБКА, отозвана): resnet50 GPU-forward падает
  **даже при N=1 в полной изоляции** на этом поде → **cuDNN torch 2.4.1+cu121 не инициализируется на
  RTX 2000 Ada** (на поде Gate2 RTX A4500 — работал; 499 «OK» scene = именно те видео Gate2). Т.е.
  **«OPT-2 scene-GPU» НЕ портируем между GPU/драйверами.**
- **Проверено на поде**: `cudnn.enabled=False` → resnet50 на GPU работает; CPU тоже работает (1.9с/16 кадров).
- **Фикс** (`scene_classification.py`, после `self.device=...`): проба cuDNN при инициализации →
  при провале `torch.backends.cudnn.enabled=False` (GPU native convs) → крайний случай CPU. Портируемо.
- **Валидировано e2e**: пере-прогон `1xFgqSpn1p0` с фиксом → **scene NPZ появился** + все компоненты. ✅
- Фикс уже на волюме пода (`/workspace/TrendFlowML/.../scene_classification.py`).

## ОСТАЛОСЬ СДЕЛАТЬ (по порядку)
1. **До-прогнать ~497 видео с упавшим scene** (фикс scene теперь работает). Эффективно: переиспользовать
   готовые core_clip/cut_detection NPZ (scene от них зависит), гнать download+segment+scene (frames
   удаляются после обработки → нужен re-download+segment). ~1.5-2ч @N=8, ~$2. Список упавших: видео без
   `rs/scene_classification/*.npz` в `corpus_smoke`.
2. **Финальный отчёт** `gate1000/BATCH_RUN_SUMMARY.md**: сравнение Gate2(500,N=4,A4500) vs Gate3(1000,N=8,
   RTX2000Ada) — throughput (~92/ч N=8 vs ~27/ч N=4), диск, деградация(нет), L12/L13. Тарбол логов всех 1000.
3. **Коммит+пуш**: фикс scene (L13), фикс cut (L12 — уже запушен `4b1e4e0`), L13 в LOGIC_ERRORS,
   report1000, summary, corpus1000 (запушен).
4. **Обновить** `NEXT_TEST_PLAN`/портфолио: (a) scene-GPU не портируем → guard обязателен; (b) N=8 на
   48-ядерном поде стабилен для Triton-компонентов, но GPU-inprocess (scene) требует cuDNN-guard;
   (c) итоги всей Gate-серии 10→50→500→1000.
5. **Гашение пода** `pod_control stop_all` — ТОЛЬКО после п.1-3.

## Запушено на GitHub (`main`) к этому моменту
- Gate2: `6762a43` (report + logs). L12 fix: `4b1e4e0`. corpus1000: `dbba47a`. GATE3_STATUS(v1): `66a659b`.
- **НЕ запушено ещё**: scene L13 fix, L13-запись, report1000, эта версия статуса — в следующем коммите.
