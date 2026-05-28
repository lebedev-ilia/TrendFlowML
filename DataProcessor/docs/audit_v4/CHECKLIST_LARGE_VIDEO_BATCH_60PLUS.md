# Чеклист готовности к запуску большого прогона (60+ видео)

**Назначение:** единый список работ; **запуск 60+** выполняется только после отметки всех обязательных пунктов и подписи в **§7**.  
**Связанный план (контекст и обоснование):** [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md)  
**Дата шаблона:** 2026-04-15 · **Последнее обновление чеклиста:** 2026-04-22 (п. **2.0.2–2.0.3**, **2.0.4–2.0.5** — классификация *empty*; waivers **W4–W6**)  

**Легенда:** `[ ]` — не сделано · `[x]` — сделано · **(О)** — обязательно для Go · **(Р)** — рекомендуется  

**Связанные артефакты (2026-04-15+):** [VIDEO_REGISTRY_60PLUS.yaml](VIDEO_REGISTRY_60PLUS.yaml) (seed 5/70) · [scripts/validate_video_registry_60plus.py](scripts/validate_video_registry_60plus.py) · [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md) · [BATCH_MODEL_VERSIONS_SNAPSHOT.md](BATCH_MODEL_VERSIONS_SNAPSHOT.md) · [COVERAGE_MATRIX_60PLUS.md](COVERAGE_MATRIX_60PLUS.md) · [BATCH_60PLUS_OWNER_TASKS.md](BATCH_60PLUS_OWNER_TASKS.md) · [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) (в т.ч. **W4–W6** — *empty* place/car/source_separation) · [METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md) · **Наблюдаемость (локальный E2E, работает):** [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md) · [METRICS_REFERENCE.md](../../monitoring/METRICS_REFERENCE.md) · **Минимальные базы `core_identity` под 70 видео:** [CORE_IDENTITY_MINIMAL_BASES_BATCH_70.md](CORE_IDENTITY_MINIMAL_BASES_BATCH_70.md)

**Порядок работ (обновление 2026-04-22):** (1) Закрыть **п. 2.0 (P0)** — в т.ч. **2.0.2**: после фикса `PYTORCH_CUDA_ALLOC_CONF` / AR — **full-max E2E** на обновлённом коде (локальный minimal Visual для AR уже **OK**, см. журнал 2026-04-22). (2) Пилот **15** `video_id` тем же full-max путём, что батч; зафиксировать отчёт (п. **5.5**). (3) При зелёном пилоте — расширение до **70** и дальше по остальным пунктам чеклиста (реестр §4, наблюдаемость §5, Go §7).

---

## Журнал выполнения

| Дата (UTC) | Сделано | Кто / примечание |
|------------|---------|------------------|
| 2026-04-15 | Созданы реестр-заготовка (5 видео набора B), шаблон матрицы покрытия, список задач владельца, черновик waivers; в `monitoring/README.md` добавлена секция «Батч 60+»; в п. **1.4** зафиксирован черновой git HEAD при подготовке документов | агент (заменить commit и hash перед Go) |
| 2026-04-15 | Добавлен черновик инвентаризации Prometheus labels ([METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md)), ссылка в п. **4.5** чеклиста | агент |
| 2026-04-15 | Код: `processor.py` — в ответ `run_processing` добавлены `processor=pipeline`, `component=main_py` для Prometheus (worker); скрипт [validate_video_registry_60plus.py](scripts/validate_video_registry_60plus.py); обновлены [METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md), [RUN_LOG.md](RUN_LOG.md) | агент |
| 2026-04-15 | **Фаза 0:** п. **0.1–0.4** — единственный разработчик (**Илья**) совмещает владельца реестра, наблюдаемости и пайплайна; целевой размер батча **70** `video_id` (≥60 по плану) | Илья |
| 2026-04-15 | П. **1.1:** зафиксирована политика **full max** (все процессоры + все переключаемые ветки из шаблона) — [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md); база `DataProcessor/configs/global_config.yaml` | Илья |
| 2026-04-15 | П. **1.2:** снимок версий моделей и зависимостей — [BATCH_MODEL_VERSIONS_SNAPSHOT.md](BATCH_MODEL_VERSIONS_SNAPSHOT.md) | Илья |
| 2026-04-15 | П. **1.3:** вычислен **config_hash** (16 hex) только по содержимому `global_config` в духе `main.py` — см. таблицу п. **1.3**; **git commit не фиксировали** (п. **1.4** без изменений) | Илья |
| 2026-04-15 | П. **1.5–1.6:** Text — **`emit_extra_metrics` и `compute_std` везде `true`**; backend векторного поиска — **только FAISS** (`require_faiss` / окружение `faiss-*`, без numpy-only) | Илья |
| 2026-04-16 | **Пилот полной цепочки E2E** (`backend/scripts/e2e_full_max_run.py`, offline mock): `youtube / -Q6fnPIybEI`, Segmenter + **Audio** + **Text** до NPZ и строк в `manifest.json` — OK (`run_id` **`437dd2f0-a239-424a-ad36-0026f63e094e`**). **Visual:** подпроцесс завершился с **`exit=1`** (~35 s) при **`--local-visual-no-triton`**; в `manifest` **нет** компонентов visual; ingestion всё же **completed** (`processors.visual.required: false`). Снимок конфига и summary: **`storage/e2e_full_max/20260416-120234_utc/`** (`global_config_e2e.yaml`, `summary.json`). Для стека: доустановлен **`embedding_service/requirements-e2e.txt`**, поднят сервис **:8005** (в штатном `start_e2e_stack.sh` он не стартовал до установки). Git при прогоне: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c` (совпадает с черновиком п. **1.4** — перепроверить перед Go). | Илья |
| 2026-04-16 | П. **2.8:** на батче **~70** видео — **включать** оркестраторную телеметрию AudioProcessor (**`AP_ORCHESTRATOR_TELEMETRY=1`**, **`…_CHILDREN=1`**; опционально **`…_LOG=1`**); отчёт **`scheduler_runtime_report.json`** — для пост-анализа владельцем наблюдаемости (**Илья**) | Илья |
| 2026-04-16 | П. **2.9:** в **`monitoring/README.md`** добавлена политика интерпретации **масок / ожидаемых NaN** для NPZ и кадровых метрик (чеклист **2.9**) | Илья |
| 2026-04-16 | П. **2.10–2.12:** политики **разных N (Segmenter)**, **`meta.models_used`** (таблица исключений), **OCR / `retain_raw_ocr_text`** — в **`monitoring/README.md`**; комментарий **2.10** в **`VIDEO_REGISTRY_60PLUS.yaml`** | Илья |
| 2026-04-17 | **E2E стек** (`E2E_RUN_ID=20260417-015311`, логи `backend/.e2e/logs/latest`): **youtube / -FOB4jpQIg8**, `run_id` **`ca818ec2-bbab-4215-8d8f-11760f7b0081`**. Ожидание оборвано ~**10 min**: у **`e2e_run_to_complete.py`** был дефолт **`--timeout 600`** — для **`--with-dataprocessor`** недостаточно на полный max-профиль; в **`manifest.json`** только **audio** (частично), **`run.status=running`**, сегментер/visual/text не завершены; в **`dataprocessor-worker`** — **Redis timeout** при чтении cancel-флага (типично при остановке стека пока `main.py` ещё в работе). **Исправление в коде:** при **`--with-dataprocessor`** без явного **`--timeout`** дефолт **7200** с (`backend/scripts/e2e_run_to_complete.py`). Повторить прогон с новым дефолтом или **`--timeout 7200+`**. Артефакт манифеста: `storage/result_store/youtube/-FOB4jpQIg8/ca818ec2-bbab-4215-8d8f-11760f7b0081/manifest.json`. | Илья |
| 2026-04-18 | **2.0.1–2.0.3 (частичная верификация):** full-max E2E прерван вручную (**Ctrl+C** при `time.sleep` в `e2e_run_to_complete`, см. `terminals/6.txt`). **2.0.1:** ~**31m** поллинга без **ReadTimeout**. **2.0.2:** в логе по-прежнему **`action_recognition` error (exit 4)** — не считать закрытым. **2.0.3:** **`micro_emotion` success** на снимке; пустая ветка — не проверена. | Илья |
| 2026-04-18 | **2.0.7 (код):** `backend/scripts/e2e_full_max_run.py` — встроенный example suite **20** планов (добавлены 11–20: повторы mock `video_id` + `audit_v3_scen_06`…`15`); `--example-suite-count` до **20** (15 и 17 допустимы). | агент |
| 2026-04-20 | Попытка запуска с **`--example-suite-count 17`** — **FAIL** до старта прогонов: `backend/.e2e/logs/e2e_terminal_20260420_065833_utc.log` (старый лимит **10**). **Исправлено в коде:** `builtin_example_suite_items` **20** слотов (п. **2.0.7**). | Илья |
| 2026-04-20 | **Example suite 10 full-max** (`e2e_terminal_latest.log` в `backend/.e2e/logs/`, `suite_manifest.json`: `example_suite_10_20260420-065944_utc`). **9/10** E2E с `e2e_exit=0`; plan **7** — **`httpx.ReadTimeout`** на опросе Backend при высокой нагрузке (не дождались `Done:`). **Повторяющиеся визуальные:** **`action_recognition` error** (subprocess exit 4) на прогонах; **`place_semantics` empty**; часто **`source_separation_extractor` empty**; plan **6** — **`micro_emotion` error** (валидация NPZ при «нет лиц» / save). **Код (после анализа):** `e2e_run_to_complete.py` — увеличен HTTP read timeout, `ReadTimeout` на poll не валит весь скрипт; `micro_emotion` — в пустом результате добавлено поле `microexpr_features` под схему v3; `action_recognition` — dtype проекции + откат батча→клип на GPU. | Илья / агент |
| 2026-04-22 | **micro_emotion (локальный minimal Visual):** [`visual_minimal_micro_emotion.yaml`](../../configs/audit_v3/visual/visual_minimal_micro_emotion.yaml) — `core_object_detections` → `core_face_landmarks` → `micro_emotion`; артефакты: `storage/result_store_me_minimal/youtube/-Q6fnPIybEI/me_minimal_cli_001/`; [RUN_LOG.md](RUN_LOG.md). **П. 2.0.3:** ветка «с лицами» + запись NPZ — **подтверждена** этим прогоном; ветка строго **no-face** по-прежнему отдельный кейс. | Илья / агент |
| 2026-04-22 | **П. 2.0.4–2.0.5 (*empty* на full-max E2E):** зафиксировано как **контентно-ожидаемые** исходы при штатных контрактах (не баг без иного `empty_reason`): **`place_semantics`** — `no_places_detected` (см. README place_semantics); **`car_semantics`** — `no_car_proposals`; **`source_separation_extractor`** — `audio_too_short` / `audio_silent` (см. extractor). Детали и ссылки: [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) **W4–W6**; на батче сверять **`meta.status` + `meta.empty_reason`** / NPZ, а не только метку *empty* в UI. | Илья / агент |
| 2026-04-22 | **Отдельные прогоны empty-компонентов (эмпирика):** `car_semantics` на `-Q6fnPIybEI` — **подтверждён** `no_car_proposals` ([`visual_minimal_car_semantics.yaml`](../../configs/audit_v3/visual/visual_minimal_car_semantics.yaml)); `place_semantics` — full pipeline не пройден (Triton :8010 down); `source_separation` на том же `video_id` — **`ok`** (empty на E2E — другие сегменты/ролики). [RUN_LOG.md](RUN_LOG.md) §«Отдельные прогоны…». | Илья / агент |
| 2026-04-22 | **action_recognition (разбор + локальная верификация):** root cause **exit 4** в E2E — не только логика SlowFast, а **`PYTORCH_CUDA_ALLOC_CONF=…,expandable_segments:…`** (из `e2e_env.sh`), не поддерживаемая старым `torch` в `modules/action_recognition/.action_recognition_venv` → при первом `cuda` init: `RuntimeError: Unrecognized CachingAllocator option: expandable_segments`. **Код:** в `action_recognition/main.py` до `import torch` — `_sanitize_pytorch_cuda_alloc_conf()` (вырезание `expandable_segments` из `PYTORCH_CUDA_ALLOC_CONF`). **Minimal Visual (весь `VisualProcessor`, только цепочка для AR):** `VisualProcessor/.vp_venv/bin/python` + [`configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml`](../../configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml) — `core_object_detections` → `action_recognition`; прогон **OK** на `-Q6fnPIybEI`, артефакты: `storage/result_store_ar_minimal/youtube/-Q6fnPIybEI/ar_minimal_cli_001/` (`detections.npz`, `action_recognition_features.npz`). Детали: [RUN_LOG.md](RUN_LOG.md) §«VisualProcessor: minimal …». **П. 2.0.2:** локальный критерий «субпроцесс не падает на той же версии кода + детекции есть» — `[~]`; **full-max E2E** без `action_recognition` error — **ещё прогнать**. | Илья / агент |

---

## Прогресс по фазам

Заполняйте колонку «Статус» по мере выполнения (например: `0/4`, `4/4 OK`).

| Фаза | Описание | Ссылка на раздел | Статус |
|------|----------|------------------|--------|
| 0 | Роли и базовая фиксация | §1 | `4/4 OK` — solo: **Илья** (роли 0.1–0.3 объединены, см. §1) |
| 1 | Конфиг и воспроизводимость (B0) | §2 | `5/7` — добавлен **пилотный** freeze п. **1.1**; по-прежнему **1.4** (commit перед Go) и уточнение «боевой» заморозки YAML для батча 60+ |
| 2 | Код, блокеры, регрессии (A) | §3 | `частично` — **2.0 (P0):** **2.0.2** — локальный AR OK, E2E — ждёт прогон; **2.0.3** — minimal **micro_emotion** OK; **2.0.4–2.0.5** — зафиксированы (W4–W6); **2.1–2.2** открыты; **2.18** — пилот **15**; **2.3–2.12**, **2.5–2.9** отмечены |
| 3 | Реестр 60+ и матрица покрытия (B) | §4 | `частично` — артефакты §4.1 созданы; заполнение до **70** (п.0.4) и теги — владелец |
| 4 | Наблюдаемость Prometheus/Grafana (C) | §5 | `частично` — **локальный E2E: Prometheus+Grafana работают** ([OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md)); **labels** `pipeline` / `main_py` в [processor.py](../../api/services/processor.py); **пилот 4.5** (labels), **4.1–4.2 прод-URL** — владелец |
| 5 | Сухой прогон (B4): **сначала 15**, при успехе **70** | §6 | `частично` — 10-видео full-max 2026-04-20 в журнале; **цель:** **15** `video_id` + отчёт **5.5** → Gate на батч **70** |
| 6 | Финальный Go / подпись | §7 | `[ ]` |

---

## 0. Стабилизация по результатам E2E (приоритет перед пилотом 15) — **п. 2.0 (P0)**

*Назначение: убрать известные **ошибки** и неоднозначные **empty** до сухого прогона на **15** видео. Код-исправления в репо — **подтвердить** повторным прогоном.*

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 2.0.1 | **E2E `e2e_run_to_complete`:** нет «ложного» падения сьюта из‑за **ReadTimeout** на `GET /api/runs/...` при нагрузке; параметр **`--http-read-timeout`** (по умолчанию 120s) + повтор при `TimeoutException` | **(О)** | `[x]` **код** · `[~]` **верификация:** ручной stop ~**31m** поллинга — **ReadTimeout в сессии не воспроизвёлся**; до **полного** критерия нет (не дождались `Done: completed`). |
| 2.0.2 | **`action_recognition`:** субпроцесс не падает с **exit 4** на full-max (или оформлен **waiver** + известный root cause) | **(О)** | `[x]` **код** (dtype/batch, санитария **`PYTORCH_CUDA_ALLOC_CONF`** / `expandable_segments` в `action_recognition/main.py`) · `[~]` **верификация:** **локально** — minimal Visual (`.vp_venv` + `visual_minimal_…yaml`, см. журнал) **OK**; **full-max E2E** после фикса — **не прогоняли** → п. **не закрыт** до зелёного E2E. |
| 2.0.3 | **`micro_emotion`:** сохранение NPZ проходит **`validate_npz`** в ветке «нет лиц в видео» (в т.ч. `microexpr_features` в пустом результате) | **(О)** | `[x]` **код** · `[x]` **верификация (часть):** **2026-04-22** — minimal Visual ([`visual_minimal_micro_emotion.yaml`](../../configs/audit_v3/visual/visual_minimal_micro_emotion.yaml), [RUN_LOG.md](RUN_LOG.md)) — `micro_emotion.npz` **ok** при наличии face-path; ветка **строго no-face** — по-прежнему отдельный контроль. |
| 2.0.4 | **`place_semantics` / `car_semantics: empty`:** зафиксировано: **ожидаемо** (нет матчей в каталоге / нет car-proposals) vs **баг** | **(Р)** | `[x]` [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) **W4**, **W6**; плейс: [place_semantics README](../../VisualProcessor/core/model_process/core_identity/place_semantics/README.md) §Empty; кар: `core_identity/car_semantics/main.py` (`no_car_proposals`). Сверка по **`meta.empty_reason`**, не только по подписи *empty*. |
| 2.0.5 | **`source_separation_extractor: empty`:** то же, что **2.0.4** | **(Р)** | `[x]` **зафиксировано** — **W5**; типично **`audio_too_short`** / **`audio_silent`** (см. `source_separation_extractor/main.py`). |
| 2.0.6 | **План 7 class:** снизить риск **load1»90** + таймаутов (параллелизм воркеров, `torch.cuda.empty_cache` между run, health backend) | **(Р)** | `[ ]` |
| 2.0.7 | **Пилот 15+ и `e2e_full_max_run`:** встроенный **example suite** расширен до **20** шагов (`builtin_example_suite_items` в [`e2e_full_max_run.py`](../../../backend/scripts/e2e_full_max_run.py)): слоты **11–20** = те же mock `platform_video_id` + `audit_v3_scen_06`…`15`. Ранее: лимит **10** — см. `e2e_terminal_20260420_065833_utc.log`. | **(О)** | `[x]` **код**; `[ ]` **верификация** (прогон `--example-suite-count 15` или **17** + мини-прогон **3** видео) |
| 2.0.8 | **Fetcher / heartbeat:** в логах E2E часто **`Fetcher 6/7`**, `fetch_api=COMPLETED`, `fetch_last=finalize` — не считать багом без сопоставления с контрактом Fetcher; при сомнении — зафиксировать в `RUN_LOG` / Fetcher docs. | **(Р)** | `[ ]` |
| 2.0.9 | **Preflight стека:** перед длинным сьютом проверять **Triton** (`/v2/health/ready` через `start_e2e_triton` или `TRITON_HTTP_URL`); **не** копить дубли **многомегабайтных** `e2e_terminal_*.log` (файл `e2e_terminal_20260420_065937_utc` по сути совпадает с `e2e_terminal_latest` по тому же прогону) — ротация / symlink / один канонический лог. | **(Р)** | `[ ]` |

**Наблюдения по указанным логам (2026-04-20):**

| Файл | Суть |
|------|------|
| `e2e_terminal_20260420_065833_utc.log` | Только preflight Triton + **мгновенный FAIL** из‑за **`--example-suite-count` > 10** (исправлено: план **20** слотов) — исторический, см. **2.0.7**. |
| `e2e_terminal_20260420_065937_utc.log` | Тот же полный прогон, что и **`e2e_terminal_latest.log`** (дубликат по объёму и содержанию, другой таймстемп в имени). |
| `e2e_terminal_latest.log` | Полный max-suite **10** видео: **ReadTimeout** E2E на plan 7, повтор **`action_recognition` error**, **`place_semantics` / `source_separation` empty**, единичный **`micro_emotion` error** на plan 6 — отражено в **2.0.1–2.0.5** и журнале. |
| (не E2E) **2026-04-22** | **Minimal Visual** по [`visual_minimal_object_detections_action_recognition.yaml`](../../configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml): валидация цепочки **YOLO → SlowFast** без full-max; см. [RUN_LOG.md](RUN_LOG.md). |

**Рецепт «только `core_object_detections` + `action_recognition`» (отладка AR без полного графа):** из `DataProcessor/` — `VisualProcessor/.vp_venv/bin/python VisualProcessor/main.py --cfg-path configs/audit_v3/visual/visual_minimal_object_detections_action_recognition.yaml` (в YAML — `frames_dir` / `rs_path`; при необходимости поменять `video_id` через каталог кадров и `run_id`).

**Инциденты *empty* на full-max E2E (2026-04-20) — нормальные причины (не считать регрессией без другого `empty_reason`):**

| Компонент | Типичный `empty_reason` / условие | Куда смотреть |
|-----------|-----------------------------------|---------------|
| `place_semantics` | `no_places_detected` | [place_semantics README §Empty](../../VisualProcessor/core/model_process/core_identity/place_semantics/README.md) |
| `car_semantics` | `no_car_proposals` | `car_semantics/main.py` (нет кандидатов после gating) |
| `source_separation_extractor` | `audio_too_short`, `audio_silent` | `AudioProcessor/.../source_separation_extractor/main.py` |

**Следующий шаг по *empty* на батче:** выборочно открывать NPZ / `meta` (или отчёты L2) и убедиться, что причина — из таблицы выше; иное — в [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) / тикет.

---

## 1. Фаза 0 — Роли и базовая фиксация

*Контекст (2026-04-15): проект ведёт **один разработчик** — **Илья**; пункты **0.1–0.3** формально разведены в таблице, фактически исполняются одним лицом.*

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 0.1 | Назначен **владелец реестра видео** (курация 60+, теги осей, sign-off набора) | **(О)** | `[x]` **Илья** (solo) |
| 0.2 | Назначен **владелец наблюдаемости** (Prometheus/Grafana, дашборды, алерты, документация URL) | **(О)** | `[x]` **Илья** (solo) |
| 0.3 | Назначен **владелец пайплайна / релиза** (заморозка кода, `config_hash`, запись в `RUN_LOG` после Go) | **(Р)** | `[x]` **Илья** (solo) |
| 0.4 | Согласован **целевой размер батча** (минимум **60** `video_id` по плану; целевое N: **70**) | **(О)** | `[x]` **70** уникальных `video_id` (2026-04-15, Илья) |

---

## 2. Фаза 1 — Конфиг и воспроизводимость (B0)

**Профиль батча (п. 1.1):** целевой режим — **максимально полный** (Segmenter + Audio + Text + Visual): все экстракторы/модули из шаблона [`global_config.yaml`](../../configs/global_config.yaml) остаются включёнными, **Visual** — все `core_providers` и `modules` в `true` (как `_enable_full_visual_inline_config` в [`e2e_full_max_run.py`](../../../backend/scripts/e2e_full_max_run.py)), **Text** — включить embeddings/связанные флаги в YAML, инфра (Triton/GPU/токены) — по возможности без отключения веток. Подробности и исключения: [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md).

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 1.1 | Зафиксирован **профиль прогона** (глобальный YAML + при необходимости тонкий profile для API): база **`DataProcessor/configs/global_config.yaml`**; **замороженный** снимок для батча (путь): `________________` | **(О)** | `[ ]` политика full max — [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md); отметить `[x]` после записи **фактического** пути замороженного файла |
| 1.2 | Зафиксированы **версии моделей / внешних сервисов** (кратко в таблице или ссылка на lockfile) | **(О)** | `[x]` [BATCH_MODEL_VERSIONS_SNAPSHOT.md](BATCH_MODEL_VERSIONS_SNAPSHOT.md) (2026-04-15); при смене стека — обновить снимок + **при пилоте** при желании приложить `pip freeze` |
| 1.3 | Вычислен и записан **`config_hash`** (или эквивалент: git tag + hash конфига): см. ниже | **(О)** | `[x]` **16-hex (только `global_config`):** `be0de63d921ffaf1` — `sha256( yaml.safe_dump({"global_config": <configs/global_config.yaml>}, sort_keys=True, allow_unicode=True).encode() )[:16]` · **SHA256 файла** `DataProcessor/configs/global_config.yaml`: `958075440cbc45a3e79f6220103a94ad09b9d9a66c67c4585c4001c8cb7f07b9` · *Фактический hash run в `main.py` может отличаться (profile, visual_cfg, chunk_size, CLI); перед Go пересчитать на **замороженном** YAML батча* |
| 1.4 | Зафиксирован **git commit** (или тег), с которого идёт батч: `________________` | **(О)** | `[ ]` черновик на 2026-04-15 при подготовке артефактов: `4c45b917c5c799c3e938ae0da78f5bcce0479b8c` — **перепроверить и заменить перед Go** |
| 1.5 | Политика **`emit_extra_metrics` / `compute_std`** для Text на батче выбрана и записана (да/нет/исключения): см. план §8.4 | **(О)** | `[x]` **везде `true`** для всех релевантных экстракторов Text в **замороженном** YAML батча (в шаблоне [`global_config.yaml`](../../configs/global_config.yaml) сейчас часто `false` — переопределить при заморозке); детали: [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md) § Text |
| 1.6 | Для Text зафиксировано в метаданных/чеклисте: **FAISS** целевой backend на батче (или осознанный numpy-only) — см. [FAISS_AND_NUMPY_BACKEND.md](../../TextProcessor/docs/FAISS_AND_NUMPY_BACKEND.md) | **(Р)** | `[x]` **только FAISS** — `faiss-cpu` или `faiss-gpu` в окружении прогона; `require_faiss: true` / режимы **не** numpy-fallback там, где поддерживается; см. [BATCH_FULL_PROFILE_REFERENCE.md](BATCH_FULL_PROFILE_REFERENCE.md) § Text |

---

## 3. Фаза 2 — Код, блокеры, регрессии (A)

### 3.1 Сквозные блокеры

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 2.1 | **TextProcessor:** на репрезентативном наборе (**≥5** видео, тот же класс конфига, что батч) **`text_processor` завершается успешно 5/5** ИЛИ оформлен **письменный waiver**: какие компоненты исключены из критерия успеха | **(О)** | `[ ]` диагностика набора B: 3/5 — **CUDA OOM** при загрузке `TitleEmbedder` / `intfloat/multilingual-e5-large` — см. [BATCH_60PLUS_OWNER_TASKS.md](BATCH_60PLUS_OWNER_TASKS.md) §3 · *Доп. эвиденс (не закрывает 5/5):* пилот E2E 2026-04-16 на `-Q6fnPIybEI` — строка `text_processor` в `manifest` **ok** (~3.5 min wall), см. журнал. |
| 2.2 | Нет известного **падения пайплайна до NPZ** на том же поднаборе (все три процессора или согласованный scope) | **(О)** | `[ ]` **Пилот 2026-04-16** (1 видео, см. журнал): Audio и Text — NPZ/manifest **OK**; Visual — **ошибка подпроцесса** (`exit=1`), артефактов visual **нет** (режим `--local-visual-no-triton`, `visual.required=false`). Критерий «все три процессора без падения до NPZ» **не выполнен**; нужен повтор с рабочим Visual (напр. Triton) или согласованный scope/waiver. |
| 2.3 | **`micro_emotion`:** исправлен PCA/вход ИЛИ компонент **исключён** из SLA батча и это задокументировано | **(О)** | `[x]` **PCA:** clamp + padding в `compute_au_pca` (историческая ошибка `-Ga4edhrfog`: `n_components=3` при `min(n_samples,n_features)=2`). Закрытие: [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) **W1** (resolved); smoke **2026-04-15** — `VisualProcessor/.vp_venv`. *L2 JSON 4 NPZ остаётся от прогона 2026-04-13; полное 5/5 после повторного E2E на `-Ga4edhrfog` — опционально.* |
| 2.4 | **`action_recognition`:** проверено на пилоте, что на выбранных видео **не вырождена** временная ось (достаточно клипов для целей батча) ИЛИ waiver | **(Р)** | `[x]` **Waiver:** пилот A+B не даёт multi-clip динамики (`tracks_with_multi_clips_total=0`, везде `num_clips=1` на трек) — см. [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) **W3**; JSON: `storage/audit_v4/action_recognition_l2/action_recognition_audit_v4_stats.json`. |

### 3.2 AudioProcessor

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 2.5 | После последних фиксов **саверов/tabular**: при необходимости выполнен **повтор опорного A** (см. audio-сводку) | **(Р)** | `[x]` **2026-04-16**, опорное видео **A** `-Q6fnPIybEI`, `run_id` **`437dd2f0-a239-424a-ad36-0026f63e094e`**: в `manifest.json` все строки **`kind=audio`** — **`status=ok`** (NPZ+render по экстракторам). Логи/сводка прогона: `storage/e2e_full_max/20260416-120234_utc/`. **Visual** в этом E2E не зафиксирован в манифесте (см. п. **2.2**); критерий пункта относится к **повтору audio** после фиксов — выполнен. Сводка: [AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md](AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md); журнал: [RUN_LOG.md](RUN_LOG.md) §«Сквозной E2E». |
| 2.6 | На пилоте проверен **сегментный режим**: полнота `run_segments` / payload для критичных экстракторов (минимум `spectral_extractor` по отчёту) | **(Р)** | `[x]` Тот же пилот **2026-04-16** / **`437dd2f0-a239-424a-ad36-0026f63e094e`** / `-Q6fnPIybEI`: **`spectral_extractor`** — `spectral_extractor_features.npz` с **канонической осью** (`segment_*` длины **12**, `segment_mask` **10** valid); в tabular **`hop_length`/`n_fft`/`duration`** конечны (**512 / 2048 / ~12.03 s**), без регрессии «4 NaN» из [spectral_extractor_audit_v4.md](components/audio_processor/spectral_extractor_audit_v4.md). Поле **`meta.run_segments`** в NPZ не сериализуется сейфером — контроль по отчёту: **payload сегментного пути** + tabular/meta STFT; здесь сегментный путь подтверждён осью и `segments_count`. Деталь: [RUN_LOG.md](RUN_LOG.md) §«Сквозной E2E». |
| 2.7 | На пилоте проверена согласованность **`meta.features_enabled`** с фактическим merge (минимум выборочно, `speech_analysis`/pitch) | **(Р)** | `[x]` Пилот **2026-04-16** / **`437dd2f0-a239-424a-ad36-0026f63e094e`**: **`speech_analysis_extractor`** — `meta.features_enabled` **`['asr_metrics']`**; в tabular **нет** `pitch_*`, **`pitch_distribution`** в NPZ пустой `{}` (сейфер не пишет pitch-скаляры без `pitch_metrics` в списке, см. `npz_savers/speech_analysis.py`). **`pitch_extractor`** — `meta.features_enabled` **`['basic_stats']`**; в NPZ **нет** `f0_series` / тайм-серийных ключей, только базовые поля + ось сегментов + `pitch_octave_distribution`. Отчёты: [speech_analysis_extractor_audit_v4.md](components/audio_processor/speech_analysis_extractor_audit_v4.md), [pitch_extractor_audit_v4.md](components/audio_processor/pitch_extractor_audit_v4.md); журнал: [RUN_LOG.md](RUN_LOG.md) §«Сквозной E2E». |
| 2.8 | Решено использование **`AP_ORCHESTRATOR_TELEMETRY`** на батче (да/нет) и кто читает `scheduler_runtime_report.json` | **(Р)** | `[x]` **Да** — для батча **~70** видео нужен **максимум** диагностики по AudioProcessor: **`AP_ORCHESTRATOR_TELEMETRY=1`**, дополнительно **`AP_ORCHESTRATOR_TELEMETRY_CHILDREN=1`** (субпроцессы / RSS детей), при необходимости отладки **`AP_ORCHESTRATOR_TELEMETRY_LOG=1`** (объём логов на **70** run оценить заранее). Итоговый JSON: **`{run_rs_path}/_reports/scheduler_runtime_report.json`** (ключ **`orchestrator_telemetry`**). **Кто читает:** владелец наблюдаемости по §**0** (**Илья**, роль совмещена с реестром и пайплайном — см. журнал выше); после батча — разбор пиков RAM/GPU и wall-time по экстракторам. Спека: [ORCHESTRATOR_TELEMETRY.md](../../AudioProcessor/docs/ORCHESTRATOR_TELEMETRY.md). |

### 3.3 VisualProcessor

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 2.9 | В дашбордах/руководстве учтены **маски и ожидаемые NaN** (не интерпретировать как баг без `present`/`valid_mask`) | **(Р)** | `[x]` Политика для батча и любых **NPZ/кадровых** панелей: [monitoring/README.md](../../monitoring/README.md) §«NPZ / визуальные ряды: маски и ожидаемые NaN»; контекст Visual — [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §8.3; эмпирика по модулям — [RUN_LOG.md](RUN_LOG.md). Штатный `dataprocessor-overview.json` — только API/очередь; при новых дашбордах по фичам — описания панелей со ссылкой на тот §. |
| 2.10 | Учтены **разные N** кадров по модулям; при необходимости в реестре/метаданных указаны политики Segmenter | **(Р)** | `[x]` Политика: не смешивать ряды разной длины без join; тег **`segmenter_N`** в реестре — про стратификацию контента, не один N на все модули; фактические оси — **metadata / NPZ**. См. [monitoring/README.md](../../monitoring/README.md) §«Разные N», [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §8.3, комментарий в [VIDEO_REGISTRY_60PLUS.yaml](VIDEO_REGISTRY_60PLUS.yaml). |
| 2.11 | **`meta.models_used`:** закрыты критичные дыры (или список известных исключений) | **(Р)** | `[x]` Зафиксирован **список согласованных исключений** и зон контроля: [monitoring/README.md](../../monitoring/README.md) §«meta.models_used»; сводка долгов Visual — [AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md](AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md), [VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md](VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md). Новые аномалии после батча — в отчёты компонентов и [RUN_LOG.md](RUN_LOG.md). |
| 2.12 | Политика **OCR / `retain_raw_ocr_text`** согласована с метриками качества (сырой текст может отсутствовать) | **(Р)** | `[x]` Батч: **`retain_raw_ocr_text: false`** в [`global_config.yaml`](../../configs/global_config.yaml) — метрики OCR по **детекциям/conf/счётчикам**, не по полному тексту кадра; отладка с сырым текстом — **отдельный dev-прогон**. См. [monitoring/README.md](../../monitoring/README.md) §«OCR», [ocr_extractor_audit_v4.md](components/visual_processor/core/ocr_extractor_audit_v4.md), [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §8.3. |

### 3.4 TextProcessor (дополнительно к2.1)

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 2.13 | В реестре заложены видео под **≥2 чанка** whisper, **`youtube_auto`**, длинный текст (для stats/shift/pairtopk/transcript_aggregator) | **(Р)** | `[ ]` *Не отдельный «текстовый» файл:* тот же [VIDEO_REGISTRY_60PLUS.yaml](VIDEO_REGISTRY_60PLUS.yaml) — тег **`transcript`** / **`notes`** (страта: много чанков whisper, трек `youtube_auto`, длинный spoken text). Разметка **вручную** при отборе или **после пилотного прогона** по факту ASR/чанков в `result_store`; квоты — [COVERAGE_MATRIX_60PLUS.md](COVERAGE_MATRIX_60PLUS.md) (ось «Транскрипт», строка Text L1 п. 23–24). |
| 2.14 | В реестре заложены **≥2** видео с **diar+ASR** для `speaker_turn_embeddings_aggregator` (или waiver) | **(Р)** | `[ ]` |
| 2.15 | Стратифицированы **комментарии** (богатые / пустые) и **QA** (вопросительные конструкции) | **(Р)** | `[ ]` |

### 3.5 Оптимизации и профилирование

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 2.16 | Критичные для батча пути прошли **профилирование** (хотя бы раз) или зафиксировано «не требуется» с обоснованием | **(О)** | `[ ]` См. **§3.5.1**. |
| 2.17 | Изменения производительности за **env-gate**; откат не требует релиза | **(Р)** | `[ ]` См. **§3.5.1**. |
| 2.18 | После существенных правок — **повтор L2-скрипта или E2E на 2–3** видео из набора B | **(Р)** | `[~]` **10** full-max E2E 2026-04-20 (`backend/.e2e/logs/e2e_terminal_latest.log`, см. журнал) — **шире** критерия; для закрытия по формулировке: **15-видео** пилот после **2.0** + ссылка на лог/артефакты. Старый пилот: `storage/e2e_full_max/20260416-120234_utc/`. |

#### 3.5.1 Пояснения к п. 2.16–2.18 (оптимизации и профилирование)

**П. 2.16 (обязательный) — профилирование критичных путей**

- **Зачем:** при батче ~70 роликов типичный риск — не «среднее», а **узкие места**: тяжёлый Visual, **OOM** у Text, **пик RAM** воркера, долгий **ASR**, **Triton** / субпроцессы; без замера можно упереться в таймаут или ресурс и не знать источник.
- **Что считается выполненным:** хотя бы **один** пилот (1–3 видео, желательно из будущего набора **B**) с замером для цепочек **того же профиля**, что батч:
  - wall-time по крупным стадиям: логи, `meta.stage_timings_ms` где есть, **`scheduler_runtime_report.json`** при **`AP_ORCHESTRATOR_TELEMETRY=1`** ([ORCHESTRATOR_TELEMETRY.md](../../AudioProcessor/docs/ORCHESTRATOR_TELEMETRY.md));
  - при необходимости — **RSS/GPU**: телеметрия оркестратора, фоновый монитор, **`VP_RESOURCE_PROFILE=1`** для части Visual (см. [components/audit_4_2/README.md](components/audit_4_2/README.md)).
- **Альтернатива:** в чеклисте / журнале явно **«профилирование не требуется»** + **краткое обоснование** (редкий случай; иначе для Go **(О)** пункт должен быть закрыт замером).
- **Процесс:** [AUDIT_4_CRITERIA_AND_PLAN.md](AUDIT_4_CRITERIA_AND_PLAN.md) §12.4.6; [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) (чекбокс про профилирование).

**П. 2.17 (рекомендуется) — ускорения только за env-gate**

- **Зачем:** экспериментальные ускорения (кеш, батчинг, отключение диагностик, иные ветки) включаются **переменной окружения** или явным флагом конфига, а не «тихим» изменением поведения по умолчанию.
- **Откат на батче:** снять env / вернуть флаг — **без** отката git-релиза.
- **Примеры в репо:** префиксы **`AP_*`** (Audio), **`VP_RESOURCE_PROFILE`** (доп. снимки в meta Visual); новые флаги — задокументировать в README / engineering log 4.2.
- **Как закрыть:** короткая запись: новые perf-изменения только за env; перечень имён — в документации модуля или [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §0 / §8.

**П. 2.18 (рекомендуется) — повтор L2 или E2E на 2–3 видео набора B**

- **Зачем:** после **существенных** правок (саверы, схемы NPZ, оркестратор, критичные экстракторы) один прогон только на опорном **A** не гарантирует отсутствие сюрпризов на **разнообразии B**.
- **Что сделать:** **2–3** разных `video_id` из набора **B** (см. [RUN_LOG.md](RUN_LOG.md): `-15jH8mtfJw`, `-5EYUqIlyJU`, `-7Ei8e05x30`, `-Ga4edhrfog` и т.д.) — полный **E2E** или **`audit_v4_npz_stats.py`** по затронутым компонентам; сверить **manifest** и ключевые NPZ.
- **Текущее состояние:** серия **10** full-max E2E 2026-04-20; для закрытия п. **2.18** в актуальном смысле — **15-видео** пилот (тот же путь, что батч) после **п. 2.0**.

---

## 4. Фаза 3 — Реестр 60+ и матрица покрытия (B)

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 3.1 | Создан артефакт реестра: **`VIDEO_REGISTRY_60PLUS.yaml`** или **`.csv`** (путь в репо или storage): `DataProcessor/docs/audit_v4/VIDEO_REGISTRY_60PLUS.yaml` | **(О)** | `[x]` seed 5 записей; дополнить до N из п. **0.4**; проверка: `python DataProcessor/docs/audit_v4/scripts/validate_video_registry_60plus.py` (перед Go — с `--strict-count`). Расшифровка: **п. 3.1** в подразделе ниже. |
| 3.2 | В реестре **≥70** уникальных `video_id` (согласованное N из п. **0.4**; не меньше **60** по плану) | **(О)** | `[ ]` См. **п. 3.2** ниже. |
| 3.3 | У каждой записи есть **теги осей** из плана §3.1 (длительность, речь/музыка, лица, OCR, язык, метаданные, транскрипт, N кадров — по применимости) | **(О)** | `[ ]` См. **п. 3.3** ниже. |
| 3.4 | Каждая ось из плана представлена **минимум 3–5** видео (кроме явно редких классов с waiver) | **(О)** | `[ ]` См. **п. 3.4** ниже. |
| 3.5 | Построена **матрица покрытия**: компонент / группа → тег оси → минимум N видео (ссылка на таблицу): [COVERAGE_MATRIX_60PLUS.md](COVERAGE_MATRIX_60PLUS.md) | **(О)** | `[x]` шаблон; заполнить факты после тегов в реестре. См. **п. 3.5** ниже. |
| 3.6 | В реестр включены **edge (C)** кейсы: минимум **2** (короткое аудио, тишина, нет лица, пустые семантики — по договорённости) | **(Р)** | `[ ]` См. **п. 3.6** ниже. |
| 3.7 | Реестр **версионирован**: hash списка id + дата + привязка к `config_hash` | **(О)** | `[ ]` См. **п. 3.7** ниже. |
| 3.8 | **Sign-off** владельца реестра (ФИО/дата): `________________` | **(О)** | `[ ]` См. **п. 3.8** ниже. |

#### Расшифровка п. 3.1–3.8 (реестр 60+ и матрица)

Источник осей и логики набора: [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §3.1–3.3. Задачи владельца: [BATCH_60PLUS_OWNER_TASKS.md](BATCH_60PLUS_OWNER_TASKS.md) §2.

**П. 3.1 — артефакт реестра**

- **Что это:** один **список видео** для батча в машиночитаемом виде. В репозитории принят YAML: [`VIDEO_REGISTRY_60PLUS.yaml`](VIDEO_REGISTRY_60PLUS.yaml) (`schema_version`, `target_video_count`, список `videos` с `video_id`, опционально `run_id`, `tags`, `notes`).
- **Зачем YAML:** удобно версионировать в git, прогонять [validate_video_registry_60plus.py](scripts/validate_video_registry_60plus.py) (структура, дубликаты, перед Go — `--strict-count` против `target_video_count`).
- **CSV:** допустим, если команда так договорится — тогда путь и формат нужно **явно** указать в этом чеклисте / журнале и не потерять связь с матрицей покрытия.
- **Сейчас:** заготовка на **5** записей; цель по п. **0.4** — **70** `video_id` (не ниже **60** по плану).

**П. 3.2 — размер и уникальность**

- **≥70** (или согласованное N из п. **0.4**) **уникальных** `video_id` — без дубликатов в одном батче (валидатор это проверяет).
- **Запас:** часть роликов может «отвалиться» (геоблок, удалённое видео, сбой fetch) — при необходимости заложить **запас** сверх минимума **60** из плана.
- После заполнения: пересчитать `current_video_count` в YAML или опереться только на длину списка + `--strict-count`.

**П. 3.3 — теги осей у каждой записи**

- Каждая строка реестра должна иметь **теги** по осям из плана [PLAN_PREP §3.1](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) («Оси контента»): например `duration`, `speech_music`, `editing_pacing`, `faces`, `on_screen_text`, `language`, `metadata_richness`, `transcript`, `segmenter_N` — как в шаблоне записи (сейчас у seed-записей стоит `todo`, перед Go нужно **конкретные** значения).
- **По применимости:** не у каждого ролика обязательны все оси в явном виде, но тогда в `notes` должно быть понятно, почему ось нерелевантна, либо значение наследуется из жанра отбора.
- Теги используются для **п. 3.4** и для заполнения [COVERAGE_MATRIX_60PLUS.md](COVERAGE_MATRIX_60PLUS.md).

**П. 3.4 — квоты по осям**

- Каждая **ось** из §3.1 плана должна быть представлена **минимум 3–5** видео в реестре (типовое правило «не один outlier на ось»).
- **Исключения:** заведомо редкие классы (например diar+ASR для `speaker_turn_embeddings_aggregator`) — **≥2** экземпляра по правилу баланса плана; при меньшем — **waiver** в [BATCH_60PLUS_WAIVERS.md](BATCH_60PLUS_WAIVERS.md) с обоснованием.
- Проверка: после проставления тегов — подсчёт по оси в матрице или простым скриптом / pivot.

**П. 3.5 — матрица покрытия**

- **COVERAGE_MATRIX_60PLUS.md** связывает **требования компонентов** (из L1/L2 аудитов) с **осями** и **минимумом N** видео.
- Шаблон таблицы уже есть; колонка «Факт» заполняется **после** тегов в реестре (сколько `video_id` попало в нужную страту).
- Дополнительные строки («хвосты») добавляются по мере разбора отчётов — см. уже внесённые примеры (Text diar, action_recognition, и т.д.).

**П. 3.6 — edge-набор (C), рекомендуется**

- Минимум **2** видео с **пограничным** поведением: очень короткое аудио, тишина, почти нет лица, пустые комментарии/семантики, экстремальная нарезка — **по договорённости** команды, но явно помечены в реестре (`notes` / отдельный тег), чтобы батч не «успешно» прошёл только на «удобном» контенте.
- Полный набор **C** не обязателен для первого батча, но несколько edge-слотов резко снижают риск сюрпризов в прод-метриках.

**П. 3.7 — версионирование**

- Перед Go: **hash** содержимого списка id (в чеклисте п. **3.7** указан `shasum -a256 VIDEO_REGISTRY_60PLUS.yaml` в комментарии к реестру) + **дата** + связь с **`config_hash`** замороженного пайплайна (чеклист п. **1.3**, [BATCH_MODEL_VERSIONS_SNAPSHOT.md](BATCH_MODEL_VERSIONS_SNAPSHOT.md)), чтобы воспроизводимо знать «какой набор под какой конфиг».

**П. 3.8 — sign-off**

- Владелец реестра (чеклист §**0**, сейчас **Илья**) подписывает готовность набора: дата, при необходимости ссылка на commit с финальным YAML и заполненной матрицей.
- До подписи не выполняются обязательные условия Go из §**7** для пунктов **3.2–3.5, 3.7**.

---

## 5. Фаза 4 — Наблюдаемость (C)

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 4.1 | **Prometheus** скрейпит **`GET /api/v1/metrics`** (и при настроенном воркере — **`/metrics` на `DP_WORKER_METRICS_PORT`**) в среде батча | **(О)** | `[x]` **Локальный E2E** — два target **UP** ([OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md)). `[ ]` **Прод/стенд** батча — зафиксировать job/URL. См. **п. 4.1** ниже. |
| 4.2 | **Grafana** доступна команде; зафиксированы **URL** и учётные данные (хранилище секретов / README): `________________` | **(О)** | `[x]` **Локально:** `localhost` + [monitoring_ports.env](../../../backend/.e2e/state/monitoring_ports.env) / дефолт **3000**, admin/admin (compose). `[ ]` **Прод/стенд.** См. **п. 4.2** ниже. |
| 4.3 | Дашборд **обзорный** (очередь, активные run, ошибки, latency по `processor`/`component`) | **(О)** | `[x]` [dataprocessor-overview.json](../../monitoring/grafana/dashboards/dataprocessor-overview.json) (provisioning в compose). `[ ]` Проверка на **боевом** datasource. См. **п. 4.3** ниже. |
| 4.4 | Дашборды или **фильтры по подсистемам** (Audio / Visual / Text) | **(Р)** | `[ ]` См. **п. 4.4** ниже. |
| 4.5 | Проведена **инвентаризация labels**: какие компоненты реально попадают в `dataprocessor_processing_seconds` при запуске батча | **(О)** | `[ ]` [METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md). См. **п. 4.5** ниже. |
| 4.6 | Закрыт пробел **CLI vs API:** если батч идёт в обход API — метрики попадают в Prometheus **единым способом** (доработка / exporter / договор о маршруте) | **(О)** | `[x]` Путь **E2E с API+worker** (план пилота 15/60+) — [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md). `[ ]` Если появляется **только** CLI-прогон — отдельное решение. См. **п. 4.6** ниже. |
| 4.7 | Документировано в **`monitoring/README.md`** и/или **`RUN_LOG.md`**: как смотреть прогон 60+, retention | **(О)** | `[x]` [monitoring/README.md](../../monitoring/README.md) § «Батч 60+». См. **п. 4.7** ниже (что ещё должен закрыть владелец). |
| 4.8 | **(Опционально)** cheap-метрики качества (`meta.status`, ключевые доли NaN) — sidecar / периодический push | **(Р)** | `[ ]` См. **п. 4.8** ниже. |
| 4.9 | **(Опционально)** алерты: рост `failures_total`, застой очереди, p95 latency по компоненту | **(Р)** | `[ ]` См. **п. 4.9** ниже. |
| 4.10 | Согласованы пороги **стоп-крана** (например, N подряд ошибок одного типа — см. план §8.7) | **(Р)** | `[ ]` См. **п. 4.10** ниже. |

#### Расшифровка п. 4.1–4.10 (наблюдаемость, фаза C)

Контекст плана: [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §4 (Prometheus/Grafana, расширения). Локальная схема: [monitoring/README.md](../../monitoring/README.md), [monitoring/prometheus/prometheus.yml](../../monitoring/prometheus/prometheus.yml), [monitoring/prometheus/alerts.yml](../../monitoring/prometheus/alerts.yml). Задачи владельца среды: [BATCH_60PLUS_OWNER_TASKS.md](BATCH_60PLUS_OWNER_TASKS.md) §5.

**П. 4.1 — Prometheus и scrape**

- **Цель:** в **той же среде**, где крутится батч (не только dev `localhost`), Prometheus **регулярно** опрашивает источник метрик DataProcessor.
- **По умолчанию в репо:** HTTP **`GET /api/v1/metrics`** у **dataprocessor-api** (см. `monitoring/README.md`, `prometheus.yml`). Для **полной** картины по `dataprocessor_processing_seconds` / `dataprocessor_failures_total` (пайплайн) — второй target на **процесс worker**: **`GET /metrics`** на порту **`DP_WORKER_METRICS_PORT`** (в E2E по умолчанию **8003**, не 8001 — Backend API).
- **Локальный E2E (проверено 2026-04-22):** [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md) — `start_e2e_stack.sh --with-infra`, compose override `prometheus.e2e_host.yml`, `host.docker.internal` для scrape с контейнера.
- **Если endpoint другой** (TLS, ingress, sidecar): зафиксировать **точный URL** и job-имя в конфиге Prometheus и здесь в чеклисте / `RUN_LOG`.
- **Проверка:** в Prometheus UI → Status → Targets: target в состоянии **UP**, после 1–2 прогонов появляются новые samples у `dataprocessor_*` (оба job’а для API+worker).

**П. 4.2 — Grafana**

- Команда, сопровождающая батч, должна **открывать** Grafana и входить под учёткой (или SSO).
- **Локально (2026-04-22):** порт **3000** (или выбранный `E2E_GRAFANA_HOST_PORT`, см. `backend/.e2e/state/monitoring_ports.env`); default login из compose: **admin** / **admin** — [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md).
- **URL** боевой (или staging) среды и **учётные данные** не хранить в открытом виде в git — password manager / внутренняя wiki; в README допустима ссылка «где взять доступ».
- Для Go желательно одна строка в [monitoring/README.md](../../monitoring/README.md) или внутренней доке: «батч 60+ → Grafana: …».

**П. 4.3 — обзорный дашборд**

- Минимум: **очередь**, **активные run**, **ошибки** (`failures_total` / crashed), **время обработки** (например p95 `dataprocessor_processing_seconds` с разрезом `processor`, `component` там, где labels реально не `unknown`).
- В репозитории есть заготовка: `monitoring/grafana/dashboards/dataprocessor-overview.json` — её нужно **импортировать/провизионить** в вашей Grafana и убедиться, что панели показывают данные при нагрузке.
- Не ожидать от этого дашборда **latency по каждому экстрактору** внутри `main.py` — для этого нужны отдельные метрики или `scheduler_runtime_report.json` / телеметрия Audio ([ORCHESTRATOR_TELEMETRY.md](../../AudioProcessor/docs/ORCHESTRATOR_TELEMETRY.md), чеклист **2.8**).

**П. 4.4 — подсистемы (рекомендуется)**

- Отдельные дашборды или **переменные-фильтры** в одном дашборде по **Audio / Visual / Text**, если в labels или в отдельных метриках это различимо.
- Сейчас при маршруте **API → worker → один subprocess `main.py`** основная гистограмма даёт **сквозной** `processor=pipeline`, `component=main_py` — фильтрация «только Visual» из одной только этой метрики **невозможна** без доработок (см. **4.5**, инвентаризацию).

**П. 4.5 — инвентаризация labels**

- Документ: [METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md) — какие labels реально выставляет worker, что означает **`pipeline` / `main_py`** после патча `processor.py`.
- **Обязательно:** на **пилоте** (1–2 run через тот же путь, что батч) в Prometheus проверить `dataprocessor_processing_seconds` и `dataprocessor_failures_total` — нет ли повсеместного `unknown`.
- Понимание ограничения: **per-extractor** latency в Prometheus из текущего API-контура **не** обязано совпадать с детализацией в `manifest` / NPZ.

**П. 4.6 — CLI vs API**

- Если батч запускается **только** через CLI / cron без постановки задач в API-очередь, **экспорт метрик** через тот же `/api/v1/metrics` может **не обновляться**.
- **Текущий план 60+ / пилот 15** — **API + worker** (как [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md)); в этом сценарии **4.6** для батча **закрыт** при условии, что прод/staging используют **тот же** контур.
- Нужен **единый согласованный** путь, если вдруг появится **только** CLI-прогон: отдельный **exporter**, **Pushgateway** — зафиксировать в `RUN_LOG` / [BATCH_60PLUS_OWNER_TASKS.md](BATCH_60PLUS_OWNER_TASKS.md).
- Условие **Go** по §**7** для **4.6** не выполнено, если **основной** путь батча — **чисто CLI** и метрики **не** покрываются.

**П. 4.7 — документация прогона и retention**

- **Сделано:** в `monitoring/README.md` есть секция **«Батч 60+»** (ссылка на чеклист, реестр, labels, предупреждение про `localhost`).
- **Остаётся за владельцем наблюдаемости:** боевые **URL** Prometheus/Grafana, **retention** и политика хранения (см. [BATCH_60PLUS_OWNER_TASKS.md](BATCH_60PLUS_OWNER_TASKS.md) §5); при необходимости — строка в `RUN_LOG` после пилота с фактическими URL.

**П. 4.8 — cheap-метрики качества (опционально)**

- Идея из плана §4: лёгкие сигналы **качества NPZ** (доля `meta.status != ok`, выборочные доли NaN) без полного L2 на каждом run — **sidecar**, периодический job, Pushgateway.
- Имеет смысл, если в Grafana нужно видеть **деградацию фич**, а не только «run завершился».

**П. 4.9 — алерты (опционально)**

- В репозитории: [monitoring/prometheus/alerts.yml](../../monitoring/prometheus/alerts.yml) (очередь, crashed runs, время, память, failure rate и т.д.).
- Перед батчем: включить нужные правила в **вашем** Prometheus, настроить **Alertmanager** / маршрут уведомлений, пороги под вашу нагрузку.

**П. 4.10 — стоп-кран**

- Организационное правило: при **N** подряд одинаковых фатальных ошибках или аномалии метрик — **пауза** батча, разбор, не «добивать» 70 видео вслепую.
- Ориентир по духу: [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §8.7; конкретные числа **N**, время окна и ответственный фиксируются командой и при желании заносятся в `RUN_LOG` / внутреннюю инструкцию.

---

## 6. Фаза 5 — Сухой прогон (B4)

| # | Пункт | О/Р | Отметка |
|---|--------|-----|---------|
| 5.1 | Выполнен сухой прогон: **минимум 15** `video_id` (согласовано 2026-04-20: «сначала 15», затем gate на **70**) из **финального** реестра на **том же** пути, что и батч (API/воркер/очередь). Ранний ориентир *5–10* из плана B4 расширен, чтобы поймать редкие взаимодействия. | **(О)** | `[ ]` **Gate:** зелёный 15-прогон → **70**; см. **п. 5.1** ниже. |
| 5.2 | В Grafana видны **ожидаемые** серии (не пустой scrape, корректные `processor`/`component` где применимо) | **(О)** | `[ ]` См. **п. 5.2** ниже. |
| 5.3 | Нет **системных дыр** (молчание метрик для целого процессора, 100% fail без объяснения) | **(О)** | `[ ]` См. **п. 5.3** ниже. |
| 5.4 | Оценено **end-to-end время** и ресурсы на пилоте (очередь, GPU, ретраи) | **(Р)** | `[ ]` Черновик E2E — **п. 5.4** ниже. |
| 5.5 | Краткий **отчёт пилота** (ссылка): `________________` | **(О)** | `[ ]` См. **п. 5.5** ниже. |

#### Расшифровка п. 5.1–5.5 (сухой прогон B4)

Веха **B4** в плане: [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) (таблица вех, §8.7 про пилот). Условие **Go** в §**7** чеклиста: **5.1–5.3**, **5.5** обязательны.

**П. 5.1 — прогон 15 видео «как в батче» (затем 70 при успехе)**

- **Зачем:** поймать проблемы **оркестрации** (очередь, воркер, Redis, лимиты параллелизма), а не только корректность кода на одном ролике у разработчика.
- **Объём (2026-04-20):** **15** `video_id` (минимум; можно расширить, если 15 зелёный) из **финального** [VIDEO_REGISTRY_60PLUS.yaml](VIDEO_REGISTRY_60PLUS.yaml) (после тегов и версионирования п. **3.x**), с **тем же** `global_config` / профилем, что запланирован на батч (**тот же `config_hash`**). При **успешном** 15-прогоне — **70**.
- **Путь:** тот же, что основной батч — как правило **API → очередь → worker → `main.py`**; если батч пойдёт через CLI — пилот тоже через CLI **и** должно быть закрыто **4.6** (метрики).
- **Не путать** с пилотом «1 видео + mock» ([`storage/e2e_full_max/`](../../../storage/e2e_full_max/)): он полезен для отладки стека, но **не заменяет** B4, если отличаются очередь, ingress или железо.

**П. 5.2 — Grafana на пилоте**

- Пока идут 5–10 run, в Grafana видны **живые** серии: длина очереди, активные run, рост счётчиков обработки, ошибки.
- **Ожидаемые labels:** для маршрута API→`main_py` — см. [METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md) (`pipeline` / `main_py`); не должно быть полного «нуля» по метрикам при работающих run.
- Если панели пустые при заведомо успешных run — искать разрыв scrape / другой job / другой namespace.

**П. 5.3 — отсутствие системных дыр**

- **Примеры дыр:** метрики **не обновляются** для всего процессора при успешных джобах; **100%** `failures` без объяснения в логах; очередь **растёт** бесконечно; воркер **молчит** при наличии сообщений.
- **Не путать** с ожидаемыми **контентными** фейлами (пустой ASR, нет лица) — они видны в `manifest` / NPZ, но не обязаны давить весь пайплайн.
- Итог пилота: явная формулировка «системных дыр нет» или список **блокеров** до Go.

**П. 5.4 — время и ресурсы (рекомендуется)**

- Зафиксировать **wall-clock** на полный цикл по **нескольким** видео (min / max / типичный), очередь ожидания, **GPU**/RAM пики (из телеметрии, `manifest` timings, `scheduler_runtime_report.json` при **`AP_ORCHESTRATOR_TELEMETRY`**).
- Экстраполяция на **70** видео — грубая оценка длительности батча и риска таймаутов.

**Черновик (не закрывает 5.1):** пилот **2026-04-16**, **1** видео `-Q6fnPIybEI`, offline mock, `local_visual_no_triton` — каталог [`storage/e2e_full_max/20260416-120234_utc/`](../../../storage/e2e_full_max/20260416-120234_utc/), `run_id` `437dd2f0-a239-424a-ad36-0026f63e094e`. По поллеру E2E ~**14.5 min** до `ingestion completed`. В `manifest.json`: `audio_processor_internal_wall_ms` ~**454.9 s**, `text_processor_internal_wall_ms` ~**211.7 s**; Visual — **exit=1** ~**35 s**, в манифест **не** попали visual-компоненты. Детализация по экстракторам: `manifest.json`, `_reports/run_manifest_summary.json`, `scheduler_runtime_report.json`.

**П. 5.5 — отчёт пилота**

- Короткий артефакт (wiki / markdown / gist / письмо): **дата**, **commit**, **список video_id**, **путь оркестрации**, **ссылка на Grafana** (snapshot или dashboard), **итог** (готовы к батчу / блокеры), при необходимости — ссылка на **RUN_LOG**.
- Строка в таблице чеклиста: подставить **URL или путь** к отчёту вместо плейсхолдера.
- После успешного B4: по плану — запись в [RUN_LOG.md](RUN_LOG.md) о готовности к батчу 60+ (дата, commit, реестр, Grafana).

---

## 7. Финальный Go — запуск 60+

Условия **Go** (все обязательные **(О)** выше отмечены `[x]`, либо оформлен письменный **waiver** с ссылкой на пункт):

- [ ] **2.0** — **2.0.1** частично; **2.0.2** — код + **локальная** проверка minimal Visual **OK** (2026-04-22), **full-max E2E** без `action_recognition` error ещё нужен; **2.0.3** частично; полная верификация — пилот **15** / завершённый E2E без stop  
- [ ] **2.1** — Text 5/5 или waiver  
- [ ] **3.2–3.5, 3.7** — реестр и матрица  
- [ ] **4.1–4.3, 4.5–4.7** — наблюдаемость и документация *(локальный E2E: **4.1–4.3**, **4.6**, **4.7** — задокументировано; **прод-URL 4.1–4.2**, **пилот 4.5** — TBD; см. [OBSERVABILITY_STACK_LOCAL_E2E.md](OBSERVABILITY_STACK_LOCAL_E2E.md))*  
- [ ] **5.1–5.3, 5.5** — сухой прогон  
- [ ] **1.1–1.5** — конфиг и политики  

**Подпись владельца аудита / техлида:** __________________ · **Дата:** __________________  

**Запуск батча разрешён с:** __________________ (дата/время UTC)  

---

## 8. После запуска (не блокирует Go, но запланировать)

| # | Пункт | Отметка |
|---|--------|---------|
| 8.1 | Запись в [RUN_LOG.md](RUN_LOG.md): дата старта батча, commit, ссылка на реестр, Grafana, итог (по завершении) | `[ ]` |
| 8.2 | Выборочный пост-аудит: **5–10** случайных run, `audit_v4_npz_stats` или лёгкие проверки | `[ ]` |
| 8.3 | Сводка по батчу: `run_id`, статусы, топ ошибок (CSV/отчёт) | `[ ]` |
| 8.4 | Проверка **диска / retention** по факту заполнения storage | `[ ]` |

---

*Конец чеклиста. При изменении плана обновляйте ссылки и нумерацию согласованно с [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md).*
