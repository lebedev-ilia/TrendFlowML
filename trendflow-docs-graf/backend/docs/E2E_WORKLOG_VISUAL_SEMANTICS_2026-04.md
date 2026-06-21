# E2E: визуальные семантические головы, Embedding Service и тайминги (апрель 2026)

Документ суммирует работу по серии полных прогонов **max-E2E** (Fetcher + DataProcessor + VisualProcessor + Embedding Service + Triton), цели («зелёные» компоненты, стабильный сьюит), внесённые правки в код и инфраструктуру, а также **итоги по логам** и ориентиры по времени.

Связанные материалы:

- Общие фиксы E2E / DP / audio: [`E2E_DP_FIXES_2026-04.md`](E2E_DP_FIXES_2026-04.md)
- Runbook и команды: [`E2E_RUNBOOK.md`](E2E_RUNBOOK.md)

---

## 1. Цели сессии

1. Семантические и смежные визуальные компоненты не должны валить прогон: **`franchise_recognition`**, **`brand_semantics`**, **`face_identity`**, **`place_semantics`**, **`car_semantics`** (и по ходу — **`action_recognition`**).
2. Полный сьюит из **10 видео** должен завершаться с **`e2e_exit=0`** и **`ingestion_status=completed`** по каждому пункту плана.
3. Зафиксировать **длительность** прогонов, **узкие места по времени**, направления оптимизации **без сознательной порчи качества** (где это возможно — например OOM-fallback с тем же inference).
4. Подготовка к следующему прогону на **17 видео**: ожидание отсутствия ошибок по перечисленным компонентам при корректной инфраструктуре.

---

## 2. Инфраструктура: Embedding Service (PostgreSQL `embeddings`)

Семантические модули делают **fail-fast**, если в категории **0 labels** в Embedding Service.

**Правка:** `backend/scripts/setup_e2e_infra.sh` — после создания таблицы `embeddings` в БД `embeddings` добавлен **multi-row seed** (один и тот же L2-нормированный вектор размерности **512**):

| UUID (пример) | `category` | `embedding_model` (как в `category_model_mapping`) |
|---------------|------------|-----------------------------------------------------|
| `…0001` | `franchise` | `clip_224` |
| `…0002` | `brand` | `clip_336` |
| `…0003` | `face` | `arcface` |
| `…0004` | `place` | `clip_448` |
| `…0005` | `car` | `clip_336` |

`ON CONFLICT (id) DO NOTHING` — повторный запуск скрипта не ломает существующие строки.

**Практика:** после смены БД или чистой установки выполнить `setup_e2e_infra.sh` (или эквивалентный SQL), иначе `get_labels` для `brand` / `face` / `place` / `car` снова даст **0 labels**.

---

## 3. Правки в коде (репозиторий)

### 3.1 DataProcessor API / очередь (контекст более ранней сессии)

- **Проблема:** расхождение экземпляров `TaskManager` между API и worker → некорректный **`active_runs`** (например «8/8» при свободных слотах).
- **Направление правки:** Redis-схема и health/process endpoints используют **согласованный подсчёт** активных ранов (см. `DataProcessor/api/services/redis_schema.py`, `api/endpoints/health.py`, `api/endpoints/process.py`).

### 3.2 `franchise_recognition`

- Устойчивое сравнение эмбеддингов «DB vs `core_clip`»: функция **`_franchise_embeddings_to_matrix`**, проверка размерности **`D` с `frame_emb.shape[1]`**, **`try/except`** с переходом на **image search** вместо падения на `np.dot`.
- Нормализация ключей UUID: **`str(...)`** для совпадения с ответами API / `numpy.str_`.
- Прогресс **`process_frames`** только при успешном direct-path; иначе отрабатывает fallback-ветка.

### 3.3 `place_semantics`

Аналогично franchise:

- **`_place_embeddings_to_matrix`**, проверка **`D`**, **`try/except`** → fallback на **поиск по кадру**.
- Исправлена ветка image search: инициализация **`processed_frames = 0`** (раньше возможен **`UnboundLocalError`** при `+=`).
- Флаг **`direct_path_completed`** вместо «полупустого» direct-path без fallback.

### 3.4 `brand_semantics` и `car_semantics`

- Вызов **`embedding_client.search`** обёрнут в **`try/except RuntimeError`**: после исчерпания ретраев клиента — **предупреждение в лог** и **пустой результат** для трека/кандидата, **без `exit 1`** у subprocess.
- Смысл: сетевые/HTTP **5xx** или временная недоступность Embedding Service не рвут весь VisualProcessor run. Качество: при ошибке нет top-K совпадений (деградация результата, не краш).

### 3.5 `cut_detection` (cascade / hard cuts)

- Подробнее в README модуля: `DataProcessor/VisualProcessor/modules/cut_detection/README.md` (блок про cascade / sparse pass 2).
- В режиме **`cascade_enabled`** второй проход больше не делает полный последовательный скан по всем кадрам: для каждой пары-кандидата **`j`** вызываются **`get(frame_indices[j])`** и **`get(frame_indices[j+1])`** (то же преобразование в grayscale, те же SSIM / локальный Farneback / deep, что и раньше для этой пары).
- При **`external_flow_mags`** (baseline: **`core_optical_flow`**) поток по-прежнему берётся из массива для **всех** пар; экономия — в **меньшем числе загрузок кадров** для SSIM (и deep, если включён).
- Исправлена ветка адаптивного **`deep_thresh`** в cascade (раньше использовалось неопределённое имя при `use_deep_features=true`).

### 3.6 `action_recognition` (SlowFast)

- **`embedding_proj`** приводится к тому же **`dtype`**, что и backbone (устранение типичных ошибок **mixed precision** на `Linear`).
- Входы **slow/fast** перед forward явно приводятся к **`device`/`dtype`** параметров модели.
- Вынесен **`_infer_batch_tensors`**; при **OOM / CUDA** на батче — **автоматический fallback на поштучные клипы** (та же модель и постобработка, иной только batching — качество не режется ради скорости).

### 3.7 `speaker_diarization_extractor` (pyannote)

- **`_load_pyannote_pipeline`**: после успешного `ModelManager.get(...)` убран ранний **`return`**, который делал недостижимым блок **`pipeline.to(cuda)`**. Теперь при **`device=cuda`** / **`auto`→CUDA** пайплайн реально выполняется на GPU, если перенос успешен (ожидаемое сокращение wall time на длинных дорожках по сравнению с «тихим» CPU).
- При OOM-fallback на CPU после успешного прогона выставляется **`self.device_str = "cpu"`**, чтобы **`device_used`** в payload совпадал с фактическим устройством.
- README компонента: `DataProcessor/AudioProcessor/src/extractors/speaker_diarization_extractor/docs/README.md` (§ Performance).

### 3.8 `text_processor` (TextProcessor orchestrator)

- **`MainProcessor.run()`**: при **`batch_enable_cpu_parallel=True`** (как в `run_cli` / `MainProcessor` из глобального конфига) подряд идущие на CPU **`LexicalStatsExtractor`** и **`ASRTextProxyExtractor`** бандлятся и гоняются в **`ThreadPoolExecutor`** (один документ). Остальные экстракторы — по-прежнему последовательно; расширять параллелизм на другие имена без аудита мутаций/`tp_artifacts` не стоит.
- **`_effective_extractor_device` + `run_batch`**: слот **`cpu2`** с **`extractor_params.device=cuda`** теперь попадает в GPU-ветку для пост-шаговой CUDA-гигиены и корректной группировки уровней DAG.
- Документация: `DataProcessor/TextProcessor/docs/audit_v3/README.md` (§ Performance).

---

## 4. Наблюдения по логам E2E

### 4.1 Файлы логов

- Компактный heartbeat: `backend/.e2e/logs/e2e_terminal_*.log`, агрегат **`e2e_terminal_latest.log`**.
- Детальные логи сервисов: `backend/.e2e/logs/<timestamp>/<service>/process.log`.

### 4.2 Сьюит 10× видео (`e2e_terminal_latest.log`)

- После правок по очереди: для каждого **`plan_item=1..10`** — **`Done: ingestion_status = completed`**, **`e2e_exit=0`**.
- **Fetcher** в heartbeat часто остаётся **`6/7`** — зафиксировано как наблюдение; на успешность **`e2e_exit`** в проверенном сьюте не влияло.
- До правок в логах фиксировались:
  - **`place_semantics`**: падение сразу после строки *«Using direct embedding comparison with N places»* (матрица / размерности / ветвление).
  - **`brand_semantics` / `car_semantics`**: **`exit 1`**, указание на **`embedding_service_client.search`** и **`raise_for_status`** (HTTP-ошибка после ретраев).
  - **`action_recognition`**: **`exit 4`**, traceback из **`torch/nn/modules`** (типично dtype/shape/OOM).

### 4.3 Ориентиры по времени (тот же лог, 10 видео)

**Стена времени на видео** (максимальный **Elapsed** в heartbeat перед каждым `Done`):

| # | Длительность |
|---|----------------|
| 1 | ~19m 05s |
| 2 | ~24m 59s |
| 3 | ~19m 45s |
| 4 | ~23m 25s |
| 5 | ~22m 26s |
| 6 | ~22m 24s |
| 7 | ~21m 17s |
| 8 | ~18m 10s |
| 9 | ~24m 15s |
| 10 | ~19m 54s |

- **Сумма по 10 видео (если строго последовательно): ~216 мин (~3h 36m).**
- **Среднее ~21.6 мин / видео.** Реальный календарное время сьюта меньше, если несколько ранов перекрываются по времени.

**Топ компонентов по пиковому зарегистрированному времени** (по строкам `visual` / `audio` / `text` в логе, значения «макс по снимкам»):

1. `cut_detection` — ~185 s  
2. `speaker_diarization_extractor` — ~161 s  
3. `text_processor` — ~150 s  
4. `frames_composition` — ~121 s  
5. `SemanticTopicExtractor` — ~92 s  
6. `shot_quality` — ~68 s  
7. `core_clip` — ~61 s  
8. `core_optical_flow` — ~55 s  
9. `core_object_detections` — ~46 s  
10. `action_recognition` — ~41 s  

Дальше по убыванию в логе следуют `micro_emotion`, `pitch_extractor`, `core_face_landmarks`, `color_light`, `source_separation_extractor`, `core_depth_midas`, и т.д.

### 4.4 Оптимизация (без ухудшения качества — принципы)

- Уже в коде: **OOM-fallback** в `action_recognition` (те же веса и постобработка); **исправлено размещение pyannote на GPU** в `speaker_diarization_extractor` (см. § 3.7); **TextProcessor** — параллель lexical/ASR на одном документе и корректный **effective GPU** в `run_batch` (§ 3.8).
- Кандидаты на отдельный профилируемый этап: **`cut_detection`**, **`speaker_diarization`**, **`text_processor`**, **`frames_composition`** — обычно упираются в объём кадров/аудио и I/O; ускорение «в лоб» (меньше кадров, ниже разрешение) без метрик — не рекомендуется.

---

## 5. Итоговое состояние

| Область | Итог |
|---------|------|
| Сьюит 10 видео | **`e2e_exit=0`**, **`ingestion completed`** по каждому пункту (на проверенном прогоне с актуальным кодом и инфрой). |
| Семантика / franchise / place | Direct-path **устойчив**; при сбое — **fallback**, не **exit 1**. |
| Brand / car | При недоступности search — **пустые результаты**, run **продолжается**. |
| Action recognition | **Согласование dtype**, **OOM → per-clip**. |
| Следующий шаг (17 видео) | Расширить manifest/план сьюта до **17** пунктов; убедиться, что **сиды Embedding Service** применены; при необходимости перезапустить worker DP после обновления кода. |

---

## 6. Чек-лист перед прогоном на 17 видео

1. `./backend/scripts/start_e2e_stack.sh` (и при необходимости `--with-infra` / Triton по runbook).
2. Выполнен **`setup_e2e_infra.sh`** (или эквивалент) — таблица `embeddings` + seed строк для **franchise, brand, face, place, car**.
3. **`TRITON_HTTP_URL`**, **`DP_MODELS_ROOT`**, переменные из **`e2e_env.sh`**.
4. Обновлён список из **17** видео в сценарии suite (тот же механизм, что для 10).
5. После merge правок: **перезапуск dataprocessor-worker** (и при изменениях API — API).

---

*Документ отражает состояние на основе обсуждения и логов с метками дат **2026-04-18 … 2026-04-19** и файла `backend/.e2e/logs/e2e_terminal_latest.log`.*
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
