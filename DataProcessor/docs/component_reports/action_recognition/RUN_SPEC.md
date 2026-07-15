# RUN_SPEC: action_recognition (v1)

Автор: Claude (static-review) → Исполнитель: Cursor. Модель взаимодействия —
`docs/CLAUDE_CURSOR_COLLAB.md`. Протокол — `DataProcessor/docs/COMPONENT_VALIDATION_PROTOCOL.md`.

## 1. Static-review (что я вижу по коду)

- **Назначение:** per-track (person) распознавание действий → на трек эмбеддинг
  `[num_clips, 256]` (L2-норм.) + метрики динамики (`max/mean_temporal_jump`,
  `stability`, `num_switches`, `num_clips`, `track_frame_count`). NPZ:
  `action_recognition/action_recognition_features.npz`, schema `action_recognition_npz_v2`.
- **Цепочка зависимостей:** `Segmenter` → **`core_object_detections`** (person-треки)
  → `action_recognition`. Кадры строго по `metadata["action_recognition"].frame_indices`.
- **Модель:** `utils/action_recognition_slowfast.py` грузит **SlowFast R50**
  (`pytorchvideo.models.hub.slowfast_r50(pretrained=False)` + ckpt из spec
  `slowfast_r50_action_recognition` → `visual/action_recognition/slowfast_r50/slowfast_r50.pyth`).
  `clip_len` по умолчанию **32** (T_slow=8) — нужно ≥32 кадров на трек.
- **Пустой результат валиден:** `status="empty"`, `empty_reason="no_person_detections"`.

### 🔴 Блокер модели (решить до прогона)
В едином репо `trendflow_models` лежит **VideoMAE**, а НЕ slowfast_r50. Файла
`.../slowfast_r50/slowfast_r50.pyth` нет → `ModelManager.resolve` упадёт. Варианты:

- **(A) Быстрый — провизионить SlowFast** (рекомендую для первого прогона): скачать
  Kinetics-веса pytorchvideo и сохранить под путь spec. Компонент запускается «как есть»,
  валидируем его реальную логику/выход. Preflight-команда (Cursor, на стеке с сетью):
  ```bash
  DataProcessor/.data_venv/bin/python - <<'PY'
  import torch
  from pytorchvideo.models.hub import slowfast_r50
  m = slowfast_r50(pretrained=True)   # Kinetics-400
  import os; p="DataProcessor/dp_models/visual/action_recognition/slowfast_r50"
  os.makedirs(p, exist_ok=True)
  torch.save(m.state_dict(), f"{p}/slowfast_r50.pyth")  # сверить state_dict_key со spec
  print("saved", p)
  PY
  ```
  Требует `pytorchvideo` в `.data_venv`. Сверить формат ckpt с ожиданием impl
  (`state_dict_key`, `strip_prefix`) — при несовпадении обернуть в `{"state_dict": ...}`.
- **(B) Стратегический — переключить на VideoMAE** (loader уже готов: провайдер
  `transformers_pretrained` + spec `videomae_kinetics400_inprocess`). Требует **переписать
  I/O модуля** (VideoMAE: 16 кадров, логиты 400 классов Kinetics вместо эмбеддинга 256).
  Это меняет семантику выхода — отдельная задача рефактора (feature-quality workstream).

**Решение для первого прогона:** идём по (A), чтобы провалидировать существующую логику.
Стратегически обсудим (B) по итогам (см. §4 model-fit).

## 2. Профиль прогона

Включить (порядок по `dag/component_graph.py`): `Segmenter` + `core_object_detections`
+ `action_recognition`. Остальное выключить (изолируем компонент). Профиль — по образцу
E2E max-run, но scope = только эти компоненты.

## 3. Матрица видео

action_recognition про людей/движение — нужны ролики **с людьми и действиями**:

| Длина | Кол-во | Контент |
|---|---|---|
| ~10 s | 2 | 1 talking-head (мало движения) + 1 спорт/танец (много движения) |
| ~30 s | 2 | влог с человеком + групповая сцена (несколько person-треков) |
| ~1 min | 2 | обучающее с человеком + экшн |
| ~2 min | 2 | интервью (стабильные треки) + динамичный клип |
| ~4 min | 1 | многосюжетный с людьми |
| ~8 min+ | 1 | длинный (проверить деградацию/стоимость) |
| контроль | 1 | видео **без людей** → ожидаем валидный `empty` (`no_person_detections`) |

Итого ~11 роликов. Зафиксировать `video_id`, факт. `duration_sec`, тип.

## 4. Что проверить (на что смотрю в отчёте)

Числа: `feature_quality_audit` (health/nan/const по метрикам динамики), `§0.2` контракт,
`batch_runs_feature_report` (метрики × длина — растёт ли число треков/клипов с длиной),
`golden_batch_compare` (повтор одного ролика).

Model-fit (§0.1 протокола) — ключевое для action_recognition:
- выход **per-track** `[num_clips,256]`, НЕ по единой оси времени. Вопрос: как Encoder
  это потребляет? Нужны **времена клипов** (`clip_center_frame_indices` → `times_s`),
  чтобы собрать time-ordered embedding-stream; проверить, что они есть и согласованы с `union_timestamps_sec`.
- эмбеддинг (256) полезен модели как токен; метрики динамики (`stability` и т.д.) —
  скорее аналитику/baseline. Пометить в ledger.
- пустые/мало-трековые случаи (короткие/без людей) — валидный empty, не падение.

## 5. Что собрать (Cursor → artifacts/)

- `action_recognition_features.npz` по каждому видео (или срез: shapes, tracks, num_clips, clip times).
- `*_batch.csv` (batch_runs_feature_report), `*_health.md` (feature_quality_audit), §0.2-json.
- Тайминги стадий (`meta_timing_*`), пики VRAM/CPU (ветка оптимизаций).
- Логи прогона; какие патчи/preflight понадобились (в стиле LOGIC_ERRORS).
- Версии: `dataprocessor_version`, `sampling_policy_version`, `model_signature`.

Заполнить `RUN_RESULT.md` (шаблон в CLAUDE_CURSOR_COLLAB) + пути к `artifacts/`.
После этого Claude пишет `REPORT_<date>.md` (шаблон PROTOCOL §3) → владелец сверяет с видео.

## 6. Решения владельца (зафиксировано)
- **(A) SlowFast для первого прогона** — согласовано.
- **Выход нужен ОБА**: эмбеддинг действия (для модели) **+** распределение классов
  действий (для аналитиков). → Целевой контракт выхода (см. ниже).

### Целевой контракт выхода action_recognition (после первого прогона — доработка)
На трек/клип отдавать оба:
- `embedding` (penultimate, 256-d, L2) — **seq-токен для Encoder** (уже есть);
- `action_topk_ids` + `action_topk_probs` (Kinetics-400, top-K) + `class_names` —
  **для аналитиков** (интерпретируемые метки действия), опц. как доп. фичи модели;
- времена клипов `clip_center_times_s` (из `clip_center_frame_indices` ⊆ `union_timestamps_sec`).

SlowFast R50 (pytorchvideo) — это Kinetics-классификатор, поэтому оба выхода снимаются
с одной модели: penultimate-фичи (эмбеддинг) + softmax головы (классы). **Первый прогон**
валидируем как есть (эмбеддинг), затем добавляем class-distribution head в рамках доработки
(отразить в schema `action_recognition_npz_v3` + `FEATURE_DESCRIPTION`).
