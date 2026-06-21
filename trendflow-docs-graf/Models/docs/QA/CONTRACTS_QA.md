# Contracts & System Q&A (multi-round)

Этот документ — **живой протокол** в формате “вопрос → ответ”, где мы фиксируем все моменты, которые сейчас **не зацементированы** в контрактах/коде/документации.

Цели:
- закрыть “серые зоны” (неопределённости) по данным, артефактам, сплитам, моделям, inference, версиям, privacy;
- зафиксировать **каноничные решения** (чтобы дальше код/CI проверял соответствие);
- собрать **список оптимизаций** (качество/скорость/масштабирование) и критерии “когда внедряем”.

Как работаем:
- Мы идём **раундами**: Round 1 → ты отвечаешь → я фиксирую ответы как “FINAL” + при необходимости предлагаю патчи в код/docs.
- Формат ответа от тебя: **копируй ID вопроса** и пиши решение (и, если нужно, короткое обоснование).
- Если решение временное: помечаем как **TEMP** + условие перехода в FINAL.

Легенда статусов:
- **OPEN**: нужен ответ/решение
- **TEMP**: временно принято (есть риск/условия)
- **FINAL**: зафиксировано (должно стать частью контрактов/валидаций)

---

## Round 1 — закрываем контракты “сквозного пайплайна”

### A) Источники данных / time semantics / leakage

**Q-A01 (FINAL)**: Что считается “prediction time” в продукте?
- **Варианты**:
  - A) время завершения DataProcessor run (`manifest.run.created_at`)
  - B) время `snapshot_0` (если оно есть/может появиться)
  - C) другое: ________
- **Почему важно**: определяет корректность age buckets, кеша, интерпретации прогнозов.
- **Предложение по умолчанию**: A (как сейчас proxy).
- **Ответ**: Вообще под prediction time подразумевается весь проход от сбора данных до предсказания модели, но только по тем алгоритмам (фичам) которые непосредственно участвуют в предсказании, так как есть алгоритмы в DataProcessor которые вычисляют аналитические фичи, которые не относяться к предсказанию. “prediction time” также можно разделить на три части: 1-сбор данных, 2-обработка(DataProcessor), 3-Inference.
  - **Decision (FINAL)**:
    - Мы фиксируем **три timestamp’а** (в отчёте прогона и в артефактах модели):
      - `data_collection_at` — время сбора snapshot/comments (если есть; иначе отсутствует)
      - `dataprocessor_run_created_at` — `manifest.run.created_at` (время формирования feature-артефактов)
      - `inference_finished_at` — время окончания инференса (время выдачи результата пользователю)
    - Термин **prediction_time** в UI/продукте = `inference_finished_at`.
    - Для вычисления age-buckets в offline training/eval используем:
      - `feature_time_at` = `dataprocessor_run_created_at` (если есть), иначе fallback на `inference_finished_at`.

**Q-A02 (TEMP)**: `video_age_hours_at_snapshot1` — это точный возраст на момент snapshot_1 или proxy?
- **Сейчас**: proxy `manifest_created_at - publishedAt` (baseline+v1).
- **Нужно зафиксировать название/семантику**:
  - A) оставить название как есть и явно считать proxy
  - B) переименовать в `video_age_hours_at_run` (или аналог) и мигрировать
- **Ответ**: `video_age_hours_at_snapshot1` я не участвовал в создании этого параметра и не знаю зачем он нужен и как вычисляется
  - **Decision (TEMP → FINAL после миграции)**:
    - Семантика: это **proxy возраста на момент run**, а не snapshot_1.
    - Переходим на новое имя: `video_age_hours_at_run` (hours) = `feature_time_at - publishedAt`.
    - Backward-compat: пока поддерживаем чтение `video_age_hours_at_snapshot1` как alias.
    - DoD: обновить docs + builders + валидаторы + (опционально) миграционный скрипт.

**Q-A03 (FINAL)**: Политика leakage: какие поля **строго запрещено** использовать во features?
- **Предложение**: всё, что derived из `snapshot_1..3` / будущих дат.
- Нужен **явный список** запрещённых колонок/паттернов для CI.
- **Ответ**: Да, все что в `snapshot_1..3` так как это и есть данные которые должна предсказать модель. Из этих снапшотов модель предсказывает только likes+views
  - **Decision (FINAL)**:
    - Запрещены любые features, derived из `snapshot_1|snapshot_2|snapshot_3`.
    - Запрещены любые колонки/ключи/артефакты содержащие паттерны: `snapshot_[123]`, `target_`, `mask_` (кроме прямого использования targets в loss).
    - В CI добавляем leakage-check: если в feature columns встречаются эти паттерны — fail.

**Q-A04 (FINAL)**: `publishedAt` — обязательное поле для всех видео?
- Если нет: какое значение/поведение по умолчанию?
  - A) считать epoch и класть в самый ранний bucket
  - B) drop из train/eval
  - C) другое: ________
- **Ответ**: Обязательное. Насколько я помню оно есть у всех видео в датасете.
  - **Decision (FINAL)**:
    - `publishedAt` required. Если отсутствует/непарсится → sample **drop** из train/eval (fail-fast в строгом режиме).

**Q-A05 (FINAL)**: `channel_id` обязателен для честного split. Как его получаем?
- **Варианты**:
  - A) всегда делаем enrichment `video_id -> channel_id` до тренировки (обязательный шаг)
  - B) допускаем fallback на `channelTitle` как TEMP
  - C) другое: ________
- **Ответ**: Я сам напишу алгоритмы через YouTube API и соберу его. Тебе стоит знать что он будет у каждого видео в датасете.
  - **Decision (FINAL)**:
    - В offline dataset/index **`channel_id` required**.
    - Split policy uses `channel_id` only (fallback на `channelTitle` допустим только как TEMP в dev до enrichment).
    - Enrichment (YouTube API) = обязательный шаг перед финальными обучениями/сравнениями.

---

### B) Артефакты / схемы / required vs optional inputs

**Q-B01 (FINAL)**: Каноничный минимальный набор файлов для v1 training sample:
- **Сейчас (факт)**:
  - `core_clip_npz_path` (frame_embeddings + frame_indices)
  - `segmenter_metadata_path` (union_timestamps_sec)
  - snapshot_0 numeric + targets
  - optional: `text_npz_path`
- **Нужно решить**:
  - audio (`clap_extractor_npz_path`) — required/optional на v1?
- **Ответ**: Реши сам. audio required.
  - **Decision (FINAL)**:
    - Для “v1.0 FINAL” audio **required** (CLAP как минимум).
    - Пока audio pipeline не готов/не прошёл аудит — допускаем TEMP режим “visual+text”, но это считается **deviation**.

**Q-B02 (FINAL)**: Контракт `core_clip` NPZ (точные ключи/shape/dtype).
- Подтвердить:
  - `frame_embeddings`: float32, shape (N, 512)
  - `frame_indices`: int64, shape (N,)
  - (опционально) `meta` и `models_used[]`?
- **Ответ**: Реши сам. Можешь посмотреть его README так как он уже прошел аудит.
  - **Decision (FINAL)**:
    - Required:
      - `frame_embeddings`: float32, shape (N, 512)
      - `frame_indices`: int64, shape (N,)
    - Recommended:
      - `meta` с `schema_version="core_clip_npz_v1"` + `producer_version` + `models_used[]` по `MODEL_SYSTEM_RULES.md`.
    - Валидатор обязан проверять dtype/shape/монотонность/уникальность `frame_indices`.

**Q-B03 (FINAL)**: Контракт `frames_dir/metadata.json` (Segmenter):
- минимально required:
  - `union_timestamps_sec` (U,)
  - (в будущем) `union_frame_indices_source`?
- надо ли требовать `analysis_fps/width/height/color_space` в v1 индексе?
- **Ответ**: Реши сам. Можно требовать если это нужно. Если этого не дает, нужно изменить сам Segmenter так как Модели важнее Segmenter.
  - **Decision (FINAL)**:
    - Segmenter metadata required:
      - `union_timestamps_sec` (U,)
      - `analysis_fps`, `analysis_width`, `analysis_height`, `color_space="RGB"`
    - v1 index не обязан дублировать эти поля, но должен хранить `segmenter_metadata_path` и валидировать наличие required keys.
    - Если Segmenter этого не выдаёт — меняем Segmenter (модели важнее).

**Q-B04 (FINAL)**: Как версионируем схемы NPZ и валидируем?
- **Уже есть**: `schema_version`, `producer_version`, `dataprocessor_version`.
- **Нужно**:
  - A) единый Python валидатор для ключевых NPZ (core_clip, clap, text tokens) + CI
  - B) только best-effort (TEMP)
- **Ответ**: Как правильнее, качественее так и делаем. Реши сам.
  - **Decision (FINAL)**:
    - Делаем **строгий валидатор** (Python) для:
      - core_clip NPZ
      - clap_extractor NPZ
      - text tokens NPZ
      - (опционально) segmenter metadata JSON
    - Включаем в CI (и как runtime fail-fast флаг).
    - Любая смена формата → bump `schema_version` + обновление docs.

**Q-B05 (FINAL)**: Где хранится “source-of-truth” спецификация входов baseline?
- **Сейчас**: `DataProcessor/DatasetBuilder/feature_spec.yaml` (хорошо).
- Нужно ли сделать аналогичный spec для v1 index (`v1_index_spec.yaml`)?
- **Ответ**: Да, и задокументировать
  - **Decision (FINAL)**:
    - Создаём `Models/v1/data/v1_index_spec.yaml` (аналогично baseline `feature_spec.yaml`).
    - `build_v1_dataset_index.py` должен писать snapshot spec и версию в `v1_dataset_metadata.json`.

---

### C) Text/comments (privacy + reproducibility)

**Q-C01 (FINAL)**: Пайплайн комментариев: мы точно **никогда** не сохраняем raw текст?
- **Контракт**: raw не храним; допускается только при explicit opt-in и OAuth.
- Подтвердить для v1 training artifacts:
  - `text_npz` хранит только embeddings/tokens/meta (без raw).
- **Ответ**: Да, может и другие фичи, но не сырой текст.
  - **Decision (FINAL)**:
    - v1 text artifacts хранят только embeddings/tokens/агрегаты/счётчики.
    - Raw текст не сохраняем (кроме отдельного DP режима с explicit opt-in и OAuth — вне v1 train artifacts).

**Q-C02 (FINAL)**: Модель текста: какая именно (name+revision) и как фиксируем “weights digest”?
- **Сейчас**: `sentence-transformers/all-MiniLM-L6-v2` без pinned revision.
- Решение:
  - A) pinned revision + digest обязателен (FINAL)
  - B) TEMP: имя модели достаточно до первой публикации модели
- **Ответ**: Как правильнее, качественее так и делаем. Реши сам.
  - **Decision (FINAL)**:
    - Требуем pinned `revision` + `weights_digest` (sha256) для текстовой модели.
    - В `text_npz.meta` фиксируем: `model_name`, `model_revision`, `weights_digest`, `device`, `precision`.
    - No-network:
      - **Inference**: запрещены downloads.
      - **Training/CI**: downloads разрешены **только** из наших репозиториев/организации и только pinned revision; после скачивания — кешируем/зеркалим.

**Q-C03 (FINAL)**: Языки: нужен реально multilingual encoder?
- Если в данных RU/ES/etc много — MiniLM может быть слабым.
- Варианты:
  - A) оставляем MiniLM как MVP
  - B) переходим на multilingual encoder (указать какой)
- **Ответ**: Да, стоит перейти на multilingual encoder, какой посоветуй сам (качественный).
  - **Decision (FINAL)**:
    - Default (quality): `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (dim=768).
    - Optional fast preset: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dim=384) + projection to D.
    - Выбор encoder фиксируется в `model_config.json` и в `text_npz.meta`.

**Q-C04 (TEMP)**: Scoring top-K комментариев:
- Сейчас heuristic: likes + replies + length_bonus.
- Хотим:
  - A) оставить heuristic как FINAL
  - B) заменить на attention pooling/learned scorer позже (V6)
- **Ответ**: Однозначно заменить
  - **Decision (TEMP → FINAL после реализации)**:
    - План: хранить `comment_embeddings (Nc<=100,D)` + `comment_mask` + агрегаты.
    - В модели: trainable attention pooling/transformer-pooling → Kc tokens (вместо offline top-K heuristic).
    - До реализации допускаем текущую heuristic как TEMP (только чтобы пайплайн работал).

---

### D) Encoder (v0/v1) и time-binning

**Q-D01 (FINAL)**: Budgets `K` (64/96/128) — FINAL?
- Условия изменения:
  - A) менять можно только с bump `model_config`/signature
  - B) можно авто-адаптировать по compute budget
- **Ответ**: Реши сам как качественнее.
  - **Decision (FINAL)**:
    - Budgets K считаются частью `model_config.json`/`model_signature`.
    - Любое изменение K-rule → bump версии/подписи и переоценка на golden sets.

**Q-D02 (FINAL)**: Encoder v0 per-bin stats:
- В контракте упомянуты quantiles/robust stats; в коде сейчас mean/max.
- Решение:
  - A) расширить v0 до mean/max/p50/p90 (и т.п.)
  - B) зафиксировать, что v0 = mean/max (TEMP)
- **Ответ**: расширить
  - **Decision (FINAL)**:
    - Encoder v0 должен использовать robust stats минимум: mean/max/p50/p90 (+ count).
    - Это нужно для устойчивости к “пикам” и для лучшего baseline качества.

**Q-D03 (FINAL)**: Trainable Encoder v1:
- Какие стабилизации обязательны?
  - gradient clipping
  - AMP (fp16/bf16)
  - dropout/norm policy
- Что считаем “готово для сравнения” с v0?
- **Ответ**: Я в этом не разбираюсь. Планировка и реализация на тебе. Мне нужно качество.
  - **Decision (FINAL)**:
    - Обязательные стабилизации:
      - gradient clipping (например norm=1.0)
      - LayerNorm в ключевых местах, dropout policy
      - AMP/bf16 (если поддерживается) как опция с фиксированием precision в артефактах
    - “Готово для сравнения с v0”, если:
      - нет NaN/inf, обучение воспроизводимо (seed фиксируется)
      - regression_mini проходит стабильно
      - на holdout не хуже v0 по north-star (или лучше) по ключевым heads (14/21).

---

### E) v1 Fusion / multimodality / heads

**Q-E01 (FINAL)**: Fusion: контракт говорит cross-attention. Сейчас реализован упрощённый вариант.
- Решение:
  - A) текущий вариант = TEMP (V2), cross-attn обязателен для “v1.0 FINAL”
  - B) допускаем encoder-only fusion как FINAL
- **Ответ**: делаем cross-attn сейчас
  - **Decision (FINAL)**:
    - Реализуем cross-attention fusion как в контракте (`V1_TRANSFORMER_MODEL.md`).
    - Текущий encoder-only fusion считаем TEMP/MVP и помечаем как deviation до замены.

**Q-E02 (TEMP)**: Audio модальность в v1:
- минимальный MVP:
  - A) CLAP embeddings → AudioEncoder → audio tokens
  - B) только global audio token
  - C) пока выключено (TEMP)
- **Ответ**: Audio еще не прошел аудит, но пока оставим вариант А.
  - **Decision (TEMP → FINAL после аудита audio)**:
    - MVP: CLAP embeddings → AudioEncoder → audio tokens (A).
    - До аудита audio допускаем отключаемый флаг, но целевая конфигурация — audio required.

**Q-E03 (FINAL)**: Quantiles:
- минимум p10/p50/p90 для каждого head — FINAL?
- enforcement:
  - A) монотонность p10<=p50<=p90 (loss/constraint/postprocess)
  - B) только sanity metric, без enforcement
- **Ответ**: Реши сам как качественее.
  - **Decision (FINAL)**:
    - Используем p10/p50/p90 для каждого head.
    - Enforce monotonicity p10<=p50<=p90:
      - предпочтительно параметризацией (например p50 + softplus(d1), p90 = p50 + softplus(d2), p10 = p50 - softplus(d0))
      - sanity metrics остаются как контроль.

**Q-E04 (FINAL)**: 7d horizon masked:
- Что считаем корректной маской?
  - A) mask_7d = 1 только если есть и views_7d и likes_7d
  - B) раздельные маски для views_7d и likes_7d (точнее)
- **Ответ**: Реши сам как качественее.
  - **Decision (FINAL)**:
    - Переходим на **раздельные маски**: `mask_views_7d`, `mask_likes_7d` (точнее).
    - Backward-compat: `mask_7d` остаётся как AND(mask_views_7d, mask_likes_7d).

---

### F) Training/Eval/Golden sets (quality gate)

**Q-F01 (FINAL)**: Golden sets:
- baseline и v1 используют один и тот же принцип (holdout=2000, regression=200).
- Нужно ли синхронизировать golden sets baseline и v1 по одинаковому списку видео?
  - A) да, один source-of-truth список видео на весь проект
  - B) нет, допускаем отдельные, но обе keyed by fingerprint
- **Ответ**: Реши сам как качественее.
  - **Decision (FINAL)**:
    - Делаем **единый source-of-truth список video_id** для golden sets, чтобы baseline vs v1 сравнивались честно.
    - Реализация: golden sets формируются детерминированно из video_id списка и сохраняются keyed by dataset fingerprint; при сравнении baseline/v1 используем пересечение/совпадающий список.

**Q-F02 (FINAL)**: Метрики:
- north star Spearman на log1p(delta) — FINAL?
- Secondary:
  - MAE
  - Spearman по age buckets
  - coverage p10/p90
- Нужно ли считать метрики отдельно по `views` и `likes` и по каждому горизонту (6 heads) — FINAL?
- **Ответ**: Реши сам как качественее.
  - **Decision (FINAL)**:
    - Считаем метрики отдельно по:
      - 2 targets (`views`, `likes`)
      - 3 horizons (7/14/21) = 6 heads
      - overall + age buckets
    - Для quantiles: coverage p10/p90 + monotonic sanity.

**Q-F03 (TEMP)**: Что считается “проходом quality gate”?
- Нужно определить минимальные пороги/неухудшения:
  - A) v1 ≥ baseline по Spearman (overall) на holdout
  - B) не хуже на большинстве age buckets
  - C) coverage в диапазоне [0.75..0.95] (пример)
- **Ответ**: Реши сам как качественее.
  - **Decision (TEMP → FINAL после первой калибровки на реальных цифрах)**:
    - Правило “не ухудшать”:
      - v1 не хуже baseline по Spearman на holdout для 14d/21d по views+likes (допуск epsilon=0.01).
      - по buckets: не хуже baseline в большинстве buckets (например ≥6 из 8).
    - Uncertainty:
      - monotonicity rate ≥ 0.98
      - coverage p10/p90 в разумном диапазоне (предварительно 0.75–0.95).
    - После первого полного прогона фиксируем пороги как FINAL.

---

### G) Inference / packaging / no-network / signatures

**Q-G01 (FINAL)**: Упаковка v1 артефакта:
- обязателен ли `model_config.json` (архитектура/версии/quantiles/K rules/Kc)?
- обязателен ли `weights_digest`?
- **Ответ**: Реши сам как качественее.
  - **Decision (FINAL)**:
    - `model_config.json` обязателен (архитектура, quantiles, K rules, Kc, text/audio encoder specs, precision policy).
    - `weights_digest` обязателен (sha256) для всех весов в артефакте.

**Q-G02 (FINAL)**: No-network policy:
- что именно запрещено в runtime?
  - downloads весов (HF)
  - calls к внешним API
  - dynamic model selection без pinned digest
- **Ответ**: Реши сам. Разрешено downloads весов (HF) только для наших моделей с наших репозиториев
  - **Decision (FINAL)**:
    - **Inference**: no-network (никаких downloads).
    - **Training/CI/dev**: downloads разрешены только из approve-list (наши репозитории), pinned revision обязателен, веса кешируются/зеркалятся.

**Q-G03 (FINAL)**: `model_signature` для обученных моделей (baseline/v1):
- где хранится и как используется?
  - в `training_run_manifest.json`
  - в `prediction.json` (output)
  - в DB (опционально)
- **Ответ**: Реши сам. Можно использовать DB или HF.
  - **Decision (FINAL)**:
    - `model_signature` + `weights_digest` фиксируются в:
      - `training_run_manifest.json`
      - output `prediction_report.json` (см. Round UI ниже)
    - DB/HF — только transport/storage (не source-of-truth по версии).

**Q-G04 (FINAL)**: Degraded-mode:
- при отсутствии text tokens / audio tokens / части visual artifacts:
  - A) fail-fast (error)
  - B) degraded prediction с masks (и статусом)
- Нужно зафиксировать per-field policy.
- **Ответ**: fail-fast (error)
  - **Decision (FINAL)**:
    - v1 inference — fail-fast при отсутствии required артефактов/модальностей (включая audio, когда станет FINAL).
    - Degraded-mode остаётся на уровне “выбор модели”: v2→v1→baseline, но каждая модель внутри себя fail-fast по своим required inputs.

---

### H) Производительность / масштабирование (cost-aware rules)

**Q-H01 (FINAL)**: Целевые бюджеты (в проде):
- max latency на prediction (после готовых токенов) для v1: 2–5s — это актуально?
- max RAM/VRAM для inference и training?
- **Ответ**: max latency на prediction: до 5 сек. max RAM/VRAM для inference и training: 48гб, но нужно явно указать что будет происходить и что менять разработчику, если он хочет запустить тренироваку или инференс например на 6, 10, 16 гб
  - **Decision (FINAL)**:
    - Target latency (post-encoder tokens): ≤ 5s.
    - Default env budget: up to 48GB VRAM/RAM.
    - Обязательная документация пресетов для меньших GPU (6/10/16GB):
      - уменьшать batch size + grad accumulation
      - уменьшать model width/depth (d_model, layers, heads)
      - уменьшать Kc, отключать часть модальностей только в dev (но это deviation)
      - использовать AMP/bf16, activation checkpointing.

**Q-H02 (TEMP)**: Где “самый дорогой” участок и как оптимизируем?
- варианты:
  - caching encoder outputs
  - AMP/bf16
  - Torch compile / ONNX / TensorRT
  - batch scheduling
- Зафиксировать приоритеты.
- **Ответ**: Где самый дорогой участок я не знаю. Оптимизировать в какой то степени можно всеми перечислеными вариантами, главное без сильной потери качества
  - **Decision (TEMP → FINAL после профилирования)**:
    - Профилируем и фиксируем top-3 bottlenecks (DataProcessor vs v1 encoder vs fusion).
    - Приоритет оптимизаций (без сильной потери качества):
      1) AMP/bf16 + grad accumulation + dataloader optimizations
      2) caching encoder outputs (в training)
      3) torch.compile / ONNX/TensorRT (в inference) после стабилизации качества
      4) batch scheduling / multi-GPU.

**Q-H03 (FINAL)**: Версионирование производных артефактов (например text_npz):
- если меняем scoring/topK/encoder — это новый `schema_version` или новый `model_signature`?
- **Ответ**: Реши сам как качественее.
  - **Decision (FINAL)**:
    - `schema_version` меняется при изменении формата файла (ключи/shape/dtype).
    - `model_signature` меняется при изменении модели/весов/engine/precision.
    - Изменение алгоритма агрегации/topK/tokenization (даже без смены формата) → bump **artifact_config_version** в meta + влияет на fingerprint.

---

### I) Риски / тестирование / CI

**Q-I01 (FINAL)**: Минимальный набор CI-проверок (FINAL):
- validate NPZ meta keys + schema_version
- leakage checks (future cols)
- reproducibility: fingerprints + deterministic splits
- smoke_e2e (минимальный прогон)
- **Ответ**: Да
  - **Decision (FINAL)**:
    - CI must include: schema/meta validation, leakage checks, deterministic splits, smoke_e2e path.

**Q-I02 (FINAL)**: Golden regression:
- при каждом PR прогоняем regression_mini=200?
- или только по релизам?
- **Ответ**: Реши сам как качественее.
  - **Decision (FINAL)**:
    - regression_mini=200 запускаем:
      - на каждый PR, который меняет Models/* (training/eval/inference) или DatasetBuilder
      - nightly (или релизно) для тяжёлых DataProcessor изменений
    - holdout=2000 — как release gate.

---

## Notes / parking lot (не блокирует Round 1)

- Context (v2): TTL=48h, деградация и contract `context_schema_version` — обсудим в Round 2.
- UI/продукт: как отображаем uncertainty/интервалы — Round 3.

---

## Round UI — “красивый прогон модели” для пользователя (stages, heads, interpretation)

Цель: пользователь должен видеть **что произошло**, **сколько заняло**, **какие данные использовались**, и **что означают 6 голов + интервалы**.

**Q-UI01 (FINAL)**: Какой “объект” показываем пользователю?
- A) “Run” (DataProcessor run_id) + привязанный prediction
- B) “Prediction job” (может включать несколько run’ов/повторов)
- **Предложение**: B (job), внутри ссылки на run_id.
- **Ответ**: Да
  - **Decision (FINAL)**: В UI показываем `prediction_job` как основной объект. Внутри — ссылки на `run_id`(ы) DataProcessor, если был пересчёт.

**Q-UI02 (FINAL)**: Каноничный формат отчёта (для backend/frontend) — `prediction_report.json`?
- **Предложение (quality)**: фиксируем JSON contract:
  - `job_id`, `platform_id`, `video_id`, `run_id`
  - timestamps: `data_collection_at`, `dataprocessor_run_created_at`, `inference_started_at`, `inference_finished_at`
  - stages[]: список этапов (Segmenter/Visual/Audio/Text/Encoder/Fusion/Heads/Postprocess) с `status`, `duration_ms`, `artifacts_used[]`, `errors[]`
  - `models_used[]` (model_signature): baseline/v1/v2 + text encoder + dp models
  - `inputs_summary`: counts (frames, audio segments, comments_used), missing flags
  - `outputs`: для 6 heads:
    - p10/p50/p90 на log1p(delta) + (опционально) инверсия в delta и “ожидаемое абсолютное значение”
    - `masked` флаги
  - `quality_warnings[]`: (например “audio missing => fail-fast” или “7d masked”)
  - `explainability` (если включено): baseline SHAP / transformer evidence summary
- **Ответ**: Ок
  - **Decision (FINAL)**: `prediction_report.json` — каноничный machine-readable отчёт для backend/frontend (плюс опциональный `report.md` для человека).

**Q-UI03 (FINAL)**: Что именно “интерпретируем” для v1?
- Варианты:
  - A) только “evidence/diagnostics” (какие модальности присутствуют, сколько токенов, sanity checks)
  - B) полноценная feature attribution (дорого/сложно/может быть misleading)
- **Предложение**: A как FINAL сейчас; B только как опциональный режим для internal/debug.
- **Ответ**: Насколько трудно реализовать вариант В и насколько он оправдан, что он дает.
  - **Decision (FINAL)**:
    - Для пользователя (default): **A) evidence/diagnostics** (без “почему так” и без причинных объяснений).
    - Вариант B (полная attribution) откладываем на отдельный будущий эпик; допускается только как internal/debug и только после отдельного дизайн‑ревью.

**Q-UI04 (FINAL)**: Представление результатов по головам:
- A) Таблица 2×3 (views/likes × 7/14/21) с p10/p50/p90
- B) Графики по горизонтам с интервалами
- C) И то и то
- **Предложение**: C (таблица + sparkline/interval chart).
- **Ответ**: И то и то
  - **Decision (FINAL)**: Таблица 2×3 + графики интервалов по горизонтам (p10/p50/p90).

**Q-UI05 (FINAL)**: Как показываем пользователю “шкалу” (log1p)?
- A) показываем только “в натуральных единицах” (delta и/или абсолют)
- B) показываем и log, и натуральные (advanced toggle)
- **Предложение**: A по умолчанию + B в advanced.
- **Ответ**: В натуральных
  - **Decision (FINAL)**: В UI показываем в натуральных единицах (delta и/или absolute). Log-scale — только internal/debug.
---

## Навигация

[Models](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
