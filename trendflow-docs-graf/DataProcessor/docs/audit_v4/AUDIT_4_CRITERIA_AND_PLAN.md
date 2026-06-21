# Audit v4 — эмпирический аудит выходов компонентов (статистика NPZ + семантика полей + полезность для Models)

Дата: 2026-04-06 (обновление 2026-04-07 — **Audit 4.2**)  
Статус: **план и критерии**; полный e2e на 5+ видео с Embedding Service см. `backend/scripts/start_e2e_stack.sh` + §12.

---

## 0) Зачем Audit v4 после Audit v3

**Audit v3** фиксирует контракты: схемы, `SCHEMA.md`, tiers (`model_facing` / `analytics` / `debug`), пустые семантики, sampling requirements, ModelManager-only, fail-fast.

**Audit v4** отвечает на вопрос: *«смотрим на реальные артефакты в `result_store` — ведут ли себя числа так, как ожидает encoder/модель/аналитика, и документирована ли каждая фича на уровне алгоритма?»*

Итог Audit v4 по компоненту — не замена v3, а **дополнение**: каталог полей в `docs/README.md`, таблица статистики по validation set, краткий engineering-вердикт (полезность, риски, предложения по изменению выхода).

---

## 1) Source-of-truth и связанные документы

### 1.1 DataProcessor

- Обзор контрактов: [`docs/contracts/CONTRACTS_OVERVIEW.md`](../contracts/CONTRACTS_OVERVIEW.md)
- Артефакты, NPZ, `meta`, `empty_reason`: [`docs/contracts/ARTIFACTS_AND_SCHEMAS.md`](../contracts/ARTIFACTS_AND_SCHEMAS.md)
- Система схем (machine + human): [`docs/contracts/SCHEMAS_SYSTEM.md`](../contracts/SCHEMAS_SYSTEM.md)
- Time-axis / Segmenter: [`docs/contracts/SEGMENTER_CONTRACT.md`](../contracts/SEGMENTER_CONTRACT.md)
- Решения Audit v3: [`docs/audit_v3/DECISIONS_AND_RULES.md`](../audit_v3/DECISIONS_AND_RULES.md)
- Процедура v3 для Audio (контекст): [`docs/audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md`](../audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md)
- Каталог компонентов: [`docs/COMPONENTS_DESC_INDEX.md`](../COMPONENTS_DESC_INDEX.md), [`docs/COMPONENTS_DESC.md`](../COMPONENTS_DESC.md)

### 1.2 Models (потребление фич)

- Интерфейс Models ↔ DataProcessor (tabular + tokens): [`Models/docs/contracts/MODEL_INTERFACE_V2.md`](../../../Models/docs/contracts/MODEL_INTERFACE_V2.md)
- AudioEncoder / типы входов (dense / events / embeddings): [`Models/docs/contracts/ENCODER_CONTRACT.md`](../../../Models/docs/contracts/ENCODER_CONTRACT.md)
- Unification variable-length → fixed budget: [`Models/docs/source_migrations/FEATURE_ENCODER_CONTRACT.md`](../../../Models/docs/source_migrations/FEATURE_ENCODER_CONTRACT.md)
- Baseline (какие аудио-модули в минимальном наборе): [`Models/docs/contracts/BASELINE_MODEL.md`](../../../Models/docs/contracts/BASELINE_MODEL.md)
- Версии и `model_signature`: [`Models/docs/contracts/MODEL_SYSTEM_RULES.md`](../../../Models/docs/contracts/MODEL_SYSTEM_RULES.md)

При оценке «пойдёт ли в модель» опираемся на **фактические** планы encoder + `MODEL_INTERFACE_V2`, а не только на локальный README компонента.

---

## 2) Scope Audit v4 (первая волна)

**Приоритет:** `AudioProcessor` — все экстракторы, которые дают NPZ в e2e run (исключения явно помечаем: «требует Embedding service», «optional upstream»).

**Расширение:** модули `VisualProcessor` / `TextProcessor` по той же сетке критериев (журнал: [`RUN_LOG.md`](RUN_LOG.md)). **Pilot TextProcessor (2026-04-06):** `asr_text_proxy_audio_features` — [`components/text_processor/asr_text_proxy_audio_features_audit_v4.md`](components/text_processor/asr_text_proxy_audio_features_audit_v4.md); `comments_embedder` — [`components/text_processor/comments_embedder_audit_v4.md`](components/text_processor/comments_embedder_audit_v4.md); `comments_aggregator` — [`components/text_processor/comments_aggregator_audit_v4.md`](components/text_processor/comments_aggregator_audit_v4.md); `description_embedder` — [`components/text_processor/description_embedder_audit_v4.md`](components/text_processor/description_embedder_audit_v4.md); `cosine_metrics_extractor` — [`components/text_processor/cosine_metrics_extractor_audit_v4.md`](components/text_processor/cosine_metrics_extractor_audit_v4.md); `embedding_pair_topk_extractor` — [`components/text_processor/embedding_pair_topk_extractor_audit_v4.md`](components/text_processor/embedding_pair_topk_extractor_audit_v4.md); `embedding_shift_indicator_extractor` — [`components/text_processor/embedding_shift_indicator_extractor_audit_v4.md`](components/text_processor/embedding_shift_indicator_extractor_audit_v4.md); `embedding_source_id_extractor` — [`components/text_processor/embedding_source_id_extractor_audit_v4.md`](components/text_processor/embedding_source_id_extractor_audit_v4.md); `embedding_stats_extractor` — [`components/text_processor/embedding_stats_extractor_audit_v4.md`](components/text_processor/embedding_stats_extractor_audit_v4.md); `hashtag_embedder` — [`components/text_processor/hashtag_embedder_audit_v4.md`](components/text_processor/hashtag_embedder_audit_v4.md); `lexico_static_features` — [`components/text_processor/lexico_static_features_audit_v4.md`](components/text_processor/lexico_static_features_audit_v4.md); `qa_embedding_pairs_extractor` — [`components/text_processor/qa_embedding_pairs_extractor_audit_v4.md`](components/text_processor/qa_embedding_pairs_extractor_audit_v4.md); `semantic_cluster_extractor` — [`components/text_processor/semantic_cluster_extractor_audit_v4.md`](components/text_processor/semantic_cluster_extractor_audit_v4.md); `semantics_topics_keyphrases` — [`components/text_processor/semantics_topics_keyphrases_audit_v4.md`](components/text_processor/semantics_topics_keyphrases_audit_v4.md); `speaker_turn_embeddings_aggregator` — [`components/text_processor/speaker_turn_embeddings_aggregator_audit_v4.md`](components/text_processor/speaker_turn_embeddings_aggregator_audit_v4.md); `tags_extractor` — [`components/text_processor/tags_extractor_audit_v4.md`](components/text_processor/tags_extractor_audit_v4.md); `title_embedder` — [`components/text_processor/title_embedder_audit_v4.md`](components/text_processor/title_embedder_audit_v4.md); `title_embedding_cluster_entropy_extractor` — [`components/text_processor/title_embedding_cluster_entropy_extractor_audit_v4.md`](components/text_processor/title_embedding_cluster_entropy_extractor_audit_v4.md); `title_to_hashtag_cosine_extractor` — [`components/text_processor/title_to_hashtag_cosine_extractor_audit_v4.md`](components/text_processor/title_to_hashtag_cosine_extractor_audit_v4.md); `topk_similar_titles_extractor` — [`components/text_processor/topk_similar_titles_extractor_audit_v4.md`](components/text_processor/topk_similar_titles_extractor_audit_v4.md); `transcript_aggregator` — [`components/text_processor/transcript_aggregator_audit_v4.md`](components/text_processor/transcript_aggregator_audit_v4.md); `transcript_chunk_embedder` — [`components/text_processor/transcript_chunk_embedder_audit_v4.md`](components/text_processor/transcript_chunk_embedder_audit_v4.md). **Pilot VisualProcessor (2026-04-06):** `action_recognition` — [`components/visual_processor/modules/action_recognition_audit_v4.md`](components/visual_processor/modules/action_recognition_audit_v4.md); `behavioral` — [`components/visual_processor/modules/behavioral_audit_v4.md`](components/visual_processor/modules/behavioral_audit_v4.md); `color_light` — [`components/visual_processor/modules/color_light_audit_v4.md`](components/visual_processor/modules/color_light_audit_v4.md); `cut_detection` — [`components/visual_processor/modules/cut_detection_audit_v4.md`](components/visual_processor/modules/cut_detection_audit_v4.md); `detalize_face` — [`components/visual_processor/modules/detalize_face_audit_v4.md`](components/visual_processor/modules/detalize_face_audit_v4.md); `emotion_face` — [`components/visual_processor/modules/emotion_face_audit_v4.md`](components/visual_processor/modules/emotion_face_audit_v4.md); `frames_composition` — [`components/visual_processor/modules/frames_composition_audit_v4.md`](components/visual_processor/modules/frames_composition_audit_v4.md); `high_level_semantic` — [`components/visual_processor/modules/high_level_semantic_audit_v4.md`](components/visual_processor/modules/high_level_semantic_audit_v4.md); `micro_emotion` — [`components/visual_processor/modules/micro_emotion_audit_v4.md`](components/visual_processor/modules/micro_emotion_audit_v4.md); `optical_flow` — [`components/visual_processor/modules/optical_flow_audit_v4.md`](components/visual_processor/modules/optical_flow_audit_v4.md); `scene_classification` — [`components/visual_processor/modules/scene_classification_audit_v4.md`](components/visual_processor/modules/scene_classification_audit_v4.md); `shot_quality` — [`components/visual_processor/modules/shot_quality_audit_v4.md`](components/visual_processor/modules/shot_quality_audit_v4.md); `similarity_metrics` — [`components/visual_processor/modules/similarity_metrics_audit_v4.md`](components/visual_processor/modules/similarity_metrics_audit_v4.md); `story_structure` — [`components/visual_processor/modules/story_structure_audit_v4.md`](components/visual_processor/modules/story_structure_audit_v4.md); `text_scoring` — [`components/visual_processor/modules/text_scoring_audit_v4.md`](components/visual_processor/modules/text_scoring_audit_v4.md); `uniqueness` — [`components/visual_processor/modules/uniqueness_audit_v4.md`](components/visual_processor/modules/uniqueness_audit_v4.md); `video_pacing` — [`components/visual_processor/modules/video_pacing_audit_v4.md`](components/visual_processor/modules/video_pacing_audit_v4.md); `core_clip` — [`components/visual_processor/core/core_clip_audit_v4.md`](components/visual_processor/core/core_clip_audit_v4.md); `core_depth_midas` — [`components/visual_processor/core/core_depth_midas_audit_v4.md`](components/visual_processor/core/core_depth_midas_audit_v4.md); `core_face_landmarks` — [`components/visual_processor/core/core_face_landmarks_audit_v4.md`](components/visual_processor/core/core_face_landmarks_audit_v4.md); `core_object_detections` — [`components/visual_processor/core/core_object_detections_audit_v4.md`](components/visual_processor/core/core_object_detections_audit_v4.md); `core_optical_flow` — [`components/visual_processor/core/core_optical_flow_audit_v4.md`](components/visual_processor/core/core_optical_flow_audit_v4.md); `ocr_extractor` — [`components/visual_processor/core/ocr_extractor_audit_v4.md`](components/visual_processor/core/ocr_extractor_audit_v4.md).

Кросс-сводка L1 (**Audio + Text + Visual**): [`AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md`](AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md). Сводка L1 по **AudioProcessor:** [`AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md`](AUDIT_V4_L1_AUDIO_PROCESSOR_SUMMARY.md). Сводка L1 по **VisualProcessor:** [`AUDIT_V4_L1_VISUAL_PROCESSOR_SUMMARY.md`](AUDIT_V4_L1_VISUAL_PROCESSOR_SUMMARY.md). Сводка L1 по **TextProcessor:** [`AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md`](AUDIT_V4_L1_TEXT_PROCESSOR_SUMMARY.md). Каталог отчётов: [`components/README.md`](components/README.md).

**Входной run:** фиксируем в `RUN_LOG.md`:

- `platform_id`, `video_id`, `run_id` (как в пути `result_store/.../run_id/<component>/`)
- ссылка на полный лог e2e (например `backend/run_e2e.txt`), git commit, `config_hash` если есть в manifest

Для каждого компонента в отчёте указываем **конкретный каталог артефакта** (как в примере с `pitch_extractor`).

---

## 3) Validation set (минимум данных для статистики)

Чтобы статистика не была «случайно про одно видео»:

| Набор | Назначение | Минимум |
|--------|------------|---------|
| **A. E2E reference** | Регрессия после правок, согласованность с прогоном команды | 1 фиксированный OK run из e2e |
| **B. Audio-present diversity** | Распределения, хвосты, жанры, тишина/музыка/речь | ≥5 видео (разная длительность/контент) |
| **C. Edge** | Пустые семантики, короткое аудио, клипы без речи и т.д. | ≥2 кейса из smoke/edge набора v3 |

Если для класса B/C нет готовых путей — заводим записи в `RUN_LOG.md` по мере появления прогонов.

**Расширенная курация (60+ видео):** операционный чек-лист перед большим прогоном — [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md).

### 3.1 Уровни готовности отчёта (не путать с «passed»)

Чтобы черновики не смешивались с закрытым Audit v4, в **`RUN_LOG.md`** для каждого компонента явно указываем **уровень готовности отчёта**:

| Уровень | Наборы данных | Что допускается | Статус в журнале |
|--------|----------------|-----------------|------------------|
| **Level 1 — draft** | только **A** (e2e reference) | Детальный разбор одного артефакта, черновик §4, черновик вердикта; корреляции/перцентили по всему diversity ещё не обязательны | `in_progress (v4 L1)` или `draft` |
| **Level 2 — product stats** | **A + B** | Распределения, хвосты, вырождение, попарные корреляции по tabular/dense на ≥5 видео; всё ещё без полного edge-покрытия | `in_progress (v4 L2)` |
| **Level 3 — DoD** | **A + B + C** | Выполнен **§8** целиком: edge по `empty`/`empty_reason`, чувствительность к sampling (хотя бы качественно на C), при необходимости регрессионная сверка на A | `passed` (полный Audit v4) |

**Правило:** строка `Status: passed` в `RUN_LOG.md` допустима **только для Level 3**. Для Level 1/2 использовать формулировки вроде «отчёт по одному run», «ожидается набор B», без объявления полного прохождения аудита.

**Связь с исключениями:** компонент может быть `blocked` (нет e2e, Embedding service и т.д.) — это отдельно от уровня L1–L3.

---

## 4) Критерии статистики по NPZ (обязательный минимум)

Для **каждого массива** в NPZ (и для скаляров внутри `meta`, если они числовые и критичны), а для вложенных структур — **рекурсивно по листьям** (или по согласованным «группам фичей», см. §5):

### 4.1 Целостность и типы

- `dtype`, `shape`, совпадение с machine schema / `SCHEMA.md`
- Наличие ожидаемых ключей при `status=ok` vs допустимое отсутствие при `empty`
- Согласованность длины времени: `frame_indices`, `times_s`, ряды `T×F` и т.д.

#### 4.1a Семантика типов, целочисленные поля и «тихие» потери при сохранении

Цель — не допускать ситуаций, когда контракт и UI ожидают одну семантику, а в tabular-представлении она ломается без явного `empty_reason`.

**Обязательные проверки (применять к `feature_names`/`feature_values`, `meta`, аналогичным «плоским» выгрузкам):**

1. **Целочисленная семантика**
   - Счётчики, индексы, номера кадров, «число скачков», бины перечислимого типа: в отчёте и в `docs/README.md` явно указать *«семантически integer»*.
   - Предпочтительно: хранить как целочисленный dtype в NPZ или выносить в `meta` / отдельный массив; если в векторе `feature_values` всё же `float32`, зафиксировать в таблице полей §5, что значения дискретны (и допустимое отклонение от целого — только float-представление).

2. **Категориальные и строковые поля**
   - Строки (`backend`, имена методов, версии протоколов) **не** должны проходить через общий числовой савер (например единый `as_float()`), если это приводит к **молчаливому NaN** в табличном векторе без объяснения в контракте.
   - Правило: строки → `meta` / object-секция / отдельный ключ с `dtype=object` или documented string array; в §5 указать, где именно живёт значение.

3. **Согласованность dtype с контрактом**
   - Если machine schema декларирует `float32`, а фактически приходит `float64` (или наоборот) — отметить как расхождение (и решить: правка схемы или кода).

4. **Чек-лист для отчёта компонента**
   - Перечислить все ключи tabular-вектора и отметить для каждого: *число непрерывное / счётчик / категория / служебное meta-only*.
   - При L2+ на наборе B: для полей-счётчиков проверить, что не появляются нецелые значения из-за багов агрегации (кроме оговорённого float-wrap).

### 4.2 Специальные значения

- Доля **`NaN`**, **`±Inf`** (по элементам; для маскированных регионов — отдельно маска vs «ядро»)
- Доля **нулей** (и отдельно: **почти нулей**, например `abs(x) < ε`, если ε осмысленлен для шкалы)
- Для целочисленных **sentinel** (`-1`, `0` как «missing») — явно сверить с контрактом (например semantic heads)

### 4.3 Распределение (числовые вещественные)

- `n_valid` (без NaN), `min`, `max`, `mean`, `std`
- Перцентили: **p01, p05, p50, p95, p99** (для тяжёлых матриц — по выборке столбцов/батчам с фиксированным seed)
- **Оценка вырождения**: доля значений в узком интервале / константа / бимодальность «подозрительная»

### 4.4 Категориальные / object

- Частоты топ значений, доля `unknown`/пустых строк
- Длины строк (p50/p95), аномально большие выбросы

### 4.5 Временная ось

- Монотонность `times_s` / согласованность с `union_timestamps_sec`
- Дубликаты индексов кадров (если применимо)
- Разрывы (gaps) между соседними точками vs ожидание sampling policy

### 4.6 Корреляции и избыточность (выборочно)

- Попарная корреляция между **скалярными** агрегатами и между **столбцами** малых матриц (например до 32 признаков) — чтобы поймать дубликаты шкал
- Явно отмечаем: «высокая корреляция с `<другим ключом>` или с `<другим компонентом>`» → кандидат на удаление/объединение в encoder-only
- Если число scalar-фич в одном NPZ велико, см. расширение в **§4.11**

### 4.7 Политика трактовки результатов

| Наблюдение | Типичный вывод |
|------------|----------------|
| Высокая доля NaN при `status=ok` | Баг, неверная маска, или контракт не описывает реальность |
| Почти константа по всем видео | Низкая информативность для модели; оставить analytics или выкинуть |
| Экстремальные хвосты / Inf | Проверить нормализацию, clip, лог-масштаб |
| Нули там, где по семантике должен быть missing | Нарушение NaN-policy; править контракт или код |
| Две фичи с \( \rho > 0.95 \) на diversity set | Рассмотреть PCA/одну фичу/encoder pooling |

### 4.8 Стабильность и регрессия между прогонами (набор A)

Цель — ловить незаметный дрейф выходов после правок экстрактора, савера или зависимостей, не полагаясь только на «глаз».

**Минимум (Level 3, на фиксированном reference run A):**

1. Зафиксировать **набор скалярных сигнатур** по компоненту: например для tabular — хэш или конкатенация округлённых `mean/std/min/max` ключевых полей; для векторов по времени — те же метрики + длина оси.
2. После изменения кода, влияющего на компонент, **перепрогнать A** и сравнить:
   - либо **допуски** (относительные/абсолютные), оговорённые в отчёте для «ожидаемо стабильных» полей;
   - либо явное объяснение в отчёте/PR, почему сдвиг ожидаем (смена алгоритма, bump схемы).
3. В `RUN_LOG.md` при смене «эталонной» сигнатуры указать **commit до/после** и ссылку на PR/issue.

**Рекомендация:** вынести в репозиторий один скрипт «`audit_v4_stats.py --component X --npz path`» с фиксированным seed для subsample-перцентилей (см. §10); тогда регрессия сводится к diff вывода или к сохранённому JSON «golden stats» для A.

### 4.9 Чувствительность к sampling, длительности и «достаточности» оси

Многие фичи теряют смысл при малом числе точек на оси (сегментов, кадров, STFT-фреймов).

**На наборе B (и подтверждение на C):**

1. **Длительность и N точек**
   - Для каждого отчёта построить простую таблицу: `duration_sec` × `N_segments` (или `T` ряда) × ключевые агрегаты (например smoothness/jumps).
   - Зафиксировать **минимальный N** (или минимальную длительность), при котором интерпретация полей в §5 остаётся валидной; ниже порога — явно помечать в вердикте «низкая надёжность» или требовать маску `low_support` в будущем.

2. **`sampling_policy_version` и параметры Segmenter**
   - Если доступны два прогона с разной политикой на одном или сопоставимом контенте — качественно описать: сдвиг распределений, сдвиг плотности точек по времени, появление вырождения.
   - Если второго прогона нет — в §6 указать **риск** и завести follow-up «измерить при смене policy».

3. **Перекрытие окон / неразбиение**
   - Для оконных компонентов (аудио-сегменты, sliding windows) явно проверить: перекрытие vs partition; в вердикте для моделей указать, нельзя ли трактовать точки как независимые.

### 4.10 Покрытие пустых исходов (`status`, `empty`, `empty_reason`)

Набор **C (edge)** обязан включать кейсы, где ожидается **не** `status=ok`: тишина, отсутствие речи, слишком короткое аудио, ошибка входа и т.д.

**Проверки:**

1. Доля run с `empty` / не-ok, таблица **`empty_reason`** (или аналог в meta) — соответствие контракту ARTIFACTS/SCHEMA.
2. Отсутствие **«тихого ok»**: при семантически пустом входе нет ли полного набора чисел без NaN, если по контракту должны быть маска/`empty_reason`/отсутствие ключей.
3. Согласованность: если компонент в одном режиме пишет `empty`, не остаются ли «хвосты» в tabular векторе без документированной политики.

### 4.11 Избыточность внутри одного NPZ (tabular: много скалярных фич)

Если число **scalar** признаков в одном артефакте **> 24** (порог по умолчанию; для компонента можно снизить/повысить в отчёте с обоснованием):

1. Обязательна **матрица корреляций** \( \rho \) на наборе **B** (не только на A).
2. Рекомендуется дополнительно:
   - либо **иерархическая кластеризация** признаков по \( | \rho | \);
   - либо **VIF** / **PCA** на стандартизированных фичах (осторожно с NaN — политика impute в отчёте) — для выявления групп «почти одна фича».
3. В §6 перечислить **группы избыточности** и кандидатов на удаление, объединение или перенос в debug tier.

Для **dense** матриц большой ширины применять §4.3 с сэмплированием столбцов; §4.11 в первую очередь про tabular-векторы и небольшие `feature_names`.

### 4.12 Лёгкий чек на утечки и использование «глобальной» информации

Полный leakage-audit — вне scope; для Audit v4 достаточно **осознанного чек-листа** в конце отчёта:

| Вопрос | Зачем |
|--------|--------|
| Все ли значения в NPZ вычислены только из данных, доступных **в момент предсказания** для продукта (например только текущее видео + метаданные снимка), без будущих таргетов? | Базовый anti-leakage для popularity |
| Есть ли нормализация **по всему датасету** или по батчу внутри экстрактора (глобальный mean стрима)? | Если да — пометить риск для обучения и для прод-пайплайна |
| Подтягивает ли компонент **внешний онлайн** сигнал (не через зафиксированный локальный артефакт)? | Должно быть явно в §5 и в `models_used`/meta; иначе риск недетерминизма |

Для большинства **AudioProcessor** экстракторов ответы тривиальны; пункт обязателен для **Text/Visual** и для агрегаторов, смешивающих несколько run.

---

## 5) Документация полей в компоненте (`docs/README.md`)

Для каждого audited компонента ведём человекочитаемый каталог в:

`DataProcessor/AudioProcessor/src/extractors/<name>/docs/README.md`

(если `docs/` нет — создаём; корневой `README.md` компонента может кратко слать сюда).

### 5.1 Таблица полей (шаблон строки)

| Key / группа | dtype / shape | Как получаем (алгоритм, параметры) | Единицы / шкала | Missing / NaN policy | Tier (model / analytics / debug) | Заметки по статистике (v4) |
|--------------|---------------|-----------------------------------|-----------------|----------------------|-------------------------------------|----------------------------|

**«Как получаем»** — кратко и инженерно: библиотека/метод (например pyinostream, YIN, STFT+chromagram), окно, hop, пороги **только** если они меняют смысл поля (не дублировать полный конфиг — он в коде/schema).

### 5.2 Связь со схемой

- Ссылка на `SCHEMA.md` и JSON schema в `AudioProcessor/schemas/` (если есть)
- Явно: какие поля **обязательны** для encoder path vs только для отчётов

### 5.3 Сверка с потребителем (tabular FeatureSpec / Encoder / Baseline)

Чтобы вердикт §6 не оставался абстрактным, для каждого компонента добавить **короткую таблицу выравнивания** (можно 5–10 строк в `docs/README.md` или в отчёте v4):

| Вопрос | Ответ (да/нет/частично) | Комментарий |
|--------|-------------------------|-------------|
| Указаны ли поля или группы полей в актуальном **tabular FeatureSpec** / плане baseline? | | Ссылка на `BASELINE_MODEL.md` или на живой YAML спецификации, если есть |
| Попадают ли **dense / time-indexed** выходы в контракт **AudioEncoder** / `MODEL_INTERFACE_V2` token path? | | Если по умолчанию только tabular — явно «encoder: не используется без флага X» |
| Совместимы ли длина оси и семантика времени с **Segmenter** и `ENCODER_CONTRACT`? | | |
| Есть ли поля только для **UI/аналитики** и не заявленные для обучения? | | Пометить tier `analytics` vs `model_facing` |

Если компонент **не** входит ни в baseline, ни в ближайший encoder path — это нормально, но тогда в §6 явно: *«вне минимального training set; опциональная фича»*.

---

## 6) Итоговая секция по компоненту (engineering verdict)

Один укороченный блок в конце отчёта компонента (или в конце `docs/README.md` раздел **«Audit v4 summary»**):

1. **Полезность для моделей (baseline / v1 encoder path):** низкая / средняя / высокая + 3–5 bullets почему.
2. **Полезность для аналитики и QA:** что реально смотрят продакт/операторы; что только для дебага.
3. **Риски:** шум, утечки (future info), нестабильность при смене sampling, тяжёлые хвосты.
4. **Предложения по изменению выхода:** удалить / объединить / пересчитать в лог-шкале / добавить маску / добавить новое поле для downstream (**с обоснованием** и влиянием на `schema_version`).
5. **Связи между компонентами:** «этот выход дублирует X», «для Y лучше отдать Z из upstream».

Ссылки на Models: какой слой потребляет (например «глобальные агрегаты → tabular FeatureSpec», «dense по времени → AudioEncoder»).

### 6.1 Опционально: числовая оценка 0–10 и операционная заметка

По договорённости команды в отчёте можно добавить **краткую шкалу** (как ориентир, не замена §6):

- Отдельные баллы: например «стабильность пайплайна», «полезность tabular», «полезность dense encoder», «аналитика/UI» — с 1–2 предложениями обоснования.
- **Операционная заметка (не цель v4, но полезно для приоритизации):** на reference run A — порядок величины wall-time шага компонента и **пиковая RAM** (если снимались), без детального профилирования. Указать железо/OS при первом замере. Если не измерялось — строка «не замерялось».

Эти пункты **не обязательны** для §8, но помогают портфолио и решению «включать ли в дешёвый профиль».

---

## 7) План работ по шагам (один компонент)

1. **Идентификация:** имя компонента, `schema_version`, путь к NPZ в фиксированном run.
2. **Загрузка артефактов и воспроизводимость:** зафиксировать в `RUN_LOG.md`:
   - интерпретатор / venv (например `DataProcessor/.data_venv`) и **версии** `numpy`/`python`, если критично;
   - **команда или путь к скрипту/ноутбуку**, которым собирались таблицы §4 (при отсутствии общего скрипта — минимум: inline-команда `python -c '...'` или ссылка на коммит ноутбука);
   - **seed** для любых subsample-оценок перцентилей/корреляций.
3. **Проверка §4** с учётом **§3.1**: на L1 — минимум A + черновик; на L2 — A+B; на L3 (DoD) — A+B+C, включая **§4.8–§4.12** там, где применимо компоненту.
4. **Заполнить §5** в `extractors/<name>/docs/README.md`, включая **§5.3** (сверка с потребителем).
5. **§6 verdict** + при необходимости **§6.1**; список follow-up задач (issue-style): must / should / could.
6. **Сверка с Audit v3:** если статистика требует смены tier или ключей — оформить как изменение схемы (bump `schema_version`, обновить machine schema, валидатор).

Порядок обхода аудио-компонентов — как в e2e dependency order или как в [`COMPONENTS_DESC_INDEX.md`](../COMPONENTS_DESC_INDEX.md); первый пациент по договорённости: **`pitch_extractor`**.

---

## 8) Definition of Done (компонент считается «Audit v4 passed»)

Условие: достигнут **Level 3** (§3.1), статус в `RUN_LOG.md` — `passed`.

- [ ] Статистика **§4.1–§4.7** собрана на наборах **A + B + C**; ключевые таблицы (перцентили, NaN, время) приложены к отчёту или воспроизводимы по командам из `RUN_LOG.md`
- [ ] Выполнены дополнительные критерии по применимости компонента:
  - [ ] **§4.1a** — разбор типов tabular/meta, нет необъяснённых NaN от строк/категорий
  - [ ] **§4.8** — регрессионная база на наборе A зафиксирована (сигнатура или golden-stats), отмечен commit при изменении
  - [ ] **§4.9** — описана чувствительность к длительности/N и при наличии данных — к sampling; минимальный порог надёжности указан в §5 или §6
  - [ ] **§4.10** — набор C покрывает пустые исходы, таблица `empty_reason` / отсутствие тихого `ok`
  - [ ] **§4.11** — если scalar-фич > 24: корреляции на B + краткий вывод об избыточности
  - [ ] **§4.12** — заполнен лёгкий anti-leakage чек-лист для компонента
- [ ] **§4.6** на наборе B для ключевых пар (внутреняя избыточность + при необходимости с другими компонентами)
- [ ] `docs/README.md` компонента: **§5.1**, **§5.2**, **§5.3** (сверка с baseline/encoder path)
- [ ] Заполнен **§6** (verdict), согласованный с `MODEL_INTERFACE_V2` / `ENCODER_CONTRACT` / `BASELINE_MODEL.md` там, где компонент относится к stack
- [ ] Если выявлены изменения контракта — оформлен план bump схемы (не «молча» менять NPZ)
- [ ] В `RUN_LOG.md`: компонент, дата, **git commit**, пути ко **всем** использованным run (A, каждый из B, каждый из C), **уровень L3**, версия tooling (§7 п.2)

---

## 9) Исключения и вне scope

- Компоненты, которые **не завершаются** в e2e без Embedding service: не объявляем «v4 passed», фиксируем «blocked by …» в `RUN_LOG.md`.
- **Глубокое** профилирование latency, стоимость GPU, детальный cost model — **не цель** Audit v4 (как и в v3).
- Исключение по **продукту**: если статистика показывает вырождение при текущих пресетах — завести отдельную задачу на пресеты или tier.
- **Операционные метрики** (wall-time, пик RAM) — по **§6.1** опционально; они не блокируют §8, но если замер есть — полезно приложить к тому же commit, что и NPZ набор A.

---

## 10) Журнал прогонов и воспроизводимость tooling

Ведём журнал: [`docs/audit_v4/RUN_LOG.md`](RUN_LOG.md).

**Расширенный обязательный минимум записи (в дополнение к шаблону внутри `RUN_LOG.md`):**

- **Уровень отчёта** (L1 / L2 / L3) и явная формулировка, если `passed` ещё рано.
- **Git commit** репозитория DataProcessor (и backend, если артефакты из e2e зависят от версии run).
- Полный список **run_id** для A, B, C (или «TODO» с датой плана закрытия).
- **Инструмент статистики:** путь к скрипту в репозитории **или** зафиксированная одноразовая команда + **hash/notebook revision**; **seed** для random subsample.
- Версии **Python / numpy** (и любых библиотек, влияющих на численный результат), если они не стандартизованы одним образом venv для всей команды.

Цель — чтобы через месяц любой член команды мог воспроизвести таблицы §4 без устных договорённостей.

---

## 11) Сводка новых разделов (навигация)

| § | Тема |
|---|------|
| 3.1 | Уровни готовности L1 / L2 / L3 и статусы в `RUN_LOG.md` |
| 4.1a | Семантика типов, integer vs float, строки vs tabular |
| 4.8 | Регрессия и стабильность на наборе A |
| 4.9 | Sampling, длительность, N точек, перекрытие окон |
| 4.10 | `empty` / `empty_reason`, набор C |
| 4.11 | Много scalar-фич: корреляции, VIF/PCA на B |
| 4.12 | Лёгкий anti-leakage чек-лист |
| 5.3 | Сверка с FeatureSpec / Encoder / Baseline |
| 6.1 | Опционально: баллы 0–10, wall-time/RAM |
| 7 | Обновлённые шаги: tooling, seed, уровни |
| 8 | Расширенный DoD под новые критерии |
| 10 | Воспроизводимость и журнал |
| 12 | **Audit 4.2**: E2E≈30 видео, Embedding Service, ресурсы, скорость, DoD |
| 12.4 | **Playbook 4.2**: единый порядок действий по компоненту (до/после L2) |
| 12.4.4 | **Gate вопросов**: до статистики / профилирования / оптимизаций и до правок кода |

---

## 12) Audit 4.2 — масштаб E2E (≈30 видео), стабильность и закрытие плана

**Цель:** подготовить прогон **~30 видео** и параллельно **дозакрыть пункты Audit v4** по компонентам (L2/L3 там, где ещё черновики), без регресса по стабильности на длинных сериях.

### 12.1 Критерии готовности (DoD 4.2)

| Область | Критерий «готово» | Журнал / артефакт |
|--------|-------------------|-------------------|
| **E2E инфра** | Postgres с **pgvector**, БД `embeddings`, **Embedding Service** на `localhost:8005` стартует вместе со стеком (`start_e2e_stack.sh`); `TRITON_BASE_URL` указывает на E2E Triton (**не** порт Fetcher). | Лог `embedding-service/process.log`, `GET /health` = 200 |
| **Semantic visual** | Модули `franchise_recognition`, `car_semantics`, `place_semantics`, `face_identity`, `brand_semantics` не падают на `Embedding Service … not running`; при пустых FAISS/БД — осмысленные **warnings** / пустые метки, а не обрыв subprocess из-за отсутствия HTTP. | Лог e2e, строки компонентов в manifest |
| **micro_emotion** | Нет `subprocess exit 3` из‑за `ValueError` после частичного OpenFace mapping; узкие выборки после drop битых строк не валят PCA. | Повтор на 2+ видео с «битым» mapping |
| **Ресурсы (long)** | Профилирование **RAM/CPU/GPU** на серии из **≥2 видео** подряд: фиксированный скрипт или notebook, логи `psutil`/NVML (или аналог), гипотезы об утечках + **явный список** follow-up в `RUN_LOG.md` или отдельном отчёте 4.2. | Ссылка на отчёт + commit |
| **Производительность (long)** | Таблица **wall-time на видео** до/после точечных оптимизаций; целевой ориентир фиксируется командой (не блокер для 4.2 L2 по компонентам, но блокер для **массового** 30‑видео при OOM). | `RUN_LOG.md` + команда замера |
| **Audit v4 компоненты** | Для оставшихся компонентов из §2: либо **L3 passed**, либо явный **L1/L2** + дата следующего шага; новые `blocked` только с причиной (зависимость, нет данных). | `RUN_LOG.md`, отчёты в `components/` |

### 12.2 Долгие ветки (не смешивать с DoD одного компонента §8)

1. **Профилирование ресурсов и утечки** — отдельный трек: повторяемые прогоны на 2 видео, снимок пиковых памятей по процессам (DataProcessor worker, subprocess extractors, Docker OpenFace), сравнение после каждого видео; при необходимости — ограничение параллелизма / явный `gc` / перезапуск worker между видео (как временная мера). **AudioProcessor:** каркас `AP_ORCHESTRATOR_TELEMETRY` + `scheduler_runtime_report.json` → [`AudioProcessor/docs/ORCHESTRATOR_TELEMETRY.md`](../../AudioProcessor/docs/ORCHESTRATOR_TELEMETRY.md) (далее — тем же API подключать другие процессоры).
2. **Ускорение E2E** — параллельно с п.1: узкие места из логов (`openface_run_ms`, Triton, тяжёлые visual), профилирование Python (`cProfile`/py-spy) по согласованию; изменения глобального конфига (отключение модулей в «быстром профиле») документировать отдельно от audit статистики.

### 12.3 Быстрые инженерные задачи (вход в 4.2)

- Поднять **Embedding Service** с инфрой (см. `Fetcher/docker-compose.yml` образ `pgvector/pgvector`, `setup_e2e_infra.sh`, `embedding_service/requirements-e2e.txt`).
- Исправление **micro_emotion** при малом `n` после фильтрации OpenFace (PCA).

### 12.4 Порядок действий по компоненту (playbook — одинаковый для всех)

Ниже зафиксирован **типовой порядок работ**, который не нужно повторно расписывать в каждом отчёте: для нового компонента достаточно ссылаться на этот §12.4 и перечислить отличия (пути, имена скриптов, флаги). **Сквозная последовательность:** **12.4.1** (якорь) → **12.4.2** (какие документы читать) → **12.4.3** (код + `result_store`) → **12.4.4** (контрольные вопросы и **`Gate OK`**) → **12.4.5** (L1→L3, статистика) → при необходимости **12.4.6** (хвост 4.2).

#### 12.4.1 Якорь и уровень отчёта

1. Зафиксировать **имя компонента**, **`schema_version`**, **уровень цели** (L1 / L2 / L3 по §3.1) и **run_id** для наборов **A** (и при L2+ — **B**, при L3 — **C**).
2. Записать в [`RUN_LOG.md`](RUN_LOG.md) пути к NPZ / manifest, **git commit**, команду или скрипт статистики, **seed** (§7 п.2, §10).

#### 12.4.2 Какие документы читать (порядок)

| Порядок | Документ | Зачем |
|--------|----------|--------|
| 1 | Этот файл — §1–§4, §3.1, §8 (DoD) | Критерии, наборы A/B/C, что считать «закрыто» |
| 2 | [`RUN_LOG.md`](RUN_LOG.md) | Фактические run, commit, tooling |
| 3 | Отчёт компонента в [`components/`](components/README.md) (после появления черновика) или заготовка по §7 | Структура L1/L2/L3 |
| 4 | `docs/README.md` / `SCHEMA.md` **в каталоге компонента** (Audio: `AudioProcessor/src/extractors/<name>/docs/…`; аналогично Visual/Text) | Контракт полей, `meta`, флаги |
| 5 | При вердикте для обучения: [`MODEL_INTERFACE_V2.md`](../../../Models/docs/contracts/MODEL_INTERFACE_V2.md), [`ENCODER_CONTRACT.md`](../../../Models/docs/contracts/ENCODER_CONTRACT.md), [`BASELINE_MODEL.md`](../../../Models/docs/contracts/BASELINE_MODEL.md) | Сверка §5.3 / §6 |
| 6 | Для 4.2 по длинным прогонам: §12.1–§12.2, при Audio — [`AudioProcessor/docs/ORCHESTRATOR_TELEMETRY.md`](../../AudioProcessor/docs/ORCHESTRATOR_TELEMETRY.md) | Ресурсы, оркестратор, не смешивать с DoD §8 одного компонента |

#### 12.4.3 Куда смотреть в коде и артефакты (типовая карта)

| Зона | Где искать (пример AudioProcessor) | Что проверять |
|------|--------------------------------------|---------------|
| Вычисление | `src/extractors/<name>/main.py` (или orchestration entry) | Логика, ветки `empty`, зависимости от CLI/env |
| Сохранение NPZ | `src/core/npz_savers/<…>.py`, регистрация в factory | Соответствие SCHEMA, `meta`, optional поля телеметрии |
| Общий раннер | `extractor_runner.py`, `main_processor.py`, CLI args | Параллелизм, batch vs single-file, прокидывание env |
| Фактические выходы | `storage/result_store/.../<component>/` (или путь из `RUN_LOG.md`) | Сверка имён файлов, `status`, типичный `meta` |
| Статистика v4 | Скрипт вида `extractors/<name>/scripts/audit_v4_npz_stats.py` или общий (если заведён) | Воспроизводимость, пути к JSON/figures |
| Профилирование / 4.2 | Утилиты компонента (`utils/…`), env с префиксом `AP_*` | Этап после L2: §12.4.6 |

#### 12.4.4 Контрольные вопросы (gate) — до скриптов, замеров и правок кода

**Правило:** после **§12.4.1–12.4.3** (якорь в журнале, чтение документов по §12.4.2, просмотр кода и **реальных артефактов** в `result_store`) исполнитель составляет **короткий Q&A** по блокам ниже и получает ответы **проверяющего** (второй инженер / владелец аудита). Пока gate не пройден — **не** пишутся скрипты статистики, **не** включается профилирование в «боевом» пути и **не** вносятся оптимизации.

**Где фиксировать ответы:** абзац или таблица в **[`RUN_LOG.md`](RUN_LOG.md)** (предпочтительно) или секция PR «Audit gate» со ссылкой на компонент и commit.

Ниже — **минимальный набор** (можно расширять под компонент). У каждого вопроса исполнитель отмечает выбранный вариант **A/B/C**; проверяющий сверяет с рекомендацией.

---

**Блок «Статистика и отчёт (L1/L2/L3)»**

1. **Какую цель уровня закрываем в этом проходе?**  
   - **A** — только **L1** (набор **A**, детальный разбор).  
   - **B** — **L2** (**A + B**, ≥5 видео, §4 на diversity).  
   - **C** — **L3 / DoD** (**A + B + C**, §4.8–§4.12, §8).  
   **Рекомендация проверяющего:** при выборе **B** или **C** без перечня `run_id` для **B** (и **C**) — вернуть на дозаполнение §12.4.1; не согласовывать старт скрипта «только на одном файле», если заявлен L2.

2. **Пути к NPZ и воспроизводимость зафиксированы?**  
   - **A** — да: **A** в `RUN_LOG.md`, для L2+ перечислены все **B** (или явные TODO с датой).  
   - **B** — есть только локальные пути у исполнителя, в журнал не перенесены.  
   - **C** — артефакты ещё не собраны.  
   **Рекомендация проверяющего:** не подтверждать начало работ по §4 до **A**; **B** блокирует объявление L2.

3. **`seed`, tooling и среда зафиксированы для сравнимости таблиц §4?**  
   - **A** — `seed` + команда/скрипт + при необходимости версии **Python/numpy** в `RUN_LOG.md`.  
   - **B** — только `seed`.  
   - **C** — не зафиксировано.  
   **Рекомендация проверяющего:** для L2+ требовать минимум **B**; для регрессий **§4.8** — **A**.

4. **Объём статистики по полям NPZ (что считаем в первую очередь)?**  
   - **A** — все листья, перечисленные в **machine schema / `SCHEMA.md`**, плюс критичные ключи `meta`.  
   - **B** — только **tabular** / скаляры, dense/time-series отложены.  
   - **C** — узкое подмножество с явным обоснованием (риск пропуска §4.1a/§4.11).  
   **Рекомендация проверяющего:** по умолчанию **A** для нового компонента; **C** только с письменным списком исключений в отчёте.

5. **Формат выхода скрипта статистики?**  
   - **A** — **JSON** агрегатов + ключевые **figures** (корреляции, хвосты).  
   - **B** — только JSON (без графиков).  
   - **C** — только интерактив без сохранения артефакта в репозитории/`storage`.  
   **Рекомендация проверяющего:** для L2 — **A** или **B** + ссылка на сохранённый JSON; **C** не считать завершённым этапом для команды.

---

**Блок «Профилирование и оркестратор (Audit 4.2 / §12.x)»**

6. **Граница замеров: куда пишем метрики?**  
   - **A** — в первую очередь **оркестратор** (`scheduler_runtime_report.json`, общий контур).  
   - **B** — **только внутри экстрактора** (тайминги / RSS/GPU в `meta` под env).  
   - **C** — **согласованно оба слоя** (один и тот же прогон объясним end-to-end).  
   **Рекомендация проверяющего:** если цель — массовый E2E (**§12.1**), предпочтительно **C**; если на этом проходе только L2 по NPZ — профилирование можно **отложить** с явной меткой в `RUN_LOG.md`.

7. **Включение профилирования в код: политика?**  
   - **A** — только за **env/CLI**, по умолчанию выкл., документировать в `docs/README.md`.  
   - **B** — всегда писать расширенный `meta` (увеличение размера NPZ).  
   - **C** — только внешний профайлер без изменений репозитория.  
   **Рекомендация проверяющего:** по умолчанию **A**; **B** нужен согласованный допуск на размер/PII в `meta`.

8. **Есть ли риск смешения с DoD §8 одного компонента?**  
   - **A** — да, отделяем: сначала **L2-статистика**, профилирование — отдельным PR/итерацией.  
   - **B** — нет, совмещаем в одном PR.  
   **Рекомендация проверяющего:** при **B** требовать явный чеклист: отчёт §4 не «размыт» правками оптимизации, регрессия на **A** при изменении выхода.

---

**Блок «Оптимизации»**

9. **Допустима ли смена семантики выхода?**  
   - **A** — **нет**, только ускорение/память при том же контракте.  
   - **B** — **да под флагом**, default прежний, документация + предупреждение о влиянии на §4.  
   - **C** — **да, меняем поведение по умолчанию**.  
   **Рекомендация проверяющего:** **C** только с **bump `schema_version`**, обновлением machine schema и планом §4.8 на **A**.

10. **Есть ли измеренное узкое место до оптимизации?**  
    - **A** — да: flame/cProfile/тайминги оркестратора/логи с цифрами.  
    - **B** — только гипотеза без замера.  
    **Рекомендация проверяющего:** не принимать крупные алгоритмические замены при **B**; разрешить микро-правки (кэш констант) по усмотрению.

11. **Критерий успеха оптимизации?**  
    - **A** — **wall-time / пик RAM** на согласованном наборе видео (до/после, тот же commit-кроме PR).  
    - **B** — только микробенчмарк на синтетике.  
    - **C** — не определён.  
    **Рекомендация проверяющего:** для изменений, влияющих на продуктовый прогон, требовать **A** или явный tech-debt ticket.

---

**Итог gate:** проверяющий ставит в `RUN_LOG.md` (или PR): **`Gate OK`** / **`Gate revise`** и 1–2 предложения. После **`Gate OK`** выполняется **§12.4.5** (этапы L1→L3: скрипт статистики, отчёт, §5) и далее по необходимости **§12.4.6** (хвост 4.2: профилирование, оптимизации, `audit_4_2`).

#### 12.4.5 Этапы до закрытия эмпирического отчёта (L1 → L2 → L3)

1. **L1:** один reference **A**, детальный разбор артефакта, черновик §4/§5/§6; в `RUN_LOG.md` без статуса `passed` (§3.1).
2. **L2:** **A + B** (≥5 видео), таблицы распределений, NaN, корреляции по §4; отчёт в `components/…/<name>_audit_v4.md`; `RUN_LOG.md` — `in_progress (v4 L2)` или эквивалент.
3. **L3:** **A + B + C**, §4.8–§4.12 по применимости, DoD §8; только тогда **`passed`** в журнале.

Инструмент: зафиксировать **JSON агрегатов** и **figures** рядом с отчётом или в оговорённом `storage/…` пути; в отчёте дать относительные ссылки.

#### 12.4.6 Этап Audit 4.2 «после L2» (инженерный хвост — тот же порядок для всех)

Выполняется, когда эмпирический отчёт **L2 уже есть**, а нужно закрыть §12 (ресурсы, скорость, наблюдаемость) **без переписывания** основного L2-документа целиком.

1. **Сверка с §12.1–§12.2:** что именно дозакрываем (RAM/GPU, wall-time, телеметрия оркестратора, узкие места).
2. **Изменения в коде (типовые, не исчерпывающие):**
   - optional **`meta`**: тайминги стадий, при необходимости — snapshot RSS/GPU (env-gated);
   - **оптимизации** без смены семантики выхода по умолчанию (или с явным флагом и предупреждением в README);
   - **исправления стабильности** (OOM, edge на малом N), если всплыли на длинной серии.
   - Любое изменение **контракта** NPZ — только через bump **`schema_version`** и обновление machine schema (§7 п.6).
3. **Документ-мост:** создать или дополнить файл в [`components/audit_4_2/`](components/audit_4_2/README.md) — **инженерный журнал** по шаблону [`asr_extractor_engineering_log_v4_2.md`](components/audit_4_2/audio_processor/asr_extractor_engineering_log_v4_2.md): ссылка на канонический L2-отчёт, краткая сводка статистик (без дубля полных таблиц), хронология версий/`producer_version`, что поменялось в коде, влияние optional флагов на §4. Для **TextProcessor** журнал кладётся в [`components/audit_4_2/text_processor/`](components/audit_4_2/README.md) (часто — срез табличных ключей в общем `text_features.npz`).
4. **Индексация:** обновить [`components/README.md`](components/README.md) (при новом журнале), [`../MAIN_INDEX.md`](../MAIN_INDEX.md) (раздел Audit v4 / 4.2), при необходимости — обратная ссылка из основного отчёта L2 на журнал `audit_4_2`.

#### 12.4.7 Пример (референс): `asr_extractor`

- L2-отчёт (канон статистики): [`components/audio_processor/asr_extractor_audit_v4.md`](components/audio_processor/asr_extractor_audit_v4.md).
- Журнал после L2: [`components/audit_4_2/audio_processor/asr_extractor_engineering_log_v4_2.md`](components/audit_4_2/audio_processor/asr_extractor_engineering_log_v4_2.md).
- В коде: профилирование `resource_profile.py`, поля `asr_stage_timings_ms` / `asr_resource_profile` в `meta`, оптимизации и env в `main.py` + документация в `docs/README.md`.

#### 12.4.8 Пример (TextProcessor, табличный срез): `asr_text_proxy_audio_features`

- L2-отчёт: [`components/text_processor/asr_text_proxy_audio_features_audit_v4.md`](components/text_processor/asr_text_proxy_audio_features_audit_v4.md).
- Журнал 4.2: [`components/audit_4_2/text_processor/asr_text_proxy_audio_features_engineering_log_v4_2.md`](components/audit_4_2/text_processor/asr_text_proxy_audio_features_engineering_log_v4_2.md).
- Статистика по префиксу `tp_asrproxy_*`: `TextProcessor/.../scripts/audit_v4_npz_stats.py` → `storage/audit_v4/asr_text_proxy_audio_features_l2/`.
---

## Навигация

[Audit v4 hub](components/audit_4_2/README.md) · [DataProcessor](../MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
