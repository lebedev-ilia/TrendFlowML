# FINAL REPORT — `core_clip`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `core_clip` (VisualProcessor **core** provider, Tier-0) |
| Версия кода (`VERSION`) | `2.1` |
| Схема NPZ (`SCHEMA_VERSION`) | `core_clip_npz_v2` |
| Артефакт | `result_store/<platform>/<video>/<run>/core_clip/embeddings.npz` |
| Модель | OpenAI **CLIP ViT-B/32**, D=512 (inprocess `clip.load` / Triton `clip_image_224\|336\|448` + `clip_text`) |
| Версия промптов | `v3_2026-01-16` |
| Дата разбора | 2026-07-17 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → core_clip ✅ (2026-07-05) |
| Отчёт валидации | [`REPORT_2026-07-05.md`](REPORT_2026-07-05.md); Audit v4 [`core_clip_audit_v4.md`](../../audit_v4/components/visual_processor/core/core_clip_audit_v4.md), [`engineering_log_v4_2`](../../audit_v4/components/audit_4_2/visual_processor/core/core_clip_engineering_log_v4_2.md) |
| Код | `DataProcessor/VisualProcessor/core/model_process/core_clip/main.py` (1434 строки) |

## 2. Резюме

`core_clip` — **CLIP-хаб всей визуальной цепочки**. На выборке кадров (union-domain, владелец выборки —
Segmenter) он считает L2-нормированные CLIP-эмбеддинги ViT-B/32 (`frame_embeddings (N,512)`) и один раз
прогоняет через CLIP text-encoder фиксированные наборы промптов (shot quality, эстетика/люкс/атмосфера
сцены, переходы монтажа, popularity-темы, 365 категорий Places365). Благодаря этому все downstream-модули
(`scene_classification`, `shot_quality`, `cut_detection`, `similarity_metrics`, `uniqueness`,
`high_level_semantic`, `story_structure`, `video_pacing`) работают **zero-shot без загрузки весов CLIP** —
единый источник истины и никаких сетевых обращений. Компонент прод-готов: контракт стабилен на 5+ прогонах
Audit v4, эмбеддинги детерминированы (golden diff 0.0), на 23 реальных артефактах — 100% finite, L2≈1.0.

## 3. Функционал

Стоит в самом начале визуального пайплайна (Tier-0, сразу после Segmenter). Делает две вещи:

1. **Image-эмбеддинги кадров** — универсальный семантический «отпечаток» каждого sampled-кадра в
   пространстве CLIP. Это базовый визуальный токен для Encoder (класс C — precomputed embeddings) и
   сырьё для 7+ downstream-модулей.
2. **Text-эмбеддинги промптов** — переводит фиксированные текстовые описания в то же 512-мерное
   пространство, чтобы downstream делал zero-shot классификацию простым скалярным произведением
   `image · text`, не таща за собой CLIP.

**Зачем продукту:** это фундамент «понимания картинки». Без него не работают ни сцена, ни оценка качества
съёмки, ни детекция монтажных склеек, ни сравнение видео между собой. Экономически он ключевой для
масштаба 200k видео: CLIP считается **один раз**, а не в каждом из семи потребителей.

## 4. Вход

Контракт строгий, **no-fallback** (компонент падает, а не молчит):

- **Кадры** — `FrameManager.get(idx)` из `frames_dir` (RGB uint8). Читаются чанками (`chunk_size`,
  `cache_size` из metadata).
- **`metadata.json.core_clip.frame_indices`** (обяз.) — какие кадры брать. Владелец выборки — **Segmenter**;
  провайдер сам не семплирует. Пустой/отсутствующий список → `RuntimeError`.
- **`metadata.json.union_timestamps_sec`** (обяз.) — source-of-truth ось времени; длина = `total_frames`.
  Из неё берётся `times_s = union_timestamps_sec[frame_indices]`. Проверяется диапазон индексов.
- **run identity** (обяз. для сохранения): `platform_id`, `video_id`, `run_id`, `config_hash`,
  `sampling_policy_version`, `dataprocessor_version`.
- **`--batch-size`** (обяз.) — задаётся scheduler/DynamicBatch, авто-подбор внутри запрещён.
- Опционально: Triton-specs (`--triton-image-model-spec`, `--triton-text-model-spec`) через ModelManager;
  `DP_MODELS_ROOT` для Places365-категорий и кэша text-эмбеддингов.

**От других компонентов не зависит** (это Tier-0 provider), кроме Segmenter (выборка) и — в Triton-режиме —
ModelManager/Triton. Places365-категории читаются из bundled_models.

## 5. Выход

NPZ `embeddings.npz`, `allow_extra_keys=false`. Три смысловых класса ключей:

- **model-facing (для Encoder):** `frame_embeddings (N,512)`, `frame_indices (N,)`, `times_s (N,)`.
  Это seq-выход по позициям во времени; агрегация — на стороне Encoder (learnable pooling), сам
  core_clip agg не считает (кроме `places365_video_topk_*`).
- **module-facing (сервисный слой):** `*_prompts (P,) object` + `*_text_embeddings (P,512)` для 7 семейств:
  `shot_quality` (10), `scene_aesthetic`/`scene_luxury`/`scene_atmosphere` (по 6),
  `cut_detection_transition` (10), `popularity_topic` (10), `places365` (365). Downstream берёт эти
  готовые text-эмбеддинги и не грузит CLIP.
- **analytics / backend-proxy:** `consecutive_cosine_prev (N,)`; per-frame score-матрицы `*_scores (N,P)`;
  `places365_topk_indices/scores (N,5)`; video-level `places365_video_topk_indices/scores (5,)`.
  Именно они уходят на сайт (raw эмбеддинги наружу **не** отдаются).

Размерности стабильны на всех прогонах: **D=512, K=5**. Единицы: эмбеддинги безразмерные L2-нормированные;
`times_s` в секундах; `*_scores` — косинусные сходства (см. §6).

## 6. Фичи (важное/неочевидное)

- **`frame_embeddings`** — L2-нормированный CLIP-вектор. Норма строки ≈1.0 (по реальным данным mean=1.0,
  Audit v4: 1.00000002…1.00000004). Косинус = скалярное произведение. Это несущая фича компонента.
- **`consecutive_cosine_prev`** — косинус между соседними по выборке кадрами; **первый кадр = NaN by
  design** (нет предыдущего). Физически это «насколько картинка изменилась». По 23 реальным видео:
  mean **0.952**, median **0.978**, p05 **0.80**, min **0.498** — т.е. большинство соседних кадров
  похожи (медленные/статичные сцены), а провалы = монтажные склейки/смены плана. **Важно:** значение
  зависит от плотности выборки Segmenter — при частом семпле cosine искусственно ближе к 1.
- **`*_scores`** — это `frame_emb · text_emb` (косинус, оба L2-норм), **НЕ softmax и НЕ вероятности**.
  Реальная сумма строки `shot_quality_scores` ≈ **2.14** (10 промптов), у Places365 top-5 сумма
  ≈1.05…1.24. Downstream обязан трактовать как similarity/logit-подобные величины (зафиксировано в
  Audit v4 §4.1a как ◐). Диапазон ~[−1,1], на практике положительные из-за общей семантики CLIP.
- **`places365_video_topk_*`** — единственная agg-фича: усреднение per-frame scores по кадрам → top-5
  сцен всего видео. Даёт словесный ответ «что это за видео» (аналитике/сайту).

## 7. Алгоритм / архитектура

- **Модель:** OpenAI CLIP **ViT-B/32** (D=512), веса — официальные openai/CLIP. Внешняя предобученная
  нейросеть, не обучается в проекте (zero-shot).
- **Препроцессинг:** resize→224² BICUBIC, нормализация CLIP mean/std, CHW float32. В Triton UINT8-режиме
  ensemble сам нормализует (компонент отдаёт NHWC uint8).
- **Инференс:** батчами `--batch-size`; image через `encode_image`, затем L2-норм. Text: `clip.tokenize`
  локально → `encode_text` (или Triton, где EOT-позиция выбирается снаружи по `argmax(tokens)`).
- **Где идёт:** inprocess (CPU/GPU, `clip.load`) — основной путь, Triton не обязателен; либо Triton-GPU
  (`clip_image_*`/`clip_text` через ModelManager, no-network).
- **Сложность:** линейна по N кадрам. Реально ~14–34 c/видео inprocess ViT-B/32 (A4500); чистый Triton
  image-inference ~0.6 c на 1 кадр … ~15 c на 304 кадра (224). Text-инференс амортизируется кэшем.

## 8. Оптимизации

- **CLIP считается один раз** и переиспользуется 7+ модулями — главная архитектурная оптимизация цепочки
  (осознанное решение, «single source-of-truth»).
- **Дисковый кэш text-эмбеддингов** по ключу (промпты в порядке + модель + версия + size), структура
  `{DP_MODELS_ROOT}/cache/core_clip_text_embeddings/size_*/`. Ускорение single-video 1.2–1.5×,
  batch 3–10×. Text-эмбеддинги модель-агностичны, но версионируются по size — осознанно.
- **Batch-processing** нескольких видео: сбор кадров → группировка → batch inference → раздача обратно.
- **Явное освобождение VRAM** (`del model` + `empty_cache`) после инференса.
- **Атомарная запись NPZ** (tmp + `os.replace`) — устойчивость к сбоям.
- **Places365 top-K** вместо полной (N,365) матрицы наружу — компактный backend-payload (тех. необходимость).
- Env-gated `resource_profile_before` (VP_RESOURCE_PROFILE) — наблюдаемость без изменения контракта.

## 9. Слабые места

- **ViT-B/32 — самый слабый CLIP** (patch 32). Для тонкой семантики/мелких объектов/текста в кадре
  ViT-B/16 или ViT-L/14 заметно точнее. Компромисс скорость↔качество в пользу скорости.
- **`consecutive_cosine_prev` зависит от плотности выборки** — не абсолютная мера «динамичности», её
  нельзя сравнивать между видео с разным sampling. Потенциальный источник ошибочных выводов на сайте.
- **`*_scores` легко спутать с вероятностями** — не нормированы, суммы >1. Требует дисциплины downstream;
  документировано, но грабли остаются.
- **Промпты только на английском** — на не-английском контенте (русские надписи, специфичные реалии)
  zero-shot Places365/topic-скора менее надёжны; для русскоязычной аудитории продукта это заметный риск.
- **Popularity_topic_scores пока analytics-only** — польза для модели не подтверждена ablation (план в
  README есть, данных нет).
- **Golden / набор C (edge) не закрыты** в Audit v4 (§4.6/§4.8 DoD) — детерминизм подтверждён отдельным
  прогоном (diff 0.0, 2026-07-05), но формальная golden-signature по A не зафиксирована.
- **Короткие видео / малый N** — на 12-кадровых роликах (медиана в реальных данных = 12!) статистика
  cosine/scene бедная; qualitytimeline почти неинформативен.
- Отдельного `LOGIC_ERRORS_FOR_CLAUDE.md` L-номера по core_clip не заведено (багов не зафиксировано).

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Ввести опциональный профиль ViT-L/14 (или ViT-B/16)** для «quality»-тарифа — заметный
   прирост различимости сцен/объектов; оставить B/32 дефолтом для скорости. D изменится (768) → версионировать схему.
2. **[выс.] Мультиязычные промпты** (RU + EN) для Places365/popularity, т.к. аудитория продукта
   русскоязычная. Либо перейти на мультиязычный CLIP (OpenCLIP `xlm-roberta`) для text-энкодера.
3. **[сред.] Нормировать `consecutive_cosine_prev` на dt** (косинус на секунду), чтобы фича была
   сравнима между видео с разной плотностью выборки — заменить/дополнить, не удаляя raw.
4. **[сред.] Экспортировать temperature-softmax версию `*_scores`** параллельно raw — чтобы сайт мог
   показывать «вероятности» без риска неверной трактовки.
5. **[низ.] Подтвердить `popularity_topic` ablation** на baseline и либо повысить до model-facing, либо
   убрать из выхода (сейчас мёртвый вес в NPZ).

## 11. Рекомендации по архитектуре / связям

- **Шарить эмбеддинги явно через артефакт-контракт, а не пересчитывать** — уже так; закрепить в Encoder,
  что `frame_embeddings` = единственный источник CLIP-сигнала (нет дублей в similarity/uniqueness).
- **Вынести Places365-скора в отдельный тонкий провайдер?** — Нет: наоборот, текущее слияние (CLIP +
  все zero-shot головы в одном проходе) оптимально; дробить не стоит.
- **Согласовать sampling-требования с Segmenter** (coverage начало/середина/конец, cap на длинных видео) —
  сейчас это «требование core_clip», но глобальная SamplingPolicy DEFERRED. Довести в Segmenter, т.к.
  качество ВСЕЙ визуальной цепочки зависит от выборки core_clip.
- **Кэш text-эмбеддингов сделать общим** для всех видео батча в проде (уже есть) + прогревать заранее.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что реально говорит |
|---|---|---|---|
| Output-валидатор (`validate_core_clip_npz.py`) | schema v2 | VALID | контракт соблюдён, типы/shape/meta ок |
| Golden-детерминизм (2026-07-05, GPU A4500) | 2 прогона тех же кадров | **max diff 0.0 (побитово)** | CLIP полностью детерминирован → воспроизводимость |
| Audit v4 L2 (A+B) | 5 run, N_total=543 | ✓ ~8.8/10 | D=512/K=5 стабильны, 0% NaN (кроме cos[0]), 0 Inf, L2≈1 |
| Реальные артефакты storage (мой прогон) | **23 видео, 555 кадров** | 100% finite, L2 mean=1.0 | эмбеддинги здоровы на проде |
| — cos_prev по 23 видео | N=532 | mean 0.952 / median 0.978 / p05 0.80 / min 0.498 | осмысленное распределение «похожести кадров» |
| — NaN cos_prev | 23/23 первых кадра | ровно по 1 на видео | NaN by design корректен, паразитных NaN нет |
| Perf (README, Triton) | 224/336/448 | 0.6–47 c | масштабируется по N и размеру входа |

Вывод: надёжность **высокая** — контракт и численные инварианты держатся и на синтетике, и на 23 реальных
видео; детерминизм подтверждён. Не хватает лишь формальной golden-signature и edge-набора C.

## 13. Интерпретируемость

**Есть:** dev-рендер (`render.py` → `_render/render.html` + `render_context.json`): timeline embedding-norm,
timeline consecutive-cosine, summary/распределения. README подробно описывает, как читать (cosine≈1 →
статика; провалы → склейки). Places365 top-K даёт словесное «что это за видео».

**Добавить (для обычного пользователя):**
- **Thumbnails sampled-кадров** (≥12 равномерно) с `time_s` — чтобы человек видел, что анализировала модель.
- **«Топ переходов»** — кадры с минимальным cosine к предыдущему = где меняется сцена, с превью до/после.
- **Places365 словами на своём видео** («в основном: студия, улица, крупный план») — самое понятное объяснение.
- **Prompt-score панели** shot_quality: «лучшие/худшие по качеству съёмки кадры» с превью.
- Приложенная визуализация (`core_clip_distributions.png`) — пример того, как показать «насколько ваше
  видео динамичное» распределением cosine.

## 14. Польза для моделей

`frame_embeddings` — **класс C (precomputed embeddings)** во [`FEATURE_ENCODER_CONTRACT`](../../../../Models/docs/source_migrations/FEATURE_ENCODER_CONTRACT.md):
базовый seq-токен, идущий в VisualEncoder → learnable pooling → VisualHead → Fusion. Это, вероятно,
**самая информативная визуальная фича** для предсказания просмотров/лайков: CLIP кодирует и объект, и
сцену, и стиль, и «вайб» кадра одним вектором — ровно то, что коррелирует с привлекательностью контента.
Text-эмбеддинги промптов и scores дают модели готовые интерпретируемые оси (качество/эстетика/тема) поверх
сырых эмбеддингов. Прямых данных о feature-importance пока нет (модель в разработке), но гипотеза сильная.

## 15. Польза для аналитиков

- **`consecutive_cosine_prev`** → «динамичность/монтажность» видео (с оговоркой про sampling).
- **`places365_video_topk_*`** → сцены/локации видео словами — понятно и сравнимо между роликами.
- **`shot_quality_scores` / `scene_*_scores`** → прокси «качество съёмки», «эстетика», «атмосфера»,
  «люкс» — для дашбордов и сравнения своих видео.
- **`popularity_topic_scores`** → грубая тематизация контента (спорт/трэвел/еда/…).
- Сравнение видео по cosine на `frame_embeddings` (через similarity_metrics) — «на что похоже моё видео».

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 5 | Фундамент всей визуальной цепочки, незаменим |
| 5. Выход (контракт) | 5 | Чистая 3-уровневая классификация ключей, стабильные D/K, строгая схема |
| 6. Фичи | 4 | Мощные, но `*_scores`/cos легко неверно трактовать |
| 8. Оптимизации | 5 | Reuse CLIP, кэш text, batch, атомарность — образцово |
| 9. Слабые места (инверсно) | 3 | ViT-B/32 слабоват, EN-only промпты, cos зависит от sampling |
| 12. Результаты тестов | 4 | Golden 0.0, 23 реальных видео чисты; нет формальной golden-sig/edge-C |
| 13. Интерпретируемость | 3 | База есть (render+Places365), но персонализация для юзера в TODO |
| 14. Польза для моделей | 5 | Базовый seq-токен, самая информативная визуальная фича |
| 15. Польза для аналитиков | 4 | Places365/cos/scores полезны, но требуют пояснений |

### Итоговые оценки

- **Польза для моделей: 5/5.** `frame_embeddings` — несущий визуальный сигнал Encoder'а и источник для
  всех downstream; без него визуальная ветка не существует. Детерминирован, стабилен, информативен.
- **Польза для аналитиков: 4/5.** Богатый интерпретируемый выход (сцены Places365, динамичность, прокси
  качества), но часть метрик требует нормализации (cos на dt) и аккуратной подачи, чтобы не вводить в
  заблуждение; персонализированный рендер ещё не доведён.

## 17. Источники

- `DataProcessor/VisualProcessor/core/model_process/core_clip/main.py`
- `.../core_clip/README.md`, `.../docs/SCHEMA.md`, `.../docs/FEATURE_DESCRIPTION.md`
- `.../core_clip/utils/validate_core_clip_npz.py`, `.../utils/render.py`, `.../scripts/audit_v4_npz_stats.py`
- `DataProcessor/docs/component_reports/core_clip/REPORT_2026-07-05.md`
- `DataProcessor/docs/audit_v4/components/visual_processor/core/core_clip_audit_v4.md`
- `DataProcessor/docs/audit_v4/components/audit_4_2/visual_processor/core/core_clip_engineering_log_v4_2.md`
- `Models/docs/source_migrations/FEATURE_ENCODER_CONTRACT.md`, `Models/docs/contracts/MODEL_CONTRACTS_V1.md`
- `automation/runner/AGENT_CONTEXT.md` (разделы 6/7: тайминги ~14–34 c, golden diff 0.0)
- Downstream-потребители (grep frame_embeddings/*_text_embeddings): `modules/{scene_classification,
  shot_quality,cut_detection,similarity_metrics,uniqueness,high_level_semantic,story_structure,video_pacing}`
- Реальные артефакты: 23× `storage/result_store/youtube/*/*/core_clip/embeddings.npz` (555 кадров)

## 18. Визуализации

![Распределения core_clip](core_clip_distributions.png)

`core_clip_distributions.png` (построено на 23 реальных артефактах): слева — распределение
`consecutive_cosine_prev` (median 0.978, p05 0.80, min 0.498 → большинство соседних кадров похожи, провалы
= склейки); справа — распределение N кадров на видео (min 12 / median 12 / max 119). Показывает, что
компонент выдаёт здоровые, осмысленные значения на проде.
