# FINAL REPORT — `scene_classification`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `scene_classification` (VisualProcessor `BaseModule`, Tier-2) |
| Версия кода | `2.0.1` |
| Схема NPZ | `scene_classification_npz_v2` |
| Артефакт | `result_store/<platform>/<video>/<run>/scene_classification/scene_classification_features.npz` |
| Модель | **Places365 ResNet50** (CSAIL, 365 классов сцен) via Triton/inprocess; CLIP zero-shot (label_fusion=clip) |
| Hard deps | `core_clip` (эмбеддинги + text-эмбеддинги) + `cut_detection` (границы сцен) — no-fallback |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → scene_classification ✅ |
| Отчёт валидации | [`REPORT_2026-07-05.md`](REPORT_2026-07-05.md) (🔄), [`RUN_SPEC.md`](RUN_SPEC.md) |
| Баг-реестр | `LOGIC_ERRORS_FOR_CLAUDE.md` L1 (Triton Places365 batch=1 → HTTP 400) |
| Код | `DataProcessor/VisualProcessor/modules/scene_classification/utils/scene_classification.py` (2244 строки) |

## 2. Резюме

`scene_classification` — **классификатор места/сцены**: Places365-ResNet50 выдаёт по кадру распределение над
365 классами сцен (discotheque, music_studio, beauty_salon…), а границы `cut_detection` режут видео на **сцены**
с усреднённой меткой, длительностью и уверенностью. Плюс — «advanced semantics» через CLIP zero-shot
(эстетика/люкс/атмосфера/настроение) и **model-facing токен `frame_scene_embedding (N,D)`**, переиспользующий
CLIP-эмбеддинг из `core_clip` (нулевая доп. инференс-стоимость). На реальном корпусе **ядро — Places365 —
живо и осмысленно** (метки правдоподобны, уверенность 0.36–0.51, различимо), но **две «model-fit» надстройки
из сессии 2026-07-05 в реальном батче мертвы**: `frame_scene_embedding` **отсутствует** во всех артефактах, а
CLIP advanced-семантики (aesthetic/luxury/epic/cozy/scary) = **0.0** (хотя `core_clip` несёт нужные text-
эмбеддинги). Батч прогнан **через Triton** (L1-путь batch=1 — узкое место; рекомендован inprocess+fp16+batch).

## 3. Функционал

Стоит в Tier-2, **после `core_clip` и `cut_detection`**. Логика:

1. **Per-frame классификация** — Places365-ResNet50 → top-k классы сцены + распределение (365); энтропия,
   top1-prob, top1↔top2 gap → уверенность.
2. **Сегментация по сценам** — берёт границы `cut_detection`, склеивает планы в сцены с `min_scene_seconds`,
   усредняет метки → `scenes` (label, start/end, length, стабильность метки).
3. **Scene-эмбеддинг (model-fit)** — `frame_scene_embedding (N,D)` L2 = переиспользованный CLIP-эмбеддинг
   кадра из core_clip (аналог penultimate у action_recognition): богатое представление сцены сверх 365 меток.
4. **Advanced semantics** — CLIP zero-shot по промптам (эстетика/люкс/атмосфера cozy/epic/scary/neutral),
   text-эмбеддинги приходят из core_clip.

**Зачем продукту:** «где происходит видео» (студия/улица/кухня/сцена) — **сильный семантический контекст**
контента: жанр, обстановка, продакшн. Метка сцены + её эмбеддинг кормят Encoder (контекст), а аналитику дают
понятный ярлык («ваше видео снято в beauty_salon») и профиль атмосферы/эстетики.

## 4. Вход

- **`core_clip`** (hard, no-fallback) — `frame_embeddings`+`frame_indices` (для scene-эмбеддинга и семантик),
  `scene_*_text_embeddings` (aesthetic/luxury/atmosphere), `places365_text_embeddings` (обяз. при `label_fusion=clip`).
- **`cut_detection`** (hard) — `shot_boundaries_frame_indices` для сегментации на сцены.
- **`union_timestamps_sec`** + Segmenter `frame_indices` (⊆ core_clip.frame_indices) — строгая ось.
- **Places365 веса** (CSAIL ResNet50; Triton `places365_resnet50_224` или inprocess ModelManager).
- **`label_fusion`** = `places` (дефолт) | `clip` (zero-shot по 365 меткам через CLIP).

## 5. Выход

- **Per-frame:** `frame_topk_ids/probs`, `frame_top1_prob`, `frame_top1_top2_gap`, `frame_entropy`, `frame_scene_id`.
- **Scene-эмбеддинг (model):** `frame_scene_embedding (N,D)` L2 — **в реальном батче ОТСУТСТВУЕТ** (§9).
- **Сцены:** `scenes` (dict: label/start/end/length/stability), `scene_ids`, `scene_change_score`.
- **Video-агрегаты:** `dominant_places_topk_ids/probs`, `class_entropy_mean`, `top1_prob_mean`,
  `fraction_high_confidence_frames`, `label_stability`.
- **Advanced:** `mean_aesthetic_score`, `mean_luxury_score`, `mean_cozy/epic/scary/neutral`, `atmosphere_entropy`
  — **в реальном батче = 0.0** (§9).
- **Ось:** `frame_indices`, `times_s`.

## 6. Фичи (важное/неочевидное)

- **`frame_scene_embedding` reuse core_clip** — умное решение: не гонять отдельную scene-CNN для эмбеддинга,
  а взять уже посчитанный CLIP-вектор (нулевая доп. стоимость), богаче 365-мерного one-hot. Но в батче его нет.
- **Сегментация от cut_detection, а не своя** — единый источник границ; качество сцен наследует точность
  cut_detection (у которого deep-канал мёртв, пороги без калибровки).
- **`class_entropy_mean`/`top1_prob`** — уверенность классификатора: на реальных данных 0.36–0.51 (норм),
  но короткий/неоднозначный ролик (-Q6fnPIyb, N=12) даёт top1=0.125, entropy=4.77 (модель «не уверена»).
- **`label_stability`** — насколько метка сцены постоянна внутри сегмента (прокси «одна локация vs мешанина»).
- **Два label_fusion** (places vs clip) — на тест-видео давали одинаковые метки (согласованность backbone'ов).
- **advanced CLIP semantics** — эстетика/люкс/атмосфера как zero-shot по промптам: концептуально ценно для
  «настроения» видео, но в батче зануллено.

## 7. Алгоритм / архитектура

- **Backbone A:** Places365 ResNet50 (CSAIL веса, missing/unexpected 0), fp16 bs=64 → **2188 img/s**, пик
  VRAM 812 МБ. **Backbone B:** CLIP zero-shot (label_fusion=clip). C (ConvNeXt/ViT) — код готов, **публичных
  Places365-fine-tuned весов нет** → не задействован.
- **Инференс:** прод-батч шёл **через Triton** (`runtime=triton`), где ensemble Places365 имеет `max_batch_size=0`
  → фактический batch=1 (L1, HTTP 400 при batch>1). Рекомендация отчёта: **inprocess+fp16+batch 32–64**.
- **Сложность:** ResNet50 forward на кадр; в цепочке ~13 c (после core_clip 14 c + cut_detection).
- **Детерминизм:** golden не догнан полностью в отчёте (🔄), но модель-слой воспроизводим.

## 8. Оптимизации

- **Scene-эмбеддинг через reuse core_clip** — ноль лишнего инференса (главная оптимизация, но не в батче).
- **Сегментация от готовых границ cut_detection** — нет второго детектора склеек.
- **fp16 + batch** для Places365 (×1.7 к fp32) — доказано на модели, но в прод-батче не задействовано (Triton bs=1).
- **Advanced-семантики через text-эмбеддинги core_clip** — не грузит CLIP повторно.

## 9. Слабые места

- **`frame_scene_embedding` отсутствует в реальном батче (главное).** Model-facing scene-токен — headline-фича
  сессии 2026-07-05 — **не материализован ни в одном storage-артефакте** (`scene_embedding_source=None`).
  Encoder не получает scene-эмбеддинга на реальных данных, только 365-мерное распределение. Отчёт помечен 🔄.
- **Advanced CLIP-семантики = 0.0 в батче** — `mean_aesthetic/luxury/epic/cozy/scary` зануллены на всех 6
  видео, хотя `core_clip` несёт `scene_aesthetic/luxury/atmosphere_text_embeddings`. Эстетика/атмосфера как
  фичи мертвы (вероятно, считаются только при `label_fusion=clip`, а батч — `places`).
- **Прод-путь через Triton batch=1 (L1)** — узкое место скорости; отчёт рекомендует inprocess+fp16+batch, но
  прод-обвязка на Triton. Для 200k — дорого/медленно.
- **Качество сцен зависит от cut_detection** — плохие границы → плохая сегментация (deep-канал cut_detection мёртв).
- **Нет свежих ConvNeXt/ViT-Places365 весов** — «современный» backbone упирается в отсутствие публичных весов
  (только ResNet/VGG16 CSAIL); качество классификации ограничено ResNet50 2016-года.
- **Отчёт валидации 🔄 «в работе»** — golden/полный прогон на 15 видео не закрыты (штамп есть, но остаток задекларирован).
- **Places365 = «места», не «сюжеты»** — 365 меток покрывают локации, а не действия/жанр; для многих
  YouTube-видео метка приблизительна (stage/indoor, balcony/interior).

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Пере-прогнать батч с `frame_scene_embedding`** — materialize model-facing scene-токен (сейчас его
   нет ни в одном артефакте, хотя код и reuse core_clip готовы). Блокер пользы для Encoder.
2. **[выс.] Включить advanced CLIP-семантики в проде** (или согласовать `label_fusion=clip`/отдельный путь) —
   aesthetic/luxury/atmosphere сейчас 0.0, хотя text-эмбеддинги в core_clip есть.
3. **[выс.] Перейти на inprocess+fp16+batch** вместо Triton bs=1 (L1) — ×1.7+ скорость, снимает HTTP 400 риск.
4. **[сред.] Догнать golden + прогон на 15+ видео** (остаток 🔄-отчёта) до полноценного штампа.
5. **[низ.] Оценить fine-tune ViT/ConvNeXt под Places365** — если качество ResNet50 окажется узким местом.

## 11. Рекомендации по архитектуре / связям

- **Reuse core_clip закреплён** (эмбеддинги + все text-эмбеддинги семантик) — правильно; убедиться, что
  scene-эмбеддинг реально пишется (сейчас теряется).
- **scene_classification — хаб для color_light/shot_quality** (они hard-зависят от `scenes`) — качество сцен
  критично для downstream; чинить совместно с cut_detection.
- **Единый scene-контекст в Encoder** — метка + эмбеддинг + атмосфера как один «где/какая обстановка» токен.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| Places365 модель на GPU | HF-видео | top-3 осмысленны (music_studio/arena…) | классификатор корректен |
| Бенчмарк fp16 bs=64 | — | 2188 img/s, VRAM 812 МБ | inprocess-путь быстр |
| End-to-end цепочка (без Triton) | 1 видео | 112 кадров, 4 сцены, эмбеддинг (112,512), метки осмысленны, оба валидатора pass | логика цепочки работает |
| label_fusion places vs clip | 1 видео | одинаковые метки | backbone'ы согласованы |
| **Реальный storage (мой прогон)** | **6 видео, все ok** | метки правдоподобны (discotheque/beauty_salon), top1 0.36–0.51; **эмбеддинг ОТСУТСТВУЕТ, advanced=0.0, runtime=triton** | ядро живо; model-fit надстройки мертвы |

Вывод: **Places365-классификация и сегментация — живы и осмысленны на реальных данных**, но обе model-fit
надстройки (scene-эмбеддинг + CLIP-семантики) в батче не материализованы, а прод-путь на медленном Triton bs=1.

## 13. Интерпретируемость

- **Сильная сторона:** метка сцены — сразу понятна («beauty_salon», «music_studio»); `render.py` есть.
- **Добавить:** словесная сводка «видео снято в: салон красоты (52%)»; таймлайн смены сцен; профиль атмосферы
  (cozy/epic/luxury) — **после** починки advanced-семантик; топ-3 меток с процентами в кабинете.

## 14. Польза для моделей

**Потенциально высокая, фактически ограниченная.** «Где происходит видео» — сильный семантический контекст,
и дизайн предусматривает идеальную форму для Encoder: 365-распределение + плотный scene-эмбеддинг (reuse CLIP).
**Но** на реальных данных Encoder получает только 365-распределение — scene-эмбеддинг отсутствует, advanced-
семантики зануллены. То есть реализованная польза = метка+уверенность (полезно, но не headline). После
пере-прогона с эмбеддингом ценность вырастет. Балл держит нереализованность.

## 15. Польза для аналитиков

**Высокая (по ядру).** Метка сцены — понятнейший креатору ярлык («салон/студия/сцена»), сегментация по сценам
+ уверенность + доминирующие локации наглядны и сравнимы. Ограничения: атмосфера/эстетика (самое «вкусное»
для креатора) сейчас 0.0; метки Places365 — «места», не жанр, иногда приблизительны; качество сцен зависит от
cut_detection.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 4 | Классификация мест + сегментация + (задуманы) эмбеддинг/семантики — богато |
| 5. Выход (контракт) | 3 | Богатый контракт, но headline scene-эмбеддинг отсутствует в батче |
| 6. Фичи | 3 | Places365-ядро сильно; scene-эмбеддинг и advanced-семантики мертвы на данных |
| 8. Оптимизации | 3 | Reuse core_clip/fp16 задуманы верно, но прод на Triton bs=1, эмбеддинг не пишется |
| 9. Слабые места (инверсно) | 2 | Эмбеддинг отсутствует, advanced=0, Triton L1, 🔄-отчёт, ResNet50-2016 |
| 12. Результаты тестов | 3 | Ядро осмысленно на реальных данных; надстройки не материализованы, golden не догнан |
| 13. Интерпретируемость | 4 | Метка сцены сразу понятна; атмосфера — потом |
| 14. Польза для моделей | 3 | Контекст ценен, но реально доступно только 365-распределение |
| 15. Польза для аналитиков | 4 | «Где снято» + сегментация наглядны; атмосфера/эстетика пока 0 |

### Итоговые оценки

- **Польза для моделей: 3/5.** Семантический контекст сцены — ценная ось, а дизайн (365-распределение +
  reuse-CLIP scene-эмбеддинг) правильный. Но фактически Encoder получает только распределение: headline
  `frame_scene_embedding` отсутствует в батче, advanced-семантики зануллены. Реализованная польза — «приемлемо»,
  потенциал — 4–5 после пере-прогона.
- **Польза для аналитиков: 4/5.** Метка места + сегментация по сценам + уверенность — понятная, наглядная,
  сравнимая аналитика («видео снято в beauty_salon»). Балл ниже 5 держат мёртвые atmosphere/aesthetic (самое
  интересное для креатора) и приблизительность Places365-меток на не-«локационном» контенте.

## 17. Источники

- `DataProcessor/VisualProcessor/modules/scene_classification/utils/scene_classification.py` (2244 строки)
- `.../utils/{validate_scene_classification_npz.py,validate_scene_classification_input.py,render.py}`, `main.py`
- `DataProcessor/VisualProcessor/modules/scene_classification/docs/SCHEMA.md`, `schemas/scene_classification_npz_v2.json`
- `DataProcessor/docs/component_reports/scene_classification/{REPORT_2026-07-05.md,RUN_SPEC.md}`
- `DataProcessor/docs/LOGIC_ERRORS_FOR_CLAUDE.md` (L1 Triton Places365 batch=1)
- `DataProcessor/docs/COMPONENT_CONTRACTS.md` (core_clip → scene_classification: frame_embeddings/places365/scene_* text-emb)
- Cross-ref: `core_clip`, `cut_detection` (FINAL_REPORTs); downstream `color_light`, `shot_quality`
- Реальные артефакты: 6 уникальных× `storage/result_store/youtube/*/*/scene_classification/scene_classification_features.npz`
  (все ok; **frame_scene_embedding отсутствует, advanced-семантики=0.0, runtime=triton**)

## 18. Визуализации

![scene_classification overview](scene_classification_overview.png)

`scene_classification_overview.png`: слева — Places365-уверенность (top1_prob 0.36–0.51, энтропия) + топ-метки
сцен на 6 реальных видео (discotheque/music_studio/beauty_salon/jewelry_shop — осмысленны, различимы); справа —
model-fit надстройки, **мёртвые в батче**: `frame_scene_embedding (N,D)` отсутствует, все advanced CLIP-семантики
(aesthetic/luxury/epic/cozy/scary) = 0.0 (хотя core_clip несёт нужные text-эмбеддинги). Подтверждает раздельный
вердикт: Places365-ядро живо и осмысленно, scene-токен и атмосфера требуют пере-прогона.
