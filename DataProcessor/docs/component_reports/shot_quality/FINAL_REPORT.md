# FINAL REPORT — `shot_quality`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `shot_quality` (VisualProcessor **module**, агрегатор технического качества) |
| Версия кода (`producer_version`) | `2.0.2` |
| Схема NPZ (`SCHEMA_VERSION`) | `shot_quality_npz_v3` |
| Артефакт | `result_store/<platform>/<video>/<run>/shot_quality/shot_quality.npz` |
| Движок | **numpy / CPU** (нет своей нейросети); F=48 фич, P=10 CLIP-промптов, K=3 top-k |
| Hard deps (5, aligned) | `core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks`, `cut_detection` |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → shot_quality ✅ (2026-07-05) |
| Отчёт валидации | [`REPORT_2026-07-05.md`](REPORT_2026-07-05.md), [`RUN_SPEC.md`](RUN_SPEC.md); Audit v4 [`shot_quality_audit_v4.md`](../../audit_v4/components/visual_processor/modules/shot_quality_audit_v4.md) |
| Код | `DataProcessor/VisualProcessor/modules/shot_quality/utils/shot_quality.py` |

## 2. Резюме

`shot_quality` — **модуль оценки технического качества съёмки** и первый в цепочке разбора модуль-агрегатор
(не core-провайдер): он не запускает свою нейросеть, а на CPU/numpy считает 48 классических CV-метрик
качества на кадр и **сшивает выходы сразу 5 core-компонентов** на строго выровненной оси `frame_indices`.
Даёт три уровня: `frame_features (N,48)` (резкость, шум, экспозиция, контраст, цвет, сжатие, линза, +depth,
+objects, +face-ROI), `quality_probs (N,10)` (zero-shot CLIP «кинематографично/смартфон/вебка/…»), и
shot-агрегации по границам из `cut_detection`. Это **самая тяжёлая по зависимостям цепочка проекта** (5
hard-deps). Компонент прод-готов по контракту: schema v3 стабильна на 5 прогонах Audit v4, оба валидатора
проходят, `quality_probs` — корректный softmax (∑≈1). Слабые места: **4 из 48 фич полностью NaN** (заглушены),
метрики — эвристики без калибровки на ground-truth, а документация фич — раздутое супермножество (дрейф).

## 3. Функционал

Стоит в визуальной цепочке **после** 5 core-провайдеров (Tier-1 модуль). На каждый sampled-кадр:

1. **Считает 48 CV-метрик качества** (numpy): резкость (Tenengrad/secondary/motion-blur/spatial-freq), шум
   (luma/chroma/ISO/grain/entropy), экспозиция (under/over/midtones/highlight/shadow), контраст
   (global/local/dynamic/clarity/microcontrast), цвет (WB/skin-tone/fidelity/uniformity), сжатие
   (blockiness/banding/ringing/bitrate/codec-entropy), линза (vignetting/chromatic/distortion), туман,
   temporal (flicker/rolling-shutter).
2. **Подмешивает выходы core-провайдеров** как фичи: `depth_mean/std/grad` (ROI-глубина из core_depth_midas),
   `objects_count/area_mean` (из core_object_detections), `face_sharpness/noise` (face-ROI из
   core_face_landmarks, NaN если лица нет).
3. **CLIP-quality** (`quality_probs`): softmax по 10 фиксированным quality-промптам через text-эмбеддинги
   core_clip (без загрузки CLIP).
4. **Shot-агрегации**: по границам шотов из `cut_detection` — mean/std/min/max каждой фичи, top-k классов
   качества, confidence/entropy per shot.

**Зачем продукту:** это ответ на вопрос **«насколько технически качественно снято видео»** — резкое ли,
не шумное ли, правильная ли экспозиция, кинематографично или «вебка». Качество съёмки — прямой драйвер
удержания и восприятия контента; для креатора это конкретный actionable-фидбек («ваше видео недоэкспонировано
/ смазано»). Для модели — компактный «технический отпечаток» кадра.

## 4. Вход

Контракт строгий, **no-fallback**, требует **идентичных `frame_indices` у всех 5 deps**
(`_ensure_same_indices` → aligned sampling group Segmenter обязателен):

- **`core_clip`** → text-эмбеддинги quality-промптов (для `quality_probs`) + CLIP-модель в meta.
- **`core_depth_midas`** → `depth_maps (N,H,W)` (пересчитывает per-frame depth_mean/std/grad).
- **`core_object_detections`** → боксы/классы (objects_count/area_mean).
- **`core_face_landmarks`** → лицевые лендмарки для face-ROI (валидная пустота → face-фичи = NaN).
- **`cut_detection`** → границы шотов (shot-агрегации).
- **Ось:** `frame_indices` из `metadata["shot_quality"]`, `times_s = union[...]` (no-fallback).
- **Кадры** — FrameManager (для самих CV-метрик).

**Empty-семантика:** компонентного empty нет; отсутствие лиц не блокирует non-face метрики (face-ROI = NaN).
Любой отсутствующий dep / рассинхрон индексов / нет ключей → **error**.

## 5. Выход

NPZ `shot_quality.npz`, `allow_extra_keys=false`. Три уровня:

- **frame-level (model-facing):** `frame_features (N,48)`, `feature_names (48,)`,
  `frame_feature_present_ratio (48,)`, `quality_probs (N,10)` (float16, softmax), `frame_indices/times_s`.
- **shot-level:** `shot_ids (N,)`, `shot_start/end_frame (S,)`, `shot_frame_count (S,)`,
  `shot_features_mean/std/min/max (S,48)`, `shot_frame_feature_present_ratio (S,48)`,
  `shot_quality_topk_ids/probs (S,3)`, `shot_quality_conf_mean/entropy_mean (S,)`.
- **debug/UI:** `meta` (+ `ui_payload` schema `shot_quality_ui_v1`, `impl_meta` с sha промптов, mappings).

**Инварианты (Audit v4 + мой прогон):** F=48, P=10, K=3; `quality_probs` ∑строки ≈1 (softmax);
`shot_quality_topk_probs` ∑ ≈0.30 (это per-shot mean top-K, **не** распределение — грабля для Encoder).
`frame_features` NaN = «нет/не определено».

## 6. Фичи (важное/неочевидное)

- **frame_features — «готовый seq-токен качества».** Решение проекта: отдельный quality-эмбеддинг не нужен
  (в отличие от scene/AR) — вектор из 48 технических метрик и есть представление кадра для Encoder.
- **4 фичи полностью NaN на всём корпусе** (подтверждено на 22 реальных видео): `vignetting_level`,
  `chromatic_aberration_level`, `lens_sharpness_drop_off`, `rolling_shutter_artifacts_score` — заглушены/не
  реализованы в текущем пайплайне. ~8% фич — мёртвый вес.
- **face-ROI фичи (`face_sharpness_tenengrad`, `face_noise_level_luma`) — present только 17%** (лица редки,
  NaN by design когда лица нет). Согласуется с разбором core_face_landmarks (20/24 видео без лиц).
- **depth/objects подмешаны как фичи**, но **пересчитываются локально**: `depth_mean/std/grad` считаются
  заново из `depth_maps` по ROI (не берутся готовые агрегаты core_depth) — это осознанно (нужна ROI-глубина
  объекта, а не глобальная), но означает дублирование вычислений.
- **`quality_probs` — настоящий softmax** (∑=1), в отличие от `*_scores` core_clip. Это интерпретируемые
  вероятности «кинематографично / смартфон-хорошо / смартфон-плохо / вебка / скринкаст / видеонаблюдение».
- **Метрики — эвристики классического CV** (Sobel/Laplacian/FFT/percentile), **не калиброваны на
  ground-truth** качества — относительные, не абсолютные (см. §9).

## 7. Алгоритм / архитектура

- **Движок:** чистый **numpy/OpenCV на CPU**, покадрово. Нет своей нейросети. Единственная «модель» — CLIP
  (через готовые text-эмбеддинги core_clip, инференс не запускается заново).
- **CV-метрики:** Sobel/Tenengrad (резкость), FFT-спектр (motion-blur, spatial-freq), Gaussian-diff (шум),
  percentile-гистограммы (экспозиция), Laplacian (контраст/микроконтраст), YUV/LAB/HSV (цвет), блочные
  8×8-различия (сжатие), центр-vs-углы (виньетка/линза).
- **Shot-агрегация:** группировка кадров по `shot_ids` (границы cut_detection) → mean/std/min/max/top-k.
- **Где идёт:** CPU. Сам модуль **дёшев** (numpy, ~единицы мс/кадр); вся стоимость — в 5 upstream-провайдерах
  (самая дорогая цепочка проекта: core_clip 34c + object_det 57c + face 16–44c + depth 44c + cut 137c).
- **Сложность:** линейна по N; тяжелее всего FFT-метрики.

## 8. Оптимизации

- **Reuse 5 upstream-артефактов** вместо пересчёта — ключевая оптимизация: core-провайдеры нужны и другим
  модулям, shot_quality лишь надстройка (осознанное архитектурное решение, критично для 200k).
- **CLIP не запускается заново** — берутся готовые text-эмбеддинги промптов из core_clip (единый источник).
- **Feature gating presets** (fast/default/quality) — можно отключать дорогие метрики по бюджету тарифа.
- **`ui_payload` готовится сразу в артефакте** (frame_confidence/entropy графики, top-k по шотам) — backend
  не пересчитывает для сайта.
- **present_ratio по фичам/шотам** — дешёвая самодиагностика полноты данных (сразу видно мёртвые фичи).
- **Атомарная запись + оба валидатора** (input-валидатор 5 deps + output schema).

## 9. Слабые места

- **[выс.] 4 из 48 фич полностью NaN** — `vignetting_level`, `chromatic_aberration_level`,
   `lens_sharpness_drop_off`, `rolling_shutter_artifacts_score` заглушены/не считаются. Либо реализовать,
   либо убрать из контракта (мёртвый вес в model-facing тензоре, снижает L2 до ~8.2).
- **[выс.] Метрики не калиброваны на ground-truth.** Все 48 — эвристики (нет датасета «хорошее/плохое
   качество»), их абсолютная корректность не проверена. `iso_estimated_value`, `bitrate_estimation_score`,
   `aesthetic` и т.п. — грубые прокси, могут вводить в заблуждение как абсолютные числа.
- **[сред.] Дрейф документации.** `FEATURES_DESCRIPTION.md` описывает раздутое супермножество (DnCNN/CBDNet
   noise-модели, `clip_embedding 768`, `quality_cinematic_prob` как frame_feature, string-типы distortion) —
   **реальный выход скромнее** (48 numpy-метрик, CLIP только в отдельном `quality_probs`). Док вводит в
   заблуждение о том, что реально считается.
- **[сред.] Наследует качество 5 deps.** Рассинхрон/ошибка любого upstream → error или мусор; face-ROI пуст
   на 83% кадров; depth относительна; objects — на COCO-микс-весах (см. разбор core_object_detections).
- **`shot_quality_topk_probs` не суммируется в 1** (per-shot mean top-K) — грабля, Encoder не должен ждать
   распределение (зафиксировано Audit v4 §4.1a ◐).
- **Короткие видео / малый N** (медиана 12) + S до 1 шота — shot-агрегации бедны.
- Отдельного `LOGIC_ERRORS_FOR_CLAUDE.md` L-номера нет.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Закрыть 4 мёртвые фичи** — либо реализовать (vignetting/chromatic/lens-dropoff/rolling-shutter),
   либо убрать из схемы. Не держать полностью-NaN колонки в model-facing тензоре.
2. **[выс.] Калибровать/заменить ключевые метрики на обучаемую оценку качества** — например NR-IQA-модель
   (BRISQUE/NIMA/MUSIQ) вместо россыпи ручных эвристик; даст валидированный, сопоставимый скор качества.
3. **[выс.] Синхронизировать `FEATURES_DESCRIPTION.md` с реальными 48 фичами** — убрать несуществующие
   (DnCNN/CBDNet/clip_embedding 768), это база для доверия к компоненту.
4. **[сред.] Не пересчитывать depth-агрегаты**, если ROI-глубина не нужна — брать готовые из core_depth
   (или задокументировать, что ROI-depth ≠ глобальной).
5. **[низ.] Нормировать эвристики в [0,1] с понятной семантикой** (сейчас часть в «сырых» единицах —
   iso/bitrate) — для интерпретируемости и стабильности для Encoder.

## 11. Рекомендации по архитектуре / связям

- **shot_quality — образцовый пример reuse-паттерна:** надстройка над 5 core-провайдерами без пересчёта.
  Закрепить как модель для остальных модулей (не дублировать тяжёлый инференс).
- **Aligned sampling group обязателен** — 5 deps на одинаковых `frame_indices`; гарантировать на Segmenter,
  а не ловить рассинхрон error'ом в самом тяжёлом модуле.
- **CLIP-quality промпты шэрятся с core_clip** (`shot_quality_text_embeddings`) — единый источник, уже так.
- **Объединить face-ROI логику с emotion/detalize** — все читают одни лицевые лендмарки; не дублировать ROI.
- **Feature gating согласовать с тарифами** сайта (fast/quality) — прямой рычаг стоимости для 200k.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что реально говорит |
|---|---|---|---|
| Полная цепочка (6 компонентов, GPU, без Triton) | 1 прогон | все 6 rc=0, выход валиден | 5-deps цепочка сходится end-to-end |
| Оба валидатора (input 5 deps + output schema) | валидация | PASS | контракт входа и выхода соблюдён |
| Audit v4 L2 (A+B) | 5 run, N_total=543 | ✓ **~8.2/10** | F=48/P=10/K=3 стабильны; softmax ∑≈1; 4 fully-NaN фичи |
| — quality_probs softmax | Audit v4 + мой прогон | ∑строки [0.9998, 1.0001] | корректные вероятности |
| — NaN профиль | Audit v4 | ~9.9% ячеек, 6 фич | 4 fully-NaN + 2 face-ROI частичные |
| **Реальные артефакты storage (мой прогон)** | **22 видео, 543 кадра** | **F=48, softmax ok** | контракт здоров на проде |
| — 4 fully-NaN фичи | 22 видео | present_ratio = 0.00 | подтверждено: vignetting/chromatic/lens-dropoff/rolling-shutter мертвы |
| — face-ROI фичи | 22 видео | present_ratio = 0.17 | лица редки (NaN by design) |
| — N / S | 22 видео | N med 12, S 1–6 | короткие ролики, мало шотов |

Вывод: **контракт и агрегация надёжны** (schema, softmax, оба валидатора, end-to-end цепочка), но **качество
самих фич не подтверждено** (эвристики без калибровки + 4 мёртвые + док-дрейф) — это самый низкий L2 (8.2)
среди разобранных компонентов.

## 13. Интерпретируемость

**Есть (сильно):** `ui_payload` (frame_confidence/entropy графики, top-k качества по шотам, video mean probs)
готов в артефакте; `quality_probs` — понятные вероятности типа съёмки; `render.py`; present_ratio.

**Добавить (для обычного пользователя):**
- **Словесный вердикт качества:** «резко, хорошая экспозиция, немного шумно» из ключевых фич — самое понятное.
- **Тип съёмки словами** из `quality_probs`: «снято как: смартфон (хорошо) / вебка / кинематографично».
- **Худшие кадры** по резкости/экспозиции с превью («здесь смазано / пересвет») — actionable для креатора.
- **Timeline качества** по шотам — где просело качество.
- Приложенная визуализация (`shot_quality_present_ratio.png`) — честно показывает, какие фичи живые/мёртвые.

## 14. Польза для моделей

`frame_features (N,48)` — **готовый model-facing seq-токен технического качества** (решение проекта: без
отдельного эмбеддинга). Плюс `quality_probs (N,10)` — интерпретируемые оси типа съёмки. Техническое качество
правдоподобно **умеренно** влияет на просмотры (плохое качество отталкивает, но вирусность определяется
контентом, а не резкостью) — скорее корректирующий, чем несущий сигнал. Для Encoder полезен как компактный
«отпечаток продакшена». Снижают ценность 4 мёртвые фичи, некалиброванность эвристик и face-ROI-разреженность.
Прямых feature-importance данных нет.

## 15. Польза для аналитиков

- **quality_probs** → «как снято ваше видео» (смартфон/вебка/кинематографично) — очень понятный ярлык.
- **Резкость/экспозиция/шум/контраст** → конкретный actionable-фидбек креатору («недоэкспонировано»,
  «смазано», «шумно») — редкий компонент с прямыми съёмочными рекомендациями.
- **Shot-агрегации** → где по таймлайну просело качество.
- Оговорка: значения относительные/некалиброванные — показывать как «лучше/хуже», не как абсолютные оценки;
  4 фичи не показывать (мёртвые).

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 5 | Уникальный слой «качество съёмки» + образцовый reuse 5 deps |
| 5. Выход (контракт) | 4 | Богатый 3-уровневый выход; 4 мёртвые фичи + topk-грабля |
| 6. Фичи | 3 | 48 метрик, но эвристики без калибровки + 4 fully-NaN + face-ROI редки |
| 8. Оптимизации | 4 | Reuse upstream, CLIP-share, gating, ui_payload; depth-пересчёт |
| 9. Слабые места (инверсно) | 3 | Мёртвые фичи, некалиброванность, док-дрейф, наследует 5 deps |
| 12. Результаты тестов | 4 | Оба валидатора+end-to-end+softmax; L2 8.2 (ниже прочих) |
| 13. Интерпретируемость | 4 | ui_payload+quality_probs очень понятны, словесный вердикт в TODO |
| 14. Польза для моделей | 4 | Готовый quality-токен, но корректирующий сигнал + мёртвые фичи |
| 15. Польза для аналитиков | 4 | Прямой съёмочный фидбек — редкая ценность; нужна калибровка подачи |

### Итоговые оценки

- **Польза для моделей: 4/5.** Компактный готовый «отпечаток технического качества» + интерпретируемые оси
  типа съёмки — полезное дополнение к контентным фичам. Снижают 4 мёртвые фичи, некалиброванные эвристики и
  то, что качество съёмки — скорее корректирующий, чем несущий предиктор просмотров.
- **Польза для аналитиков: 4/5.** Один из немногих компонентов с **прямым actionable-фидбеком креатору**
  («снято как смартфон, недоэкспонировано, смазано») — высокая продуктовая ценность. Ограничивают
  некалиброванность метрик (подавать как относительные) и мёртвые/разреженные фичи.

## 17. Источники

- `DataProcessor/VisualProcessor/modules/shot_quality/utils/shot_quality.py` (48-фич stable order, depth_metrics)
- `.../shot_quality/utils/{validate_shot_quality_input.py, validate_shot_quality_npz.py, render.py}`
- `.../shot_quality/docs/{SCHEMA.md, FEATURE_DESCRIPTION.md, FEATURES_DESCRIPTION.md}`, `README.md`
- `DataProcessor/docs/component_reports/shot_quality/{REPORT_2026-07-05.md, RUN_SPEC.md}`
- `DataProcessor/docs/audit_v4/components/visual_processor/modules/shot_quality_audit_v4.md`
- Upstream deps (contracts): `core_clip`, `core_depth_midas`, `core_object_detections`,
  `core_face_landmarks`, `cut_detection` (см. их FINAL_REPORT/COMPONENT_CONTRACTS.md)
- `automation/runner/AGENT_CONTEXT.md` (разделы 6/7: 5-deps выравнивание, тайминги цепочки, NaN by design)
- Реальные артефакты: 22× `storage/result_store/youtube/*/*/shot_quality/shot_quality.npz` (543 кадра)

## 18. Визуализации

![present_ratio по 48 фичам](shot_quality_present_ratio.png)

`shot_quality_present_ratio.png` (построено на 22 реальных видео, 543 кадра): доля конечных значений по всем
48 фичам. Красным — **4 полностью-NaN фичи** (vignetting_level, chromatic_aberration_level,
lens_sharpness_drop_off, rolling_shutter_artifacts_score — заглушены); оранжевым — 2 face-ROI фичи (present
0.17, лица редки); синим — 42 рабочие фичи. Наглядно показывает, что ~8% контракта — мёртвый вес, а
face-ветка разрежена; остальное считается стабильно.
