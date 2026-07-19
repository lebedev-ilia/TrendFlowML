# FINAL REPORT — `core_face_landmarks`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Отдельная задача от валидации (штамп ✅ уже стоит). Здесь — «насколько компонент хорош и полезен».

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `core_face_landmarks` (VisualProcessor **core** provider, Tier-0) |
| Версия кода (`VERSION`) | `2.1` |
| Схема NPZ (`SCHEMA_VERSION`) | `core_face_landmarks_npz_v2` |
| Артефакт | `result_store/<platform>/<video>/<run>/core_face_landmarks/landmarks.npz` |
| Модель | **MediaPipe** FaceMesh (468 3D-точек), опц. Pose (33×4), Hands (21×3 ×2); Solutions API |
| FACES=1, HANDS=2 · person-mask гейтинг · OneEuro temporal-фильтр (raw+filtered) |
| Deploy-требование | **mediapipe `<0.10.15`** (0.10.35 удалил `mp.solutions` → AttributeError) |
| Дата разбора | 2026-07-17 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → core_face_landmarks ✅ (2026-07-05) |
| Отчёт валидации | [`REPORT_2026-07-05.md`](REPORT_2026-07-05.md); Audit v4 [`core_face_landmarks_audit_v4.md`](../../audit_v4/components/visual_processor/core/core_face_landmarks_audit_v4.md) |
| Код | `DataProcessor/VisualProcessor/core/model_process/core_face_landmarks/main.py` (1515 строк) |

## 2. Резюме

`core_face_landmarks` — **Tier-0 провайдер лицевой геометрии** и фундамент всей человеко-центричной ветки
продукта. На primary-выборке кадров (владелец — Segmenter) он через MediaPipe FaceMesh извлекает **468 3D-
точек лица** на кадр (+ опционально позу 33 и руки 21×2), с гейтингом по **person-mask из
`core_object_detections`** (запускает mesh только там, где детектор нашёл человека, в окне ±radius). Выход —
`face_landmarks (N,FACES,468,3)` + богатая диагностика присутствия (`face_present`, `has_any_face`,
`face_mesh_ran`, `*_empty_reason`) и OneEuro-сглаживание (raw + filtered). Его **жёстко потребляет вся
face/emotion-ветка**: `shot_quality` (face-ROI), `emotion_face`, `micro_emotion`, `detalize_face`,
`behavioral`, `frames_composition`, `high_level_semantic`, `story_structure`. Компонент прод-готов: schema v2
стабильна на 5 прогонах Audit v4, **NaN-политика слотов соблюдена строго** (present→без NaN, absent→все NaN;
0 нарушений на 24 реальных видео), **valid-empty на видео без лиц — ключевая фича** (не error).

## 3. Функционал

Стоит в начале визуального пайплайна (Tier-0, после Segmenter и логически после core_object_detections —
нужна person-mask). Двухстадийный:

1. **Stage-1 (detect):** по person-маске из детектора отбирает кадры, где есть человек (окно
   `person_window_radius` вокруг person-кадров, stride по длине видео), и запускает MediaPipe.
2. **Stage-2 (mesh):** FaceMesh даёт 468 3D-лендмарков на лицо; опционально Pose (33 точки скелета) и Hands
   (21×2). OneEuro temporal-фильтр сглаживает дрожание между кадрами (сохраняются и raw, и filtered).

**Зачем продукту:** это **геометрия лица во времени** — фундамент анализа человека в кадре. Крупность и
позиция лица определяют качество портретной съёмки (`shot_quality`); мимика/геометрия → эмоции
(`emotion_face`, `micro_emotion`); детальный разбор лица (`detalize_face`); поза/жесты → поведение
(`behavioral`). Лица и эмоции — сильнейший драйвер вовлечённости в контенте с людьми, поэтому этот провайдер
критичен для человеко-центричных видео (блогеры, talking-head, реакции).

## 4. Вход

Контракт строгий:

- **Кадры** — `FrameManager.get(idx)` из `frames_dir`, RGB uint8.
- **`metadata.json.core_face_landmarks.frame_indices`** (обяз.) — Segmenter-выборка; `times_s = union[...]`.
- **person-mask из `core_object_detections`** (обяз. при `--use-person-mask`, дефолт вкл.) — читает
  detections.npz, берёт person-боксы (class 0), строит маску «где есть человек» + окно ±radius. **Это
  зависимость от детектора** — face_landmarks логически идёт ПОСЛЕ core_object_detections.
- **run identity** + `--batch-size`. Опции: `--use-pose`, `--use-hands`, `--enable/disable-temporal-filter`,
  `--temporal-filter-min-cutoff/beta`, `--person-window-radius`.
- **MediaPipe `<0.10.15`** (deploy-хард-требование).

Работает на том же shared-sampling `frame_indices`, что core_clip/depth/optical_flow/object_detections.

## 5. Выход

NPZ `landmarks.npz`, `allow_extra_keys=false`. Классы ключей:

- **model-facing:** `face_landmarks (N,FACES,468,3)`, `face_present (N,FACES)`, `frame_indices/times_s (N,)`.
  Это seq лицевой геометрии по времени; при отсутствии лица слот = **NaN by design**.
- **analytics / диагностика:** `has_any_face ()`, `face_mesh_ran (N,)`, `person_present (N,)`,
  `empty_reason/face_empty_reason/pose_empty_reason/hands_empty_reason`, опц. `pose_present`,
  `hands_present`, `has_any_pose/hands`.
- **опц. pose/hands:** `pose_landmarks (N,33,4)`, `hands_landmarks (N,HANDS,21,3)` + present-маски (если
  `--use-pose/--use-hands`).
- **debug:** `face_landmarks_raw`/`pose_raw`/`hands_raw` (до OneEuro), `meta`, legacy top-level дубли.

**Инварианты (проверено Audit v4 + мой прогон):** FACES=1, HANDS=2; `face_present=True ⇒ без NaN`,
`face_present=False ⇒ все 468×3 = NaN`; `face_present ⇒ face_mesh_ran`; координаты нормализованы MediaPipe
(x,y ∈[0,1] отн. кадра, z — относительная глубина).

## 6. Фичи (важное/неочевидное)

- **`face_landmarks` 468×3 — несущая фича.** MediaPipe FaceMesh: плотная 3D-сетка лица (контур, глаза,
  брови, губы, нос). x,y нормированы к кадру, z — относительная глубина точки. Основа для геометрии мимики,
  ориентации головы, открытости глаз/рта downstream.
- **NaN by design при отсутствии лица** — не баг, а контракт: downstream (shot_quality) ждёт NaN и
  превращает его в NaN-фичи «лица нет». На 24 реальных видео **0 нарушений** политики (present→finite,
  absent→all-NaN).
- **`face_mesh_ran` vs `face_present`** — тонкая диагностика: mesh мог запуститься (`face_mesh_ran=True`), но
  лицо не найдено (`face_present=False`) → NaN-слот. Audit v4: такие кадры есть (8/48 на A) — ожидаемо при
  гейтинге по person-mask. Это отделяет «не проверяли» от «проверили, лица нет».
- **person-mask гейтинг** — mesh запускается только где детектор видел человека (экономия + меньше ложных
  срабатываний на фоне). По реальным данным `face_mesh_ran` ratio mean 0.184 (проверяется ~18% кадров).
- **OneEuro temporal-фильтр** — сглаживает дрожание лендмарков между кадрами (raw и filtered оба в NPZ);
  адаптивный (min_cutoff + beta·скорость), убирает джиттер без лага на быстрых движениях.
- **`empty_reason` строки** (`skipped_due_to_person_mask_no_person`, `no_faces_in_video`, …) — человекочитаемо
  объясняют, почему пусто; сильная QA-диагностика.

## 7. Алгоритм / архитектура

- **Модель:** **MediaPipe FaceMesh** (Google) — 468-точечная 3D-сетка лица (attention mesh); опц.
  MediaPipe Pose (33 точки, BlazePose) и Hands (21×2). Solutions API (не Tasks) → зависимость от версии.
  Внешняя предобученная модель, не обучается.
- **Гейтинг:** person-mask из детектора → детект-стадия выбирает кадры с человеком (stride + окно radius).
- **Пост:** OneEuro-фильтр по времени (класс `OneEuroFilter`), raw+filtered.
- **Где идёт:** **CPU** (MediaPipe Solutions — CPU-инференс, models_used device=cpu в Audit v4). Не GPU/Triton.
- **Сложность:** линейна по числу проверяемых кадров; стоимость ~16–44 c/видео (FaceMesh+pose+hands).
  person-mask снижает объём mesh-прогонов.

## 8. Оптимизации

- **person-mask гейтинг** — не гонять FaceMesh на кадрах без людей (экономия CPU + чистота); осознанное
  архитектурное решение, переиспользует уже посчитанные person-боксы детектора.
- **Stride + window radius** — на длинных видео детект-стадия семплит реже, mesh — в окне вокруг person-кадров.
- **OneEuro вместо простого сглаживания** — адаптивный фильтр: гладко на статике, отзывчиво на движении
  (осознанный выбор качества трекинга лендмарков).
- **raw + filtered оба сохранены** — downstream выбирает (эмоции могут хотеть raw, композиция — filtered).
- **Богатая empty-диагностика** (`*_empty_reason`, `face_mesh_ran`) — дёшево, но резко упрощает QA/дебаг.
- **`meta_json`-подобные legacy-дубли** + object-safe запись — cross-venv надёжность.
- **Атомарная запись NPZ** + пост-валидация схемы.

## 9. Слабые места

- **FACES=1 — только одно лицо.** Мульти-лицо (интервью, групповой контент) не покрыто; Audit v4 явно
  отметил «сценарий FACES>1 не закрыт». Для контента с несколькими людьми теряется геометрия всех, кроме
  одного. Существенное ограничение для части контента.
- **Зависимость от детектора (person-mask).** Если core_object_detections прогнан на неканоничном/COCO-весе
  или пропустил человека — mesh не запустится (false-empty). Т.е. качество лица наследует качество person-
  детекции (а мы видели COCO-микс в object_detections). На 24 реальных видео **20/24 valid-empty** — сигнал
  лица очень разрежен (частично из-за строгого гейтинга + короткие ролики + мало людей крупным планом).
- **CPU-инференс MediaPipe** — не масштабируется на GPU/Triton как остальное ядро; для 200k — узкое место
  (16–44 c/видео на CPU), нужен пул CPU-воркеров.
- **Хрупкая версионная зависимость** — Solutions API удалён в mediapipe ≥0.10.15; жёстко прибит `<0.10.15`.
  Технический долг: перейти на MediaPipe **Tasks API** (актуальный, не депрекейтнутый).
- **Короткие видео / малый N** — медиана N=12: лицевая динамика по нескольким кадрам бедна.
- **z-координата FaceMesh — относительная**, не метрическая (как и depth) — межвидовое сравнение только по
  форме/нормированным величинам.
- Отдельного `LOGIC_ERRORS_FOR_CLAUDE.md` L-номера нет.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс.] Перейти на MediaPipe Tasks API** (FaceLandmarker) — убрать хрупкую зависимость `<0.10.15`,
   получить поддерживаемый рантайм, blendshapes (готовые коэффициенты мимики — прямой сигнал для эмоций).
2. **[выс.] Поддержать FACES>1** (мульти-лицо) — расширить FACES-ось, привязать лица к person-трекам
   детектора; критично для интервью/групп. Требует версионирования схемы.
3. **[сред.] Не наследовать слепо person-mask** — опциональный fallback-детект лица без person-гейтинга на
   кадрах, где детектор мог пропустить человека (снизить false-empty), с флагом источника.
4. **[сред.] GPU-путь для MediaPipe** (или замена на GPU-face-mesh) — снять CPU-bottleneck для 200k.
5. **[низ.] Экспортировать производные геометрии** (yaw/pitch/roll головы, EAR/MAR — открытость глаз/рта)
   прямо здесь, чтобы downstream не пересчитывал из 468 точек каждый.

## 11. Рекомендации по архитектуре / связям

- **Закрепить порядок Tier:** core_object_detections → core_face_landmarks (person-mask) → face/emotion-ветка.
  Face зависит от детектора — задокументировать явно.
- **Единый источник лица** для shot_quality/emotion_face/micro_emotion/detalize_face/behavioral — все читают
  этот артефакт, не запускают FaceMesh заново. Уже так — закрепить в контракте.
- **Blendshapes из Tasks API шэрить** как готовый сигнал мимики → упростит emotion_face/micro_emotion.
- **Связать face-слоты с track_ids детектора** (когда появятся) — чтобы мульти-лицо и эмоции привязывались
  к конкретному человеку во времени.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что реально говорит |
|---|---|---|---|
| Output-валидатор (schema v2) | 4 видео | VALID, структура 468×3 ок | контракт соблюдён |
| Valid-empty путь (видео без лиц) | валидация | has_any_face=False, все формы верны | ключевая фича для shot_quality работает |
| Face-present путь | видео с людьми | 245/245 кадров с лицами, 468 детектятся | оба пути (лицо есть/нет) корректны |
| Audit v4 L2 (A+B) | 5 run, N_total=543 | ✓ ~8.8/10 | FACES=1, HANDS=2; NaN-инварианты 0 нарушений |
| — NaN-политика | Audit v4 | absent→all-NaN, present→no-NaN (violations=0) | контракт слотов строг |
| — mediapipe | Audit v4 | 0.10.14, device=cpu | версия/устройство зафиксированы |
| **Реальные артефакты storage (мой прогон)** | **24 видео, 567 кадр** | **0 NaN-нарушений, 0 Inf, FACES=1** | контракт безупречен на проде |
| — has_any_face | 24 видео | **4 с лицом / 20 valid-empty** | сигнал лица разрежен; empty-путь доминирует |
| — face_present ratio | 24 видео | mean 0.049, max 0.615 | лица редки на этом корпусе |
| — face_mesh_ran ratio | 24 видео | mean 0.184 | person-гейтинг работает |

Вывод: **контракт и NaN-политика — образцовы** (0 нарушений на 24 реальных видео, оба пути корректны). Не
хватает мульти-лица (FACES>1) и подтверждён факт разреженности лицевого сигнала на реальном корпусе — что
делает надёжность valid-empty критически важной (и она держится).

## 13. Интерпретируемость

**Есть:** dev-рендер (`utils/render.py`) — 468 точек поверх лица; `*_empty_reason` человекочитаемо;
`face_present`/`face_mesh_ran` — понятная диагностика присутствия.

**Добавить (для обычного пользователя):**
- **Лицо с наложенной сеткой** (K превью-кадров) — «модель видит ваше лицо так».
- **Timeline присутствия лица** (`face_present` по времени) — «когда вы в кадре крупным планом».
- **Словесная сводка:** «лицо в кадре 35% времени, в основном крупный план» / «лиц не обнаружено».
- **Производные простыми словами** (после экспорта): «часто улыбаетесь / смотрите в камеру / отводите взгляд».
- Приложенная визуализация (`core_face_landmarks_distributions.png`) — доля видео с лицом и распределения.

## 14. Польза для моделей

`face_landmarks (N,FACES,468,3)` + `face_present` — model-facing seq. Для Encoder это **геометрия лица во
времени** — присутствие человека, ориентация/крупность лица, динамика мимики. Лица и эмоции правдоподобно
**сильно** влияют на вовлечённость в человеко-центричном контенте. Но: (а) 468×3 на кадр — тяжёлый и
избыточный тензор для трансформера (нужен пулинг/производные, а не сырые точки); (б) NaN на кадрах без лица
требует аккуратной обработки маской; (в) на не-человеческом контенте фича пуста (20/24 видео). Практически
для модели полезнее **производные** (присутствие, поза головы, открытость глаз/рта) + агрегаты эмоций из
downstream, чем сырые 468 точек. Гипотеза: сильный сигнал на человеческом контенте, нулевой на остальном.

## 15. Польза для аналитиков

- **Присутствие лица во времени** (`face_present`, `has_any_face`) → «сколько вы в кадре» — понятно и ценно.
- **Face-ROI и крупность** → качество портретной съёмки (через shot_quality).
- **Геометрия/мимика** → эмоции (через emotion_face/micro_emotion) — понятный инсайт «эмоциональность контента».
- **Поза/руки** (опц.) → жестикуляция, энергичность подачи.
- Оговорка: сырые 468 точек аналитику не нужны — ценны производные и downstream-агрегаты; на видео без людей
  выход пуст (это корректно, но надо честно показывать «лиц нет»).

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 5 | Фундамент всей face/emotion-ветки (8+ модулей) |
| 5. Выход (контракт) | 5 | Богатая диагностика, строгая NaN-политика, valid-empty by design |
| 6. Фичи | 4 | 468×3 мощны, но сырые тяжелы; отличная диагностика присутствия |
| 8. Оптимизации | 4 | person-гейтинг, OneEuro, raw+filtered; CPU-only, версия хрупкая |
| 9. Слабые места (инверсно) | 3 | FACES=1, CPU-bottleneck, зависимость от детектора, `<0.10.15` |
| 12. Результаты тестов | 4 | 0 NaN-нарушений на 24 видео, оба пути; FACES>1 не закрыт |
| 13. Интерпретируемость | 3 | Рендер сетки+диагностика есть, словесная/overlay-подача в TODO |
| 14. Польза для моделей | 4 | Сильный сигнал на людях, но сырьё тяжело + пусто на не-человеч. контенте |
| 15. Польза для аналитиков | 4 | Присутствие/эмоции очень понятны; сырые точки не для аналитика |

### Итоговые оценки

- **Польза для моделей: 4/5.** Даёт Encoder'у уникальный слой лицевой геометрии/присутствия — сильный
  предиктор вовлечённости на человеко-центричном контенте. Снижают оценку тяжесть сырого тензора 468×3
  (нужны производные), NaN-маскирование и полная пустота на не-человеческом контенте (20/24 видео здесь).
- **Польза для аналитиков: 4/5.** Присутствие лица и (через downstream) эмоции — один из самых понятных
  человеку выходов. Ограничивают FACES=1 (групповой контент), необходимость производных вместо сырых точек
  и пока отсутствующая словесная подача.

## 17. Источники

- `DataProcessor/VisualProcessor/core/model_process/core_face_landmarks/main.py`
- `.../core_face_landmarks/README.md`, `.../docs/SCHEMA.md`, `.../docs/FEATURE_DESCRIPTION.md`
- `.../core_face_landmarks/utils/{validate_core_face_landmarks_npz.py, render.py}`
- `DataProcessor/docs/component_reports/core_face_landmarks/REPORT_2026-07-05.md`
- `DataProcessor/docs/audit_v4/components/visual_processor/core/core_face_landmarks_audit_v4.md`
- Downstream (grep face_landmarks/face_present): `modules/{shot_quality, emotion_face, micro_emotion,
  detalize_face, behavioral, frames_composition, high_level_semantic, story_structure}`
- Upstream-зависимость: `modules`/`core_object_detections` (person-mask, class 0)
- `automation/runner/AGENT_CONTEXT.md` (разделы 6/7: тайминги 16–44 c, mediapipe<0.10.15, valid-empty фича)
- Реальные артефакты: 24× `storage/result_store/youtube/*/*/core_face_landmarks/landmarks.npz` (567 кадр)

## 18. Визуализации

![Распределения core_face_landmarks](core_face_landmarks_distributions.png)

`core_face_landmarks_distributions.png` (построено на 24 реальных артефактах, 567 кадр): доля видео с лицом
(4) против valid-empty (20 — empty-путь доминирует на этом корпусе), распределение `face_present ratio`
(mean 0.049 — лица редки), `face_mesh_ran ratio` (person-гейтинг ~18%) и N кадров. Подтверждает: NaN-политика
безупречна (0 нарушений, 0 Inf), а надёжность valid-empty-пути критична, т.к. большинство видео здесь — без лиц.
