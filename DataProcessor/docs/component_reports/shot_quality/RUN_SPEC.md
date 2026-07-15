# RUN_SPEC: shot_quality (v1, стартовый — static-review Claude)

Протокол — `DataProcessor/docs/COMPONENT_VALIDATION_PROTOCOL.md`. По коду до прогона; уточнить после прогона.

## 1. Static-review (по коду)
- **Назначение:** техническое качество видео — **frame-level** (`frame_features (N,F)`: sharpness
  tenengrad/laplacian-var, exposure, blur и т.д.), **shot-level** агрегации по шотам (`cut_detection`),
  **CLIP-quality** (zero-shot вероятности по фикс. промптам из core_clip). NPZ `shot_quality.npz`,
  schema `shot_quality_npz_v3` (+ JSON-схема `VisualProcessor/schemas/shot_quality_npz_v3.json`), producer 2.0.2.
- **Цепочка зависимостей (hard, no-fallback, aligned frame_indices):**
  `Segmenter → core_clip + core_depth_midas + core_object_detections + core_face_landmarks + cut_detection → shot_quality`.
  `core_face_landmarks`: валидная пустота ок → face-ROI фичи = `NaN` (не error).
- **Семантика статусов:** `empty` НЕ используется; отсутствие лиц не блокирует non-face метрики;
  любая отсутствующая зависимость / рассинхрон `frame_indices` / отсутствующие ключи ⇒ **error (raise)**.
- **Ось:** `frame_indices` строго от Segmenter; `times_s = union_timestamps_sec[frame_indices]`.

### 🔴 Отличие / сложность
Самая тяжёлая цепочка из трёх: нужны **5 upstream** (core_clip, depth_midas, object_detections,
face_landmarks, cut_detection) — каждый со своей моделью. Профиль прогона должен поднять их все.
Это дороже по времени/VRAM — важно для ledger и решения по 200k.

## 2. Что валидировать (4 оси)
- **Корректность:** `frame_features` конечны (кроме преднамеренных face-NaN), диапазоны sharpness/exposure
  осмысленны; shot-агрегации соответствуют границам `cut_detection`; CLIP-quality вероятности в [0,1];
  `frame_indices` sorted/unique/выровнены со всеми deps; schema/JSON-контракт.
- **Стабильность:** golden-повтор (детерминизм; face-NaN политика стабильна).
- **Различимость:** health/nan/const по каждой фиче; NaN-политика (face-ROI) — by design, не «сломано».
- **Модель-fit:** frame_features как seq-качество для Encoder; shot-агрегации + CLIP-quality для аналитики.
  Уточнить, какие фичи реально полезны модели vs аналитике (ledger).

## 3. Матрица видео
Разное качество: резкое/размытое, пере/недоэкспонированное, дрожащая камера, чистая студия; со сменами
шотов (проверка shot-агрегаций) + ролики с лицами и без (проверка face-NaN пути).

## 4. Что собрать / измерить (ledger)
Тайминги ВСЕХ upstream + shot_quality + пик VRAM (5 моделей!); стоимость на 200k (эта цепочка —
кандидат в самые дорогие). NPZ shapes, health по фичам, доля face-NaN, golden, оба валидатора.

## 5. Порядок (следующая сессия)
1. Раннер, поднимающий полную цепочку (5 deps + shot_quality) или профиль visual.
2. Прогон на поде → NPZ → REPORT (4 оси, особое внимание NaN-политике и стоимости цепочки) → правки →
   валидаторы вход/выход + метрики → ledger/checklist.
> Замечание по стоимости: если 5-deps цепочка слишком дорога для 200k — рассмотреть переиспользование
> уже посчитанных upstream-артефактов (они и так нужны другим компонентам), а не пересчёт под shot_quality.
