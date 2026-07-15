# RUN_SPEC: scene_classification (v1, стартовый — static-review Claude)

Протокол — `DataProcessor/docs/COMPONENT_VALIDATION_PROTOCOL.md`. Заполнен по коду до прогона;
следующая сессия уточняет после реального прогона на поде.

## 1. Static-review (по коду)
- **Назначение:** классификация сцены (Places365, 365 классов) на кадр + advanced/семантические
  фичи. NPZ `scene_classification_features.npz`, schema `scene_classification_npz_v2`, producer 2.0.1.
- **Модель:** Places365 **ResNet50**, runtime `inprocess | triton` (spec `places365_resnet50_224_triton`).
  L1-бэклог: Triton Places365 требует **batch=1** (иначе HTTP 400) — по умолчанию `--batch-size 1`.
- **Метки:** `label_fusion = places | clip` (clip — zero-shot по core_clip над теми же 365 метками); дефолт `places`.
- **Семантика:** строго из **core_clip** эмбеддингов + core_clip-provided prompt-эмбеддингов
  (`places365_text_embeddings`, `scene_*_text_embeddings`). Локального CLIP нет.
- **Цепочка зависимостей (hard, no-fallback):**
  `Segmenter → core_clip (embeddings.npz + text_embeddings) → cut_detection (shot_boundaries) → scene_classification`.
  Ограничение: `scene_classification.frame_indices ⊆ core_clip.frame_indices` (иначе fail-fast).

### 🔴 Отличие от action_recognition
Цепочка тяжелее: нужны **core_clip** (CLIP-эмбеддинги + text-эмбеддинги Places365/scene) и
**cut_detection**. Профиль прогона должен включать эти компоненты (и их модели: CLIP через Triton/inprocess).
Places365-веса: `provision_base_models.py --only places365_resnet50` (manual) ИЛИ Triton ONNX.

## 2. Что валидировать (4 оси)
- **Корректность:** scene-вероятности (N,365) в [0,1] сумма≈1; топ-класс правдоподобен (сверка с роликом);
  `frame_indices ⊆ core_clip`, `times_s = union[frame_indices]`; schema-контракт.
- **Стабильность:** golden-повтор (детерминизм inprocess resnet50 / Triton).
- **Различимость:** распределение сцен не вырождено; advanced-фичи (ontology/atmosphere) — health/nan/const.
- **Модель-fit:** что идёт в Encoder — вероятности 365 (мягкая метка сцены) и/или эмбеддинг? Уточнить
  назначение (seq по кадрам ⊆ union → сцена как временной сигнал). Классы Places для аналитики.

## 3. Матрица видео
Разные сцены: улица/интерьер/природа/спорт/студия + быстрая смена сцен (проверка cut_detection-агрегации).
Реальные фикстуры из `fixtures/` (люди) + добавить пары с явной сменой локаций.

## 4. Что собрать / измерить (ledger)
Тайминги стадий (core_clip, cut_detection, scene) + пик VRAM (ResNet50/Triton); стоимость на 200k
(inprocess vs triton batch=1 → пропускная способность). NPZ shapes, health, golden, оба валидатора.

## 5. Порядок (следующая сессия)
1. Раннер `run_component_local.py` (Segmenter+core_clip+cut_detection+scene_classification) или профиль.
2. Прогон на поде (GPU) → NPZ → REPORT (4 оси) → правки логики при необходимости → валидаторы вход/выход
   + метрики → ledger/checklist. Модель Places365 provision (manual/Triton) — заранее.
