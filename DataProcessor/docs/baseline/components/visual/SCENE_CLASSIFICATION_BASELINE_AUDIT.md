# ✅ Baseline Audit — `scene_classification`

Компонент: `DataProcessor/VisualProcessor/modules/scene_classification/`  
Тип: Visual module (Places365 scene segmentation + semantics from `core_clip`)  
Статус: **✅ CLOSED (baseline)**  

---

## Резюме

`scene_classification` выполняет:
- классификацию сцен (Places365, 365 классов) на выбранных кадрах
- агрегацию по “сценам” на основе **hard shot boundaries** из `cut_detection`
- добавляет семантические агрегаты из `core_clip` (aesthetic/luxury/atmosphere)

Baseline‑политика:
- **runtime=triton** (GPU-only path)
- **батчинг — knob scheduler’а** (`--batch-size`), модуль сам batch size не выбирает
- sampling строго Segmenter‑owned, а `scene_classification ⊆ core_clip` гарантируется через deps alignment
- **эвристики запрещены**: удалены keyword‑онтологии (indoor/outdoor, nature/urban) и `label_fusion=fused`
- **error-only semantics**: partial results запрещены

---

## Оценки (1–10)

- **Качество кода и алгоритмов**: **7/10**
- **Логика алгоритмов**: **7/10**
- **Логика глобального взаимодействия**: **8/10**
- **Оптимизации (параллелизм/батчинг)**: **7/10** *(после фикса батчинга в Triton‑path)*

---

## Соответствие `BASELINE_COMPONENT_AUDIT_CRITERIA.md`

### 1) Контракты входа/выхода

- `frame_indices` — строго из `frames_dir/metadata.json` (Segmenter).  
- `times_s` — строго из `union_timestamps_sec[frame_indices]` (no‑fallback).  
- `core_clip` — **hard dependency** (no‑fallback): без `core_clip/embeddings.npz` модуль падает.
- `cut_detection` — **hard dependency** (precision policy): без shot boundaries модуль падает.

### 2) Per-run storage + atomic save + validation

- Артефакт: `rs_path/scene_classification/scene_classification_features.npz` (**фиксированное имя**).
- Сохранение: atomic (`mkstemp(..., suffix=".npz")` → `np.savez_compressed` → `os.replace`).
- Валидация: `validate_npz()` **fail-fast**, при провале файл удаляется.

### 3) Batching / scheduler contract (Triton)

- `runtime=triton` используется для baseline GPU.
- Батчинг контролируется извне: модуль принимает `--batch-size` и отправляет кадры батчами в Triton.
- Triton модели batch-enabled: `max_batch_size > 0` и ONNX экспорт с dynamic batch.

---

## Модели

### Places365 (Triton)

Ветки (обязательные):
- `places365_resnet50_224`
- `places365_resnet50_336`
- `places365_resnet50_448`

Spec’и:
- `dp_models/spec_catalog/vision/places365_resnet50_{224,336,448}_triton.yaml`

Triton repo (minimal, для тестов/локального запуска):
- `DataProcessor/triton/models_places365/`
  - `preprocess_places365_*` (python backend, batch-enabled)
  - `places365_resnet50_*_onnx` (onnxruntime)
  - `places365_resnet50_*` (ensemble, внешний контракт `UINT8 NHWC`)

---

## Производительность / resource costs

Источник:
- unit-cost (B=1): `docs/models_docs/resource_costs/scene_classification_costs_v1.json`  
  Evidence: `storage/reports/out/checklist-places365-b1/`
- throughput (B=8): `docs/models_docs/resource_costs/scene_classification_costs_b8_v1.json`  
  Evidence: `storage/reports/out/checklist-places365-b8/`

Единица обработки: `frame`

*(VRAM фиксируется по процессу `tritonserver` как `vram_triton_*`.)*

---

## Проверка качества (human-friendly)

Скрипт:
- `scripts/baseline/demo_scene_classification_quality.py`

Evidence (последний прогон, `label_fusion=clip`):
- `storage/reports/out_fused/demo_scene_classification_quality_NSumhkOwSg_20260116-025726-502152.html`
- `storage/reports/out_fused/demo_scene_classification_quality_-3s8SdV4bsU_20260116-025747-119093.html`

Что показывает:
- список сцен (scene_id, label, длительность)
- ключевые агрегаты Places365 + CLIP semantics
- thumbnails (first/mid/last) для каждой сцены
- sanity checks через `validate_npz()`

---

## Замечания / рекомендации по функционалу (важно для “популярности”)

### A) Почему Places365 может “плохо классифицировать”

Типичные причины:
- domain shift (видео-кадры, motion blur, low-light, экраны, мемы) vs фото‑датасет
- Places365 классы — **сцены**, а не **места/достопримечательности**
- top‑1 label по кадру шумный без temporal smoothing / top‑k агрегации

Практичные улучшения (без смены архитектуры):
- агрегация по **top‑k** + “mass” по классам вместо жесткого top‑1
- включить/усилить temporal smoothing (в модуле есть флаги; baseline использует smoothing + weighted smoothing)
- выбирать ветку 336/448 на “сложных” видео (качество ↑, latency ↑)

Дополнительно реализовано в baseline:
- **label_fusion=clip**: CLIP zero‑shot по тем же 365 лейблам (через `core_clip.places365_text_embeddings`)
- Исправлен парсинг категорий Places365 (сохраняются подкатегории типа `apartment_building/outdoor`)
- Для коротких видео Segmenter использует более плотный primary sampling (≈0.25s gap) для лучшего покрытия
- `core_clip` умеет чанковать `clip_text` запросы (Triton `max_batch_size=64`)

### B) “Эрмитаж / популярные места”

Это **не задача Places365** (по сути это landmark recognition / geo‑localization).

Рекомендация по архитектуре baseline:
- оставить `scene_classification` как “scene segmentation + coarse scene labels”
- добавить отдельный модуль (рекомендуемое имя): **`place_recognition`**
  - вход: `core_clip` embeddings + кадры
  - метод: retrieval по базе эталонных мест (CLIP image embeddings) +/или специальная geo модель

Что можно сделать быстро (v1, без тяжёлой разметки):
- собрать “галерею мест” (из Википедии/официальных фото) → посчитать CLIP embeddings оффлайн
- на инференсе: cosine similarity кадра → top‑K мест + confidence + coverage по времени

Что лучше для качества (v2):
- модель geo‑localization (GeoCLIP / PlaNet / DELG / NetVLAD‑подобные) + обучение на landmarks датасете

### C) “Мультик / видеоигра / аниме / какая игра”

Это тоже лучше как отдельный модуль над `core_clip`:
- `content_domain` (real vs animation vs game vs screen-recording)
- `game_title` (если нужно “какая игра”) — почти всегда требует **OCR+текст** (интерфейс/меню) + CLIP

Почему отдельным модулем:
- `scene_classification` отвечает за “scene segmentation”, а домены/титулы — это другой label space и другие метрики качества
- проще эволюционировать и обучать отдельно (и не ломать downstream, который ждёт scenes)

---

## Итог

Компонент соответствует baseline‑критериям по контрактам, сохранению/валидации, batching‑контракту и перф‑измерениям.  
Крупные “функциональные” улучшения (landmarks, games/anime) предлагаются как отдельные модули поверх `core_clip` для качества и чистоты архитектуры.

Зафиксированная оговорка:
- На части данных остаётся bias к некоторым indoor‑классам (например, `beauty_salon`). Дальнейший шаг: domain routing + отдельный domain head поверх `core_clip`.


