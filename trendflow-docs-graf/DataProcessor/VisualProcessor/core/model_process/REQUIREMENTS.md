# Зависимости между Core Providers

**Источник правды**: `VisualProcessor/main.py:CORE_DEPS` и README каждого компонента  
**Последнее обновление**: 2025-01-13

---

## Дерево зависимостей

```
┌─────────────────────────────────────────────────────────────┐
│                    Независимые Core Providers               │
│  (Tier-0 baseline, могут выполняться параллельно)           │
└─────────────────────────────────────────────────────────────┘

  core_clip
  ├─ Назначение: Универсальные CLIP embeddings (image + text)
  ├─ Выход: embeddings.npz
  └─ Зависимости: нет

  core_object_detections
  ├─ Назначение: Детекция объектов (YOLO) + трекинг (ByteTrack)
  ├─ Выход: detections.npz
  └─ Зависимости: нет

  core_optical_flow
  ├─ Назначение: Оптический поток (RAFT)
  ├─ Выход: flow.npz
  └─ Зависимости: нет

  core_depth_midas
  ├─ Назначение: Оценка глубины (MiDaS)
  ├─ Выход: depth.npz
  └─ Зависимости: нет

  core_face_landmarks
  ├─ Назначение: Landmarks лица (MediaPipe FaceMesh)
  ├─ Выход: landmarks.npz
  └─ Зависимости: нет


┌─────────────────────────────────────────────────────────────┐
│              Semantic Heads (Tier-1, зависят от базовых)     │
└─────────────────────────────────────────────────────────────┘

  core_place_semantics
  ├─ Назначение: Retrieval мест/лэндмарков по CLIP embeddings
  ├─ Выход: place_semantics.npz
  └─ Зависимости:
      ├─ core_object_detections (для frame_indices)
      └─ core_clip (для frame embeddings)

  core_car_semantics
  ├─ Назначение: Семантика автомобилей (make/model/segment)
  ├─ Выход: car_semantics.npz
  ├─ Использует: CLIP через Triton (не требует core_clip как зависимость)
  └─ Зависимости:
      └─ core_object_detections (для bbox proposals и tracks)

  core_brand_semantics
  ├─ Назначение: Семантика брендов/логотипов (CLIP-matching)
  ├─ Выход: brand_semantics.npz
  ├─ Использует: CLIP через Triton (не требует core_clip как зависимость)
  └─ Зависимости:
      └─ core_object_detections (для bbox proposals и tracks)

  core_face_identity
  ├─ Назначение: Идентификация известных людей (celebrity retrieval)
  ├─ Выход: face_identity.npz
  └─ Зависимости:
      ├─ core_object_detections (для frame_indices)
      └─ core_face_landmarks (для face bbox из landmarks)
```

---

## Детальное описание зависимостей

### Tier-0: Независимые Core Providers

Эти компоненты не зависят от других core providers и могут выполняться параллельно.

#### `core_clip`
- **Зависимости**: нет
- **Назначение**: Универсальные CLIP embeddings для изображений и текста
- **Используется**: `core_place_semantics` (требует как зависимость)

#### `core_object_detections`
- **Зависимости**: нет
- **Назначение**: Детекция объектов (YOLO) + трекинг (ByteTrack)
- **Используется**: 
  - `core_place_semantics` (frame_indices)
  - `core_car_semantics` (bbox proposals, tracks)
  - `core_brand_semantics` (bbox proposals, tracks)
  - `core_face_identity` (frame_indices)

#### `core_optical_flow`
- **Зависимости**: нет
- **Назначение**: Оптический поток (RAFT)
- **Используется**: модулями (например, `story_structure`)

#### `core_depth_midas`
- **Зависимости**: нет
- **Назначение**: Оценка глубины (MiDaS)
- **Используется**: модулями (например, `shot_quality`)

#### `core_face_landmarks`
- **Зависимости**: нет
- **Назначение**: Landmarks лица (MediaPipe FaceMesh)
- **Используется**: 
  - `core_face_identity` (требует как зависимость)
  - модулями (например, `shot_quality`, `story_structure`)

---

### Tier-1: Semantic Heads

Эти компоненты зависят от базовых core providers и выполняются после них.

#### `core_place_semantics`
- **Зависимости**: 
  - `core_object_detections` (обязательно)
  - `core_clip` (обязательно)
- **Назначение**: Retrieval мест/лэндмарков по CLIP frame embeddings
- **Причина зависимости от `core_clip`**: Использует frame embeddings из `core_clip/embeddings.npz` для cosine similarity с gallery embeddings
- **Причина зависимости от `core_object_detections`**: Использует `frame_indices` для выравнивания по shared sampling group

#### `core_car_semantics`
- **Зависимости**: 
  - `core_object_detections` (обязательно)
- **Назначение**: Семантика автомобилей (make/model/segment/body_type/price_bucket)
- **Причина зависимости**: Использует bbox proposals и tracks из `core_object_detections` для извлечения car crops
- **Примечание**: Использует CLIP через Triton напрямую (не требует `core_clip` как зависимость, т.к. работает на crops, а не на frame embeddings)

#### `core_brand_semantics`
- **Зависимости**: 
  - `core_object_detections` (обязательно)
- **Назначение**: Семантика брендов/логотипов через CLIP-matching по bbox crops
- **Причина зависимости**: Использует bbox proposals (особенно `logo_region`) и tracks из `core_object_detections`
- **Примечание**: Использует CLIP через Triton напрямую (не требует `core_clip` как зависимость, т.к. работает на crops, а не на frame embeddings)

#### `core_face_identity`
- **Зависимости**: 
  - `core_object_detections` (обязательно)
  - `core_face_landmarks` (обязательно)
- **Назначение**: Идентификация известных людей (celebrity retrieval) по face embeddings
- **Причина зависимости от `core_face_landmarks`**: Использует face landmarks для извлечения face bbox и crops
- **Причина зависимости от `core_object_detections`**: Использует `frame_indices` для выравнивания по shared sampling group

---

## Важные замечания

### CLIP через Triton vs `core_clip`

**Различие**:
- `core_clip` — это core provider, который вычисляет **frame-level embeddings** и сохраняет их в NPZ
- CLIP через Triton — это прямой вызов модели CLIP для **crop-level embeddings** (bbox crops)

**Семантические heads используют CLIP напрямую**:
- `core_car_semantics`: CLIP для car crop embeddings (не требует `core_clip`)
- `core_brand_semantics`: CLIP для logo/bbox crop embeddings (не требует `core_clip`)

**Только `core_place_semantics` требует `core_clip`**:
- Использует **frame embeddings** из `core_clip/embeddings.npz`
- Не вычисляет embeddings самостоятельно

### Shared Sampling Group

Все semantic heads, зависящие от `core_object_detections`, используют **тот же `frame_indices`**:
- `core_object_detections.frame_indices` — источник правды
- Все downstream компоненты выравниваются по этим индексам
- Это обеспечивает консистентность данных для downstream модулей

### Порядок выполнения (Topological Sort)

Оркестратор автоматически определяет порядок выполнения через topological sort:
1. **Tier-0** (независимые): `core_clip`, `core_object_detections`, `core_optical_flow`, `core_depth_midas`, `core_face_landmarks`
2. **Tier-1** (semantic heads): `core_place_semantics`, `core_car_semantics`, `core_brand_semantics`, `core_face_identity`

---

## Ссылки

- **Код зависимостей**: `VisualProcessor/main.py:CORE_DEPS` (строка 468)
- **README компонентов**: `VisualProcessor/core/model_process/<component>/README.md`
- **Taxonomy**: `VisualProcessor/core/model_process/core_object_detections/TAXONOMY_V1.yaml`
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
