# Фаза 5 — Прогноз расширения VisualProcessor до всех компонентов

В 300-видео прогоне участвовали **7 из 29** VP-компонентов. Этот документ — прогноз по добавлению
оставшихся (исключая заблокированные владельцем/багом). Основа: аудит кода (`import`/`--runtime`/Docker)
+ портфолио-оценка + тайминги реального прогона. Смоук-числа появятся после теста расширенного набора.

## Исключены (Фаза 0 — не трогать)
- `core_object_detections` — идёт retrain YOLO 41→34 (владелец §8).
- `ocr_extractor`, `text_scoring` — 100% пустые из-за detector-бага (каскад от object_detections), чинить рано.
- `brand_semantics`, `car_semantics`, `franchise_recognition`, `place_semantics`, `face_identity` — stub-БД,
  деприоритезированы (нет данных). Код готов, но оценивать нечего.

## Аудит инфраструктуры — 15 кандидатов

| Компонент | Runtime / зависимость | Инфра готова? | Прогноз времени | Блокеры/примечания |
|---|---|---|---|---|
| `core_face_landmarks` | inprocess: **mediapipe** (CPU) + torch | ⚠️ mediapipe нужен в venv (`<0.10.15`, портфолио); torch — только телеметрия | средне (CPU, покадрово) | mediapipe уже стоял в валидации; проверить на свежем venv |
| `content_domain` | читает CLIP-эмбеддинги core_clip (Triton) | ✅ (core_clip уже в бандле) | **дёшево** (numpy + порог) | зависит от core_clip в том же прогоне; порог 0.23 не откалиброван |
| `shot_quality` | inprocess torch (эвристики) | ⚠️ torch на CPU (драйвер) | средне | эвристики «на глаз», не откалибровано |
| `story_structure` | **CLIP** + torch | ✅ через Triton core_clip | дёшево-средне | hook_ratio баг был (портфолио §3.1), проверить |
| `emotion_face` | torch **EmoNet** | ❌ EmoNet source+веса нужны вручную (портфолио §1: не на volume, копировать 8KB source + .pth) | средне (GPU-модель на CPU медленно) | добавить EmoNet в setup |
| `detalize_face` | геометрия (нет тяжёлых импортов) | ✅ | **дёшево** (CPU) | потребляет core_face_landmarks |
| `behavioral` | геометрия/поза (нет тяжёлых) | ✅ | **дёшево** (CPU) | потребляет landmarks; body_lean баг был |
| `action_recognition` | **SlowFast** + torch (GPU-heavy) | ❌ SlowFast веса + torch-GPU (драйвер!) | **дорого** (пик VRAM ~1.4-1.6ГБ, портфолио §6; на CPU — очень медленно) | нужен рабочий torch-GPU (см. драйвер-проблема) или Triton-экспорт SlowFast |
| `color_light` | cv2/numpy (нет тяжёлых) | ✅ | **дёшево** (CPU) | 7 мёртвых temporal-фич (портфолио §3.1), пересчитать |
| `frames_composition` | inprocess torch | ⚠️ torch CPU | средне | style_dominant_id баг был (портфолио §3.1) |
| `similarity_metrics` | torch + **референс-корпус** | ⚠️ torch CPU + нужен наполненный референс | средне | центральная фича, 62% NaN без базы сравнения (владелец §8: реализовать все оси) |
| `high_level_semantic` | inprocess torch | ⚠️ torch CPU | средне | |
| `micro_emotion` | **OpenFace через Docker** | ❌❌ **Docker не работает на RunPod** (overlay/userns, то же что с Triton) | — | **БЛОКЕР**: либо OpenFace-бандл через skopeo (как Triton), либо CPU-only без Docker, либо отложить. Данные и так были «мусор» (портфолио §4) |
| `optical_flow` (модуль) | consumer core_optical_flow (Triton) | ✅ (flow уже в бандле) | дёшево | модуль-потребитель, не инференс |

## Сводка готовности

- **✅ Готовы сразу (дёшево, инфра есть):** `content_domain`, `detalize_face`, `behavioral`, `color_light`,
  `optical_flow`, `story_structure` (через Triton CLIP). ~6 компонентов, малый доп. оверхед.
- **⚠️ Работают, но torch-inprocess на CPU (медленно, как scene_classification):** `core_face_landmarks`,
  `shot_quality`, `frames_composition`, `similarity_metrics`, `high_level_semantic`. Выиграют от общего
  фикса torch-GPU (см. драйвер-проблема ниже) или Triton-экспорта.
- **❌ Требуют доп. setup/решения:** `emotion_face` (EmoNet вручную), `action_recognition` (SlowFast + torch-GPU),
  `micro_emotion` (**Docker-блокер на RunPod**).

## Сквозной блокер — torch-GPU на RunPod
venv torch собран под CUDA 13 (cu130), драйвер пода — CUDA 12.4 → **inprocess-torch падает на GPU, идёт на CPU**
(это уже видно на scene_classification: 101с CPU). Пока не решено, ВСЕ torch-inprocess компоненты будут на CPU
(медленно). Решения: (а) поставить torch cu121 (совместим с драйвером 550) в venv/side-venv; (б) экспортировать
модели в Triton ONNX (как CLIP/MiDaS/RAFT). Это ключевое решение для дешёвого добавления torch-GPU компонентов.
Валидация в Фазе 2 (scene GPU) даст ответ.

## Прогноз диска (доп. МБ/видео)
Базовый набор 7 компонентов = ~151 МБ/видео (8 NPZ). Доп. компоненты по типу выхода:
- Дёшево (скаляры/малые векторы): content_domain, color_light, behavioral, detalize_face, shot_quality,
  frames_composition, high_level_semantic, story_structure, optical_flow — **~+2-10 МБ/видео каждый**.
- Эмбеддинги/тяжёлые: emotion_face (per-face), action_recognition (temporal), similarity_metrics, micro_emotion,
  core_face_landmarks (per-frame landmarks) — **~+15-50 МБ/видео каждый**.
- Грубый прогноз полного VP (29 комп.): ~300-450 МБ/видео → 300 видео ≈ 90-135 ГБ (близко к лимиту тома 120ГБ
  → **нужна выгрузка в HF по ходу**, см. disk-прогноз/Фаза 7).

## Рекомендация Фазы 5
1. Сначала добавить **6 «дешёвых готовых»** (content_domain/detalize_face/behavioral/color_light/optical_flow/
   story_structure) — почти бесплатно по времени, обобщив `run_corpus.sh` (список компонентов конфигом).
2. Решить **torch-GPU** (Фаза 2 scene-эксперимент) → разблокирует 5 «медленных на CPU».
3. `emotion_face`/`action_recognition` — после решения torch-GPU + доставки весов (EmoNet/SlowFast) в setup.
4. `micro_emotion` — отдельное решение по Docker/OpenFace (или отложить; данные и так были плохие).

*Смоук-тест расширенного набора (5-10 видео) даст реальные числа времени/памяти/диска — до него оценки выше
помечены как прогноз.*
