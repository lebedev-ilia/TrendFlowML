# Дизайн: action_recognition schema v3 + оконный сэмплинг Segmenter

Статус: **спека к реализации** (2026-07-03). Автор: Claude → исполнитель: владелец/Cursor.
Основание: REPORT 2026-07-02 (R1/R3) + решения владельца (нужны **эмбеддинг И классы**; трекинг —
свой appearance-трекер, см. `EMBEDDING_TRACKER.md`; schema v3 согласована).

Две независимые правки: **(A)** плоский per-clip выход с классами; **(B)** плотные окна кадров.

---

## A. Schema `action_recognition_npz_v3` — плоский per-clip stream

Проблема v2: seq выдавался **per-track внутри `results_json` (object-dict)** — Encoder'у пришлось бы
парсить T словарей; классов действий не было вовсе. v3 — единый time-ordered поток клипов.

### Поля NPZ (топ-уровневые массивы)
| key | tier | dtype | shape | описание |
|---|---|---|---|---|
| `clip_embeddings` | model_facing | float32 | `(C, 256)` | penultimate-фичи SlowFast, **L2-норм** — seq-токен для Encoder |
| `clip_times_s` | model_facing | float32 | `(C,)` | центр клипа, сек; **⊆ `union_timestamps_sec`** |
| `clip_frame_indices` | model_facing | int32 | `(C,)` | центральный кадр клипа |
| `clip_topk_action_ids` | analytics | int32 | `(C, K)` | top-K Kinetics-400 id (дефолт K=5) |
| `clip_topk_probs` | analytics | float32 | `(C, K)` | softmax-вероятности top-K |
| `clip_track_id` | model_facing | int32 | `(C,)` | id трека (из `core_object_detections.track_ids`), `-1` если клип не привязан к треку |
| `class_names` | analytics | str | `(400,)` | стабильная карта Kinetics-400 `id:name` |
| `clip_count` | analytics | int32 | `()` | = C |

Порядок: клипы отсортированы по `clip_times_s` (возрастание). Один трек может дать несколько
клипов (несколько окон) — они идут в общем потоке, связь через `clip_track_id`.

### Агрегаты для аналитиков (из stream, top-уровневые — чтобы их видел feature_quality_audit)
| key | dtype | shape | описание |
|---|---|---|---|
| `video_action_hist` | float32 | `(400,)` | нормир. распределение действий по видео (Σ probs по клипам / C) |
| `dominant_action_ids` | int32 | `(top,)` | top-действия видео (дефолт top=10) |
| `dominant_action_probs` | float32 | `(top,)` | их агрег. вес |
| `num_tracks` | int32 | `()` | сколько треков дали ≥1 клип |
| `mean_clips_per_track` | float32 | `()` | когерентность (растёт → меньше фрагментации) |

Per-track динамика (`stability/temporal_jump/num_switches`) из v2 — **депрецируется как
топ-контракт** (была недостоверна без трекинга). При желании пересчитать поверх stream и
положить в `meta_json.per_track` (debug), не в модель-facing поля.

### Head (обе выхода с одной модели)
SlowFast R50 (pytorchvideo) — Kinetics-классификатор. С одного forward снимаем оба:
- **penultimate** (перед `blocks[-1].proj`) → `clip_embeddings` (256, L2);
- **logits → softmax** → `clip_topk_*` + вклад в `video_action_hist`.

Реализация: хук на penultimate ИЛИ `model.blocks[-1].proj = Identity()` для фич + отдельный forward
на logits (проще — один forward с сохранением обоих тензоров через forward-hook). Приложить
`class_names` из официальной карты Kinetics-400 (положить `kinetics400_labels.txt` в spec-ассеты).

### Пустой/контроль
`status="empty"`, `empty_reason="no_person_detections"`, `clip_count=0`, все `clip_*` — пустые
массивы правильного dtype. Валидатор считает это **валидным** (не падение).

### Валидация NPZ (component-scoped, R5)
Добавить `utils/validate_action_recognition_npz.py` (по образцу `validate_core_object_detections_npz.py`):
проверяет наличие/типы/shape полей v3, L2-норму `clip_embeddings`, `clip_times_s ⊆ union`,
монотонность времени, `clip_topk_probs` в [0,1] и сорт по убыванию, `clip_track_id` ∈ известных id.
Это заменяет неработающий на одном компоненте §0.2 (`e2e_validate_output_quality` ждёт full-manifest).

---

## B. Оконный сэмплинг Segmenter под action_recognition

Проблема v2: union-кадры **разрежены** → на трек попадает <32 кадров → SlowFast (clip_len=32)
делает 1 клип/трек. Нужна плотность.

### Политика `action_recognition` в Segmenter
Компонент получает **не разреженные union-точки, а набор непрерывных окон**:
- окно = `clip_len` (32) **подряд идущих нативных кадров** (stride=1 внутри окна);
- окна расставляются по таймлайну с шагом `window_hop_s` (дефлот ~2 c) → перекрытие/тайлинг;
- итоговые `frame_indices` компонента = **объединение кадров всех окон** (по-прежнему ⊆ union
  как множество, но union **расширяется** плотными кадрами в этих зонах);
- метадата окон в `metadata["action_recognition"].windows = [{start_frame,end_frame,center_s}]`
  — action_recognition берёт клипы **по окнам**, а не «по треку из разреженных кадров».

Это убирает зависимость «Segmenter должен заранее знать, где люди»: окна ставятся по всему
таймлайну (или, как оптимизация 2-го прохода — только вокруг person-зон, если детекция уже была).
Трекер затем связывает person'ов сквозь плотные кадры → `clip_track_id`.

Правки: `Segmenter/_build_default_component_budgets` (+ оконный билдер), учесть окна в
`sampling_policy_version` (меняем → бамп версии, идемпотентность через config_hash).

### Выравнивание детекции под окна (R1, решение 2026-07-04 — качество)
Прогон 07-04 показал: если `core_object_detections` сэмплирует **разреженно**, а окна плотные,
то внутри окна у person-трека мало кадров → `_make_clips` **паддит** → шумные клипы
(`mean_clips_per_track=1.0`, 18/22 клипов из коротких треков). Решение (элегантнее 2-го прохода
детекции): в Segmenter dense-кадры окон **добавляются в выборку `core_object_detections`** →
детекция покрывает те же кадры → треки полные (≥`clip_len`) без паддинга, а multi-clip на трек
возникает через re-ID между окнами. Доп. guard в компоненте: `min_clip_real_frames` (дефолт 16) —
трек с меньшим числом РЕАЛЬНЫХ кадров клип не эмитит. Это даёт per-track траекторию действий
(полезная фича) без тяжёлой 2-проходной детекции.

### Стоимость (200k)
Плотные окна дороже разреженных точек. Ограничители: `max_windows_per_video`,
`window_hop_s` (реже окна на длинных видео), кап клипов (`max_clips`), батч-инференс клипов.
Зафиксировать в отчёте фактические тайминги на 8:00 ролике.

---

## C. Связь компонентов (обновлённый контракт)
```
Segmenter[action_recognition: dense windows ≥32] ─► core_object_detections[+appearance tracker → track_ids]
                                                            │
                                                            ▼
                                          action_recognition (per-clip stream v3)
   clip_embeddings(C,256) L2  +  clip_times_s ⊆ union  +  clip_topk_action_*(Kinetics)  +  clip_track_id
```
Входная валидация action_recognition: есть окна (иначе empty), есть person-детекции с `track_ids`;
клип строится по окну (≥clip_len кадров гарантирован Segmenter'ом), `clip_track_id` = мода
`track_ids` person-боксов в окне (или `-1`).

---

## D. Порядок внедрения
1. Segmenter: оконная политика для action_recognition (бамп `sampling_policy_version`).
2. core_object_detections: appearance-tracker → `track_ids` (schema v3) — `EMBEDDING_TRACKER.md`.
3. action_recognition: v3-выход (плоский stream + классы + агрегаты) + `validate_*_npz.py`.
4. Перепрогон по `RUN_SPEC_v2.md` на реальных видео → REPORT → сверка владельцем → штамп.
