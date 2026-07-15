# RUN_SPEC: action_recognition (v2 — перепрогон после доработки)

Автор: Claude → Исполнитель: Cursor. Основание: REPORT 2026-07-02 (🔁) + решения владельца.
Дизайн правок: `docs/design/EMBEDDING_TRACKER.md`, `docs/design/ACTION_RECOGNITION_V3.md`.

## 0. Что изменилось относительно v1
1. **Segmenter**: плотные окна ≥32 кадров для action_recognition (не разреженные union-точки).
2. **core_object_detections**: appearance-embedding **трекер** → реальные `track_ids` (schema v3).
3. **action_recognition**: плоский **per-clip stream v3** (эмбеддинг + классы Kinetics + `clip_track_id`
   + агрегаты) + component-scoped `validate_action_recognition_npz.py`.
4. **Реальные видео** вместо synthetic-склеек (см. §2).

Порядок реализации — раздел D в `ACTION_RECOGNITION_V3.md`. Прогон — после всех трёх правок.

## 1. Профиль
Включить: `Segmenter` + `core_object_detections` (tracking.enabled=true, embedder=osnet|clip) +
`action_recognition` (v3). Остальное выкл. `clip_len=32`, `topk K=5`.
Если OSNet-вес не готов к первому прогону — `tracking.embedder: clip` (CLIP уже в Triton), пометить в RUN_RESULT.

## 2. Матрица видео (реальные, предоставлены владельцем)
Файлы: `component_reports/action_recognition/fixtures/`.

| video_id | файл | длина | тип | что проверяем |
|---|---|---:|---|---|
| `ar_real_4m35_people` | ar_real_4m35_people.mp4 | 275 s (4:35) | люди, реальный длинный | когерентность треков, mean_clips_per_track, стоимость |
| `ar_real_8m00_people` | ar_real_8m00_people.mp4 | 481 s (8:00) | люди, самый длинный | деградация/стоимость, re-ID при уходах из кадра |
| `ar_real_2m47_control_nopeople` | ar_real_2m47_control_nopeople.mp4 | 167 s (2:47) | **без людей** | валидный `empty (no_person_detections)`, `num_tracks≈0`, FP детектора |

Доп. (по возможности, из прошлой матрицы — для сравнения короткого): 1 короткий talking-head (~10 s)
и 1 динамичный (~30 s) с людьми. Не обязательно, но полезно для оси «различимость».

Зафиксировать фактические `duration_sec`, fps, разрешение каждого.

## 3. Что собрать (Cursor → artifacts/)
Per video:
- `action_recognition_features.npz` (v3): shapes всех `clip_*`, `clip_count`, `num_tracks`,
  `mean_clips_per_track`, `dominant_action_ids/probs` (расшифровать `class_names`).
- `detections.npz` срез: `num_tracks`, распределение длин треков, `tracks_json`.
- **Прокси-метрики трекера** (критерий приёмки §8 EMBEDDING_TRACKER):
  - гистограмма intra-track vs inter-track cosine (разделимость id);
  - mean/median track_len, доля 1-клиповых треков (было ~100% — цель сильно меньше);
  - для 8:00: пример re-ID (id сохраняется после ухода из кадра) — рендер боксов с id (несколько кадров).
- `*_health.md` (feature_quality_audit — теперь видит топ-уровневые агрегаты и `clip_*`),
  `validate_action_recognition_npz.py` вывод (pass/fail по контракту v3).
- **golden-повтор** одного видео (напр. 4:35) ×2 → идентичность `track_ids` и `clip_embeddings`.
- Тайминги стадий (Segmenter/detection+tracking/action) + пик VRAM/CPU; **overhead трекинга** отдельно.
- Версии: `dataprocessor_version`, `sampling_policy_version` (должна смениться!), `model_signature`
  (SlowFast + embedder), какие патчи/preflight понадобились (стиль LOGIC_ERRORS).

## 4. На что смотрю в REPORT (оси)
- **Корректность:** v3-контракт (validate pass), `clip_embeddings` L2/finite, `clip_times_s ⊆ union`,
  классы осмысленны (совпадают ли top-действия с реальным содержимым — сверяет владелец).
- **Стабильность:** golden-повтор идентичен.
- **Различимость:** intra≫inter cosine треков; `mean_clips_per_track>1`; распределение действий не вырождено.
- **Фрагментация (главное):** число треков на 4:35/8:00 **не** взрывается; доля 1-клиповых мала.
- **Контроль:** people-free → empty; FP person измерен.
- **Стоимость (200k):** тайминги на 8:00 + overhead трекера; нужны ли капы (`max_windows/max_clips`).

## 5. DoD перепрогона (когда штампуем v3)
- Все 3 видео `ok`/валидный `empty`; контракт v3 проходит валидатор.
- Фрагментация устранена (метрика §4); re-ID виден на 8:00.
- Классы Kinetics присутствуют и правдоподобны (сверка владельцем с реальным видео).
- golden идентичен; стоимость на 8:00 в приемлемом бюджете (или введены капы).
- Владелец сверяет top-действия/треки с самими роликами → штамп `action_recognition v3`,
  обновление `FEATURE_DESCRIPTION.md` + чеклиста + ledger фич.

## 6. Открытые решения (по итогам прогона)
- Финальный `embedder` (osnet vs clip) — по метрике различимости id.
- Капы стоимости под 200k (`window_hop_s`, `max_windows_per_video`, `max_clips`).
- Депрецировать ли per-track динамику v2 полностью или оставить в debug.
