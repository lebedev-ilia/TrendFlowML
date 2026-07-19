# FINAL REPORT — `action_recognition`

> Глубокий разбор по [`COMPONENT_DEEP_DIVE_PROTOCOL.md`](../../COMPONENT_DEEP_DIVE_PROTOCOL.md).
> Целостный разбор компонента. Три его model-facing выхода разобраны отдельно:
> [`clip_embeddings`](../clip_embeddings/FINAL_REPORT.md),
> [`clip_times_s/clip_frame_indices`](../clip_times_s_clip_frame_indices/FINAL_REPORT.md),
> [`clip_track_id`](../clip_track_id/FINAL_REPORT.md) — здесь модуль как единое целое + аналитическая (Kinetics) сторона.

## 1. Метаданные

| Поле | Значение |
|---|---|
| Компонент | `action_recognition` (VisualProcessor `BaseModule`, Tier-2, GPU) |
| Версия схемы (целевая) | `action_recognition_npz_v3` (реальный батч — **v2**) |
| Артефакт | `result_store/<platform>/<video>/<run>/action_recognition/action_recognition_features.npz` |
| Модель | **SlowFast R50** (pytorchvideo, Kinetics-400); backbone-абстракция (VideoMAE/v2/Hiera опц.) |
| Hard deps | `core_object_detections` (person + appearance-`track_ids`) + Segmenter dense-окна |
| Потребители | Models/Encoder (`clip_embeddings`), аналитика (Kinetics-распределение) |
| Дата разбора | 2026-07-18 |
| Штамп валидации | `COMPONENT_VALIDATION_CHECKLIST.md` → action_recognition v3 ✅ (2026-07-05) |
| Отчёт валидации | [`REPORT_2026-07-05_FINAL.md`](REPORT_2026-07-05_FINAL.md), `RUN_RESULT_v3*.md`, `ASSESSMENT_action_recognition.md` |
| Дизайн | [`design/ACTION_RECOGNITION_V3.md`](../../design/ACTION_RECOGNITION_V3.md), [`design/EMBEDDING_TRACKER.md`](../../design/EMBEDDING_TRACKER.md) |
| Код | `modules/action_recognition/utils/{action_recognition_slowfast.py, action_recognition_v3.py, backbones.py}` |

## 2. Резюме

`action_recognition` — **единственный компонент, распознающий действия в движении** (не по кадру, а по короткому
клипу). SlowFast прогоняет окна вокруг треков людей и выдаёт: **action-токен** `clip_embeddings (C,2304)` (penultimate
L2 — для Encoder), **распределение действий Kinetics-400** (clip_topk / video_action_hist / dominant_action — для
аналитики), привязку к персонажу (`clip_track_id`, appearance-трекер), per-person tubelet и temporal localization.
Архитектурно это **сильный, продуманный компонент** (penultimate-фичи, свой appearance-трекер, косто-нейтральные
окна), доказанный на тест-прогонах v3 (44/65 клипов, golden=0, mean_clips/track 4.0). **Но в реальном storage-
корпусе не материализован ничего из ценного:** все артефакты — **схема v2** (256-d per-track эмбеддинги + трекинг-
метрики, **без Kinetics-классов и без плоского стрима**), фрагментация ~1 клип/трек (до-v3.2), доминирующий ролик
`empty` (no_person). Итог: **ни model-токен (2304), ни аналитические классы действий не доступны на реальных данных**
— нужен пере-прогон батча на v3.

## 3. Функционал

Стоит в Tier-2, после `core_object_detections` (person + track_ids) и Segmenter (dense-окна ≥`clip_len`):

1. **Нарезка клипов по трекам** — окна `clip_len×window_len_mult` (v3.2=96 кадров) вокруг каждого персонажа.
2. **SlowFast инференс** — на клип: **логиты Kinetics-400** (что за действие) + **penultimate-фичи** (богатое
   представление действия, снимается forward-hook'ом).
3. **Сборка v3-стрима** (`build_v3_arrays`) — плоский time-ordered поток: эмбеддинги + классы + track_id + ось.
4. **Агрегаты** — `video_action_hist (400)`, `dominant_action_ids/probs`, `num_tracks`, `mean_clips_per_track`.
5. **Tubelet + temporal localization** — разные действия разным трекам, границы действий (`clip_segment_id`).

**Зачем продукту:** тип и динамика действия (танец/спорт/готовка/разговор) — **сильный семантический сигнал
формата и жанра** контента, драйвер вовлечённости. Для Encoder — action-токен (что происходит во времени), для
аналитика — «какие действия в вашем видео».

## 4. Вход

- **`core_object_detections`** (hard) — person-детекции + appearance-`track_ids` (schema v3); нет person →
  valid `empty` (`no_person_detections`).
- **Segmenter dense-окна** (≥`clip_len`) — иначе SlowFast = 1 клип/трек (корневая проблема v1).
- **общий `frame_indices`** (shared sampling group), **`union_timestamps_sec`** — ось.
- **SlowFast веса** (pytorchvideo, Kinetics-400), fp16-опция.

## 5. Выход

Плоский v3-стрим (целевой): `clip_embeddings (C,D≈2304)`, `clip_times_s`/`clip_frame_indices`, `clip_track_id`,
`clip_segment_id`, `clip_topk_action_ids/probs`, `class_names (400)`, `video_action_hist (400)`,
`dominant_action_ids/probs`, `num_tracks`, `mean_clips_per_track`, `clip_count`. **Реальный батч (v2):** вместо
этого — object-array per-track эмбеддингов (256-d) + трекинг-метрики (stability/switches/temporal_jumps),
**без классов и стрима**. Детальный разбор трёх model-выходов — в отдельных FINAL_REPORT (§0).

## 6. Фичи (важное/неочевидное)

- **Kinetics-400 классификация (аналитическая сторона)** — `dominant_action` + `video_action_hist`: «какое
  действие преобладает». На тест-видео метки правдоподобны (clarinet/harmonica, yoga/marching). **В v2-батче
  ОТСУТСТВУЕТ полностью** (0 action-ключей per-track) → аналитик не получает действий на реальных данных.
- **penultimate-эмбеддинг вместо логитов** — ключевое улучшение v3 (2304-d богатая репрезентация vs 400
  «уверенностей»). См. [`clip_embeddings`](../clip_embeddings/FINAL_REPORT.md).
- **appearance-трекер** — свой ReID (OSNet/CLIP) внутри core_object_detections чинит фрагментацию (205→11 треков).
  См. [`clip_track_id`](../clip_track_id/FINAL_REPORT.md).
- **tubelet per-person** — разным трекам на групповой сцене приписываются разные действия (не одно на весь кадр).
- **temporal localization** (`clip_segment_id`) — слабая: `num_action_segments = num_tracks` (1 сегмент/трек,
  change-point внутри трека не срабатывает) — аналитическая тонкость.

## 7. Алгоритм / архитектура

- **SlowFast R50** (Kinetics-400, pytorchvideo, веса `model_state`). Backbone-абстракция (`backbones.py`)
  поддерживает VideoMAE/VideoMAEv2/Hiera — по baseline-ablation.
- **penultimate** снимается `register_forward_pre_hook` на классификаторе; L2-нормировка.
- **Сложность:** тяжёлый 3D-CNN на клип (окно 96 кадров); бюджет 1536 кадров (16 окон×96), диск ~10.7 GB/видео,
  fp16-опция. Детерминизм: fp32-backbone + детерминированный трекер → golden ×2 идентичны.

## 8. Оптимизации

- **Косто-нейтральные окна (v3.2)** — окно×3, окон втрое меньше → тот же бюджет, но mean_clips/track 1→4.
- **penultimate через hook** без модификации модели (общий `_HFVideoBackbone`).
- **Reuse appearance-`track_ids`** из core_object_detections (не считает трекинг дважды).
- **fp16 + батч клипов**, очистка frames после NPZ (`cleanup_frames_after_npz`).
- **Чистая `build_v3_arrays` (numpy)** — юнит-тестируема без GPU, golden тривиален.

## 9. Слабые места

- **Реальный батч — schema v2, ничего ценного не материализовано (главное).** Все 6 storage-видео: 256-d
  per-track эмбеддинги + трекинг-метрики, **без Kinetics-классов, без плоского стрима, mean_clips/track≈1**
  (до-v3.2), доминирующий ролик `empty`. И model-токен (2304), и аналитические действия **недоступны** —
  блокер и для Encoder, и для дашборда. Требует пере-прогона на v3.
- **Доковый дрейф** — `SCHEMA.md` описывает v2 (256-d), контракт/код — v3 (2304). См. clip_embeddings §9.
- **Зависимость от person-детекций** — нет людей → `empty`; контент без людей (природа/графика/скринкаст)
  action-токена не получает (тихая дыра).
- **Дорого/диск-ёмко** (~10.7 GB/видео, тяжёлый 3D-CNN) — для 200k критична политика очистки кадров.
- **temporal localization слабая** (1 сегмент/трек) — тонкие границы действий не выделяются.
- **D нестабильна между backbone** (2304 slowfast / 768 VideoMAE / 400 fallback) — Encoder обязан подстраиваться.

## 10. Рекомендации по улучшению алгоритма (приоритизированные)

1. **[выс., блокер] Пере-прогнать storage-корпус на v3** — материализовать и `clip_embeddings (C,2304)`, и
   Kinetics-классы; без этого весь компонент бесполезен на реальных данных (общий блокер трёх выходов).
2. **[выс.] Обновить `SCHEMA.md` до v3** — устранить доковый дрейф v2↔v3.
3. **[сред.] Зафиксировать прод-backbone и D** (slowfast_r50 / 2304) — чтобы Encoder не подстраивал размерность.
4. **[сред.] Fallback для не-people-контента** — scene-level action-эмбеддинг без трека, чтобы природа/графика
   получали хотя бы минимальный action-сигнал.
5. **[низ.] Улучшить temporal localization** (сегментация по логит-классу, а не эмбеддингу) — для тонких границ.

## 11. Рекомендации по архитектуре / связям

- **Shared sampling group** (core_object_detections ↔ action_recognition) — гарантировать совпадение
  `frame_indices` на Segmenter, иначе клип↔трек рассыпается.
- **Единая размерность-политика с Encoder** — читать D из `meta.embedding_dim`, запретить хардкод.
- **Reuse appearance-трекера** для других person-компонентов (emotion/behavioral) — общий person-id слой.
- **Провенанс backbone в meta** (`model_id`, `embedding_mode`) обязателен — Encoder должен знать, на чём учился.

## 12. Результаты тестов и оценка

| Прогон | Объём | Результат | Что говорит |
|---|---|---|---|
| v3.2 валидация (NPZ) | 4:35 / 8:00 | (44,2304)/(65,·) L2, классы ✅, golden идентичен, cl/trk 4.0/4.06 | контракт+токен корректны на ТЕСТЕ |
| Контроль no_person | 2:47 | `empty`, golden ✅ | valid-empty корректен |
| Различимость треков/tubelet | групповая сцена | intra≫inter, разные действия разным трекам | id/action разделяются |
| Классы Kinetics | сверка с роликами | clarinet/yoga/marching правдоподобны | семантика осмысленна |
| Эволюция v1→v3.2 | — | фрагментация 205→11 треков, cl/trk 1→4 | корневая проблема решена |
| **Реальный storage (мой прогон)** | **6 видео, все v2** | 256-d per-track, **0 action-классов, стрим отсутствует, cl/trk≈1, 1 empty** | ничего ценного не материализовано |

Вывод: **дизайн и логика доказаны на тест-прогонах** (golden=0, penultimate, классы, tubelet, cl/trk>1), но
**реальный корпус целиком на v2** — ни model-токен, ни классы не доступны; штамп основан на тест-данных.

## 13. Интерпретируемость

- **Есть:** `dominant_action` + `class_names` (Kinetics) — человекочитаемая метка действия (когда есть v3);
  `video_action_hist` — профиль действий. Эмбеддинг сам не интерпретируем (показывать классы).
- **Добавить:** «в видео преобладает: танец/готовка/спорт»; timeline действий по `clip_times_s`; всё — **после**
  пере-прогона на v3 (сейчас классов в данных нет).

## 14. Польза для моделей

**Потенциально высокая, фактически нулевая на реальных данных.** `clip_embeddings (C,2304)` — уникальный
action-токен (семантика действия во времени), идеальная форма для Encoder — **но в v2-батче его нет** (только
256-d per-track проекция). После пере-прогона на v3 ценность станет одной из самых высоких среди визуальных
входов (см. clip_embeddings: 4/5). Сейчас — незрелость данных.

## 15. Польза для аналитиков

**Потенциально высокая, фактически нулевая.** «Какие действия в видео» (Kinetics-профиль, доминирующее действие)
— ценная и наглядная аналитика — **но в v2-батче классов нет вовсе** (0 action-ключей). После v3-прогона —
понятный action-профиль для сравнения видео. Сейчас аналитик действий не получает.

## 16. Оценки

| Пункт | Балл | Обоснование |
|---|---:|---|
| 3. Функционал | 5 | Единственный источник семантики действия во времени; богатый (токен+классы+tubelet) |
| 5. Выход (контракт) | 3 | Богатый v3-контракт, но в реальном батче — v2 без классов/стрима |
| 6. Фичи | 3 | penultimate/классы/tubelet сильны по дизайну, но не материализованы (v2, no classes) |
| 8. Оптимизации | 4 | Косто-нейтральные окна, hook, reuse трекера, fp16, golden=0 |
| 9. Слабые места (инверсно) | 2 | Весь ценный выход не в батче (v2), доковый дрейф, no-people, дорого |
| 12. Результаты тестов | 3 | Дизайн доказан на тесте (golden=0, cl/trk 4), но реальный корпус v2 |
| 13. Интерпретируемость | 3 | Kinetics-метки понятны, но в данных их нет |
| 14. Польза для моделей | 4 | Уникальный action-токен по дизайну; фактически ждёт v3-прогона |
| 15. Польза для аналитиков | 3 | Action-профиль ценен, но в реальных данных отсутствует |

### Итоговые оценки

- **Польза для моделей: 4/5.** По дизайну — уникальный и семантически богатый action-токен (penultimate 2304,
  L2, time-ordered, track-linked), потенциально один из самых информативных визуальных входов Encoder'а
  (жанр/формат/динамика). Балл держит потенциал (сильная, доказанная на тесте архитектура), но фактически на
  реальных данных токен не материализован (v2) — до пере-прогона польза нулевая.
- **Польза для аналитиков: 3/5.** «Какие действия в видео» (Kinetics-профиль) — наглядная ценная аналитика, но
  в реальном батче классы отсутствуют полностью (v2 без классификации). Потенциал 4, факт — 0 до v3-прогона;
  средний балл отражает разрыв.

## 17. Источники

- `modules/action_recognition/utils/{action_recognition_slowfast.py, action_recognition_v3.py, backbones.py}`, `main.py`
- `modules/action_recognition/docs/SCHEMA.md` (⚠ v2, доковый дрейф)
- `DataProcessor/docs/component_reports/action_recognition/{REPORT_2026-07-05_FINAL.md, RUN_RESULT_v3*.md, ASSESSMENT_action_recognition.md}`
- `DataProcessor/docs/design/{ACTION_RECOGNITION_V3.md, EMBEDDING_TRACKER.md}`
- `DataProcessor/docs/COMPONENT_CONTRACTS.md` (core_object_detections → action_recognition → Models/Encoder)
- Sub-reports: `clip_embeddings/FINAL_REPORT.md`, `clip_times_s_clip_frame_indices/FINAL_REPORT.md`, `clip_track_id/FINAL_REPORT.md`
- Реальные артефакты: 12× (6 уникальных) `storage/.../action_recognition/action_recognition_features.npz`
  (**все schema v2: 256-d per-track, 0 action-классов, стрим отсутствует, 1 empty**)

## 18. Визуализации

![action_recognition overview](action_recognition_overview.png)

`action_recognition_overview.png`: слева — эволюция v1→v3.2 (appearance-трекер починил фрагментацию 205→11
треков, окна×3 → mean_clips/track 1→4); справа — «дизайн (v3) vs реальность (v2-батч)»: ни `clip_embeddings 2304`,
ни Kinetics-классификация, ни плоский стрим не материализованы в storage (256-d per-track, 0 action-ключей,
cl/trk≈1, доминирующий ролик empty). Подтверждает: сильная архитектура доказана на тесте, но реальный корпус
целиком на v2 и требует пере-прогона.
