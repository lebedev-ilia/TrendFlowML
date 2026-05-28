# VisualProcessor — Audit v4: общий итог (L1, набор **A**)

**Дата сводки:** 2026-04-06  
**Опорный run (набор A):** `youtube / -Q6fnPIybEI / 4c3bf25b-e300-47b3-915e-4699c72ab190`  
**План и критерии:** [AUDIT_4_CRITERIA_AND_PLAN.md](AUDIT_4_CRITERIA_AND_PLAN.md)  
**Журнал прогонов:** [RUN_LOG.md](RUN_LOG.md)  
**Каталог отчётов:** [components/README.md](components/README.md)

## Статус волны

Все перечисленные ниже компоненты имеют отчёты уровня **L1 (draft)** на **одном** артефакте **A**: эмпирика, сверка с machine-schema (`allow_extra_keys: false` где задано), без закрытия **§8 DoD**, без наборов **B/C**. В `RUN_LOG` статусы помечены **in_progress**.

**Сводная оценка L1 по пилотной волне (среднее по вердиктам отчётов, округлённо): ~8.3 / 10.**

Интерпретация: контракты NPZ и фактические ключи в целом **согласованы**; типичные замечания **L1** — семантика вероятностей/top‑k (не всегда softmax), маски vs NaN, пустые `models_used` там, где upstream очевиден. Ранний рассинхрон схемы и артефакта в `action_recognition` (плоские `metric__*` и списочные поля) закрыт: новый артефакт в `storage/result_store` записывается как `action_recognition_features.npz` и проходит `validate_npz`. До «продуктово зелёного» уровня нужны **B/C** и **§4.8**.

## Структура отчётов


| Зона   | Путь                                                                         |
| ------ | ---------------------------------------------------------------------------- |
| Модули | [components/visual_processor/modules/](components/visual_processor/modules/) |
| Core   | [components/visual_processor/core/](components/visual_processor/core/)       |


## Сквозные темы (на A)

1. **Маски и NaN.** `core_face_landmarks`, `emotion_face`, `shot_quality`, `optical_flow` (модуль vs core) — потребители должны явно использовать `*_present` / `valid_mask` / `processed_mask`, а не догадываться по одной только плотности float.
2. **Вероятности и top‑k.** `core_clip`, `scene_classification`, `shot_quality`, `similarity_metrics` — суммы по подмножеству классов **не обязаны** быть 1; энкодеру нужен явный контракт или постобработка.
3. **Ось N по модулям.** Один и тот же run может иметь **N=48** у части задач и **N=120** у `text_scoring` — нормально при разных политиках Segmenter.
4. **Core без «треков».** `core_object_detections` **v2** не содержит track id; downstream, ожидающий треки, должен получать контракт или отдельный артефакт.
5. **Приватность OCR.** `ocr_extractor` и часть `text_scoring`: сырой текст может отсутствовать при `retain_raw_ocr_text=false`.

## Таблица по компонентам

Оценка в колонке «вердикт» — из соответствующего отчёта (см. §6 / итог L1 в файле).

### Модули (`visual_processor/modules/`)


| Компонент              | Отчёт                                                                                                    | Вердикт (L1) | Кратко на **A**                                         |
| ---------------------- | -------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------- |
| `action_recognition`   | [action_recognition_audit_v4.md](components/visual_processor/modules/action_recognition_audit_v4.md)     | **~8**/10 | **фиксы в коде:** `metric__`/схема/имя NPZ/валидация; **A** уже есть в `storage/result_store` (`action_recognition_features.npz`) |
| `behavioral`           | [behavioral_audit_v4.md](components/visual_processor/modules/behavioral_audit_v4.md)                     | ~**8.5**/10  | маски рук/тела vs float рядов                           |
| `color_light`          | [color_light_audit_v4.md](components/visual_processor/modules/color_light_audit_v4.md)                   | ~**8.5**/10  | крупный `video_features`; часть NaN                     |
| `cut_detection`        | [cut_detection_audit_v4.md](components/visual_processor/modules/cut_detection_audit_v4.md)               | ~**8.5**/10  | два NPZ; SSIM/глубина — маски NaN                       |
| `detalize_face`        | [detalize_face_audit_v4.md](components/visual_processor/modules/detalize_face_audit_v4.md)               | ~**8.5**/10  | масштаб compact L2 ≠ 1                                  |
| `emotion_face`         | [emotion_face_audit_v4.md](components/visual_processor/modules/emotion_face_audit_v4.md)                 | ~**8.5**/10  | `processed_mask` vs доля NaN                            |
| `frames_composition`   | [frames_composition_audit_v4.md](components/visual_processor/modules/frames_composition_audit_v4.md)     | ~**8.5**/10  | face‑зависимые поля и NaN                               |
| `high_level_semantic`  | [high_level_semantic_audit_v4.md](components/visual_processor/modules/high_level_semantic_audit_v4.md)   | ~**8.5**/10  | опциональный audio → NaN в колонках                     |
| `micro_emotion`        | [micro_emotion_audit_v4.md](components/visual_processor/modules/micro_emotion_audit_v4.md)               | ~**8.5**/10  | Docker OpenFace; K=0 events                             |
| `optical_flow`         | [optical_flow_audit_v4.md](components/visual_processor/modules/optical_flow_audit_v4.md)                 | ~**8.5**/10  | высокий missing vs core OF                              |
| `scene_classification` | [scene_classification_audit_v4.md](components/visual_processor/modules/scene_classification_audit_v4.md) | ~**8.5**/10  | top‑5 не суммируется в 1                                |
| `shot_quality`         | [shot_quality_audit_v4.md](components/visual_processor/modules/shot_quality_audit_v4.md)                 | ~**8**/10    | lens/face NaN; пустой `models_used`                     |
| `similarity_metrics`   | [similarity_metrics_audit_v4.md](components/visual_processor/modules/similarity_metrics_audit_v4.md)     | ~**8.5**/10  | без reference — много NaN в tabular                     |
| `story_structure`      | [story_structure_audit_v4.md](components/visual_processor/modules/story_structure_audit_v4.md)           | ~**8**/10    | topic ветка off; экстремальный скаляр                   |
| `text_scoring`         | [text_scoring_audit_v4.md](components/visual_processor/modules/text_scoring_audit_v4.md)                 | ~**8**/10    | разреженный OCR; CTA/flags NaN                          |
| `uniqueness`           | [uniqueness_audit_v4.md](components/visual_processor/modules/uniqueness_audit_v4.md)                     | ~**8.5**/10  | высокая repetition_ratio на A                           |
| `video_pacing`         | [video_pacing_audit_v4.md](components/visual_processor/modules/video_pacing_audit_v4.md)                 | ~**8**/10    | NaN при выключенных `enable_*`                          |


### Core (`visual_processor/core/`)


| Компонент                | Отчёт                                                                                                     | Вердикт (L1) | Кратко на **A**                                      |
| ------------------------ | --------------------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------- |
| `core_clip`              | [core_clip_audit_v4.md](components/visual_processor/core/core_clip_audit_v4.md)                           | ~**8.5**/10  | скоры не softmax; NaN у `consecutive_cosine_prev[0]` |
| `core_depth_midas`       | [core_depth_midas_audit_v4.md](components/visual_processor/core/core_depth_midas_audit_v4.md)             | ~**8.5**/10  | плотные карты; preview K=10                          |
| `core_face_landmarks`    | [core_face_landmarks_audit_v4.md](components/visual_processor/core/core_face_landmarks_audit_v4.md)       | ~**8.5**/10  | NaN при `face_present`; mesh vs présence             |
| `core_object_detections` | [core_object_detections_audit_v4.md](components/visual_processor/core/core_object_detections_audit_v4.md) | ~**8.5**/10  | только `valid_mask`; нет tracks                      |
| `core_optical_flow`      | [core_optical_flow_audit_v4.md](components/visual_processor/core/core_optical_flow_audit_v4.md)           | ~**8.5**/10  | первый кадр NaN в flow‑рядах                         |
| `ocr_extractor`          | [ocr_extractor_audit_v4.md](components/visual_processor/core/ocr_extractor_audit_v4.md)                   | ~**8.5**/10  | redacted текст; R строк vs N кадров                  |


**Всего компонентов VisualProcessor с отчётом L1 в этой волне:** **23** (17 модулей + 6 core).

## Следующие шаги

1. **action_recognition:** схема ↔ продюсер и доки выровнены; артефакт **A** присутствует в `storage/result_store` и `manifest` отмечает `status=ok`. Дальше — наборы **B/C** и golden (§4.8).
2. Наборы **B/C** и **§4.8** для модулей с высокой долей NaN или спорной семантикой.
3. Унифицировать заполнение **`meta.models_used`** в остальных модулях, где по отчётам пусто при работающем upstream (сделано точечно для **`shot_quality`**).

