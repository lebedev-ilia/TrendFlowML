# `micro_emotion` — выходы и фичи

**Модуль:** [`../utils/micro_emotion_processor.py`](../utils/micro_emotion_processor.py) · **Схема NPZ:** [`../../../schemas/micro_emotion_npz_v3.json`](../../../schemas/micro_emotion_npz_v3.json) · **Валидатор:** [`../utils/validate_micro_emotion.py`](../utils/validate_micro_emotion.py) (и `validate_micro_emotion_npz` при наличии).

## Содержимое NPZ (смысл)

| Группа | Смысл |
|--------|--------|
| `frame_indices`, `times_s` | Ось кадров/времени, выравнивание с сэмплингом |
| `face_present_any` | Наличие лица на кадре |
| `frame_feature_names` / `frame_features` | Плотные признаки OpenFace [N, F] |
| `compact22` / `compact22_feature_names` | 22 «компактных» фичи на кадр |
| `event_*` | События микро-выражений (время, тип, сила) |
| `feature_names` / `feature_values` | Глобальные агрегаты для CSV/melt |
| `microexpr_features`, `summary` | Сводки (object) |
| `meta` | `status`, тайминги, OpenFace/ Docker параметры |

**Нормальные диапазоны (ориентиры):** `times_s` неубывающие/монотонность по валидатору; AUs и compact — зависят от OpenFace; события — `event_strength` ≥ 0. Детальные проверки — в `MicroEmotionValidator`.

## HTML / melt

Подписи: `view_csv_feature_descriptions_ru.json` (по именам колонок из batch); QA-подсветка — при наличии секции `micro_emotion` в `view_csv_feature_qa.json`.
