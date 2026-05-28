# Отчёт о тестировании scene_classification компонента

**Дата**: 2026-03-09  
**Компонент**: `scene_classification`  
**Версия схемы**: `scene_classification_npz_v2`  
**Producer version**: 2.0.1

---

## Резюме

✅ **Все тесты пройдены успешно**

- **Протестировано видео**: 20 (+ smoke)
- **Успешных прогонов**: 20/20 (100%)
- **Валидных артефактов**: 20/20 (100%)

Зависимости: `core_clip`, `cut_detection`, `core_optical_flow`, `core_object_detections`, `core_face_landmarks`. Валидация без ошибок.

---

## Качество данных

- ✅ Все обязательные ключи присутствуют
- ✅ Frame-level и scene-level данные согласованы
- ✅ Ошибок валидации: 0

---

## Файлы

- **Скрипт запуска**: `DataProcessor/VisualProcessor/modules/scene_classification/scripts/run_tests.sh`
- **Валидатор**: `DataProcessor/VisualProcessor/modules/scene_classification/utils/validate_scene_classification.py`
- **Результаты**: `DataProcessor/dp_results/youtube/test_scene_classification_*/`

---

## Заключение

Компонент `scene_classification` успешно протестирован на 20 видео. Все артефакты валидны. Компонент готов к использованию.
