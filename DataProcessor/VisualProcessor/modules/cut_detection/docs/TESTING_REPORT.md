# Отчёт о тестировании cut_detection компонента

**Дата**: 2026-03-09  
**Компонент**: `cut_detection`  
**Версия**: 2.0  
**Schema**: `cut_detection_npz_v1`

---

## Резюме

✅ **Все тесты пройдены успешно**

- **Протестировано видео**: 20
- **Успешных прогонов**: 20/20 (100%)
- **Валидных артефактов**: 20/20 (100%)

Все артефакты соответствуют схеме `cut_detection_npz_v1`. Обязательные поля и размерности проверены.

---

## Качество данных

- ✅ Все обязательные поля присутствуют
- ✅ `frame_indices` и `times_s` — 1D, размерности корректны
- ✅ `features` и `detections` — валидные объекты

---

## Файлы

- **Скрипт запуска**: `DataProcessor/VisualProcessor/modules/cut_detection/scripts/run_tests.sh`
- **Валидатор**: `DataProcessor/VisualProcessor/modules/cut_detection/utils/validate_cut_detection.py`
- **Результаты**: `DataProcessor/dp_results/youtube/test_cut_detection_*/`

---

## Заключение

Компонент `cut_detection` успешно протестирован на 20 видео. Все артефакты валидны. Компонент готов к использованию.
