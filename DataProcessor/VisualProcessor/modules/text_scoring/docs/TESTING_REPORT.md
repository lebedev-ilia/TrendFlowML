# Отчёт о тестировании text_scoring компонента

**Дата**: 2026-03-09  
**Компонент**: `text_scoring`  
**Версия схемы**: `text_scoring_npz_v2`  
**Producer version**: 2.0.1

---

## Резюме

✅ **Все тесты пройдены успешно**

- **Протестировано видео**: 20 (+ smoke = 21)
- **Успешных прогонов**: 21/21 (100%)
- **Валидных артефактов**: 21/21 (100%)

Модуль является consumer OCR-артефакта. Все тесты завершены успешно, валидация без ошибок.

---

## Качество данных

- ✅ Все артефакты соответствуют схеме `text_scoring_npz_v2`
- ✅ Обязательные ключи, размерности, sanity-checks пройдены

---

## Файлы

- **Скрипт запуска**: `DataProcessor/VisualProcessor/modules/text_scoring/scripts/run_tests.sh`
- **Валидатор**: `DataProcessor/VisualProcessor/modules/text_scoring/utils/validate_text_scoring.py`
- **Результаты**: `DataProcessor/dp_results/youtube/test_text_scoring_*/`

---

## Заключение

Компонент `text_scoring` успешно протестирован на 20+ видео. Все артефакты валидны. Компонент готов к использованию.
