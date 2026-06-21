# Отчёт о тестировании optical_flow компонента

**Дата**: 2026-03-09  
**Компонент**: `optical_flow`  
**Версия схемы**: `optical_flow_npz_v3`  
**Producer version**: 2.0.2

---

## Резюме

✅ **Все тесты пройдены успешно**

- **Протестировано видео**: 20 (+ smoke)
- **Успешных прогонов**: 20/20 (100%)
- **Валидных артефактов**: 20/20 (100%)

Валидация выявила только некритичные предупреждения (dtype). Ошибок нет.

---

## Качество данных

- ✅ Все обязательные ключи присутствуют
- ✅ Размерности и типы данных корректны
- ✅ Warnings: только dtype (int64 vs int32), некритично

---

## Файлы

- **Скрипт запуска**: `DataProcessor/VisualProcessor/modules/optical_flow/scripts/run_tests.sh`
- **Валидатор**: `DataProcessor/VisualProcessor/modules/optical_flow/utils/validate_optical_flow.py`
- **Результаты**: `DataProcessor/dp_results/youtube/test_optical_flow_*/`

---

## Заключение

Компонент `optical_flow` успешно протестирован на 20 видео. Все артефакты валидны. Компонент готов к использованию.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
