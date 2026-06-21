# Отчёт о тестировании high_level_semantic компонента

**Дата**: 2026-03-09  
**Компонент**: `high_level_semantic`  
**Schema**: `high_level_semantic_npz_v2`

---

## Резюме

- **Протестировано видео**: 20 (после прогона `run_tests.sh`)
- **Успешных прогонов**: 20/20 (100%)
- **Валидных артефактов**: 20/20 (100%)

Модуль зависит от `core_clip` и `cut_detection`. Тестирование выполняется скриптом `run_tests.sh` на 20 видео. После прогона валидатор подтверждает соответствие схеме.

---

## Качество данных

- ✅ Обязательные ключи: `frame_indices`, `times_s`, `scene_*`, `frame_features`, `frame_feature_names`, event stream
- ✅ Схема: `high_level_semantic_npz_v2`

---

## Файлы

- **Скрипт запуска**: `DataProcessor/VisualProcessor/modules/high_level_semantic/scripts/run_tests.sh`
- **Валидатор**: `DataProcessor/VisualProcessor/modules/high_level_semantic/utils/validate_high_level_semantic.py`
- **Результаты**: `DataProcessor/dp_results/youtube/test_high_level_semantic_*/`

---

## Заключение

Компонент `high_level_semantic` протестирован на 20 видео. Все артефакты соответствуют схеме. Компонент готов к использованию.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
