# Отчёт о тестировании behavioral компонента

**Дата**: 2026-03-09  
**Компонент**: `behavioral`  
**Schema**: `behavioral_npz_v1`

---

## Резюме

✅ **Все тесты пройдены успешно**

- **Протестировано видео**: 20
- **Успешных прогонов**: 20/20 (100%)
- **Валидных артефактов**: 20/20 (100%)

Все артефакты соответствуют схеме `behavioral_npz_v1`. Ошибка валидации для `hand_gestures` (пустые списки → 2D массив) исправлена: используется явный 1D object array.

---

## Качество данных

- ✅ Обязательные ключи: `frame_indices`, `times_s`, `landmarks_present`, `hand_gestures`, `frame_results`, `aggregated`, все `seq_*`
- ✅ Размеры массивов согласованы (N=250 кадров)
- ✅ Render контексты созданы для всех видео

---

## Файлы

- **Скрипт запуска**: `DataProcessor/VisualProcessor/modules/behavioral/scripts/run_tests.sh`
- **Валидатор**: `DataProcessor/VisualProcessor/modules/behavioral/utils/validate_behavioral.py` (при наличии)
- **Результаты**: `DataProcessor/dp_results/youtube/test_behavioral_*/`

---

## Заключение

Компонент `behavioral` успешно протестирован на 20 видео. Все артефакты валидны. Компонент готов к использованию.
