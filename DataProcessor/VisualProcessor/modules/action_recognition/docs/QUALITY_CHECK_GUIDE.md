# Руководство по проверке качества action_recognition компонента

Это руководство поможет вам самостоятельно проверить качество компонента `action_recognition` и его фичей.

---

## 1. Визуальная проверка через HTML рендеры

### Шаг 1: Открыть HTML рендеры

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
# Откройте в браузере:
firefox dp_results/youtube/test_action_recognition_shortest/test_action_recognition_shortest/action_recognition/_render/render.html
# или
google-chrome dp_results/youtube/test_action_recognition_*/test_action_recognition_*/action_recognition/_render/render.html
```

### Что проверить в рендере:

1. **Summary Metrics**:
   - Количество треков соответствует ожидаемому?
   - Total Clips разумное?
   - Avg Stability в диапазоне [0, 1]?

2. **Stability by Track** (bar chart):
   - Есть ли треки с низкой стабильностью (< 0.5)?
   - Распределение стабильности выглядит разумным?

3. **Timeline: Embedding Norms**:
   - График показывает изменения по времени?
   - Есть ли резкие скачки (возможные смены действий)?
   - Несколько треков видны одновременно?

4. **Top 10 Clips with Highest Temporal Jumps**:
   - Какие треки имеют наибольшие скачки?
   - Времена скачков соответствуют визуальным изменениям в видео?

5. **Tracks Details**:
   - Все треки имеют корректные метрики?
   - `num_clips` > 1 для треков с несколькими клипами?

---

## 2. Статистический анализ метрик

### Запустить скрипт анализа всех результатов:

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
.data_venv/bin/python3 VisualProcessor/modules/action_recognition/analyze_all_results.py
```

Этот скрипт покажет:
- Распределение метрик по всем видео
- Корреляции между метриками
- Выбросы и аномалии
- Сравнение между короткими и длинными видео

---

## 3. Проверка качества эмбеддингов

### Проверить нормализацию и распределение:

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
.data_venv/bin/python3 VisualProcessor/modules/action_recognition/check_embeddings_quality.py \
  dp_results/youtube/test_action_recognition_v8/test_action_recognition_v8/action_recognition/action_recognition_features.npz
```

Проверяет:
- L2 нормализация (должна быть ≈ 1.0)
- Распределение эмбеддингов (PCA визуализация)
- Кластеризация похожих действий
- Разнообразие эмбеддингов

---

## 4. Сравнение результатов между видео

### Создать сравнительный отчёт:

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
.data_venv/bin/python3 VisualProcessor/modules/action_recognition/compare_videos.py \
  --video-ids test_action_recognition_v3 test_action_recognition_v8 test_action_recognition_v15 \
  --output comparison_report.html
```

Сравнивает:
- Количество треков vs длительность видео
- Средние метрики стабильности
- Распределение temporal jumps
- Качество детекций (сколько person детекций найдено)

---

## 5. Визуальная проверка на конкретных видео

### Выбрать несколько видео и проверить вручную:

1. **Короткое видео** (12-30 сек):
   - Открыть видео в плеере
   - Открыть соответствующий HTML рендер
   - Сопоставить временные метки треков с визуальными событиями
   - Проверить, что треки соответствуют появлению людей

2. **Среднее видео** (30-120 сек):
   - Проверить, что треки не разрываются на короткие сегменты без причины
   - Проверить temporal jumps на местах смены действий
   - Убедиться, что stability корректна для длинных треков

3. **Длинное видео** (120+ сек):
   - Проверить, что компонент обработал всё видео
   - Проверить распределение треков по времени
   - Убедиться, что метрики не деградируют на длинных видео

---

## 6. Проверка edge cases

### Тестирование граничных случаев:

```bash
# Видео без людей (должно быть empty)
# Видео с множественными людьми
# Видео с быстрой сменой сцен
# Видео с длинными статичными сценами
```

Проверить:
- Корректная обработка empty cases
- Группировка person детекций в треки
- Разрыв сегментов при больших временных разрывах

---

## 7. Проверка согласованности с детекциями

### Сравнить с core_object_detections:

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
.data_venv/bin/python3 VisualProcessor/modules/action_recognition/verify_detections_alignment.py \
  dp_results/youtube/test_action_recognition_v8/test_action_recognition_v8/
```

Проверяет:
- Все person детекции использованы?
- Треки соответствуют последовательным детекциям?
- Временные метки согласованы?

---

## 8. Анализ производительности

### Проверить время обработки:

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
.data_venv/bin/python3 VisualProcessor/modules/action_recognition/analyze_performance.py
```

Показывает:
- Время обработки vs количество треков
- Время обработки vs длительность видео
- GPU utilization
- Память

---

## 9. Проверка метрик на репрезентативных примерах

### Выбрать видео с известными действиями:

1. **Видео с одним человеком**:
   - Проверить, что создан один длинный трек
   - Stability должна быть высокой для статичных действий
   - Temporal jumps должны быть низкими для плавных движений

2. **Видео с несколькими людьми**:
   - Проверить, что созданы отдельные треки для каждого человека
   - Треки не должны пересекаться по времени (если люди не взаимодействуют)

3. **Видео со сменой действий**:
   - Проверить, что temporal jumps высокие на местах смены действий
   - Stability должна быть ниже для видео с частыми сменами

---

## 10. Сравнение с ground truth (если доступно)

Если у вас есть разметка действий в видео:

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
.data_venv/bin/python3 VisualProcessor/modules/action_recognition/compare_with_ground_truth.py \
  --npz-path dp_results/youtube/test_action_recognition_v8/test_action_recognition_v8/action_recognition/action_recognition_features.npz \
  --ground-truth-path /path/to/ground_truth.json
```

Сравнивает:
- Временные метки треков с разметкой
- Количество действий
- Типы действий (если доступны)

---

## Чеклист для быстрой проверки

- [ ] Открыть 3-5 HTML рендеров разных видео
- [ ] Проверить, что все метрики в разумных диапазонах
- [ ] Убедиться, что temporal jumps соответствуют визуальным изменениям
- [ ] Проверить, что stability корректна для разных типов действий
- [ ] Запустить валидатор на всех результатах
- [ ] Сравнить результаты между похожими видео
- [ ] Проверить edge cases (пустые результаты, множественные люди)

---

## Автоматизированные проверки

Все автоматизированные проверки можно запустить одной командой:

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
bash scripts/run_all_quality_checks.sh
```

Этот скрипт запустит:
1. Валидацию всех артефактов
2. Статистический анализ
3. Проверку качества эмбеддингов
4. Сравнение между видео
5. Генерацию сводного отчёта

---

## Интерпретация результатов

### Хорошие признаки:
- ✅ Stability в диапазоне [0.5, 1.0] для большинства треков
- ✅ Temporal jumps > 0.1 на местах смены действий
- ✅ Количество треков пропорционально количеству person детекций
- ✅ Эмбеддинги нормализованы (L2 norm ≈ 1.0)
- ✅ Нет выбросов в метриках

### Плохие признаки:
- ❌ Все stability = 1.0 и temporal jumps = 0.0 (слишком короткие треки)
- ❌ Много треков с 1 клипом (недостаточно данных для анализа)
- ❌ Temporal jumps не соответствуют визуальным изменениям
- ❌ Эмбеддинги не нормализованы
- ❌ Много warnings в валидаторе

---

## Дополнительные ресурсы

- **README.md**: Полная документация компонента
- **SCHEMA.md**: Описание схемы данных
- **FEATURES_DESCRIPTION.md**: Описание всех фичей
- **TESTING_REPORT.md**: Отчёт о тестировании

