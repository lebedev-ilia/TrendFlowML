# Быстрая проверка качества action_recognition

## 🚀 Быстрый старт (5 минут)

### 1. Запустить автоматические проверки

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
bash scripts/run_all_quality_checks.sh
```

Это покажет:
- ✅ Валидность всех артефактов
- 📊 Статистику по метрикам
- ⚠️ Предупреждения и рекомендации

### 2. Открыть HTML рендеры (визуальная проверка)

```bash
# Откройте несколько рендеров в браузере:
firefox dp_results/youtube/test_action_recognition_v3/test_action_recognition_v3/action_recognition/_render/render.html &
firefox dp_results/youtube/test_action_recognition_v8/test_action_recognition_v8/action_recognition/_render/render.html &
firefox dp_results/youtube/test_action_recognition_v15/test_action_recognition_v15/action_recognition/_render/render.html &
```

**Что проверить:**
- ✅ Summary Metrics выглядят разумно
- ✅ Timeline графики показывают изменения
- ✅ Stability bar chart визуализирован
- ✅ Top jumps таблица заполнена

### 3. Проверить конкретное видео вручную

```bash
# Выберите одно видео и откройте:
# 1. Видео файл в плеере
# 2. Соответствующий HTML рендер
# 3. Сопоставьте временные метки треков с визуальными событиями
```

---

## 📋 Детальная проверка (15-30 минут)

### Шаг 1: Статистический анализ

```bash
cd /media/ilya/Новый том/TrendFlowML/DataProcessor
.data_venv/bin/python3 VisualProcessor/modules/action_recognition/analyze_all_results.py
```

**Обратите внимание на:**
- Количество треков на видео (должно быть разумным)
- Распределение метрик (нет выбросов)
- Предупреждения в конце отчёта

### Шаг 2: Валидация конкретных файлов

```bash
# Проверить несколько файлов детально:
.data_venv/bin/python3 VisualProcessor/modules/action_recognition/validate_action_recognition.py \
  dp_results/youtube/test_action_recognition_v8/test_action_recognition_v8/action_recognition/action_recognition_features.npz -v
```

**Проверьте:**
- ✅ Нет errors
- ⚠️ Количество warnings минимально
- 📊 Metrics distribution выглядит разумно

### Шаг 3: Сравнение между видео

Откройте HTML рендеры для:
- **Короткого видео** (v1, v2) - проверьте базовую функциональность
- **Среднего видео** (v8, v9) - проверьте качество метрик
- **Длинного видео** (v15, v18) - проверьте стабильность на длинных видео

**Сравните:**
- Количество треков vs длительность
- Средние метрики стабильности
- Распределение temporal jumps

---

## 🔍 Что проверить в рендере

### Summary Metrics
- `Total Tracks` - соответствует ли количеству людей в видео?
- `Total Clips` - разумное ли количество?
- `Avg Stability` - в диапазоне [0, 1]?

### Timeline: Embedding Norms
- График показывает изменения?
- Есть ли резкие скачки (смены действий)?
- Несколько треков видны?

### Top 10 Clips with Highest Temporal Jumps
- Какие треки имеют наибольшие скачки?
- Времена скачков разумны?

### Tracks Details
- Все треки имеют корректные метрики?
- `num_clips` > 1 для длинных треков?

---

## ⚠️ Текущие наблюдения

Анализ показал:
- **Все треки имеют num_clips=1** - это означает, что треки слишком короткие
- **Temporal jumps = 0** - нет соседних клипов для сравнения
- **Stability = 1.0** - максимальная стабильность (ожидаемо для 1 клипа)

**Это нормально для:**
- Коротких видео
- Редких person детекций
- Коротких сегментов между разрывами

**Для более информативных метрик нужно:**
- Видео с длинными непрерывными треками людей
- Настройка `segment_gap_sec` для объединения близких сегментов
- Увеличение `clip_len` или уменьшение `stride` для большего количества клипов

---

## 📝 Чеклист качества

- [ ] Все артефакты валидны (запустить валидатор)
- [ ] HTML рендеры открываются и показывают данные
- [ ] Метрики в разумных диапазонах
- [ ] Количество треков соответствует видео
- [ ] Временные метки согласованы
- [ ] Нет критических ошибок в логах

---

## 🛠️ Дополнительные инструменты

### Детальный анализ одного файла

```bash
.data_venv/bin/python3 VisualProcessor/modules/action_recognition/validate_action_recognition.py \
  dp_results/youtube/test_action_recognition_v8/test_action_recognition_v8/action_recognition/action_recognition_features.npz \
  --json > analysis.json
```

### Сравнение метрик между видео

```bash
# Создать таблицу сравнения
.data_venv/bin/python3 -c "
import numpy as np
import sys
sys.path.insert(0, 'VisualProcessor')
from modules.action_recognition.analyze_all_results import analyze_all_results
result = analyze_all_results()
print('Video ID | Tracks | Clips | Avg Stability')
print('-' * 50)
for vid in result['per_video']:
    print(f\"{vid['video_id']:30} | {vid['tracks_count']:6} | {vid['total_clips']:5} | {result['summary']['stability']['mean']:.3f}\")
"
```

---

## 📚 Дополнительная документация

- **QUALITY_CHECK_GUIDE.md** - полное руководство по проверке качества
- **TESTING_REPORT.md** - отчёт о тестировании
- **README.md** - документация компонента
- **SCHEMA.md** - описание схемы данных

---

## ❓ Вопросы для самопроверки

1. **Соответствуют ли треки визуальным событиям?**
   - Откройте видео и рендер, сопоставьте временные метки

2. **Разумно ли количество треков?**
   - Сравните с количеством person детекций в core_object_detections

3. **Корректны ли метрики?**
   - Stability должна быть высокой для статичных действий
   - Temporal jumps должны быть выше на местах смены действий

4. **Нет ли выбросов?**
   - Проверьте распределение метрик в analyze_all_results.py

5. **Работает ли компонент на разных типах видео?**
   - Короткие, средние, длинные
   - С одним человеком, с несколькими
   - Со статичными и динамичными сценами

