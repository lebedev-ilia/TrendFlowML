# Рекомендации по улучшению логики и алгоритмов behavioral компонента

**Дата**: 2026-02-27  
**Основано на**: Анализе результатов тестирования на 19 видео

---

## 1. Критические рекомендации

### 1.1 Обработка низкого покрытия landmarks

**Проблема**: Среднее покрытие landmarks составляет всего 9.22%, что приводит к:
- Низким valid_ratios для sequence features (< 10% для многих метрик)
- Потенциально ненадежным агрегированным метрикам при малом количестве данных

**Рекомендации**:

1. **Добавить quality flags в aggregated**:
   ```python
   aggregated['data_quality'] = {
       'landmarks_present_ratio': landmarks_present_ratio,
       'min_valid_samples_required': 10,  # минимум для надежности
       'is_reliable': landmarks_present_ratio > 0.1,  # флаг надежности
   }
   ```

2. **Использовать weighted aggregation** вместо простого mean:
   - Взвешивать метрики по количеству валидных кадров
   - Учитывать временное распределение валидных кадров (равномерность)

3. **Добавить confidence intervals** для агрегированных метрик:
   ```python
   aggregated['avg_engagement_ci'] = {
       'mean': avg_engagement,
       'std': std_engagement,
       'n_samples': valid_count,
       'ci_95': (mean - 1.96*std/sqrt(n), mean + 1.96*std/sqrt(n))
   }
   ```

### 1.2 Нормализация сигналов для engagement/confidence/stress

**Проблема**: Текущие формулы используют фиксированные веса без учета масштаба входных сигналов:
- `arm_openness` может иметь значения > 1.0 (видно в тестах: до 2.0)
- `body_lean_angle` часто = 1.0 (мало вариативности)
- `speech_activity_proxy` в диапазоне [0.5, 0.95], но используется как есть

**Рекомендации**:

1. **Нормализовать входные сигналы перед агрегацией**:
   ```python
   # Вместо прямого использования
   engagement_signal = 0.5 * speech_proxy + 0.3 * norm_sig(arm_open) + 0.2 * norm_sig(body_lean)
   
   # Использовать нормализованные версии
   arm_open_norm = np.clip(arm_open / 2.0, 0, 1)  # нормализация к [0, 1]
   body_lean_norm = np.clip(body_lean, 0, 1)  # уже в [0, 1], но проверить
   speech_norm = (speech_proxy - 0.5) / 0.5  # нормализация к [0, 1] из [0.5, 1.0]
   
   engagement_signal = 0.5 * speech_norm + 0.3 * arm_open_norm + 0.2 * body_lean_norm
   ```

2. **Использовать adaptive weights** на основе валидности данных:
   ```python
   # Веса зависят от количества валидных кадров для каждого сигнала
   speech_weight = valid_speech_ratio / (valid_speech_ratio + valid_arm_ratio + valid_body_ratio)
   arm_weight = valid_arm_ratio / (valid_speech_ratio + valid_arm_ratio + valid_body_ratio)
   body_weight = valid_body_ratio / (valid_speech_ratio + valid_arm_ratio + valid_body_ratio)
   ```

### 1.3 Улучшение детекции стресса

**Проблема**: 
- `blink_rate_short` часто = 0.0 (редкое моргание на sparse sampling)
- `self_touch_flag` бинарный, не учитывает частоту
- `fidgeting_energy` имеет очень низкие значения (0-0.01)

**Рекомендации**:

1. **Улучшить детекцию моргания**:
   - Использовать более длинное окно для blink_rate (не только последние 30 кадров)
   - Добавить temporal smoothing для EAR (Eye Aspect Ratio)
   - Учитывать частоту кадров при вычислении blink_rate

2. **Сделать self_touch непрерывной метрикой**:
   ```python
   # Вместо бинарного флага
   self_touch_intensity = calculate_self_touch_intensity(hand_landmarks, pose_landmarks)
   # Возвращает значение [0, 1] вместо 0/1
   ```

3. **Нормализовать fidgeting_energy**:
   - Текущие значения слишком малы (0-0.01)
   - Использовать log-scale или другую нормализацию
   - Добавить пороги для разных уровней fidgeting

---

## 2. Улучшения алгоритмов агрегации

### 2.1 Temporal segmentation для early/late метрик

**Проблема**: Текущая логика использует фиксированные 20% от начала и конца:
```python
split = max(int(0.2 * n), 1)
early = engagement_signal[:split]
late = engagement_signal[-split:]
```

**Рекомендации**:

1. **Использовать временные сегменты вместо кадров**:
   ```python
   # Использовать times_s для реального времени
   duration = times_s[-1] - times_s[0]
   early_threshold = duration * 0.2  # первые 20% времени
   late_threshold = duration * 0.8  # последние 20% времени
   
   early_mask = times_s <= early_threshold
   late_mask = times_s >= late_threshold
   ```

2. **Добавить middle segment** для анализа динамики:
   ```python
   aggregated['middle_engagement_mean'] = float(np.nanmean(engagement_signal[middle_mask]))
   aggregated['engagement_trend'] = 'increasing' if late_mean > early_mean else 'decreasing'
   ```

### 2.2 Улучшение детекции peaks

**Проблема**: Текущий алгоритм слишком простой (только локальные максимумы):
```python
if engagement_signal[i] > engagement_signal[i - 1] and engagement_signal[i] > engagement_signal[i + 1]:
    engagement_peaks += 1
```

**Рекомендации**:

1. **Добавить пороги для значимости peaks**:
   ```python
   mean_val = np.nanmean(engagement_signal)
   std_val = np.nanstd(engagement_signal)
   threshold = mean_val + 0.5 * std_val  # peak должен быть выше среднего + 0.5*std
   
   for i in range(1, engagement_signal.size - 1):
       if (engagement_signal[i] > engagement_signal[i - 1] and 
           engagement_signal[i] > engagement_signal[i + 1] and
           engagement_signal[i] > threshold):
           engagement_peaks += 1
   ```

2. **Использовать scipy.signal.find_peaks** для более надежной детекции:
   ```python
   from scipy.signal import find_peaks
   peaks, properties = find_peaks(engagement_signal, 
                                  height=mean_val + 0.5*std_val,
                                  distance=5)  # минимум 5 кадров между peaks
   ```

### 2.3 Улучшение gesture rate calculation

**Проблема**: Текущий расчет использует общее количество жестов / длительность:
```python
aggregated['gesture_rate_per_sec'] = float(len(all_gestures) / duration_sec)
```

**Рекомендации**:

1. **Учитывать только кадры с валидными landmarks**:
   ```python
   valid_frames = np.sum(landmarks_present)
   valid_duration = duration_sec * (valid_frames / total_frames)
   gesture_rate = len(all_gestures) / max(valid_duration, 1e-6)
   ```

2. **Добавить gesture density по времени** (не только rate):
   ```python
   # Плотность жестов в активных сегментах
   gesture_times = [times_s[i] for i, present in enumerate(landmarks_present) if present]
   if gesture_times:
       gesture_density = len(all_gestures) / (max(gesture_times) - min(gesture_times))
   ```

---

## 3. Улучшения sequence features

### 3.1 Обработка NaN и missing data

**Проблема**: Много NaN значений из-за низкого покрытия landmarks

**Рекомендации**:

1. **Добавить temporal interpolation** для последовательных NaN:
   ```python
   # Интерполировать только короткие пропуски (< 5 кадров)
   from scipy.interpolate import interp1d
   
   valid_mask = np.isfinite(seq_feature)
   if np.sum(valid_mask) >= 2:
       valid_indices = np.where(valid_mask)[0]
       valid_values = seq_feature[valid_mask]
       interp_func = interp1d(valid_indices, valid_values, 
                              kind='linear', 
                              fill_value='extrapolate',
                              bounds_error=False)
       # Интерполировать только короткие пропуски
       for i in range(len(seq_feature)):
           if not valid_mask[i]:
               # Проверить, что пропуск короткий
               gap_size = count_gap_size(i, valid_mask)
               if gap_size < 5:
                   seq_feature[i] = interp_func(i)
   ```

2. **Добавить forward/backward fill** для edge cases:
   ```python
   # Заполнить начальные/конечные NaN последним/первым валидным значением
   seq_feature = pd.Series(seq_feature).fillna(method='ffill').fillna(method='bfill')
   ```

### 3.2 Улучшение speech_activity_proxy

**Проблема**: Valid ratio очень низкий (< 10% для многих видео), значения в узком диапазоне [0.5, 0.95]

**Рекомендации**:

1. **Расширить детекцию речи**:
   - Использовать не только mouth dynamics, но и head motion
   - Добавить audio features если доступны (через AudioProcessor)
   - Использовать более длинное окно для анализа mouth velocity

2. **Улучшить нормализацию**:
   ```python
   # Текущий диапазон [0.5, 0.95] слишком узкий
   # Нормализовать к [0, 1] для лучшей интерпретации
   speech_normalized = (speech_activity_proxy - 0.5) / 0.45  # [0.5, 0.95] -> [0, 1]
   ```

---

## 4. Улучшения классификации жестов

### 4.1 Soft representation для жестов

**Текущее состояние**: Используется soft representation (gesture_probs), но также есть hard classification

**Рекомендации**:

1. **Использовать только soft representation** для sequence features:
   - Убрать hard classification из основных метрик
   - Использовать gesture_probs напрямую для engagement/confidence расчетов

2. **Добавить gesture transitions**:
   ```python
   # Анализ переходов между жестами
   gesture_transitions = []
   for i in range(1, len(gesture_sequence)):
       prev_gesture = get_dominant_gesture(gesture_probs[i-1])
       curr_gesture = get_dominant_gesture(gesture_probs[i])
       if prev_gesture != curr_gesture:
           gesture_transitions.append({
               'time': times_s[i],
               'from': prev_gesture,
               'to': curr_gesture
           })
   aggregated['gesture_transition_count'] = len(gesture_transitions)
   aggregated['gesture_transition_rate'] = len(gesture_transitions) / duration_sec
   ```

### 4.2 Улучшение классификации self_touch

**Проблема**: Self_touch детектируется только через жест, может пропускать другие формы

**Рекомендации**:

1. **Добавить геометрический анализ**:
   ```python
   # Проверить расстояние между руками и телом
   hand_to_body_distance = calculate_hand_to_body_distance(hand_landmarks, pose_landmarks)
   if hand_to_body_distance < threshold:
       self_touch_intensity = 1.0 - (hand_to_body_distance / threshold)
   ```

2. **Учитывать частоту и длительность**:
   ```python
   # Вместо бинарного флага, использовать интенсивность и длительность
   self_touch_duration = count_consecutive_frames_with_self_touch()
   self_touch_frequency = count_self_touch_episodes() / duration_sec
   ```

---

## 5. Статистические улучшения

### 5.1 Добавить robust statistics

**Рекомендации**:

1. **Использовать median вместо mean** для устойчивости к выбросам:
   ```python
   aggregated['median_engagement'] = float(np.nanmedian(engagement_signal))
   aggregated['engagement_iqr'] = float(np.nanpercentile(engagement_signal, 75) - 
                                        np.nanpercentile(engagement_signal, 25))
   ```

2. **Добавить percentiles**:
   ```python
   aggregated['engagement_percentiles'] = {
       'p25': float(np.nanpercentile(engagement_signal, 25)),
       'p50': float(np.nanpercentile(engagement_signal, 50)),
       'p75': float(np.nanpercentile(engagement_signal, 75)),
       'p90': float(np.nanpercentile(engagement_signal, 90)),
   }
   ```

### 5.2 Добавить distribution analysis

**Рекомендации**:

1. **Анализ распределения метрик**:
   ```python
   # Проверить нормальность распределения
   from scipy.stats import shapiro, normaltest
   
   valid_engagement = engagement_signal[np.isfinite(engagement_signal)]
   if len(valid_engagement) > 3:
       stat, p_value = normaltest(valid_engagement)
       aggregated['engagement_distribution'] = {
           'is_normal': p_value > 0.05,
           'skewness': float(scipy.stats.skew(valid_engagement)),
           'kurtosis': float(scipy.stats.kurtosis(valid_engagement)),
       }
   ```

---

## 6. Производительность и оптимизация

### 6.1 Кэширование вычислений

**Рекомендации**:

1. **Кэшировать нормализованные сигналы**:
   ```python
   # Вычислять нормализацию один раз
   self._cached_normalizations = {}
   ```

2. **Векторизовать операции**:
   ```python
   # Использовать numpy vectorized operations вместо циклов
   engagement_signal = (0.5 * speech_proxy + 
                        0.3 * norm_sig_vectorized(arm_open) + 
                        0.2 * norm_sig_vectorized(body_lean))
   ```

### 6.2 Оптимизация gesture classification

**Рекомендации**:

1. **Batch processing для gesture classification**:
   - Классифицировать жесты для всех кадров сразу, если возможно
   - Использовать vectorized operations для finger states

---

## 7. Документация и конфигурируемость

### 7.1 Вынести веса в конфиг

**Рекомендации**:

1. **Сделать веса конфигурируемыми**:
   ```yaml
   behavioral:
     aggregation_weights:
       engagement:
         speech_activity: 0.5
         arm_openness: 0.3
         body_lean: 0.2
       confidence:
         arm_openness: 0.6
         body_lean: 0.4
       stress:
         blink_rate: 0.4
         self_touch: 0.3
         fidgeting: 0.3
   ```

2. **Добавить параметры для порогов**:
   ```yaml
   behavioral:
     thresholds:
       blink_ear_threshold: 0.2
       self_touch_distance_threshold: 0.1
       peak_significance_std_multiplier: 0.5
   ```

---

## 8. Приоритеты внедрения

### Высокий приоритет (критично для качества)
1. ✅ Обработка низкого покрытия landmarks (quality flags)
2. ✅ Нормализация сигналов для engagement/confidence
3. ✅ Улучшение детекции стресса (blink_rate, self_touch)

### Средний приоритет (улучшение качества)
4. Temporal segmentation с использованием times_s
5. Улучшение детекции peaks
6. Обработка NaN с интерполяцией

### Низкий приоритет (оптимизация и расширение)
7. Robust statistics (median, percentiles)
8. Distribution analysis
9. Конфигурируемые веса
10. Производительность (кэширование, векторизация)

---

## 9. Примеры улучшенного кода

### Пример 1: Улучшенная агрегация engagement

```python
def _aggregate_engagement_improved(self, results, times_s, landmarks_present):
    """Улучшенная агрегация engagement с нормализацией и quality flags."""
    
    # Получить sequence features
    speech_proxy = self._get_series(results, 'speech_activity_proxy')
    arm_open = self._get_series(results, 'arm_openness')
    body_lean = self._get_series(results, 'body_lean_angle')
    
    # Нормализация сигналов
    speech_norm = np.clip((speech_proxy - 0.5) / 0.45, 0, 1)  # [0.5, 0.95] -> [0, 1]
    arm_norm = np.clip(arm_open / 2.0, 0, 1)  # [0, 2] -> [0, 1]
    body_norm = np.clip(body_lean, 0, 1)  # уже в [0, 1]
    
    # Adaptive weights на основе валидности
    valid_speech = np.sum(np.isfinite(speech_norm))
    valid_arm = np.sum(np.isfinite(arm_norm))
    valid_body = np.sum(np.isfinite(body_norm))
    total_valid = valid_speech + valid_arm + valid_body
    
    if total_valid > 0:
        w_speech = valid_speech / total_valid
        w_arm = valid_arm / total_valid
        w_body = valid_body / total_valid
    else:
        w_speech, w_arm, w_body = 0.5, 0.3, 0.2  # fallback
    
    # Вычислить engagement signal
    engagement_signal = (w_speech * np.nan_to_num(speech_norm, nan=0) +
                         w_arm * np.nan_to_num(arm_norm, nan=0) +
                         w_body * np.nan_to_num(body_norm, nan=0))
    
    # Quality assessment
    valid_count = np.sum(np.isfinite(engagement_signal))
    landmarks_ratio = np.mean(landmarks_present)
    
    aggregated = {
        'avg_engagement': float(np.nanmean(engagement_signal)),
        'median_engagement': float(np.nanmedian(engagement_signal)),
        'engagement_std': float(np.nanstd(engagement_signal)),
        'engagement_quality': {
            'valid_samples': int(valid_count),
            'landmarks_ratio': float(landmarks_ratio),
            'is_reliable': valid_count >= 10 and landmarks_ratio > 0.1,
        }
    }
    
    # Temporal analysis с использованием times_s
    if times_s is not None and len(times_s) > 0:
        duration = times_s[-1] - times_s[0]
        if duration > 0:
            early_mask = times_s <= (times_s[0] + duration * 0.2)
            late_mask = times_s >= (times_s[0] + duration * 0.8)
            
            early_engagement = engagement_signal[early_mask]
            late_engagement = engagement_signal[late_mask]
            
            aggregated['early_engagement_mean'] = float(np.nanmean(early_engagement))
            aggregated['late_engagement_mean'] = float(np.nanmean(late_engagement))
            aggregated['engagement_trend'] = 'increasing' if (
                aggregated['late_engagement_mean'] > aggregated['early_engagement_mean']
            ) else 'decreasing'
    
    return aggregated
```

---

## 10. Заключение

Основные направления улучшений:
1. **Надежность**: Обработка низкого покрытия данных, quality flags
2. **Точность**: Нормализация сигналов, улучшенная детекция
3. **Информативность**: Robust statistics, temporal analysis
4. **Гибкость**: Конфигурируемые параметры

Рекомендуется начать с высокоприоритетных улучшений, которые критично влияют на качество метрик.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
