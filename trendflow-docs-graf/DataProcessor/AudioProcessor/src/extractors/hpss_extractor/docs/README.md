## `hpss_extractor` (Audio Tier‑1, optional)

### Назначение

Извлекает **Harmonic-Percussive Source Separation (HPSS)** признаки — разложение аудио на гармоническую и перкуссионную компоненты. Вычисляет доли энергии каждой компоненты, спектральные фичи из разделённых компонент, и опционально сохраняет восстановленные временные сигналы.

**Версия**: 2.1.1 (Audit v4.2 observability)  
**Категория**: source_separation  
**GPU**: не требуется  
**Схема**: `hpss_extractor_npz_v1`

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** family `hpss` (для `run_segments()`)

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/hpss_extractor/hpss_extractor_features.npz` (**фиксированное имя**)

Схема: `hpss_extractor_npz_v1` (см. `schemas/hpss_extractor_npz_v1.json`, `SCHEMA.md`).

#### Audit v4 — NPZ (ключи)

| Группа | Заметка |
|--------|---------|
| `feature_names` / `feature_values` | Зависят от `meta.features_enabled`; порядок — реализация `npz_savers/hpss.py`. |
| Сегменты | `segment_*`, `segment_mask`, `hpss_*_share_by_segment` — строгое **N**. |
| Ряды долей | `hpss_*_share_series` — только если реально вычислены (**legacy `run()`**); в **`run_segments()`** ключи могут быть пустыми `(0,)`, а `waveforms`/`time_series` **не** попадают в `features_enabled`. |
| `meta` | `hpss_dominance`, `hpss_contract_version`, `features_enabled`, … |

#### Audit v4.2 observability:
- `meta.stage_timings_ms` (dict): покомпонентные тайминги (ms), пишутся в NPZ meta
- `meta.hpss_resource_profile` (dict|None): best-effort snapshot RSS/VMS/VRAM (если включено)
  - включение: `AP_HPSS_RESOURCE_PROFILE=1`

**`run_segments()` (Audit v3):** глобальные энергии и спектральные скаляры в tabular — **средние по успешным окнам** (после фикса агрегации); до фикса при включённых флагах часть полей давала **NaN** в tabular из-за отсутствия ключей в payload.

#### Полезные поля payload:

**Энергетические метрики** (включены по умолчанию, отключить: `--hpss-disable-energy-metrics`):
- `hpss_harmonic_share`: доля гармонической энергии (float, 0.0-1.0)
- `hpss_percussive_share`: доля перкуссионной энергии (float, 0.0-1.0)
- `hpss_energy_total`: общая энергия спектра (float)
- `hpss_energy_harmonic`: энергия гармонической компоненты (float)
- `hpss_energy_percussive`: энергия перкуссионной компоненты (float)
- `hpss_harmonic_stability`: стабильность гармонической компоненты (float, 0.0-1.0)
- `hpss_percussive_stability`: стабильность перкуссионной компоненты (float, 0.0-1.0)
- `hpss_separation_quality`: качество разделения (float, 0.0-1.0)
- `hpss_balance_score`: баланс между компонентами (float, 0.0-1.0)
- `hpss_dominance`: доминирующая компонента (`"harmonic"`, `"percussive"`, `"mixed"`) — в NPZ обычно в **`meta`**, не в `feature_values`
- `hpss_harmonic_share_mean`: среднее значение harmonic share по сегментам (float, для `run_segments()`)
- `hpss_harmonic_share_std`: стандартное отклонение harmonic share по сегментам (float, для `run_segments()`)
- `hpss_percussive_share_mean`: среднее значение percussive share по сегментам (float, для `run_segments()`)
- `hpss_percussive_share_std`: стандартное отклонение percussive share по сегментам (float, для `run_segments()`)

**Спектральные фичи** (feature-gated: `--hpss-enable-spectral-features`):
- `hpss_harmonic_centroid_mean/std`: спектральный центроид гармонической компоненты (Hz)
- `hpss_harmonic_bandwidth_mean/std`: спектральная ширина полосы гармонической компоненты (Hz)
- `hpss_harmonic_rolloff_mean/std`: спектральный rolloff гармонической компоненты (Hz)
- `hpss_percussive_centroid_mean/std`: спектральный центроид перкуссионной компоненты (Hz)
- `hpss_percussive_bandwidth_mean/std`: спектральная ширина полосы перкуссионной компоненты (Hz)
- `hpss_percussive_rolloff_mean/std`: спектральный rolloff перкуссионной компоненты (Hz)

**Временные сигналы** (feature-gated: `--hpss-enable-waveforms`):
- `hpss_harmonic_npy`: относительный путь к сохраненному .npy файлу с гармоническим сигналом (`_artifacts/harmonic.npy`)
- `hpss_percussive_npy`: относительный путь к сохраненному .npy файлу с перкуссионным сигналом (`_artifacts/percussive.npy`)
- `hpss_waveform_length`: длина восстановленных сигналов в сэмплах (int)

**Временные серии** (feature-gated: `--hpss-enable-time-series`):
- `hpss_harmonic_share_series`: временная серия доли гармонической энергии (float32[N] или путь к .npy)
- `hpss_percussive_share_series`: временная серия доли перкуссионной энергии (float32[N] или путь к .npy)
- `hpss_harmonic_share_series_npy`: относительный путь к сохраненному .npy файлу с гармонической share series (если размер > threshold)
- `hpss_percussive_share_series_npy`: относительный путь к сохраненному .npy файлу с перкуссионной share series (если размер > threshold)

**Метаданные**:
- `sample_rate`: частота дискретизации (int)
- `n_fft`: размер FFT окна (int)
- `hop_length`: размер hop для STFT (int)
- `duration`: длительность аудио в секундах (float)
- `device_used`: устройство обработки (str)
- `hpss_frames`: количество кадров в STFT (int)
- `hpss_kernel_size`: размер ядра для HPSS фильтрации (int)
- `hpss_margin`: отступ для границ HPSS (float)
- `hpss_power`: степень для нормализации HPSS (float)
- `segments_count`: количество обработанных сегментов (int, для `run_segments()`)
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: временные границы сегментов (float32[N])
- `segment_mask`: маска валидных сегментов (bool[N], false = failed)
- `hpss_harmonic_share_by_segment`, `hpss_percussive_share_by_segment`: per-segment shares (float32[N], NaN для failed)
- `hpss_contract_version`: версия контракта (`"hpss_contract_v1"`)
- `_features_enabled`: список включённых фичей (для feature gating)

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.

**Библиотеки**:
- **librosa**: основная библиотека для HPSS разложения (`librosa.decompose.hpss`) и спектральных фичей

### Конфигурация

```python
{
    "sample_rate": 22050,                    # Частота дискретизации
    "n_fft": 2048,                          # Размер FFT окна
    "hop_length": 512,                      # Размер hop для STFT
    "average_channels": True,               # Усреднять каналы для многоканального аудио
    "hpss_kernel_size": 31,                 # Размер ядра для HPSS фильтрации (должен быть нечётным)
    "hpss_margin": 1.0,                     # Отступ для границ HPSS
    "hpss_power": 2.0,                      # Степень для нормализации HPSS
    "enable_audio_normalization": False,    # Нормализация аудио перед обработкой
    "enable_energy_metrics": False,         # Включить энергетические метрики
    "enable_waveforms": False,              # Включить восстановленные временные сигналы
    "enable_spectral_features": False,      # Включить спектральные фичи из разделённых компонент
    "enable_time_series": False,            # Включить временные серии
    "device": "auto"                         # "auto" | "cuda" | "cpu"
}
```

### Feature Gating

- `--hpss-enable-energy-metrics`: энергетические метрики (default: True)
- `--hpss-disable-energy-metrics`: отключить энергетические метрики
- `--hpss-enable-waveforms`: восстановленные временные сигналы (harmonic и percussive)
- `--hpss-enable-spectral-features`: спектральные фичи из разделённых компонент (centroid, bandwidth, rolloff для H и P)
- `--hpss-enable-time-series`: временные серии (harmonic и percussive share series)

### Feature Dependencies

- **Энергетические метрики** не зависят от других фичей
- **Спектральные фичи** требуют вычисления HPSS разложения (автоматически выполняется)
- **Waveforms** требуют вычисления HPSS разложения (автоматически выполняется)
- **Временные серии** требуют включения энергетических метрик (для вычисления shares по времени)

### Алгоритм работы

1. **Загрузка аудио**: Audit v3 — только `run_segments()` по `families.hpss`; `run()` отключён. Загрузка через `AudioUtils.load_audio_segment()` с ресемплированием до `sample_rate`
2. **Нормализация аудио** (опционально): если `enable_audio_normalization=True`, нормализация по максимуму
3. **Сведение в моно**: если аудио стерео и `average_channels=True`, усреднение каналов
4. **Вычисление STFT**: через `librosa.stft()` с параметрами `n_fft` и `hop_length`
5. **HPSS разложение** (fail-fast, no-fallback): через `librosa.decompose.hpss()` на спектрограмме
   - Разделяет спектрограмму на гармоническую (H) и перкуссионную (P) компоненты
   - Гармоническая: горизонтальные структуры (стабильные тона)
   - Перкуссионная: вертикальные структуры (удары, перкуссия)
   - Параметры: `kernel_size`, `margin`, `power` (fail-fast при ошибке)
6. **Вычисление метрик**:
   - Энергетические метрики (если `enable_energy_metrics=True`): shares, energies, stability, separation quality, balance score, dominance
   - Спектральные фичи (если `enable_spectral_features=True`): centroid, bandwidth, rolloff для H и P отдельно
7. **Восстановление сигналов** (если `enable_waveforms=True`):
   - Использование фазы из исходного STFT
   - Обратное STFT для гармонической и перкуссионной компонент
   - Сохранение в .npy файлы в per-run storage (`_artifacts/harmonic.npy`, `_artifacts/percussive.npy`)
8. **Временные серии** (если `enable_time_series=True`):
   - Сохранение shares по времени (в NPZ или .npy для больших массивов)

### Особенности

- **Segmenter contract**: поддерживает `run_segments()` для обработки сегментов из Segmenter (family `hpss`)
- **No-fallback policy**: fail-fast при ошибках HPSS разложения (no-fallback для всех операций)
- **Per-run storage**: waveforms и временные серии сохраняются в `_artifacts/` с фиксированными именами
- **Feature gating**: все фичи управляются отдельными флагами (opt-in)
- **Валидация**: полная валидация выходных данных (диапазоны, NaN/inf, консистентность) и параметров (fail-fast)
- **Progress reporting**: обновление прогресса для каждого этапа и сегмента (каждые 10%)
- **Contract versioning**: `hpss_contract_version="hpss_contract_v1"` для валидации совместимости

### Performance characteristics

**Resource costs**:
- **CPU**: O(N * log(N)) для STFT, где N — длина аудио
- **Память**: O(n_fft * frames) для спектрограммы
- **Estimated duration**: ~1.3 секунд для типичного аудио файла (полное аудио)
- **Per-segment**: ~0.1-0.3 секунды на сегмент (зависит от размера сегмента и включенных фичей)
- **Waveforms overhead**: если `enable_waveforms=True`, дополнительное время на обратное STFT (~+50-100%)
- **Spectral features overhead**: если `enable_spectral_features=True`, дополнительное время на вычисление спектральных фичей (~+20-30%)

**Единица обработки**: 
- `run()`: весь аудио файл (legacy mode)
- `run_segments()`: сегменты из Segmenter (family `hpss`)

**Batch processing**:
- **CPU parallelism**: поддерживается через `extract_batch_segments()` с `ThreadPoolExecutor`
- **Масштабирование**: линейное ускорение при увеличении количества CPU ядер (до лимита I/O)
- **Изоляция**: каждый файл обрабатывается изолированно, нет shared mutable state между файлами
- **Артефакты**: каждый файл имеет свой `artifacts_dir` для сохранения `.npy` файлов (per-run storage)

**Критически важные оптимизации для производительности**:
- **Segment parallelism**: **ОБЯЗАТЕЛЬНО** для ускорения обработки
  - По умолчанию: `segment_parallelism=1` (последовательная обработка) → **очень медленно**
  - Рекомендуется: `segment_parallelism=4-8` для CPU
  - Ожидаемое ускорение: **~4-8x** при `segment_parallelism=8` на многоядерном CPU
  - Пример: 20 сегментов × 0.2 сек = 4 сек → с `segment_parallelism=8`: ~0.5-1 сек

### Parallelization

#### Внутренний параллелизм (внутри компонента)

- **Segment-level parallelism**: поддерживается через `segment_parallelism` и `max_inflight` (для `run_segments()`)
  - Использует `ThreadPoolExecutor` для параллельной обработки сегментов внутри одного файла
  - Количество потоков контролируется параметром `segment_parallelism` (по умолчанию 1)
  - Thread-safety: экстрактор thread-safe для параллельной обработки сегментов
  - Ограничения: параллелизм ограничен I/O операциями (загрузка аудио) и CPU вычислениями (STFT, HPSS)

#### Внешний параллелизм (выше компонента)

- **Batch processing**: поддерживается через `extract_batch_segments()` с CPU parallelism для обработки сегментов из нескольких видео одновременно
  - Использует `ThreadPoolExecutor` для параллельной обработки файлов
  - Количество воркеров контролируется параметром `max_workers` (по умолчанию `os.cpu_count()`)
  - Каждый файл обрабатывается изолированно через `run_segments()`
  - Изоляция артефактов: каждый файл имеет свой `artifacts_dir` для сохранения `.npy` файлов
  - Требования к изоляции: разные `run_id`, разные `result_store` пути, разные `artifacts_dir`
  - Ограничения: нет shared mutable state между файлами (кроме read-only моделей/корпусов)

#### Комбинированный подход

- Внутренний параллелизм (сегменты внутри файла) + внешний параллелизм (несколько файлов)
- Рекомендуется: использовать внешний параллелизм для обработки нескольких файлов, внутренний параллелизм для обработки сегментов внутри одного файла
- GPU: не используется (CPU-only)

### Примеры использования

#### Загрузка результатов

```python
import numpy as np

data = np.load("hpss_extractor_features.npz", allow_pickle=True)
payload = data["payload"].item()

# Энергетические метрики
harmonic_share = payload["hpss_harmonic_share"]
percussive_share = payload["hpss_percussive_share"]
total_energy = payload["hpss_energy_total"]
balance_score = payload["hpss_balance_score"]
dominance = payload["hpss_dominance"]

print(f"Harmonic share: {harmonic_share:.3f}")
print(f"Percussive share: {percussive_share:.3f}")
print(f"Balance score: {balance_score:.3f}")
print(f"Dominance: {dominance}")
```

#### Анализ структуры аудио

```python
# Определить доминирующий тип контента
if dominance == "harmonic":
    print("Доминирует гармонический контент (музыка, вокал)")
elif dominance == "percussive":
    print("Доминирует перкуссионный контент (удары, ритм)")
else:
    print("Смешанный контент")
```

#### Загрузка восстановленных сигналов

```python
# Если enable_waveforms=True
if "hpss_harmonic_npy" in payload:
    harmonic_npy_path = payload["hpss_harmonic_npy"]
    # Загрузка из per-run storage
    harmonic_wav = np.load(f"result_store/.../hpss_extractor/{harmonic_npy_path}")
    percussive_wav = np.load(f"result_store/.../hpss_extractor/{payload['hpss_percussive_npy']}")
    
    # Воспроизведение или дальнейшая обработка
    # harmonic_wav содержит только гармонические компоненты
    # percussive_wav содержит только перкуссионные компоненты
```

#### Спектральные фичи

```python
# Если enable_spectral_features=True
harmonic_centroid = payload["hpss_harmonic_centroid_mean"]
percussive_centroid = payload["hpss_percussive_centroid_mean"]

print(f"Harmonic centroid: {harmonic_centroid:.2f} Hz")
print(f"Percussive centroid: {percussive_centroid:.2f} Hz")
```

### Связанные компоненты

- **Segmenter**: предоставляет аудио файл и сегменты (family `hpss`)
- **librosa**: основная библиотека для HPSS и STFT
- **Другие экстракторы**: могут использовать результаты HPSS для дальнейшего анализа

### Visualization

**Рекомендуемые типы визуализации**:

1. **Timeline с shares**:
   - Линейный график с двумя линиями: harmonic share и percussive share по времени
   - Ось X: время (секунды или frame index)
   - Ось Y: share (0.0-1.0)
   - Интерактивные элементы: tooltips с точными значениями, zoom, фильтры по времени

2. **Распределение shares**:
   - Гистограмма или box plot для harmonic и percussive shares
   - Показывает вариацию shares по времени

3. **Spectral features comparison**:
   - Сравнительные графики для harmonic и percussive компонент (centroid, bandwidth, rolloff)
   - Bar charts или line charts для mean/std значений

4. **Balance score и dominance**:
   - Gauge chart или pie chart для визуализации баланса между компонентами
   - Цветовая индикация доминирующей компоненты

5. **Separation quality**:
   - Gauge chart для визуализации качества разделения (0.0-1.0)

**HTML renderer**: доступен через `render_hpss_extractor_html()` для локального дебага (debug-only, не в production артефактах)

### Примечания

1. **HPSS алгоритм**: использует медианную фильтрацию в частотной и временной областях для разделения компонент
2. **No-fallback policy**: все ошибки обрабатываются fail-fast с детальными error codes
3. **Waveforms**: восстановленные сигналы требуют дополнительной памяти и времени обработки
4. **Энергетические метрики**: простые и эффективные для анализа структуры аудио
5. **Спектральные фичи**: позволяют анализировать характеристики разделённых компонент отдельно
6. **Моно обработка**: стерео аудио сводится в моно (если `average_channels=True`)
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
