## `band_energy_extractor` (Audio signal processing extractor)

### Назначение

Извлекает **доли энергии по 3 фиксированным полосам** (низ/середина/высокие) из Segmenter-окон (shared family `spectral`). Audit v3: **librosa-only**, **no-fallback**, опционально возвращает segment-aligned последовательности shares с `segment_mask`.

**Версия**: 2.1.1 (Audit v4.2: stage timings + profiling + perf)  
**Категория**: spectral  
**GPU**: не требуется

**Audit v4:** [`DataProcessor/docs/audit_v4/components/audio_processor/band_energy_extractor_audit_v4.md`](../../../../../docs/audit_v4/components/audio_processor/band_energy_extractor_audit_v4.md) (уровень **L2** на **A+B**). Воспроизводимая статистика: `scripts/audit_v4_npz_stats.py` → JSON/`figures` (см. `RUN_LOG.md` и шапку отчёта).

### Каталог полей NPZ (Audit v4)

| Поле | Где | Как получают |
|------|-----|--------------|
| `band_edges_hz` | `float32[3,2]` | Канон: \([0,200), [200,2000), [2000, nyq]\) Hz при `sample_rate` |
| `band_energy_shares` | `float32[3]` | Энергия STFT в полосе / сумма энергий по всем полосам; **в `run_segments`** — усреднение долей по валидным сегментам (`nanmean`, маска коротких окон) |
| `band_share_{low,mid,high}` | tabular | Те же три числа, что и `band_energy_shares` (удобство FeatureSpec) |
| `segment_*`, `band_shares_by_segment` | опционально | `--band-energy-enable-time-series`: выравнивание по `families.spectral.segments`, `segment_mask=false` → NaN в строке |
| `band_balance_score`, `band_contrast`, `band_dominant_band` | опционально | `--band-energy-enable-balance-metrics`: энтропия/норма, max−min, argmax полосы (в tabular доминирующий индекс как float) |
| `meta.method`, `n_fft`, `hop_length`, `sample_rate` | meta | Параметры STFT / метод **только librosa** (audit v3) |
| `meta.stage_timings_ms` | meta | Тайминги стадий (мс): загрузка/нормализация/бэнды/валидация/total; в `run_segments` дополнительно `segments_*` |
| `meta.band_energy_resource_profile` | meta (optional) | Best-effort снимки RSS/VMS/GPU (env-gated) для отладки скорости/памяти: `AP_BAND_ENERGY_RESOURCE_PROFILE=1` |
| `meta.duration` | meta | В **`run()`** — длина клипа; в **`run_segments()`** — span по окнам Segmenter (max(end_sec)−min(start_sec)) |

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter) — полный аудио файл
- **`audio/segments.json`** (Segmenter contract `audio_segments_v1`) — для `run_segments()`:
  - **required family**: `families.spectral.segments[]` — shared sampling policy (Audit v3). Это не runtime fallback: это объявленное требование к Segmenter.

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/band_energy_extractor/band_energy_extractor_features.npz` (**фиксированное имя**)

Схема (Audit v3): `band_energy_extractor_npz_v1` (см. `schemas/band_energy_extractor_npz_v1.json` + `SCHEMA.md`).

#### Полезные поля payload:

**Основной результат (Audit v3, всегда включен):**
- `band_edges_hz`: границы полос `float32[3,2]` (Hz), канонично: low/mid/high
- `band_energy_shares`: доли энергии `float32[3]`, сумма = 1.0
- `feature_names/feature_values`: включает **только shares** как model_facing (`band_share_low/mid/high`)

**Segment-aligned sequences (feature-gated, analytics):**
- `segment_centers_sec`: центры сегментов (float32[N])
- `segment_durations_sec`: длительности сегментов (float32[N])
- `segment_mask`: маска валидных сегментов (bool[N]) — `false` если сегмент слишком короткий/ошибка
- `band_shares_by_segment`: доли по сегментам (float32[N,3]); для `segment_mask=false` значения = `NaN`

**Метрики баланса (feature-gated):**
- `band_balance_score`: оценка баланса между полосами (float, 0.0-1.0, энтропия распределения, если `enable_balance_metrics=True`)
- `band_dominance`: индекс доминирующей полосы (int, если `enable_balance_metrics=True`)
- `band_contrast`: контраст между полосами (float, max - min, если `enable_balance_metrics=True`)

**Примечание (Audit v3):**
- базовые/расширенные статистики и dynamics‑метрики **исключены** из audited контракта и приводят к fail-fast, если включены.

**Метаданные:**
- `sample_rate`: частота дискретизации (int)
- `n_fft`: размер FFT окна (int)
- `hop_length`: размер hop для STFT (int)
- `duration`: длительность аудио в секундах (float)
- `device_used`: устройство обработки (str)
- `band_energy_contract_version`: версия контракта ("band_energy_contract_v1")

### Feature Dependencies

**Зависимости между фичами (Audit v3):**
- Базовый результат: три доли и `band_edges` всегда после успешного прогона.
- `band_balance_score` / `band_contrast` / `band_dominance` только при `enable_balance_metrics=True`.
- Segment-aligned ряд только при `enable_time_series=True`.
- `enable_basic_stats` / `enable_extended_stats` / `enable_dynamics` **запрещены** в audited контракте (fail-fast).

**Зависимости от других extractors:**
- **spectral_extractor** (опционально): может использовать предвычисленный `stft_magnitude` и `frequencies` из `shared_features` для оптимизации (избегает повторного вычисления STFT)

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.

**Библиотеки:**
- **librosa**: единственный разрешённый путь в Audit v3 (`band_method` должен быть `librosa`).

### Конфигурация

```python
{
    "sample_rate": 22050,           # Частота дискретизации
    "bands": None,                   # Список полос [(lo, hi), ...] в Hz, по умолчанию: [(0, 200), (200, 2000), (2000, nyq)]
    "n_fft": 2048,                   # Размер FFT окна
    "hop_length": 512,              # Размер hop для STFT
    "use_mel_bands": False,         # Audit v3: только фиксированные 3 полосы
    "n_mels": 3,                    # (не используется в audited профиле)
    "band_method": "librosa",       # Audit v3: только librosa (no-fallback)
    "average_channels": True,        # Усреднять каналы для многоканального аудио
    "enable_audio_normalization": True,   # Audit v3: включено по умолчанию
    # Feature gating flags (Audit v3)
    "enable_time_series": False,     # Включить segment-aligned sequences (band_shares_by_segment + segment_mask)
    "enable_balance_metrics": False  # Включить метрики баланса
    "device": "auto"                 # "auto" | "cuda" | "cpu"
}
```

### Feature Gating

Все фичи контролируются через персональные флаги (по умолчанию все выключены, кроме базовых полей):

- `--band-energy-enable-time-series`: Включить segment-aligned sequences (`band_shares_by_segment`, `segment_mask`, `segment_centers_sec`, `segment_durations_sec`)
- `--band-energy-enable-balance-metrics`: Включить метрики баланса (`band_balance_score`, `band_dominance`, `band_contrast`)

**Зависимости фичей:**
- (Audit v3) dynamics/statistics не поддерживаются в audited профиле.

### Алгоритм работы

#### Метод (Audit v3, no-fallback):

1. **Только librosa** (fail-fast для `band_method != "librosa"`).

#### Librosa путь:

1. **Загрузка аудио**: через `AudioUtils.load_audio()` с ресемплированием до `sample_rate`
2. **Опциональная нормализация аудио**: если `enable_audio_normalization=True`
3. **Сведение в моно**: усреднение каналов для стерео аудио
4. **Проверка минимальной длительности**: fail-fast если аудио < 1 секунды
5. **Вычисление STFT**:
   - **Оптимизация**: если `shared_features` содержит `stft_magnitude`, используется он
   - **STFT**: вычисление спектрограммы мощности через `librosa.stft()`
6. **Частотные полосы (Audit v3)**:
   - **Фиксированные 3 полосы**: `[(0, 200), (200, 2000), (2000, nyq)]` Hz (канонично)
   - mel‑полосы в audited профиле **отключены** (fail-fast если `use_mel_bands=True`)
7. **Векторизованный биннинг**: эффективное вычисление энергий по полосам через матричное умножение масок
8. **Метрики баланса** (feature-gated, analytics): `band_balance_score`, `band_dominance`, `band_contrast`
9. **Временные последовательности** (feature-gated, analytics): `band_shares_by_segment` + `segment_mask`

#### Segment-based обработка (`run_segments()`, Audit v3):

1. **Загрузка сегментов**: из `families.spectral.segments[]` от Segmenter (shared family)
2. **Обработка каждого сегмента**: вычисление энергий по полосам для каждого сегмента
3. **Агрегация результатов**: усреднение энергий и shares по сегментам
4. **Strict alignment** (если `enable_time_series=True`): возвращаем массивы длины `N` (как в Segmenter), а “пропуски” кодируем через `segment_mask=false` и `NaN` в `band_shares_by_segment`.

### Особенности

- **Явный выбор метода**: контроль через `band_method` (essentia/librosa/auto), no-fallback policy
- **Векторизованный биннинг**: эффективное вычисление через матричные операции (librosa)
- **Гибкие полосы**: поддержка фиксированных полос или мел-шкалы
- **Shared STFT**: может использовать предвычисленный STFT из `shared_features` для оптимизации
- **Segment-based processing**: поддержка `run_segments()` для отслеживания изменений частотного баланса
- **Dynamics metrics**: метрики динамики для анализа изменений по времени
- **Balance metrics**: метрики баланса для анализа частотного распределения

### Error Handling

Детальные error codes:
- `audio_load_failed`: Ошибка загрузки аудио
- `audio_too_short`: Аудио слишком короткое (< 1 секунды)
- `stft_computation_failed`: Ошибка вычисления STFT
- `band_computation_failed`: Ошибка вычисления энергий по полосам
- (Audit v3) Essentia не используется.
- `invalid_parameters`: Невалидные параметры (bands, n_fft, и т.д.)

**No-fallback policy:**
- Отсутствие обязательного входа → fail-fast с `RuntimeError`
- Пустой список segments → `ValueError("segments is empty (no-fallback)")`
- Невалидные параметры → `ValueError` с описанием
- Ошибки вычисления → `status="error"` с error_code

### Performance characteristics

**Resource costs:**
- **CPU**: O(N * log(N)) для STFT, где N — длина аудио
- **Память**: O(freq_bins * frames) для спектрограммы
- **Estimated duration**: ~0.9 секунд для типичного аудио (для `run()`)
- **Segment-based overhead**: для `run_segments()`, время пропорционально количеству сегментов

**Единица обработки:**
- `run()`: весь аудио файл
- `run_segments()`: сегменты от Segmenter (`families.spectral.segments[]`)

### Parallelization

**Внутренний параллелизм** (внутри компонента):
- **CPU parallelism**: компонент поддерживает параллельную обработку сегментов из нескольких видео через `extract_batch_segments()`
- **ThreadPoolExecutor**: используется для параллельной обработки файлов в batch режиме
- **Количество потоков**: автоматически определяется как `min(len(audio_files), os.cpu_count())` или задаётся через `max_workers`
- **Thread-safety**: компонент thread-safe для параллельной обработки разных файлов (каждый файл имеет свой `artifacts_dir`)

**Внешний параллелизм** (выше компонента):
- Можно запускать несколько экземпляров компонента параллельно на разных видео (разные `run_id`, разные `result_store` пути)
- Требования к изоляции:
  - Каждый файл должен иметь свой `artifacts_dir` для изоляции артефактов
  - Каждый файл должен иметь свой `tmp_path` для временных файлов
  - Сегменты из разных файлов не должны пересекаться

**Batch processing:**
- Компонент поддерживает batch processing через `extract_batch_segments()`:
  - Сбор сегментов из всех видео
  - Параллельная обработка файлов через ThreadPoolExecutor
  - Изоляция результатов для каждого файла
- Для включения batch processing установите `supports_batch = True` (уже установлено)
- Batch processing контролируется через параметры:
  - `max_workers`: количество параллельных воркеров (None = auto)
  - `enable_cpu_parallel`: включение CPU параллелизма (через MainProcessor)

**Ограничения:**
- Память: peak memory при параллельном выполнении пропорционален количеству параллельных воркеров
- CPU: рекомендуется использовать не более `os.cpu_count()` воркеров для оптимальной производительности
- GPU: не используется (CPU-only компонент)

### Примеры использования (Audit v3)

#### Загрузка результатов

```python
import numpy as np

data = np.load("band_energy_extractor_features.npz", allow_pickle=True)

# Основной результат
band_edges = data["band_edges_hz"]          # float32[3,2]
band_shares = data["band_energy_shares"]    # float32[3]

print(f"Band shares: {band_shares}")
```

#### Анализ частотного баланса

```python
# Проверка доминирующей полосы
shares = payload["band_energy_shares"]
dominant_band = np.argmax(shares)
band_names = ["Low", "Mid", "High"]
print(f"Доминирующая полоса: {band_names[dominant_band]} ({shares[dominant_band]:.2%})")

# Если включены метрики баланса
if "band_balance_score" in payload:
    print(f"Balance score: {payload['band_balance_score']:.3f}")
    print(f"Contrast: {payload['band_contrast']:.3f}")
```

#### Анализ статистик

```python
# Если включены базовые статистики
if "band_energy_mean" in payload:
    mean = payload["band_energy_mean"]
    std = payload["band_energy_std"]
    median = payload["band_energy_median"]
    print(f"Mean energies: {mean}")
    print(f"Std energies: {std}")
    print(f"Median energies: {median}")
```

#### Анализ динамики (для run_segments)

```python
# Если включены метрики динамики
if "band_energy_stability" in payload:
    print(f"Stability: {payload['band_energy_stability']:.3f}")
    print(f"Transitions: {payload['band_transitions_count']} transitions")
    print(f"Transition rate: {payload['band_transitions_rate']:.4f} transitions/sec")
    print(f"Band diversity: {payload['band_diversity']} unique bands")
    
    # Распределение времени по полосам
    distribution = payload["band_distribution"]
    print("Band distribution:")
    for band_idx, proportion in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  Band {band_idx}: {proportion:.2%}")
```

#### Использование с shared_features

```python
# Если spectral_extractor был запущен ранее
shared_features = {
    "stft_magnitude": stft_magnitude,  # shape: (freq_bins, frames)
    "frequencies": frequencies,  # shape: (freq_bins,)
}

result = band_energy_extractor.run(input_uri, tmp_path, shared_features=shared_features)
# Band energy extractor использует предвычисленный STFT, избегая повторного вычисления
```

### Связанные компоненты

- **Segmenter**: предоставляет аудио файл и сегменты (`families.spectral.segments[]`)
- **librosa**: основная библиотека для STFT и мел-шкалы
- (Audit v3) Essentia не используется
- **spectral_extractor**: может предоставлять `stft_magnitude` и `frequencies` в `shared_features` для оптимизации

### Visualization

**Рекомендуемые типы визуализации:**

1. **Bar chart энергий**:
   - Столбчатая диаграмма энергий по полосам
   - Логарифмическая шкала для энергий

2. **Pie chart долей**:
   - Круговая диаграмма долей энергий по полосам
   - Процентное соотношение каждой полосы

3. **Timeline визуализация** (для `run_segments()`):
   - График изменений энергий по времени
   - Отдельные линии для каждой полосы
   - Логарифмическая шкала для энергий

4. **Statistics comparison**:
   - Сравнение статистик (mean, std, median) по полосам
   - Grouped bar chart

5. **Balance metrics**:
   - Индикатор баланса (balance score)
   - Индикатор доминирующей полосы
   - Контраст между полосами

**Интерактивные элементы:**
- Tooltips с детальной информацией о полосах
- Zoom для детального просмотра timeline
- Фильтры по полосам
- Переключение между различными метриками

### Примечания

1. **По умолчанию 3 полосы**: low [0-200 Hz), mid [200-2000 Hz), high [2000-nyq Hz)
2. **Мел-шкала**: если `use_mel_bands=True`, полосы строятся через мел-шкалу (более музыкально релевантно)
3. **Essentia vs librosa**: Essentia может быть быстрее для frame-by-frame обработки, librosa — более гибкий
4. **Векторизованный биннинг**: librosa использует матричное умножение для эффективного вычисления
5. **Shared STFT**: оптимизация для случаев, когда STFT уже вычислен другим экстрактором
6. **Segment-based processing**: полезно для длинных видео и отслеживания изменений частотного баланса
7. **Dynamics metrics**: метрики динамики полезны для ML моделей и аналитики
8. **Balance metrics**: метрики баланса помогают анализировать частотное распределение
