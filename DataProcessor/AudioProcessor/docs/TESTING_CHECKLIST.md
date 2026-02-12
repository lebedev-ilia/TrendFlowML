# Чеклист готовности к тестированию AudioProcessor

## ✅ Готово для тестирования

### 1. Запуск через верхний оркестратор
- ✅ `DataProcessor/main.py` поддерживает запуск AudioProcessor через `--run-audio`
- ✅ Поддержка `--global-config` для конфигурации через `global_config.yaml`
- ✅ Автоматическая передача параметров batch processing через `config_parser.get_audio_cli_args()`

### 2. Поочередное добавление компонентов
- ✅ Все extractors в `global_config.yaml` имеют `enabled: false` по умолчанию
- ✅ Можно включать по одному через `enabled: true`
- ✅ Порядок extractors в конфиге: clap, tempo, loudness, asr, speaker_diarization, emotion_diarization, source_separation, ...

### 3. Два режима тестирования
- ✅ **Без оптимизаций**: 
  - `batch_processing.enabled: false`
  - `scheduler.segment_parallelism: 1`
  - `scheduler.clap_batch_size: 1`
- ✅ **С оптимизациями** (фиксированные значения из `global_config.yaml`):
  - `scheduler.segment_parallelism: 16` (параллелизм для CPU extractors)
  - `scheduler.clap_batch_size: 16` (размер батча для GPU extractors)
  - `batch_processing.enabled: true`
  - `batch_processing.enable_gpu_batching: true`
  - `batch_processing.enable_cpu_parallel: true`
  - `batch_processing.enable_video_parallel: true`

### 4. Фиксация времени
- ✅ Время сохраняется в NPZ meta: `stage_timings_ms` и `timings_by_extractor`
- ✅ Время доступно для каждого extractor'а отдельно
- ✅ Общее время обработки в `timings.wall_clock.elapsed_s`
- ✅ В batch mode время сохраняется для каждого файла в `batch_results`

### 5. HTML Render
- ✅ Render-context JSON генерируется автоматически для каждого extractor'а
- ✅ Расположение: `result_store/.../<component_name>/_render/render_context.json`
- ✅ Render вызывается в single-file mode (строка 3906 в run_cli.py)
- ✅ Render вызывается в batch mode (добавлено в строке 3123+ в run_cli.py)

## 📋 Процесс тестирования

### Шаг 1: Подготовка
1. Откройте `DataProcessor/configs/global_config.yaml`
2. Включите только первый extractor (например, `clap: enabled: true`)
3. Установите `batch_processing.enabled: false` для первого теста

### Шаг 2: Тест без оптимизаций
```bash
python3 DataProcessor/main.py \
  --video-path /path/to/video.mp4 \
  --global-config DataProcessor/configs/global_config.yaml \
  --run-audio \
  --platform-id youtube \
  --video-id test_video_1 \
  --run-id test_run_1_no_optimizations
```

**Проверьте**:
- [ ] NPZ файл создан: `result_store/.../clap_extractor/clap_extractor_features.npz`
- [ ] Render-context создан: `result_store/.../clap_extractor/_render/render_context.json`
- [ ] Время обработки записано в NPZ meta
- [ ] HTML render корректен (если генерируется)

**Запишите**:
- Время обработки из `stage_timings_ms.run_extractors_ms`
- Время extractor'а из `timings_by_extractor.clap.wall_ms`

### Шаг 3: Тест с оптимизациями
1. Обновите `global_config.yaml` (фиксированные значения для тестов):
   ```yaml
   scheduler:
     segment_parallelism: 16
     max_inflight: null
     clap_batch_size: 16
   
   batch_processing:
     enabled: true
     max_video_workers: null
     enable_video_parallel: true
     max_segment_workers: null
     enable_segment_parallel: true
     enable_gpu_batching: true
     max_segments_per_gpu_batch: null
     enable_cpu_parallel: true
   ```

2. Запустите с новым `--run-id`:
```bash
python3 DataProcessor/main.py \
  --video-path /path/to/video.mp4 \
  --global-config DataProcessor/configs/global_config.yaml \
  --run-audio \
  --platform-id youtube \
  --video-id test_video_1 \
  --run-id test_run_1_with_optimizations
```

**Проверьте**:
- [ ] Время обработки меньше, чем без оптимизаций
- [ ] Render-context корректен
- [ ] NPZ файл корректен

**Запишите**:
- Время обработки
- Ускорение (время без оптимизаций / время с оптимизациями)

### Шаг 4: Добавление следующего компонента
1. Включите следующий extractor в `global_config.yaml`
2. Повторите шаги 2 и 3

## 📊 Документирование результатов

Создайте файл `docs/TEST_RESULTS.md` со следующей структурой:

```markdown
# Результаты тестирования AudioProcessor

## Тестовая конфигурация
- Дата: YYYY-MM-DD
- Видео: <video_id>
- Платформа: youtube
- Устройство: cuda/cpu
- Версия: <dataprocessor_version>

## Результаты по компонентам

### clap_extractor

#### Без оптимизаций
- Время обработки: X.XX секунд
- Stage timings:
  - load_input_ms: XX
  - run_extractors_ms: XX
  - save_npz_ms: XX
- Per-extractor timing: XX ms
- HTML render: ✅ / ❌
- NPZ валидация: ✅ / ❌

#### С оптимизациями
- Время обработки: X.XX секунд
- Ускорение: X.XXx
- Stage timings: ...
- HTML render: ✅ / ❌
- NPZ валидация: ✅ / ❌

### tempo_extractor
...
```

## 🔍 Извлечение метрик

### Из NPZ файла (Python)
```python
import numpy as np

npz = np.load("result_store/.../clap_extractor/clap_extractor_features.npz", allow_pickle=True)
meta = npz["meta"].item() if hasattr(npz["meta"], "item") else npz["meta"]

stage_timings = meta.get("stage_timings_ms", {})
timings_by_extractor = meta.get("timings_by_extractor", {})

print(f"Total time: {stage_timings.get('run_extractors_ms', 0)} ms")
print(f"CLAP time: {timings_by_extractor.get('clap', {}).get('wall_ms', 0)} ms")
```

### Из manifest.json
```python
import json

with open("result_store/.../manifest.json", "r") as f:
    manifest = json.load(f)

for component in manifest.get("components", []):
    print(f"{component['name']}: {component.get('duration_ms', 0)} ms")
```

## ⚠️ Важные замечания

1. **Single-file mode vs Batch mode**:
   - Через `DataProcessor/main.py` работает только **single-file mode** (один видео файл)
   - Для настоящего batch mode (несколько файлов) запускайте напрямую `AudioProcessor/run_cli.py` с `--audio-input-dir`

2. **Оптимизации в single-file mode**:
   - `batch_processing.enabled: true` включает GPU batching сегментов и CPU parallelism **внутри одного файла**
   - Это не то же самое, что batch processing нескольких файлов

3. **Render генерируется автоматически**:
   - Для каждого успешно обработанного extractor'а
   - В single-file mode и batch mode
   - Расположение: `<component_name>/_render/render_context.json`

4. **Время сохраняется автоматически**:
   - В NPZ meta для каждого extractor'а
   - В manifest.json для каждого компонента
   - В batch mode: для каждого файла отдельно

## ✅ Все готово к тестированию!

См. подробное руководство: `docs/TESTING_GUIDE.md`

