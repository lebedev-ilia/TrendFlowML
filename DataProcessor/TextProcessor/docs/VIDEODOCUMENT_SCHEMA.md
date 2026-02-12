# VideoDocument JSON Schema

## Описание

`VideoDocument` — это входной JSON-файл для TextProcessor, содержащий все текстовые данные о видео (title, description, transcripts, comments и т.д.).

## Схема

См. `TextProcessor/src/schemas/models.py` для полной схемы.

### Обязательные поля

- **`title`** (string): Заголовок видео
- **`description`** (string): Описание видео

### Опциональные поля

- **`hashtags`** (array of strings): Список хештегов (без символа '#'). Обычно заполняется автоматически `TagsExtractor` из title/description.
- **`transcripts`** (object): Словарь транскриптов по источникам:
  ```json
  {
    "whisper": "текст транскрипта от Whisper",
    "youtube_auto": "текст транскрипта от YouTube"
  }
  ```
- **`transcripts_token_ids`** (object): Токенизированные транскрипты (предпочтительно для privacy):
  ```json
  {
    "whisper": [101, 2023, ...],
    "youtube_auto": [...]
  }
  ```
- **`audio_duration_sec`** (float): Длительность аудио в секундах. Обязательно для extractors, которые вычисляют time-normalized метрики.
- **`asr`** (object): ASR payload от AudioProcessor (предпочтительный источник транскрипта):
  ```json
  {
    "schema_version": "asr_v1",
    "segments": [
      {
        "text": "текст сегмента",
        "confidence": 0.95,
        "start_sec": 0.0,
        "end_sec": 5.2
      }
    ]
  }
  ```
- **`transcripts_meta`** (object): Legacy alias для ASR (используется старыми extractors). Предпочтительно использовать `asr`.
- **`video_description_by_neuro`** (string, optional): Описание видео, сгенерированное нейросетью.
- **`trend_words`** (string, optional): Трендовые слова.
- **`comments`** (array): Список комментариев. Каждый комментарий может быть:
  - Строкой: `"Great video!"`
  - Объектом: `{"text": "Great video!"}`
- **`speakers`** (object, optional): Информация о спикерах:
  ```json
  {
    "speaker_1": {
      "name": "Speaker Name",
      "segments": [...]
    }
  }
  ```

## Пример минимального файла

```json
{
  "title": "My Video Title",
  "description": "Video description",
  "transcripts": {
    "whisper": "Transcript text here"
  },
  "comments": [
    {"text": "Comment 1"},
    {"text": "Comment 2"}
  ]
}
```

## Пример полного файла

См. `TextProcessor/example_video_document.json`.

## Источники данных

### ASR (Automatic Speech Recognition)

**Предпочтительный источник**: `asr` поле (от AudioProcessor)

```json
{
  "asr": {
    "schema_version": "asr_v1",
    "segments": [
      {
        "text": "текст сегмента",
        "confidence": 0.95,
        "start_sec": 0.0,
        "end_sec": 5.2
      }
    ]
  }
}
```

**Legacy источник**: `transcripts` поле

```json
{
  "transcripts": {
    "whisper": "полный текст транскрипта",
    "youtube_auto": "альтернативный транскрипт"
  }
}
```

### Комментарии

Комментарии могут быть представлены как:
- Массив строк: `["Comment 1", "Comment 2"]`
- Массив объектов: `[{"text": "Comment 1"}, {"text": "Comment 2"}]`

## Использование в TextProcessor

TextProcessor читает `VideoDocument` из JSON файла, указанного через `--input-json`:

```bash
python3 TextProcessor/run_cli.py \
  --input-json /path/to/video_document.json \
  --rs-base ./result_store \
  --platform-id youtube \
  --video-id <video_id> \
  --run-id <run_id>
```

Или через глобальный конфиг:

```yaml
processors:
  text:
    enabled: true
    input_json: "/path/to/video_document.json"
```

## Зависимости от других процессоров

### AudioProcessor

Если AudioProcessor включен и выполняется ASR, результат должен быть записан в `VideoDocument.asr` перед запуском TextProcessor.

### Segmenter

Если Segmenter включен, `audio_duration_sec` должен быть заполнен для корректной работы time-normalized метрик.

## Privacy considerations

По умолчанию TextProcessor **не сохраняет raw текст** в NPZ артефактах (только фичи/статистики).

Для дебага можно использовать `--store-raw-payload`, но это создает privacy risk.

## Валидация

TextProcessor валидирует входной JSON при загрузке:
- Обязательные поля (`title`, `description`) должны присутствовать
- Опциональные поля могут быть `null` или отсутствовать
- Неправильный формат комментариев будет автоматически исправлен (строка → объект с `text`)

## Примеры использования

### Минимальный пример (только title и description)

```json
{
  "title": "Video Title",
  "description": "Video description"
}
```

### С транскриптом

```json
{
  "title": "Video Title",
  "description": "Video description",
  "transcripts": {
    "whisper": "Full transcript text here"
  }
}
```

### С ASR от AudioProcessor

```json
{
  "title": "Video Title",
  "description": "Video description",
  "audio_duration_sec": 120.5,
  "asr": {
    "schema_version": "asr_v1",
    "segments": [
      {
        "text": "First segment",
        "confidence": 0.95,
        "start_sec": 0.0,
        "end_sec": 5.0
      }
    ]
  }
}
```

### С комментариями

```json
{
  "title": "Video Title",
  "description": "Video description",
  "comments": [
    "Comment 1",
    "Comment 2",
    {"text": "Comment 3"}
  ]
}
```

