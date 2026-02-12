# TextProcessor Renderer Guide

Руководство по созданию renderer'ов для TextProcessor extractors.

> **Важно**: См. также `RENDERER_CHECKLIST.md` для полного чеклиста требований и шаблонов.

## Обзор

TextProcessor использует систему рендеринга для генерации human-friendly JSON из NPZ артефактов. Система состоит из:

1. **Глобальный renderer** (`src/core/renderer.py`): обрабатывает весь TextProcessor NPZ и координирует per-extractor renderer'ы
2. **Per-extractor renderer'ы** (`src/extractors/<name>/render.py`): генерируют детализированные render-context'ы для конкретных extractors

## Структура render-context

Render-context — это JSON словарь со следующей структурой:

```json
{
  "component": "extractor_name",
  "summary": {
    "key_stat": "value"
  },
  "features": {
    "category": {
      "feature_name": value
    }
  },
  "statistics": {},
  "meta": {
    "status": "ok",
    "producer_version": "1.0.0",
    "schema_version": "text_npz_v1"
  }
}
```

## Создание render.py для extractor'а

### 1. Создайте файл `render.py`

В директории extractor'а создайте файл:
```
src/extractors/<extractor_name>/render.py
```

### 2. Реализуйте функцию renderer'а

Функция должна иметь сигнатуру:

```python
def render_<extractor_name>(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для extractor'а.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_<extractor>_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "<extractor_name>",
        "summary": {},
        # ... другие секции
    }
    
    # Обработка фич
    # ...
    
    return render
```

### 3. Экспортируйте функцию

Добавьте в конец файла:

```python
__all__ = ["render_<extractor_name>"]
```

## Пример: lexico_static_features

См. `src/extractors/lexico_static_features/render.py` для полного примера.

Ключевые моменты:
- Фичи имеют префикс `tp_lex_*` (для lexico_static_features)
- Группировка фич по категориям (title, description, transcript)
- Извлечение summary статистики

## Префиксы фич

TextProcessor extractors используют префиксы для идентификации:

- `tp_lex_*` → `lexico_static_features`
- `tp_tags_*` → `tags_extractor`
- `tp_asr_*` → `asr_text_proxy_audio_features`
- `tp_title_*` → `title_embedder`
- `tp_desc_*` → `description_embedder`
- `tp_hashtag_*` → `hashtag_embedder`
- `tp_transcript_*` → `transcript_chunk_embedder`
- `tp_comments_*` → `comments_embedder`
- `tp_cosine_*` → `cosine_metrics_extractor`
- `tp_embedding_*` → `embedding_stats_extractor`
- `tp_qa_*` → `qa_embedding_pairs_extractor`
- `tp_semantic_*` → `semantics_topics_keyphrases`

Глобальный renderer автоматически группирует фичи по этим префиксам.

## Best Practices

1. **Privacy-safe**: Не включайте raw текст в render-context (только статистики и метрики)
2. **Структурированность**: Группируйте фичи по логическим категориям
3. **Summary**: Включайте ключевые статистики в `summary` для быстрого обзора
4. **Обработка NaN**: Корректно обрабатывайте NaN значения (можно использовать `None` или пропускать)
5. **Ошибки**: Renderer не должен падать при ошибках — используйте try/except и возвращайте базовый render

## Интеграция

Renderer автоматически вызывается из `run_cli.py` после сохранения NPZ:

```python
from src.core.renderer import render_text_processor
render = render_text_processor(npz_path, output_dir)
```

Render-context сохраняется в:
- `result_store/.../text_processor/_render/render_context.json`

И добавляется в `manifest.json.artifacts[]` (type=`"render"`).

## Тестирование

Для тестирования renderer'а:

```python
import numpy as np
from src.core.renderer import load_npz, extract_meta
from src.extractors.<extractor_name>.render import render_<extractor_name>

# Load NPZ
npz_data = load_npz("path/to/text_features.npz")
meta = extract_meta(npz_data)

# Extract extractor features (by prefix)
extractor_features = {k: v for k, v in npz_data.get("features_flat", {}).items() 
                      if k.startswith("tp_<extractor>_")}

# Generate render
render = render_<extractor_name>(extractor_features, npz_data.get("payload", {}), meta)
print(json.dumps(render, indent=2, ensure_ascii=False))
```

