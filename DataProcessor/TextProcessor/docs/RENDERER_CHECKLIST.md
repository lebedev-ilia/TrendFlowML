# Чеклист Renderer для TextProcessor Extractors

Этот документ фиксирует все требования и шаблоны для приведения extractors TextProcessor к production-ready состоянию с поддержкой renderer'ов.

## Обязательные компоненты для каждого extractor'а

### 1. JSON Renderer (`render.py`)

**Файл**: `src/extractors/<extractor_name>/render.py`

**Обязательная функция**:
```python
def render_<extractor_name>(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для <extractor_name> extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_<extractor>_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    # 1. Очистка NaN значений
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # 2. Группировка фич по категориям
    # 3. Построение summary
    # 4. Возврат структурированного render-context
    
    return {
        "component": "<extractor_name>",
        "summary": {...},
        "categories": {...},
        ...
    }
```

**Требования**:
- ✅ Все значения должны быть очищены от NaN/Infinity (через `_clean_value()`)
- ✅ Структура должна быть логичной и группировать фичи по категориям
- ✅ Summary должен содержать ключевые метрики
- ✅ Экспорт через `__all__ = ["render_<extractor_name>"]`

### 2. HTML Renderer (`render.py`)

**Обязательная функция**:
```python
def render_<extractor_name>_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага <extractor_name> результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    import os
    import json
    import math
    
    # 1. Получить JSON render
    render = render_<extractor_name>(extractor_features, payload, meta)
    
    # 2. Подготовить данные для визуализации
    # 3. Сгенерировать HTML с:
    #    - Заголовками на русском
    #    - Таблицами с описаниями фич на русском
    #    - Графиками Plotly (если есть числовые данные)
    #    - Метаданными (status, версии)
    
    # 4. Atomic write
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    
    return output_path
```

**Требования к HTML**:
- ✅ **Все тексты на русском языке** (заголовки, описания, метки)
- ✅ `<meta charset="UTF-8">` для корректного отображения кириллицы
- ✅ Plotly для графиков (если есть числовые данные)
- ✅ Таблицы с описаниями всех фич
- ✅ Summary секция с ключевыми метриками
- ✅ Стилизация (CSS) для читаемости
- ✅ Atomic write (tmp → replace)

**Шаблон HTML структуры**:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Название компонента</title>
    <meta charset="UTF-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* Стили */
    </style>
</head>
<body>
    <div class="container">
        <h1>Название компонента</h1>
        <p><strong>Компонент:</strong> <extractor_name></p>
        <p><strong>Статус:</strong> <span style="color: green;">✓ OK</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <!-- Карточки с ключевыми метриками -->
        </div>
        
        <div class="section">
            <h2>Категория признаков</h2>
            <div class="plot-container">
                <div id="features-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <!-- Строки с фичами -->
            </table>
        </div>
        
        <!-- Другие секции -->
    </div>
    
    <script>
        // Plotly графики
    </script>
</body>
</html>
```

## Чеклист для каждого extractor'а

### ✅ Базовые требования

- [ ] Файл `src/extractors/<extractor_name>/render.py` существует
- [ ] Функция `render_<extractor_name>()` реализована
- [ ] Функция `render_<extractor_name>_html()` реализована
- [ ] Обе функции экспортированы через `__all__`

### ✅ JSON Renderer

- [ ] Очистка NaN/Infinity значений через `_clean_value()`
- [ ] Логичная группировка фич по категориям
- [ ] Summary с ключевыми метриками
- [ ] Структура соответствует паттерну других extractors

### ✅ HTML Renderer

- [ ] Все тексты на русском языке
- [ ] `<meta charset="UTF-8">` присутствует
- [ ] Таблицы с описаниями всех фич
- [ ] Графики Plotly (если есть числовые данные)
- [ ] Summary секция с ключевыми метриками
- [ ] Atomic write (tmp → replace)
- [ ] Стилизация для читаемости

### ✅ Интеграция

- [ ] Renderer автоматически вызывается из `src/core/renderer.py`
- [ ] HTML отчеты сохраняются в `_render/<extractor_name>_report.html`
- [ ] HTML отчеты добавляются в `manifest.json.artifacts[]` (type=`"html_report"`)

## Примеры реализованных extractors

### lexico_static_features

**Реализовано**:
- ✅ `render_lexico_static_features()` - JSON renderer
- ✅ `render_lexico_static_features_html()` - HTML renderer
- ✅ Все тексты на русском
- ✅ Графики Plotly для title и transcript фич
- ✅ Таблицы с описаниями
- ✅ Atomic write

**Файл**: `src/extractors/lexico_static_features/render.py`

### tags_extractor

**Реализовано**:
- ✅ `render_tags_extractor()` - JSON renderer
- ✅ `render_tags_extractor_html()` - HTML renderer
- ✅ Все тексты на русском
- ✅ Графики Plotly для распределения хэштегов и длин топ-K
- ✅ Таблицы с описаниями всех фич
- ✅ Отображение топ-K хэштегов (privacy-safe)
- ✅ Atomic write

**Файл**: `src/extractors/tags_extractor/render.py`

### asr_text_proxy_audio_features

**Реализовано**:
- ✅ `render_asr_text_proxy_audio_features()` - JSON renderer
- ✅ `render_asr_text_proxy_audio_features_html()` - HTML renderer
- ✅ Все тексты на русском
- ✅ Графики Plotly для confidence, rhythm и noise метрик
- ✅ Таблицы с описаниями всех фич
- ✅ Группировка по категориям (presence, audio_meta, confidence, noise, rhythm, intonation)
- ✅ Atomic write

**Файл**: `src/extractors/asr_text_proxy_audio_features/render.py`

### title_embedder

**Реализовано**:
- ✅ `render_title_embedder()` - JSON renderer
- ✅ `render_title_embedder_html()` - HTML renderer
- ✅ Все тексты на русском
- ✅ График Plotly для характеристик эмбеддинга
- ✅ Таблицы с описаниями всех фич
- ✅ Группировка по категориям (embedding, cache, performance, configuration)
- ✅ Atomic write

**Файл**: `src/extractors/title_embedder/render.py`

## Шаблон для нового extractor'а

1. Скопировать `src/extractors/lexico_static_features/render.py` как основу
2. Заменить `lexico_static_features` на `<extractor_name>`
3. Адаптировать группировку фич под структуру extractor'а
4. Обновить описания фич на русском
5. Добавить графики для числовых данных (если есть)
6. Протестировать генерацию HTML

## Проверка перед коммитом

Для каждого extractor'а:

```bash
# 1. Проверить синтаксис
python3 -m py_compile src/extractors/<extractor_name>/render.py

# 2. Проверить импорты
python3 -c "from src.extractors.<extractor_name>.render import render_<extractor_name>, render_<extractor_name>_html; print('OK')"

# 3. Запустить TextProcessor и проверить:
#    - render_context.json генерируется
#    - <extractor_name>_report.html генерируется
#    - HTML открывается в браузере и все тексты на русском
#    - Графики отображаются корректно
```

## Известные проблемы и решения

### Проблема: NaN в JSON
**Решение**: Использовать `_clean_value()` для всех значений перед добавлением в render-context

### Проблема: Некорректное отображение кириллицы
**Решение**: Убедиться, что `<meta charset="UTF-8">` присутствует и файл сохраняется с `encoding="utf-8"`

### Проблема: HTML renderer не вызывается
**Решение**: Проверить, что функция экспортирована через `__all__` и имя функции точно `render_<extractor_name>_html`

### Проблема: Графики не отображаются
**Решение**: Проверить, что Plotly скрипт загружен и данные фильтруются от NaN перед передачей в Plotly

## Статус extractors

| Extractor | JSON Renderer | HTML Renderer | Русский язык | Статус |
|-----------|---------------|---------------|--------------|--------|
| `lexico_static_features` | ✅ | ✅ | ✅ | **Готово** |
| `tags_extractor` | ✅ | ✅ | ✅ | **Готово** |
| `asr_text_proxy_audio_features` | ✅ | ✅ | ✅ | **Готово** |
| `title_embedder` | ✅ | ✅ | ✅ | **Готово** |
| `description_embedder` | ✅ | ✅ | `tp_descemb_*` | Готово |
| `hashtag_embedder` | ✅ | ✅ | `tp_hashemb_*` | Готово |
| `transcript_chunk_embedder` | ✅ | ✅ | `tp_tchunk_*` | Готово |
| `comments_embedder` | ❌ | ❌ | - | Требуется |
| `cosine_metrics_extractor` | ❌ | ❌ | - | Требуется |
| `embedding_stats_extractor` | ❌ | ❌ | - | Требуется |
| ... | ... | ... | ... | ... |

## Примечания

- HTML отчеты генерируются только для включенных extractors (из `global_config.yaml`)
- Если extractor не имеет renderer'а, генерируется только базовый JSON render-context
- HTML отчеты предназначены для локального дебага и не должны содержать raw текст (privacy-safe)

