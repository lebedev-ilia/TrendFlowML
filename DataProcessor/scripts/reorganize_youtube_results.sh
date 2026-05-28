#!/bin/bash
# Скрипт для реорганизации структуры папок в dp_results/youtube

# Не используем set -e, чтобы скрипт продолжал работу при несущественных ошибках

YOUTUBE_DIR="/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/youtube"
BACKUP_DIR="/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/youtube_backup_$(date +%Y%m%d_%H%M%S)"

cd "$YOUTUBE_DIR"

echo "=================================================================================="
echo "РЕОРГАНИЗАЦИЯ СТРУКТУРЫ dp_results/youtube"
echo "=================================================================================="
echo ""

# Создаём резервную копию
echo "1. Создание резервной копии..."
mkdir -p "$BACKUP_DIR"
cp -r "$YOUTUBE_DIR"/* "$BACKUP_DIR/" 2>/dev/null || true
echo "   ✅ Резервная копия создана: $BACKUP_DIR"
echo ""

# Создаём новую структуру
echo "2. Создание новой структуры папок..."
mkdir -p tests/action_recognition
mkdir -p audit/v3/components
mkdir -p audit/v3/smoke
mkdir -p archive/old_tests
mkdir -p archive/old_videos
echo "   ✅ Структура создана"
echo ""

# Функция для безопасного перемещения
move_safe() {
    local src="$1"
    local dst="$2"
    if [ -d "$src" ] && [ ! -d "$dst" ]; then
        mv "$src" "$dst"
        echo "   ✅ $src -> $dst"
        return 0
    elif [ -d "$dst" ]; then
        echo "   ⚠️  Пропущено (уже существует): $dst"
        return 1
    else
        echo "   ⚠️  Пропущено (не найдено): $src"
        return 1
    fi
}

# Перемещаем тесты action_recognition
echo "3. Перемещение тестов action_recognition..."
for dir in test_action_recognition_*; do
    if [ -d "$dir" ]; then
        move_safe "$dir" "tests/action_recognition/$dir"
    fi
done
echo ""

# Перемещаем smoke тесты аудита
echo "4. Перемещение smoke тестов аудита..."
for dir in audit3_*_smoke_*; do
    if [ -d "$dir" ]; then
        move_safe "$dir" "audit/v3/smoke/$dir"
    fi
done
echo ""

# Перемещаем тесты аудита компонентов
echo "5. Перемещение тестов аудита компонентов..."
for dir in test_*_audit_3; do
    if [ -d "$dir" ]; then
        move_safe "$dir" "audit/v3/components/$dir"
    fi
done
echo ""

# Перемещаем старые видео
echo "6. Перемещение старых видео в архив..."
for dir in video[0-9]* test_video_*; do
    if [ -d "$dir" ]; then
        move_safe "$dir" "archive/old_videos/$dir"
    fi
done
echo ""

# Создаём README с описанием структуры
echo "7. Создание документации..."
cat > README.md << 'EOF'
# Структура dp_results/youtube

Эта директория содержит результаты обработки видео для платформы YouTube.

## Структура

```
youtube/
├── tests/                    # Тестовые прогоны компонентов
│   └── action_recognition/   # Тесты компонента action_recognition
│
├── audit/                    # Результаты аудита
│   └── v3/                   # Audit v3 результаты
│       ├── components/      # Тесты отдельных компонентов
│       └── smoke/            # Smoke тесты
│
├── archive/                  # Архив старых результатов
│   ├── old_tests/            # Старые тесты
│   └── old_videos/           # Старые видео
│
└── README.md                 # Этот файл
```

## Описание категорий

### tests/
Тестовые прогоны для проверки качества и функциональности компонентов.
- `tests/action_recognition/` - результаты тестирования компонента action_recognition

### audit/
Результаты аудита компонентов (Audit v3).
- `audit/v3/components/` - тесты отдельных компонентов
- `audit/v3/smoke/` - smoke тесты для быстрой проверки

### archive/
Архив старых результатов, которые больше не используются активно.

## Формат run_id

Стандартный формат: `<component>_<purpose>_<identifier>`

Примеры:
- `test_action_recognition_v1` - тест action_recognition, версия 1
- `audit3_core_clip_smoke_1` - smoke тест core_clip для audit v3
- `test_core_object_detections_audit_3` - тест core_object_detections для audit v3

## Структура run

Каждый run имеет структуру:
```
<video_id>/
└── <run_id>/
    ├── manifest.json
    ├── <component_name>/
    │   └── *.npz
    ├── _render/
    ├── _logs/
    └── ...
```

См. также: `DataProcessor/docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
EOF

echo "   ✅ README.md создан"
echo ""

# Подсчитываем результаты
echo "8. Подсчёт результатов..."
tests_count=$(find tests -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
audit_count=$(find audit -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)
archive_count=$(find archive -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)

echo ""
echo "=================================================================================="
echo "РЕОРГАНИЗАЦИЯ ЗАВЕРШЕНА"
echo "=================================================================================="
echo ""
echo "Статистика:"
echo "  - Тесты: $tests_count директорий"
echo "  - Аудит: $audit_count директорий"
echo "  - Архив: $archive_count директорий"
echo ""
echo "Резервная копия: $BACKUP_DIR"
echo ""
echo "Новая структура:"
tree -L 3 -d tests audit archive 2>/dev/null || find tests audit archive -type d -maxdepth 3 | head -20
echo ""

