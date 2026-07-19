# Системные требования для Fetcher

## Обзор

Fetcher — это сервис для сбора видеоконтента с YouTube и других платформ. Помимо Python зависимостей, требуются несколько системных компонентов и программ.

## Обязательные системные пакеты

### Для yt-dlp (YouTube метаданные и видео)

```bash
# Node.js и npm — КРИТИЧНО для решения YouTube signature challenges
# Без Node.js yt-dlp не может получить прямые ссылки на видео
apt-get install -y nodejs npm

# ffmpeg — для обработки видео и аудио
apt-get install -y ffmpeg

# curl — для HTTP запросов
apt-get install -y curl

# ca-certificates — для HTTPS
apt-get install -y ca-certificates
```

### Для других платформ (опционально)

```bash
# Для TikTok и Instagram
apt-get install -y chromium-browser

# Для асинхронных операций
apt-get install -y libpq-dev  # PostgreSQL client libraries
```

## Python зависимости

```bash
pip install -r requirements.txt
```

Основные пакеты:
- `yt-dlp>=2023.0.0` — YouTube видео и метаданные
- `httpx[socks]` — HTTP клиент с proxy поддержкой
- `fastapi` — REST API
- `sqlalchemy` — ORM для БД
- `redis` + `celery` — очереди задач

## Процесс установки (для RunPod)

### 1. Обновить apt (если нужно)

```bash
apt-get update
apt-get upgrade -y
```

### 2. Установить системные пакеты

```bash
apt-get install -y \
  nodejs npm \
  ffmpeg \
  curl \
  ca-certificates \
  git \
  python3 python3-pip python3-venv
```

### 3. Создать venv и установить Python пакеты

```bash
cd /workspace
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r Fetcher/requirements.txt
```

### 4. Проверить установку

```bash
# Проверить Node.js
node --version
npm --version

# Проверить yt-dlp
python3 -m yt_dlp --version

# Проверить ffmpeg
ffmpeg -version | head -n 1

# Проверить подключение к YouTube (может быть медленно)
python3 -c "import yt_dlp; yt_dlp.YoutubeDL({'quiet': True}).extract_info('dQw4w9WgXcQ', download=False)" && echo "✓ yt-dlp работает"
```

## Проблемы и решения

### "yt-dlp enrich failed" ошибки в логах

**Симптомы**:
- Массовые ошибки "yt-dlp enrich failed"
- Растущий `queue_dead_letter.jsonl` файл
- Логи показывают "Signature solving failed"

**Причина**:
- Node.js не установлен или недоступен
- yt-dlp требует JavaScript runtime для решения YouTube signature challenges

**Решение**:
```bash
# Установить Node.js
apt-get install -y nodejs npm

# Перезапустить worker процессы
pkill -f "dataset_collector"  # или kill через deploy.py
```

### "missing local file" ошибки

**Симптомы**:
- Ошибки вида "missing local file downloads/videos/..."
- Download worker не находит скачанные файлы

**Причины**:
- Недостаточно дискового пространства
- Race condition между download и upload worker'ами
- Проблемы с сетью при скачивании

**Проверки**:
```bash
# Проверить свободное место
df -h /workspace

# Проверить целостность скачанных файлов
find /workspace/dataset_runs -name "*.mp4" | wc -l
```

## Версии (проверено)

- **Node.js**: v12.22.9+ (может быть и более новое)
- **npm**: 8.5.1+
- **Python**: 3.9+
- **yt-dlp**: 2023.0.0+ (или 2026.07.04 для новых features)
- **ffmpeg**: 4.4+

## Мониторинг установки

Проверить готовность систем можно через:

```bash
python3 -c "
import subprocess
import sys

checks = [
    ('node', ['node', '--version']),
    ('npm', ['npm', '--version']),
    ('ffmpeg', ['ffmpeg', '-version']),
    ('yt-dlp', ['python3', '-m', 'yt_dlp', '--version']),
]

print('System checks:')
for name, cmd in checks:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f'  ✓ {name}: {result.stdout.split(chr(10))[0]}')
        else:
            print(f'  ✗ {name}: FAILED ({result.stderr[:50]})')
    except Exception as e:
        print(f'  ✗ {name}: NOT FOUND')
"
```

## Для разработки

Дополнительные пакеты для локальной разработки:

```bash
pip install -r requirements-test.txt  # pytest, coverage, etc
```

---

**Последнее обновление**: 2026-07-17  
**Автор**: Claude Code (Fetcher Monitor Agent)  
**Status**: 📝 Документировано после инцидента с Node.js на worker-b
