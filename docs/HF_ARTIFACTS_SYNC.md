# HF Artifacts Sync

Скрипты для синхронизации артефактов и моделей между локальным репозиторием `TrendFlowML` и Hugging Face Hub.

## Файлы

- `DataProcessor/scripts/hf_artifacts_sync.py` - основной CLI sync (`upload`/`download`)
- `DataProcessor/scripts/hf_upload_all.sh` - массовая выгрузка
- `DataProcessor/scripts/hf_download_all.sh` - массовая загрузка
- `configs/hf_artifacts_manifest.json` - список путей и маппинг local->remote

## Подготовка

1. Установить зависимость:
   - `pip install huggingface_hub`
2. Авторизоваться:
   - экспортировать `HF_TOKEN` (или `HUGGINGFACE_HUB_TOKEN`)
3. Указать ваш HF репозиторий в `configs/hf_artifacts_manifest.json`:
   - `repo_id: "your-hf-username/trendflowml-artifacts"`

## Быстрые команды

Проверить, что будет выгружено:

```bash
./DataProcessor/scripts/hf_upload_all.sh --dry-run
```

Выгрузить все артефакты:

```bash
./DataProcessor/scripts/hf_upload_all.sh --create-repo --private
```

Проверить, что будет загружено:

```bash
./DataProcessor/scripts/hf_download_all.sh --dry-run
```

Загрузить все артефакты в локальный репозиторий:

```bash
./DataProcessor/scripts/hf_download_all.sh
```

## Для ноутбука и ПК

- После каждого коммита в git:
  - при необходимости обновить артефакты: `hf_upload_all.sh`
- На второй машине после `git pull`:
  - восстановить артефакты: `hf_download_all.sh`

Так код синхронизируется через git, а тяжелые данные - через HF.
