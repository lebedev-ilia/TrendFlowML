# Профили DataProcessor для seed API

На старте приложение читает `dataproc_root/profiles/*.yaml` (см. `app/main.py`, `PROFILES.md`).

- **Монорепозиторий TrendFlowML:** по умолчанию `dataproc_root` указывает на каталог **`DataProcessor`** у корня монорепо; профили берутcя из `DataProcessor/profiles/` — этот каталог **`backend/profiles`** можно не заполнять.
- **Автономный репозиторий (только backend):** скопируйте сюда содержимое `DataProcessor/profiles/` из монорепозитория и задайте в `.env` при необходимости `TF_BACKEND_DATAPROC_ROOT` на корень backend-репо (обычно не нужно: см. `resolve_paths` в `app/config.py`).

```bash
# Пример из корня монорепо TrendFlowML (один раз перед пушем в отдельный ре по):
rsync -a --delete DataProcessor/profiles/ backend/profiles/
```

Файлы `*.yaml` в git для портфолио — по желанию (лицензии/размер); минимум для демо — хотя бы несколько публичных профилей.
---

## Навигация

[Backend](../docs/MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
