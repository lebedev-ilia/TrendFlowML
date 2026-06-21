# Dataset Balancer

Балансировщик подключается из campaign config через поле `balancer_config_file`.
Текущий стартовый конфиг: `dataset_balancer.json`.

## Первый тестовый запуск

Запускайте сначала одну категорию, чтобы проверить скорость и перекосы в Grafana:

```bash
python -m fetcher.dataset_collector.cli discover dataset_campaign.json \
  --category Avto_i_transport \
  --limit 500 \
  --metrics-port 9095
```

После запуска сравните в Grafana распределения `language`, `country`,
`duration_seconds`, `view_count` и `balancer_*` reject reasons. Если сбор
замедлился слишком сильно, уменьшайте `coefficient` у самого проблемного поля.

## Как читать coefficient

- `0.0` — поле не влияет на acceptance.
- `0.3` — мягкая коррекция перекоса.
- `0.6` — заметная коррекция без жёсткой остановки сбора.
- `0.85+` — почти жёсткое стремление к целевому распределению.

Поля из `post_enrich_fields` пока используются как report-only: они помогают
видеть перекосы после enrich, но не удаляют уже принятые видео.
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
