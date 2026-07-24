"""Заполняет core.subscription_plans тарифами base / pro / personal.

Идемпотентно: планы сопоставляются по имени.

TODO(продукт): лимиты и цены — плейсхолдеры. По SITE_SPECIFICATION.md §12.1
детали тарифов определяются «после обучения моделей». Значения ниже —
структура, а не утверждённый оффер.

Запуск: cd backend && .venv/bin/python -m scripts.seed_subscription_plans
"""

from __future__ import annotations

from app.db import SessionLocal
from app.dbv2.models import SubscriptionPlan

PLANS = [
    {
        "id": 1,
        "name": "Base",
        "max_videos_per_month": 10,
        "max_analyses_per_month": 20,
        "max_channels": 1,
        "max_storage_gb": 5,
        "has_api_access": False,
        "has_advanced_explainability": False,
        "price": 0.0,
    },
    {
        "id": 2,
        "name": "Pro",
        "max_videos_per_month": 50,
        "max_analyses_per_month": 40,
        "max_channels": 3,
        "max_storage_gb": 25,
        "has_api_access": False,
        "has_advanced_explainability": True,
        "price": 990.0,
    },
    {
        "id": 3,
        "name": "Personal",
        "max_videos_per_month": 200,
        "max_analyses_per_month": 200,
        "max_channels": 10,
        "max_storage_gb": 100,
        "has_api_access": True,
        "has_advanced_explainability": True,
        "price": 2490.0,
    },
]


def main() -> None:
    db = SessionLocal()
    try:
        created, updated = 0, 0
        for spec in PLANS:
            existing = (
                db.query(SubscriptionPlan)
                .filter(SubscriptionPlan.name == spec["name"])
                .first()
            )
            if existing:
                for key, value in spec.items():
                    if key != "id":
                        setattr(existing, key, value)
                updated += 1
            else:
                db.add(SubscriptionPlan(**spec))
                created += 1
        db.commit()
        print(f"Тарифы: создано {created}, обновлено {updated}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
