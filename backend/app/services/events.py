from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict

import redis.asyncio as redis

from ..config import Settings

settings = Settings()


def run_channel(run_id: str) -> str:
    return f"run:{run_id}"


async def publish_run_event(run_id: str, payload: Dict[str, Any]) -> None:
    client = redis.from_url(settings.redis_url, decode_responses=True)
    await client.publish(run_channel(run_id), json.dumps(payload, ensure_ascii=False))
    await client.close()


async def subscribe_run_events(run_id: str) -> AsyncGenerator[Dict[str, Any], None]:
    client = redis.from_url(settings.redis_url, decode_responses=True)
    pubsub = client.pubsub()
    await pubsub.subscribe(run_channel(run_id))
    try:
        async for msg in pubsub.listen():
            if msg.get("type") != "message":
                continue
            data = msg.get("data")
            if not data:
                continue
            try:
                payload = json.loads(data)
                if isinstance(payload, dict):
                    yield payload
            except Exception:
                continue
    finally:
        await pubsub.unsubscribe(run_channel(run_id))
        await pubsub.close()
        await client.close()

