#!/usr/bin/env python3
"""
Линт согласованности docker-compose.prod.yml (перед `docker compose up`).

Ловит рантайм-ломающие ссылки, которые `docker compose config` не всегда явно
подсвечивает:
  * depends_on -> сервис существует;
  * тома контейнера объявлены в top-level volumes:;
  * build.context / dockerfile существуют на диске;
  * host-ссылки в env (http://HOST, @HOST) резолвятся в имя сервиса.

Запуск:  python deploy/validate_compose.py [путь_к_compose]   (exit 0 ок / 2 проблемы)
"""
from __future__ import annotations

import os
import re
import sys

try:
    import yaml
except ImportError:
    raise SystemExit("нужен PyYAML")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERNAL_HOSTS = {
    "postgres", "redis", "minio", "triton", "embedding-service",
    "backend-api", "fetcher-api", "dataprocessor-api",
}
HOST_RE = re.compile(r"(?:https?://|@)([a-z][a-z0-9-]+)(?::\d+|/|$)")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "docker-compose.prod.yml")
    d = yaml.safe_load(open(path))
    svcs = d.get("services", {}) or {}
    vols = set((d.get("volumes") or {}).keys())
    names = set(svcs)
    errs = []

    for s, cfg in svcs.items():
        dep = cfg.get("depends_on")
        deps = list(dep.keys()) if isinstance(dep, dict) else (dep or [])
        for t in deps:
            if t not in names:
                errs.append(f"{s}: depends_on -> '{t}' не сервис")
        for v in cfg.get("volumes", []) or []:
            vs = str(v).split(":")[0]
            if not vs.startswith((".", "/")) and vs not in vols:
                errs.append(f"{s}: том '{vs}' не объявлен в volumes:")
        b = cfg.get("build")
        if isinstance(b, dict):
            ctx, df = b.get("context"), b.get("dockerfile")
            ctx_abs = os.path.join(REPO_ROOT, ctx) if ctx and not os.path.isabs(ctx) else ctx
            if ctx and not os.path.exists(ctx_abs):
                errs.append(f"{s}: build context '{ctx}' нет")
            if df and ctx and not os.path.exists(os.path.join(ctx_abs, df)):
                errs.append(f"{s}: dockerfile '{ctx}/{df}' нет")
        env = cfg.get("environment", {})
        items = env.items() if isinstance(env, dict) else [x.split("=", 1) for x in (env or []) if "=" in x]
        for _k, val in items:
            for m in HOST_RE.finditer(str(val)):
                h = m.group(1)
                if h in INTERNAL_HOSTS and h not in names:
                    errs.append(f"{s}: env host '{h}' не сервис compose")

    errs = sorted(set(errs))
    if errs:
        print("compose lint: ПРОБЛЕМЫ")
        for e in errs:
            print("  -", e)
        return 2
    print(f"compose lint: OK ({len(names)} сервисов, {len(vols)} томов)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
