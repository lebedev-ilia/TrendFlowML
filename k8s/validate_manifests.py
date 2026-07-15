#!/usr/bin/env python3
"""
Линт согласованности k8s-манифестов (перед `kubectl apply -k k8s/`).

Проверяет внутренние ссылки, которые kubectl НЕ ловит на этапе apply, но которые
ломают рантайм:
  * secretKeyRef.key существует в Secret;
  * configMapRef.name определён как ConfigMap;
  * persistentVolumeClaim.claimName определён как PVC;
  * trendflow-* image присутствует в kustomization images;
  * host-ссылки в env/командах (http://HOST, @HOST, -h HOST) резолвятся в Service.

Запуск:  python k8s/validate_manifests.py   (exit 0 = ок, 2 = есть проблемы)
"""
from __future__ import annotations

import collections
import glob
import os
import re
import sys

try:
    import yaml
except ImportError:
    raise SystemExit("нужен PyYAML")

K8S_DIR = os.path.dirname(os.path.abspath(__file__))
INFRA_HOSTS = {  # внутренние сервисы, чьи host-ссылки должны резолвиться
    "postgres", "redis", "minio", "triton", "embedding-service",
    "fetcher-orchestrator", "dataprocessor", "backend-service",
}
HOST_RE = re.compile(r"(?:https?://|@|-h\s+)([a-z][a-z0-9-]+)(?::\d+|\s|/|$)")


def _iter_docs():
    for f in glob.glob(os.path.join(K8S_DIR, "**", "*.yaml"), recursive=True):
        for d in yaml.safe_load_all(open(f)):
            if isinstance(d, dict):
                yield os.path.relpath(f, os.path.dirname(K8S_DIR)), d


def _walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k, v
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def _strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def main() -> int:
    docs = list(_iter_docs())
    secrets = collections.defaultdict(set)
    configmaps = collections.defaultdict(set)
    pvcs, services = set(), set()
    for _f, d in docs:
        kind = d.get("kind")
        name = (d.get("metadata") or {}).get("name")
        if kind == "Secret":
            secrets[name] |= set((d.get("stringData") or {}).keys()) | set((d.get("data") or {}).keys())
        elif kind == "ConfigMap":
            configmaps[name] |= set((d.get("data") or {}).keys())
        elif kind == "PersistentVolumeClaim":
            pvcs.add(name)
        elif kind == "Service":
            services.add(name)

    images = set()
    kz_path = os.path.join(K8S_DIR, "kustomization.yaml")
    if os.path.isfile(kz_path):
        for im in (yaml.safe_load(open(kz_path)) or {}).get("images", []):
            images.add(im.get("name"))

    errs = []
    for f, d in docs:
        for k, v in _walk(d):
            if k == "secretKeyRef" and isinstance(v, dict):
                nm, key = v.get("name"), v.get("key")
                if nm not in secrets:
                    errs.append(f"{f}: secretKeyRef -> secret '{nm}' не определён")
                elif key not in secrets[nm] and not v.get("optional"):
                    errs.append(f"{f}: secret '{nm}' без ключа '{key}'")
            elif k == "configMapRef" and isinstance(v, dict):
                if v.get("name") not in configmaps:
                    errs.append(f"{f}: configMapRef '{v.get('name')}' не определён")
            elif k == "persistentVolumeClaim" and isinstance(v, dict):
                if v.get("claimName") not in pvcs:
                    errs.append(f"{f}: claimName '{v.get('claimName')}' не определён как PVC")
            elif k == "image" and isinstance(v, str):
                base = v.split(":")[0]
                if base.startswith("trendflow-") and base not in images:
                    errs.append(f"{f}: image '{base}' не в kustomization images")
        for s in _strings(d):
            for m in HOST_RE.finditer(s):
                h = m.group(1)
                if h in INFRA_HOSTS and h not in services and (h + "-service") not in services:
                    errs.append(f"{f}: host '{h}' не резолвится (нет Service '{h}')")

    errs = sorted(set(errs))
    if errs:
        print("k8s manifest lint: ПРОБЛЕМЫ")
        for e in errs:
            print("  -", e)
        return 2
    print(f"k8s manifest lint: OK ({len(docs)} документов, {len(services)} сервисов)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
