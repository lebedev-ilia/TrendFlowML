#!/usr/bin/env python3
"""
Полный локальный прогон VisualProcessor: все core_providers и modules из шаблона global_config.yaml.
Audio и Text отключены (только Segmenter → кадры → Visual).

Требования (как у e2e_full_max_run.py без --visual-minimal):
  - Triton с CLIP / MiDaS / RAFT / Places365 и др. (TRITON_HTTP_URL, по E2E часто 8010)
  - GPU там, где это нужно модулю
  - embedding_service_url для brand/car/face_identity/place (см. inline_config в global_config)
  - micro_emotion: Docker OpenFace (может упасть без Docker)

Запуск (из корня репозитория или из DataProcessor)::

    export TRITON_HTTP_URL=http://127.0.0.1:8010   # или только порт E2E:
    export TRITON_E2E_HTTP_PORT=8010
    python scripts/run_visual_full_test.py --video-path /path/to/video.mp4

или из каталога DataProcessor::

    python scripts/run_visual_full_test.py --video-path ../example/example_videos/-Q6fnPIybEI.mp4

С `--write-config` только записать сгенерированный YAML (без main.py).
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _dp_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_triton_http_url(cli_value: str | None) -> str:
    """Triton HTTP base URL for visual_full_test.

    Приоритет: ``--triton-http-url``, затем ``TRITON_HTTP_URL`` / ``TRITON_HTTP``,
    затем ``http://127.0.0.1:$TRITON_E2E_HTTP_PORT`` (как в E2E; по умолчанию порт 8010).

    Без этого в конфиге остаётся placeholder из ``global_config.yaml``
    (``http://localhost:8000`` — порт Fetcher, не Triton).
    """
    if cli_value and str(cli_value).strip():
        return str(cli_value).strip()
    for key in ("TRITON_HTTP_URL", "TRITON_HTTP"):
        u = (os.environ.get(key) or "").strip()
        if u:
            return u
    port = (os.environ.get("TRITON_E2E_HTTP_PORT") or "8010").strip()
    return f"http://127.0.0.1:{port}"


def _enable_full_visual_inline(cfg: dict) -> None:
    vis = cfg.get("processors", {}).get("visual")
    if not isinstance(vis, dict):
        return
    ic = vis.get("inline_config")
    if not isinstance(ic, dict):
        return
    cp = ic.get("core_providers")
    if isinstance(cp, dict):
        for k in cp:
            cp[k] = True
    md = ic.get("modules")
    if isinstance(md, dict):
        for k in md:
            md[k] = True


def build_visual_only_config(
    *,
    triton_http_url: str | None,
    dataproc_version: str,
) -> dict:
    root = _repo_root_from_here()
    src = root / "DataProcessor" / "configs" / "global_config.yaml"
    if not src.is_file():
        raise FileNotFoundError(f"Missing template: {src}")

    import yaml

    with open(src, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(cfg)
    g = cfg.setdefault("global", {})
    g["dataprocessor_version"] = dataproc_version

    procs = cfg.setdefault("processors", {})
    for name in ("audio", "text"):
        p = procs.setdefault(name, {})
        if isinstance(p, dict):
            p["enabled"] = False
    vis = procs.setdefault("visual", {})
    if not isinstance(vis, dict):
        raise ValueError("processors.visual must be a mapping")
    vis["enabled"] = True
    vis["required"] = False

    _enable_full_visual_inline(cfg)

    if triton_http_url:
        ic = vis.setdefault("inline_config", {})
        glob = ic.setdefault("global", {})
        if isinstance(glob, dict):
            glob["triton_http_url"] = triton_http_url

    return cfg


def main() -> int:
    dp = _dp_root()
    root = _repo_root_from_here()
    p = argparse.ArgumentParser(description="Run DataProcessor with all Visual components enabled.")
    p.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Входное видео для Segmenter (или задайте VISUAL_FULL_TEST_VIDEO).",
    )
    p.add_argument(
        "--rs-base",
        type=Path,
        default=dp / "dp_results" / "visual_full_test",
        help="Каталог result_store для прогона",
    )
    p.add_argument(
        "--triton-http-url",
        default=None,
        help=(
            "Triton HTTP URL. По умолчанию: TRITON_HTTP_URL, иначе "
            "http://127.0.0.1:$TRITON_E2E_HTTP_PORT (default 8010). "
            "Иначе в YAML остаётся localhost:8000 (Fetcher — неверно для infer)."
        ),
    )
    p.add_argument(
        "--dataprocessor-version",
        default="visual_full_test",
        help="global.dataprocessor_version в сгенерированном конфиге",
    )
    p.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help="Если задано — только записать YAML по этому пути и выйти",
    )
    args = p.parse_args()

    import yaml

    cfg = build_visual_only_config(
        triton_http_url=_resolve_triton_http_url(args.triton_http_url),
        dataproc_version=str(args.dataprocessor_version),
    )

    if args.write_config:
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"Wrote {args.write_config}")
        return 0

    vid = args.video_path
    if vid is None:
        raw = (os.environ.get("VISUAL_FULL_TEST_VIDEO") or "").strip()
        vid = Path(raw) if raw else None
    if not vid or not vid.is_file():
        print(
            "FAIL: укажите существующий файл: --video-path … "
            "или переменную VISUAL_FULL_TEST_VIDEO\n"
            f"  пример: example/example_videos/-Q6fnPIybEI.mp4 (от корня репо: {root})",
            file=sys.stderr,
        )
        return 2

    args.rs_base.mkdir(parents=True, exist_ok=True)
    cfg_path = args.rs_base / "visual_full_global_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    main_py = dp / "main.py"
    cmd = [
        sys.executable,
        str(main_py),
        "--video-path",
        str(vid.resolve()),
        "--global-config",
        str(cfg_path),
        "--rs-base",
        str(args.rs_base.resolve()),
        "--platform-id",
        str(cfg.get("global", {}).get("platform_id") or "youtube"),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(dp))


if __name__ == "__main__":
    raise SystemExit(main())
