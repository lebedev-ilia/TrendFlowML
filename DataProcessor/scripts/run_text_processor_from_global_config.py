#!/usr/bin/env python3
"""
Запуск TextProcessor/run_cli.py с теми же CLI-аргументами, что генерирует GlobalConfigParser.

Важно: нельзя подставлять get_text_cli_args() в bash через $(python ...), если значения
содержат пробелы (типичный json.dumps для --devices-config-json / --extractor-params-json) —
shell разобьёт строку на слова и argparse увидит «левые» токены.

Использование (из каталога DataProcessor):

  export DP_MODELS_ROOT="$PWD/dp_models/bundled_models"
  ./TextProcessor/.tp_venv/bin/python scripts/run_text_processor_from_global_config.py \\
    --input-json /path/to/video_document.json \\
    --video-id=-Q6fnPIybEI \\
    --rs-base /path/to/result_store_parent
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    dp_root = Path(__file__).resolve().parent.parent
    default_cfg = dp_root / "configs" / "global_config.yaml"
    default_tp_venv = dp_root / "TextProcessor" / ".tp_venv" / "bin" / "python"
    default_run_cli = dp_root / "TextProcessor" / "run_cli.py"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--global-config", type=Path, default=default_cfg)
    p.add_argument("--python", type=Path, default=default_tp_venv, help="Интерпретатор (.tp_venv)")
    p.add_argument("--run-cli", type=Path, default=default_run_cli)
    p.add_argument("--input-json", type=Path, required=True)
    p.add_argument("--video-id", type=str, required=True)
    p.add_argument("--rs-base", type=Path, required=True)
    p.add_argument("--platform-id", type=str, default="youtube")
    p.add_argument("--run-id", type=str, default="pilot_from_global_yaml")
    p.add_argument("--config-hash", type=str, default="from_global_config_yaml")
    p.add_argument("--dataprocessor-version", type=str, default="local_pilot")
    p.add_argument("--sampling-policy-version", type=str, default="v1")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать команду (argv), без запуска",
    )
    args = p.parse_args()

    sys.path.insert(0, str(dp_root))
    from configs.config_parser import GlobalConfigParser  # noqa: E402

    cfg_path = args.global_config.resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2

    text_extra = GlobalConfigParser(str(cfg_path)).get_text_cli_args()

    cmd: list[str] = [
        str(args.python),
        str(args.run_cli),
        "--rs-base",
        str(args.rs_base.resolve()),
        "--platform-id",
        args.platform_id,
        f"--video-id={args.video_id}",
        "--run-id",
        args.run_id,
        "--sampling-policy-version",
        args.sampling_policy_version,
        "--config-hash",
        args.config_hash,
        "--dataprocessor-version",
        args.dataprocessor_version,
        "--input-json",
        str(args.input_json.resolve()),
    ]
    cmd.extend(text_extra)

    if args.dry_run:
        for i, a in enumerate(cmd):
            print(f"{i}\t{a}")
        return 0

    env = os.environ.copy()
    if not env.get("DP_MODELS_ROOT", "").strip():
        bundled = dp_root / "dp_models" / "bundled_models"
        if bundled.is_dir():
            env["DP_MODELS_ROOT"] = str(bundled.resolve())

    r = subprocess.run(cmd, cwd=str(dp_root), env=env)
    return int(r.returncode) if r.returncode is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
