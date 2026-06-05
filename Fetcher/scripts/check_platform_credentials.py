#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fetcher.services.credentials import CredentialsStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fetcher platform credentials configuration.")
    parser.add_argument(
        "--credentials-dir",
        default="fetcher/credentials",
        help="Directory with credential files",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args()

    store = CredentialsStore(credentials_dir=args.credentials_dir)
    summary = store.masked_summary()

    checks = {
        "youtube": bool(store.youtube().api_keys),
        "tiktok_api": store.tiktok().api_configured,
        "tiktok_sdk": store.tiktok().sdk_configured,
        "instagram_api": store.instagram().api_configured,
        "instagram_sdk": store.instagram().sdk_configured,
        "twitch_api": store.twitch().api_configured,
        "twitch_sdk": store.twitch().sdk_configured,
    }
    summary["configured"] = checks

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Credentials dir: {summary['credentials_dir']}")
        for name, ok in checks.items():
            status = "OK" if ok else "MISSING"
            print(f"  {name}: {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
