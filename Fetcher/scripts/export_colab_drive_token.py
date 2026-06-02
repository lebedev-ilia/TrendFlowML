#!/usr/bin/env python3
"""Export Drive API credentials for dataset collector worker subprocesses.

Run in Colab *after* drive.mount() and one notebook auth cell:

    from google.colab import auth
    auth.authenticate_user()

Then:

    python scripts/export_colab_drive_token.py \\
        --output-dir /content/drive/MyDrive/dataset_runs/20k-test-2
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
TOKEN_NAME = ".dataset_drive_token.pickle"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Campaign output_dir; token is written as .dataset_drive_token.pickle inside it.",
    )
    args = parser.parse_args(argv)

    try:
        import google.auth
        from google.auth.transport.requests import Request
    except ImportError:
        print("Install: pip install google-auth", file=sys.stderr)
        return 1

    try:
        credentials, _ = google.auth.default(scopes=[DRIVE_SCOPE])
    except Exception as exc:
        print(
            "Could not load Google credentials. In a Colab notebook cell run first:\n"
            "  from google.colab import auth\n"
            "  auth.authenticate_user()\n"
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    if getattr(credentials, "expired", False) and getattr(credentials, "refresh_token", None):
        credentials.refresh(Request())

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / TOKEN_NAME
    token_path.write_bytes(pickle.dumps(credentials))
    print(f"Wrote Drive token for workers: {token_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
