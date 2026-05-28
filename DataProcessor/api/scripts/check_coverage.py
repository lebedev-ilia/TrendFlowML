#!/usr/bin/env python3
"""
Скрипт для проверки coverage и генерации отчета

Использование:
    python scripts/check_coverage.py
    python scripts/check_coverage.py --min-coverage 80
    python scripts/check_coverage.py --html
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check test coverage")
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=80,
        help="Minimum coverage percentage (default: 80)"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--fail-under",
        type=int,
        help="Exit with non-zero status if coverage is below this percentage"
    )
    
    args = parser.parse_args()
    
    # Определить корневую директорию API
    api_dir = Path(__file__).parent.parent
    
    # Команда для запуска pytest с coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=api",
        "--cov-report=term-missing",
    ]
    
    if args.html:
        cmd.append("--cov-report=html")
    
    if args.fail_under:
        cmd.append(f"--cov-fail-under={args.fail_under}")
    
    # Добавить путь к тестам
    cmd.append("tests/")
    
    # Запустить pytest
    result = subprocess.run(cmd, cwd=api_dir)
    
    if result.returncode != 0:
        print(f"\n❌ Coverage check failed!")
        print(f"Minimum coverage required: {args.min_coverage}%")
        sys.exit(1)
    
    print(f"\n✅ Coverage check passed!")
    if args.html:
        print(f"HTML report generated in: {api_dir / 'htmlcov' / 'index.html'}")


if __name__ == "__main__":
    main()

