#!/bin/bash
# Запуск анализа результатов voice_quality_extractor после прогона тестов

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"}"
RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/.data_venv/bin/python"}"
ANALYZER="${SCRIPT_DIR}/../utils/analyze_all_results.py"

cd "${BASE_DIR}/DataProcessor"

echo "=========================================="
echo "Анализ результатов voice_quality_extractor"
echo "=========================================="

PYTHONPATH="${BASE_DIR}/DataProcessor/AudioProcessor/src:${PYTHONPATH}" "${PYTHON}" "${ANALYZER}" \
    --rs-base "${RESULTS_DIR}/youtube"

echo ""
echo "=========================================="
echo "Готово"
echo "=========================================="
