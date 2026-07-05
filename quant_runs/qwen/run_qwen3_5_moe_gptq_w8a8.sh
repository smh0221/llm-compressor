#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/qwen3_5_moe_gptq_w8a8.py"
CONFIG="${CONFIG:-${SCRIPT_DIR}/qwen3_5_moe_gptq_w8a8.config.yaml}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

cd "${REPO_ROOT}"
python "${PY_SCRIPT}" config="${CONFIG}" 2>&1 | tee "${LOG_FILE}"
