#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/qwen3_5_moe_gptq_w8a8.py"
CONFIG="${CONFIG:-${SCRIPT_DIR}/qwen3_5_moe_gptq_w8a8.config.json}"

cd "${REPO_ROOT}"; python "${PY_SCRIPT}" --config "${CONFIG}"
