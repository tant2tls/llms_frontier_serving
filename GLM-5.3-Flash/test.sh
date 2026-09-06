#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-zai-org/GLM-5.3-Flash}"
VLLM_BIN="${VLLM_BIN:-/usr/local/bin/vllm}"

CACHE_ROOT="/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/LLMs_serving_report/cache"

export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$CACHE_ROOT/hub"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"

# Prevent older Transformers settings from overriding the cache.
unset TRANSFORMERS_CACHE

exec "$VLLM_BIN" serve "$MODEL" \
  --port 8001 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.82 \
  --block-size 128 \
  --no-enable-flashinfer-autotune \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.82 \
  --reasoning-parser glm45 