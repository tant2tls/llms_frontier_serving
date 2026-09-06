#!/usr/bin/env bash
# GLM-5.3-Flash -- BF16 KV **on the FlashInfer 0.6.18 overlay**. THE CONTROL ARM.
#
#   ./run_bf16kv_fi618.sh
#
# ---------------------------------------------------------------------------
# WHY THIS ARM IS MANDATORY, NOT OPTIONAL
# ---------------------------------------------------------------------------
# results/fp8kv-fi618/ is 28-42% SLOWER than results/bf16kv/. Two things changed
# between those runs, not one:
#
#     KV dtype          BF16  ->  fp8
#     attention backend FLASH_ATTN_MLA_SPARSE -> FLASHINFER_MLA_SPARSE_SM90
#     flashinfer        0.6.17 -> 0.6.18 (+ moe_backend forced to deep_gemm)
#
# So "FP8 KV costs ~30% throughput" is NOT a supportable claim from those two dirs.
# The slowdown could be the dtype, the kernel, the MoE backend, or any mix.
#
# THIS ARM HOLDS EVERYTHING FIXED EXCEPT THE KV DTYPE: same overlay, same
# moe_backend=deep_gemm, same flags -- only `--kv-cache-dtype fp8` is dropped.
#
#   results/bf16kv-fi618/  vs  results/fp8kv-fi618/   = CLEAN dtype A/B
#   results/bf16kv/        vs  results/bf16kv-fi618/  = the BACKEND/version effect
#
# Those two deltas decompose the confound. Report both or neither.
#
# ⚠️ Same caveat as the FP8 arm: 0.6.18 Python against the image's 0.6.17 compiled
# artifacts, so FLASHINFER_DISABLE_VERSION_CHECK=1 and moe_backend=deep_gemm are
# required. Not a production configuration. Validate with ../sending.sh chat.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

export PYTHONPATH=/tmp/fi618${PYTHONPATH:+:$PYTHONPATH}
export FLASHINFER_DISABLE_VERSION_CHECK=1

preflight_gpus 8 || exit 1
preflight_caches || exit 1

printf '\n  \033[1mFlashInfer overlay check\033[0m\n'
"$PY_BIN" - <<'PY'
import flashinfer
from vllm.utils.flashinfer import has_flashinfer_sm90_nope_mla as gate
print(f"    flashinfer {flashinfer.__version__}  vllm_gate={gate()}")
raise SystemExit(0 if gate() else 1)
PY
[[ $? == 0 ]] || { printf '  \033[31mFAIL\033[0m overlay not active\n'; exit 1; }

# NO --kv-cache-dtype  -> auto -> BF16. Everything else identical to the FP8 arm.
launch "bf16kv-fi618" "${COMMON_ARGS[@]}" \
  --kernel-config '{"moe_backend":"deep_gemm"}'
