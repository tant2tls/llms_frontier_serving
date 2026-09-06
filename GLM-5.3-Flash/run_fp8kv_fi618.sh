#!/usr/bin/env bash
# GLM-5.3-Flash -- FP8 KV via the SM90 NoPE sparse-MLA backend, unlocked by
# overlaying FlashInfer 0.6.18. **SECOND DTYPE ARM, TAKE 2.**
#
#   ./run_fp8kv_fi618.sh
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS, AND WHY THE FIRST FP8 ATTEMPT WAS THE WRONG EXPERIMENT
# ---------------------------------------------------------------------------
# run_fp8kv.sh asked for `--kv-cache-dtype fp8_ds_mla` and died with
#     pe_dim must be 64 for fp8_ds_mla
# That was a REAL result but the WRONG ROUTE. `fp8_ds_mla` is DeepSeek-V3.2's KV
# layout and assumes a decoupled-RoPE dim of 64; GLM-5.3 is NoPE
# (qk_rope_head_dim=0), so that layout can never represent its KV.
#
# THERE IS A SECOND FP8-KV PATH BUILT FOR EXACTLY THIS CASE:
# FLASHINFER_MLA_SPARSE_SM90. Its architectural requirements
# (flashinfer_mla_sparse_sm90.py:150-158) are:
#     kv_lora_rank == 512                  GLM: 512   ✅
#     qk_rope_head_dim in (0, 64)          GLM: 0     ✅  <- NoPE EXPLICITLY allowed
#     hasattr(hf, "index_topk")            GLM: 2048  ✅
# GLM-5.3 satisfies ALL THREE. cuda.py:150-157 even PREFERS this backend for
# qk_rope_head_dim==0 sparse models. The only thing gating it is a FlashInfer
# feature probe for the `ckv_scale_arr` kwarg on
# BatchMLAPagedAttentionWrapper.run, added in FlashInfer >= 0.6.18. The image
# ships 0.6.17 -> has_flashinfer_sm90_nope_mla() == False.
#
# So "FP8 KV is impossible for GLM-5.3 on Hopper" was WRONG as an architectural
# claim. It is a LIBRARY-VERSION limitation on this image, and it is testable.
#
# ⚠️ WHAT THIS ARM CONFOUNDS -- STATE IT WITH EVERY NUMBER.
# The BF16 baseline (results/bf16kv/) ran on FlashInfer 0.6.17. This arm runs on
# 0.6.18 via a PYTHONPATH overlay, so an FP8-vs-BF16 delta measured against that
# baseline mixes TWO changes: the KV dtype AND the FlashInfer version (which also
# changes the selected ATTENTION BACKEND, from FLASH_ATTN_MLA_SPARSE to
# FLASHINFER_MLA_SPARSE_SM90). That is a backend swap, not just a dtype swap.
# To get a clean dtype A/B you must ALSO re-run BF16 under this same overlay --
# see run_bf16kv_fi618.sh. Do not publish the cross-version delta as "FP8 KV costs/
# saves X" without that control.
#
# ⚠️ FLASHINFER_DISABLE_VERSION_CHECK=1 is REQUIRED and is a REAL RISK.
# flashinfer 0.6.18 wants flashinfer-cubin 0.6.18, but this mirror only carries
# cubin up to 0.6.13, so the overlay falls back to the image's 0.6.17 cubins. The
# version check is bypassed deliberately. If you see kernel-level garbage,
# illegal-memory-access, or nonsense outputs, SUSPECT THIS FIRST -- validate
# correctness with ../sending.sh chat before trusting any throughput number.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

# The overlay must precede the image's site-packages.
export PYTHONPATH=/tmp/fi618${PYTHONPATH:+:$PYTHONPATH}
export FLASHINFER_DISABLE_VERSION_CHECK=1

preflight_gpus 8 || exit 1
preflight_caches || exit 1

printf '\n  \033[1mFlashInfer overlay check\033[0m\n'
"$PY_BIN" - <<'PY'
import flashinfer, inspect
from flashinfer.mla import BatchMLAPagedAttentionWrapper as W
from vllm.utils.flashinfer import has_flashinfer_sm90_nope_mla as gate
ok = "ckv_scale_arr" in inspect.signature(W.run).parameters
print(f"    flashinfer {flashinfer.__version__}  ckv_scale_arr={ok}  vllm_gate={gate()}")
raise SystemExit(0 if gate() else 1)
PY
[[ $? == 0 ]] || { printf '  \033[31mFAIL\033[0m SM90 NoPE MLA still gated off -- overlay not active\n'; exit 1; }

# `fp8` (not fp8_ds_mla): on this backend it means FP8 storage for the 512-dim
# latent with IN-KERNEL DEQUANT, which is the whole point.
# ⚠️ --kernel-config moe_backend=deep_gemm IS REQUIRED WITH THIS OVERLAY.
# With moe_backend=auto, vLLM picks a FlashInfer fused-MoE kernel. The overlay's
# 0.6.18 PYTHON then calls into the image's pinned 0.6.17 COMPILED artifacts
# (flashinfer_cubin 0.6.17 + flashinfer_jit_cache 0.6.17+cu130) and the ABI does not
# match:
#     TypeError: Mismatched number of arguments when calling `init(...)`:
#               Expected 8 but got 9 arguments
#     from /tmp/fi618/flashinfer/fused_moe/core.py:693
# This mirror has no cubin > 0.6.13 and cannot reach flashinfer-jit-cache at all, so
# the binaries CANNOT be upgraded to match. Routing MoE to DeepGEMM sidesteps
# FlashInfer's MoE path while still letting the FLASHINFER_MLA_SPARSE_SM90 ATTENTION
# backend (the thing we actually want) come from the overlay.
launch "fp8kv-fi618" "${COMMON_ARGS[@]}" --kv-cache-dtype fp8 \
  --kernel-config '{"moe_backend":"deep_gemm"}'
