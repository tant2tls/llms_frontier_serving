#!/usr/bin/env bash
# GLM-5.3-Flash -- FP8 KV cache (fp8_ds_mla). **SECOND DTYPE ARM. MAY NOT BOOT.**
#
#   ./run_fp8kv.sh              # serve on :8001
#   DRY_RUN=1 ./run_fp8kv.sh
#
# Then, from the REPO ROOT:
#   MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
#   MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/fp8kv \
#     ./bench.sh batch context prefix
#
# ---------------------------------------------------------------------------
# WHAT THIS ARM IS TESTING, AND WHY IT IS EXPECTED TO BE FRAGILE
# ---------------------------------------------------------------------------
# The recipe says Hopper "must run BF16 KV" for this model. This arm exists to
# test that claim directly instead of trusting it, and to get a same-model
# BF16-vs-FP8 KV A/B that isolates the KV dtype from everything else.
#
# The reasoning, from source in THIS image:
#
#   1) The backend that gives FP8 KV *with in-kernel dequant* on Hopper is
#      FLASHINFER_MLA_SPARSE_SM90. It is GATED OFF here: it feature-probes for
#      `ckv_scale_arr` on BatchMLAPagedAttentionWrapper.run (FlashInfer >= 0.6.18)
#      and this image ships 0.6.17, so has_flashinfer_sm90_nope_mla() == False.
#      Verified by running the probe, not by reading the version number.
#
#   2) That leaves FLASHMLA_SPARSE, which DOES advertise fp8:
#         supported_kv_cache_dtypes = ["auto","bfloat16","fp8_ds_mla","fp8"]
#         supports_compute_capability: capability.major in [9, 10]   <- SM90 ok
#         get_supported_kernel_block_sizes: [64]
#
#   3) ⚠️ THE BLOCK-SIZE CONFLICT IS THE LIKELY FAILURE MODE. FLASHMLA_SPARSE
#      advertises exactly [64], but GLM's kpool indexer REQUIRES a multiple of 128
#      (index_kpool=4 -> block_size % 128 == 0, attention.py:140-150). 64 is not a
#      multiple of 128 and 128 is not in [64], so the two constraints may be
#      unsatisfiable at once. vllm/v1/worker/utils.py resolves a common kernel
#      block size and RAISES "No common block size" when none exists. If that is
#      what happens, this arm is impossible on this image -- which is itself the
#      finding, and confirms the recipe's Hopper note from the other direction.
#
#   4) `fp8_ds_mla` is a 656-byte-per-token DeepSeek-V3.2 storage format. GLM-5.3's
#      KV is a NoPE 512-latent + kpool-compressed indexer -- a DIFFERENT structure.
#      Even if it boots, the layout may not mean what it means for V4, so DO NOT
#      report this as "matched to the V4 run" without saying so.
#
# WHATEVER HAPPENS, RECORD IT. A clean failure with the exact error message is a
# real result (rule 7: don't invent results, state what didn't run and why). The
# log lands in logs/serve_fp8kv_*.log.

# ===========================================================================
# ⚠️ RESOLVED 2026-09-02: THIS ARM IS IMPOSSIBLE. DO NOT SPEND A RUN ON IT.
# ===========================================================================
# Result + full evidence: results/fp8kv/RESULT-arm-impossible.md
#
# It failed at the FIRST KV WRITE, not at backend selection:
#     RuntimeError: concat_and_cache_mla, cache_kernels.cu:866,
#                   pe_dim must be 64 for fp8_ds_mla
#
# CAUSE IS ARCHITECTURAL, NOT A VERSION GATE. `fp8_ds_mla` is DeepSeek-V3.2's KV
# layout and hardcodes a decoupled-RoPE dim of 64. GLM-5.3 is a **NoPE** model:
# config.json text_config has qk_rope_head_dim = 0 and mla_use_nope = true. So
# pe_dim = 0 != 64, no flag can change it, and this layout cannot represent GLM's
# KV on ANY hardware -- not just SM90.
#
# ⚠️ THE BLOCK-SIZE PREDICTION BELOW WAS WRONG. It is kept because a well-reasoned
# wrong hypothesis is worth recording: FLASHMLA_SPARSE resolved WITHOUT complaint
# (vLLM auto-raises block size to 640, legal for both the [64] kernel and the
# kpool's multiple-of-128 rule), so "No common block size" never happened.
# ===========================================================================

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

preflight_gpus 8 || exit 1
preflight_caches || exit 1

cat <<'EOS'

  NOTE: this arm may fail at startup by design. Two things to look for in the log:
    - "No common block size"        -> block-size constraints are unsatisfiable
                                       (FLASHMLA_SPARSE wants 64, kpool wants 128)
    - "only supports fp8 kv-cache" / a backend-selection error
                                       -> fp8 rejected for this model on SM90
  Either outcome is a reportable finding. Do not retry with --block-size 64:
  that violates the kpool assert and fails deeper, with a less legible error.

EOS

# --kv-cache-dtype fp8_ds_mla : the explicit format name rather than the `fp8`
# alias, so the log and manifest record WHICH fp8 layout was requested.
launch "fp8kv" "${COMMON_ARGS[@]}" --kv-cache-dtype fp8_ds_mla
