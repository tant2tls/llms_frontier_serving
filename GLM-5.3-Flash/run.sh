#!/usr/bin/env bash
# GLM-5.3-Flash -- BF16 KV cache. **THIS IS THE HEADLINE ARM.**
#
#   ./run.sh                # serve on :8001
#   DRY_RUN=1 ./run.sh      # print the command, launch nothing
#
# Then, from the REPO ROOT:
#   MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
#   MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/bf16kv \
#     ./bench.sh batch context prefix
#
# ---------------------------------------------------------------------------
# WHY BF16 KV IS THE HEADLINE, AND WHY IT CANNOT MATCH THE V4-FLASH RUN
# ---------------------------------------------------------------------------
# The vLLM recipe (recipes.vllm.ai/zai-org/GLM-5.3-Flash) states plainly:
#
#     "Hopper does not support FP8 KV cache for this model and must run BF16 KV."
#
# Verified in THIS image rather than taken on faith. FP8 KV for GLM-5.3 needs the
# FLASHINFER_MLA_SPARSE_SM90 backend, which feature-probes FlashInfer for the
# `ckv_scale_arr` kwarg on BatchMLAPagedAttentionWrapper.run (>= 0.6.18):
#
#     flashinfer-python          0.6.17     <- this image
#     has_flashinfer_sm90_nope_mla()  ->  False
#     run() params: [... 'o_scale', 'ckv_scale', 'kpe_scale']   # no ckv_scale_arr
#
# So that backend is gated OFF and the FP8-KV-with-in-kernel-dequant path is
# unavailable. run_fp8kv.sh probes the other route (fp8_ds_mla on FLASHMLA_SPARSE)
# as an explicit second arm -- see that file.
#
# ⚠️ THE V4 COMPARISON HAS AN UNCLOSABLE KV-DTYPE ASYMMETRY. Do not paper over it:
#
#   DeepSeek-V4-Flash on H100 : fp8_ds_mla ONLY. Its BF16-KV backend
#     (DeepseekV4FlashInferMLASparseBackend, use_fp8_ds_mla_layout=False) is gated
#     to `capability.major in [10, 12]` -- Blackwell/SM120 only
#     (models/deepseek_v4/nvidia/flashinfer_sparse.py:113). On SM90 the SM90
#     backend hard-sets use_fp8_ds_mla_layout=True.
#   GLM-5.3-Flash on H100     : BF16 ONLY (the FlashInfer gate above).
#
# Each model can only run the dtype the other cannot. This is HARDWARE/KERNEL
# AVAILABILITY, not a choice, and it must be REPORTED rather than resolved.
#
# Why it is nonetheless a survivable mismatch:
#   - KV is not the binding resource in either run. GLM BF16 KV is ~11.35 KiB/tok
#     (11 DSA layers x 512 latent x 2 B = 11.00, + kpool indexer ~0.35), so a
#     131K sequence is ~1.42 GiB against ~27 GiB/GPU free. V4 measured peak KV
#     30.2% even at 256K x 8.
#   - The WEIGHT dtype -- which drives the expert-read term the report is actually
#     about -- is FP8 e4m3 on both. That comparison is intact.
#   - The remaining weight difference (V4's MXFP4 experts vs GLM's FP8 + a
#     1509-entry modules_to_not_convert BF16 exclusion list) is far larger than
#     the KV term and had to be recorded anyway.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

preflight_gpus 8 || exit 1
preflight_caches || exit 1

# NO --kv-cache-dtype flag at all. "auto" resolves to the model dtype (BF16) for
# GLM-5.3; passing `--kv-cache-dtype bfloat16` explicitly is rejected by vLLM for
# models that do not default to fp8 ("this is an invalid option for models that do
# not default to fp8"), so omission is the correct way to ask for BF16 here.
#
# EAGER=1 ./run.sh  -- fallback for the cudagraph-capture crash (see _common.sh's
# open-issue block). --enforce-eager skips capture entirely, so it TRADES DECODE
# THROUGHPUT for the ability to boot at all. ⚠️ Numbers from an eager server are NOT
# comparable to the DeepSeek-V4-Flash baseline, which ran WITH cudagraphs: eager
# costs per-step kernel-launch overhead that hurts small-batch decode most. If the
# headline numbers come from here, the whole GLM arm must be LABELLED EAGER, and the
# V4 comparison needs a matching eager rerun to stay honest.
if [[ ${EAGER:-0} == 1 ]]; then
  printf '\n  \033[33mnote\033[0m EAGER=1 -- cudagraphs DISABLED. Not comparable to the\n'
  printf '        cudagraph-enabled V4-Flash baseline. Label results accordingly.\n'
  launch "bf16kv-eager" "${COMMON_ARGS[@]}" --enforce-eager
else
  launch "bf16kv" "${COMMON_ARGS[@]}"
fi
