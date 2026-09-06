#!/usr/bin/env bash
# DeepSeek-V4-Flash, base model (MTP OFF), **IN THE GLM IMAGE'S ENGINE**.
# THE FAIRNESS-CLOSING RERUN. This is the arm that makes the GLM-vs-V4 comparison
# an ARCHITECTURE result instead of an engine artifact.
#
#   ./run_nomtp_image.sh          # serves on :8002
#
# Then from the REPO ROOT:
#   MODEL=deepseek-ai/DeepSeek-V4-Flash PORT=8002 \
#   VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
#   MODELDIR=deepseek_v4_flash OUTDIR=deepseek_v4_flash/results/mtp-off-image \
#   QUANT=fp8-attn-dense+mxfp4-experts KV_DTYPE=fp8_ds_mla \
#     ./bench.sh batch context prefix
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
# ---------------------------------------------------------------------------
# The original V4 baseline (results/mtp-off/) ran on the CONDA env:
#     vLLM 0.28.0, torch 2.13.0+cu129
# GLM-5.3-Flash can only run in the IMAGE:
#     vLLM 0.1.dev20051+g487ecf187, torch 2.13.0+cu130
# So every published GLM-vs-V4 ratio mixed an ARCHITECTURE difference with an
# ENGINE + CUDA difference.
#
# ⚠️ THAT CONFOUND IS NOT HYPOTHETICAL -- IT IS MEASURED. In the GLM FP8-KV work,
# swapping ONLY the attention backend (FLASH_ATTN_MLA_SPARSE ->
# FLASHINFER_MLA_SPARSE_SM90, same model, same dtype) cost **27.3% throughput**.
# An engine+CUDA delta is easily that large. A cross-engine architecture claim is
# therefore indefensible until this arm lands.
#
# ---------------------------------------------------------------------------
# WHAT IS HELD FIXED vs THE GLM RUN (the whole point)
# ---------------------------------------------------------------------------
#   engine            /usr/local/bin/vllm 0.1.dev20051+g487ecf187   SAME
#   CUDA / torch      13.0 / 2.13.0+cu130                           SAME
#   TP / EP           8 / on                                        SAME
#   gpu-mem-util      0.82                                          SAME
#   max-model-len     262144                                        SAME
#   max-num-seqs      256                                           SAME  <- see below
#   spec decode       OFF                                           SAME
#   bench grid        bench.sh batch/context/prefix, unique seeds    SAME
#   node              this container, 8xH100-80GB                    SAME
#
# WHAT STILL CANNOT MATCH, AND MUST BE REPORTED:
#   KV dtype   V4 = fp8_ds_mla (its BF16 backend is Blackwell-gated), GLM = BF16.
#              Each model can only run the dtype the other cannot. HARDWARE/KERNEL
#              availability, not a choice.
#   block-size V4 needs 256 (sparse_mla.py per-layer compress_ratios); GLM needs a
#              multiple of 128 and vLLM auto-raises it to 640. Different constraints
#              from different attention designs -- not a knob we are free to match.
#   tokenizer  different vocabularies -> tok/s is not strictly commensurable.
#
# ⚠️ --max-num-seqs 256 IS SET DELIBERATELY. The original V4 baseline passed no
# such flag, so it ran at the H100 auto-default of 1024 (arg_utils.py
# get_batch_defaults: any non-A100 GPU >=70 GiB). GLM is pinned to 256 because its
# 34 KDA layers need one recurrent-state block per sequence. Pinning V4 to 256 here
# removes that asymmetry. It cannot clip any measured point -- the grid tops out at
# concurrency 64.
#
# PORT 8002, not 8001: keeps this arm separable from a GLM server, and 8000 is the
# image entrypoint's Qwen3-0.6B.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Reuse the GLM scaffolding for the NFS-quota cache fixes and GPU preflight --
# those are environment facts, not model facts.
source ../GLM-5.3-Flash/_common.sh

MODEL=deepseek-ai/DeepSeek-V4-Flash
PORT=8002

preflight_gpus 8 || exit 1
preflight_caches || exit 1

launch "v4-nomtp-image" \
  --port "$PORT" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-num-seqs 256 \
  --max-cudagraph-capture-size 256 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.82 \
  --max-model-len 262144 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --no-enable-flashinfer-autotune \
  --enable-mfu-metrics
