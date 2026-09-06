#!/usr/bin/env bash
# Qwen3.8-Flash-Next-FP8 -- base model, spec decode OFF. **THIS IS THE HEADLINE ARM.**
#
#   ./run.sh                      # serve on :8001 over GPUs 4,5,6,7 at TP4
#   DRY_RUN=1 ./run.sh            # print the command, launch nothing
#   GPUS=0,1,2,3,4,5,6,7 TP=8 ./run.sh    # only after GPU 0 is freed (see _common.sh)
#
# Then, from the REPO ROOT -- note BOTH client overrides, they are load-bearing:
#
#   MODEL=Qwen/Qwen3.8-Flash-Next-FP8 PORT=8001 \
#   VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
#   MODELDIR=Qwen3.8-Flash-Next-FP8 OUTDIR=Qwen3.8-Flash-Next-FP8/results/base-tp4 \
#     ./bench.sh batch context prefix
#
# bench.sh defaults VLLM=/PY= to the conda env vllm-py12 (vLLM 0.28.0), which
# does not know model_type=qwen4_exp. `vllm bench serve` tokenizes prompts and
# reads max_model_len from the config, so the OLD CLIENT FAILS on this model
# even though the SERVER is fine. Override both or the sweep dies in preflight.
#
# ---------------------------------------------------------------------------
# WHAT THIS ARM CAN AND CANNOT MATCH IN THE V4 / GLM COMPARISON
# ---------------------------------------------------------------------------
# Matched deliberately:
#   - --max-model-len 262144      identical to the GLM-5.3-Flash arm
#   - vision tower disabled       text-to-text, same fairness flag as GLM-5.3
#   - spec decode OFF             base-model headline (rule 1)
#   - bench grid, unique seeds, --ignore-eos: the shared bench.sh
#
# NOT matched, and NOT matchable -- report, do not paper over:
#
#   1) KV DTYPE. Every model in this comparison is pinned to a DIFFERENT KV dtype
#      by kernel availability on SM90, and no two of them overlap:
#        V4-Flash  : fp8_ds_mla ONLY (its BF16 backend is gated to Blackwell)
#        GLM-5.3   : BF16 ONLY (FlashInfer 0.6.17 < 0.6.18 gates the FP8 path off)
#        Qwen3.8   : BF16 ONLY (every QSA backend declares
#                    supported_kv_cache_dtypes = ["auto","bfloat16"], and the
#                    impl raises NotImplementedError otherwise; the indexer also
#                    requires BF16 model dtype outright)
#      Qwen3.8 and GLM-5.3 therefore MATCH each other on KV dtype -- that pair is
#      the clean KV comparison. V4 is the odd one out.
#
#   2) GPU COUNT. This arm runs TP4 on GPUs 4-7 because the image entrypoint
#      (PID 1) squats ~77 GiB on GPU 0 and cannot be killed without risking the
#      container. V4-Flash and GLM-5.3 were both TP8. So RAW tok/s IS NOT
#      COMPARABLE ACROSS THESE RUNS -- use tok/s per GPU (this project's rule).
#      TP4 is arguably the FAIRER per-GPU number here: 172.76 GiB of weights need
#      ~3 H100s minimum, so TP4 is close to a real deployment, whereas V4-Flash's
#      TP8 on a 3-GPU-capable model made its per-GPU figure pessimistic.
#
#   3) gpu_memory_utilization 0.90 vs 0.82 for V4/GLM. Recipe-sanctioned for TP4.
#      Affects KV CAPACITY only, not the decode cost model.
#
#   4) TOKENIZER. vocab_size=248,320 here vs V4's 129,536. A denser tokenizer
#      does more work per token, so cross-family tok/s is not commensurable --
#      already a standing caveat in this project, and it is larger here than
#      anywhere else in the set.
#
# ---------------------------------------------------------------------------
# WHY THIS CHECKPOINT IS NOT THE "DENSE CONTROL" THE PLAN EXPECTED
# ---------------------------------------------------------------------------
# ⚠️ CLAUDE.md's architecture table lists "Qwen3.8-27B | 64 layers | dense" and
# designates it the mandatory dense control. THAT IS A DIFFERENT MODEL. Measured
# from THIS checkpoint's config.json and safetensors headers:
#
#   architectures      : Qwen4ExpForConditionalGeneration  (model_type qwen4_exp)
#   num_hidden_layers  : 48   (NOT 64)
#   layer_types        : 36 linear_attention + 12 full_attention, interval 4
#   num_experts        : 512, num_experts_per_tok 10  ->  E/k = 51.2x   <- MoE!
#   weights            : 172.76 GiB, 94.1% of bytes FP8 e4m3, 5.9% BF16
#   params             : 180.0 B total (152,089 tensors, 131 shards)
#
# So this is a 512-expert MoE, not a dense model. **The dense control is still
# missing** and the layer-reduction ladder still has no validation gate. Do not
# let this run be reported as the control -- it is a third MoE data point.
#
# What it uniquely adds instead: the most internally heterogeneous stack in the
# set, which is exactly the report's thesis. One decode step here touches
#   36 GDN linear-attention layers  -> O(1) recurrent state, 0.1055 GiB/seq
#   12 full-attention layers        -> O(ctx) KV at 24.00 KiB/tok BF16
#   12 QSA compressed-key caches    -> O(ctx/4) at 0.75 KiB/tok
#   1  PLE n-gram layer (layer 2)   -> 51.2 B params of EMBEDDING, 28.5% of the
#                                      model, touched by table lookup, not GEMM
# Four different per-layer cost classes and three different memory-growth laws in
# one forward pass. The uniform per-layer cost model is not just wrong here, it
# is wrong four ways.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

preflight_gpus || exit 1
preflight_caches || exit 1

# NO --kv-cache-dtype: "auto" resolves to the model dtype (BF16), the only KV
# dtype any QSA backend in this build accepts. Passing bfloat16 explicitly is
# rejected by vLLM for models that do not default to fp8, so omission is the
# correct way to ask for BF16 here (same as the GLM-5.3 arm).
launch "base-tp${TP}" "${COMMON_ARGS[@]}"
