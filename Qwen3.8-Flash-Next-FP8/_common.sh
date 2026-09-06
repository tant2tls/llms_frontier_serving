#!/usr/bin/env bash
# Shared launch scaffolding for the Qwen3.8-Flash-Next-FP8 arms. Sourced, not executed.
#
#   run.sh      base model, spec decode OFF  <- HEADLINE (primary numbers)
#   run_mtp.sh  same + MTP                   <- spec-decode A/B, never headline
#
# Every non-default flag is justified in README.md. The ones that are NOT free
# choices are explained inline because getting them wrong fails confusingly.

set -uo pipefail

MODEL=${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}

# ---------------------------------------------------------------------------
# 1) THE ENGINE MUST BE THE IMAGE'S, NOT THE CONDA ENV'S.
# ---------------------------------------------------------------------------
# This container (vllm/vllm-openai:qwen38-flash-next) ships a dev build that
# REGISTERS qwen4_exp / Qwen3_8FlashNext*; the conda env vllm-py12 does not:
#
#   image  /usr/local/bin/vllm   0.1.dev20073+g8e685d198  -> Qwen3_8FlashNext* present
#   conda  .../vllm-py12/bin/vllm  0.28.0                 -> ABSENT
#
# The shell profile activates vllm-py12 and puts it FIRST on PATH, so a bare
# `vllm serve` silently resolves to 0.28.0 and dies with "architecture not
# supported". Always use the absolute image path.
#
# ⚠️ THIS APPLIES TO THE BENCH CLIENT TOO. `vllm bench serve` builds random
# prompts through the model's tokenizer and reads its config for max_model_len,
# so the 0.28.0 client cannot parse model_type=qwen4_exp either. bench.sh
# defaults VLLM=/PY= to the conda env -- OVERRIDE BOTH (see run.sh header).
VLLM_BIN=${VLLM_BIN:-/usr/local/bin/vllm}
PY_BIN=${PY_BIN:-/usr/bin/python3}

# ---------------------------------------------------------------------------
# 2) PORT 8001, NOT 8000.
# ---------------------------------------------------------------------------
# The image ENTRYPOINT is itself `vllm serve` (PID 1) serving the default
# Qwen/Qwen3-0.6B. It answers /health 200 on port 8000 and parks ~77 GiB on
# GPU 0 via its VLLM::EngineCore child (PID 403). We cannot kill PID 1 -- it is
# container init. See preflight_gpus below.
PORT=${PORT:-8001}

# ---------------------------------------------------------------------------
# 3) GPU SET AND TP DEGREE ARE COUPLED -- and GPU 0 is NOT usable.
# ---------------------------------------------------------------------------
# The entrypoint squatter leaves GPU 0 with ~4 GiB free, so any layout that
# includes GPU 0 cannot hold its 21.6 GiB (TP8) shard. Default to the four
# GPUs the squatter does not touch.
#
# FP8 weights are 172.76 GiB (measured from the safetensors index), so:
#   TP4 -> 43.2 GiB/GPU of 79.6 GiB  => fits, ~28 GiB left for KV+state at 0.90
#   TP8 -> 21.6 GiB/GPU              => fits, but NEEDS GPU 0 freed first
#
# ⚠️ TP MUST BE PAIRED WITH EXPERT PARALLEL ON THIS CHECKPOINT. The recipe
# (recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next-FP8) states for Hopper: "Plain TP8 is
# incompatible with the FP8 checkpoint; use TEP8" -- the cause is the
# checkpoint's 128-wide FP8 quantization blocks, which plain TP would split
# across ranks mid-block. --enable-expert-parallel is therefore MANDATORY here,
# not a throughput tuning knob as it was for V4-Flash.
GPUS=${GPUS:-4,5,6,7}
TP=${TP:-4}
export CUDA_VISIBLE_DEVICES=$GPUS

# Recipe-sanctioned utilization per layout: 0.90 at TP4, 0.85 at TP8 (Hopper).
# NOTE this differs from the 0.82 used for V4-Flash and GLM-5.3. It changes KV
# CAPACITY, not the decode cost model -- and V4 peaked at 30.2% KV even at
# 256K x 8, so KV was not the binding resource there either. Record it; do not
# silently treat the runs as identically provisioned.
if [[ -z ${GPU_UTIL:-} ]]; then
  if [[ $TP -ge 8 ]]; then GPU_UTIL=0.85; else GPU_UTIL=0.90; fi
fi

# ---------------------------------------------------------------------------
# 4) ABSOLUTE HF_HOME.
# ---------------------------------------------------------------------------
# The inherited env has HF_HOME=cache / HF_HUB_CACHE=cache/hub -- RELATIVE paths
# that resolve against $PWD, so from a $HOME-side directory a 173 GiB download
# lands on the fleet-shared NFS export (623 GiB free, 93% full). The model is
# ALREADY cached (174 GiB, snapshot 236dfdf2) under the absolute path below.
# Do not re-download.
export HF_HOME=/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/LLMs_serving_report/cache
export HF_HUB_CACHE=$HF_HOME/hub

# ===========================================================================
# PORTED FROM GLM-5.3-Flash/_common.sh (2026-09-02) -- DO NOT DROP THIS.
# Without it the engine dies ~8 min into startup with
#     RuntimeError: Worker failed with error '[Errno 122] Disk quota exceeded'
# or a MISLEADING bare `CUDA error: invalid argument`. Both are the NFS $HOME
# quota, not a CUDA or config problem. See ../fix_bug.md bugs 2 and 3.
# ===========================================================================
# ---------------------------------------------------------------------------
# 5) NODE-LOCAL TRITON / INDUCTOR / VLLM CACHES -- MANDATORY, NOT AN OPTIMIZATION.
# ---------------------------------------------------------------------------
# ⚠️ WITHOUT THIS THE ENGINE DIES AT determine_available_memory WITH:
#       RuntimeError: Worker failed with error '[Errno 122] Disk quota exceeded'
#   raised from triton/runtime/cache.py:120 `with open(temp_path, mode) as f`.
#
# MEASURED 2026-09-02: the NFS $HOME is AT ITS PER-USER QUOTA. This is NOT a full
# volume -- `df -h /usr2/tanngo` reports 622G avail (93% used), which makes the
# failure look impossible until you try to write:
#
#       dd if=/dev/zero of=$HOME/.triton/_probe bs=1M count=20
#       dd: closing output file ...: Disk quota exceeded
#
#   $HOME/.triton              28M   (unwritable)
#   $HOME/.torchinductor_cache 1.2G  (unwritable)
#
# The inherited env actively points Inductor at NFS
# (TORCHINDUCTOR_CACHE_DIR=/usr2/tanngo/.torchinductor_cache) and Triton defaults to
# $HOME/.triton. Both are the SAME NFS export mounted in every container, so this is
# also the fleet-shared-state hazard described for TileLang above -- a Triton cubin
# compiled under one CUDA runtime can be picked up under another.
#
# /tmp is node-local overlay with 2.3T free, so pin everything JIT there. Keyed by
# uid + CUDA major so a cu129 and a cu130 container never share compiled binaries.
_CACHE_TAG="$(id -u)_$(cut -d. -f1 <<<"${CUDA_VERSION:-unknown}")"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_$_CACHE_TAG}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_$_CACHE_TAG
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/vllm_cache_$_CACHE_TAG}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/xdg_cache_$_CACHE_TAG}

# ⚠️ THE TWO BELOW ARE THE ONES THAT ACTUALLY KILLED RUNS 2-6, and they are NOT
# covered by TRITON_CACHE_DIR / XDG_CACHE_HOME. Both are DeepGEMM/FlashInfer JIT
# dirs that resolve off $HOME independently:
#
#   TRTLLM_DG_CACHE_DIR -> defaults to $HOME/.tensorrt_llm, and the path is built in
#     C++ (flashinfer/data/csrc/nv_internal/tensorrt_llm/deep_gemm/compiler.cuh:65-90
#     getDefaultUserDir()), so no Python-level cache setting touches it. THIS is what
#     produced the misleading `CUDA error: invalid argument` startup crash: the real
#     error is one frame up and says
#         tvm.error.InternalError: filesystem error: cannot create directories:
#         Disk quota exceeded [/usr2/tanngo/.tensorrt_llm/tmp/gemm_swapAB_3072_...]
#     from fp8_blockscale_gemm_sm90 -> run_flashinfer_deepgemm_swapAB. The GEMM then
#     launches with an unbuilt kernel and CUDA reports `invalid argument`. Chasing the
#     CUDA error (cudagraph sizes, VLLM_USE_BREAKABLE_CUDAGRAPH, --enforce-eager) is
#     chasing a SYMPTOM -- all three were tried and none of them helped, because the
#     quota write happens on the first FP8 block-scale GEMM either way.
#   FLASHINFER_WORKSPACE_BASE -> defaults to pathlib.Path.home() (flashinfer/jit/
#     env.py:59). Same NFS quota, same failure class.
export TRTLLM_DG_CACHE_DIR=${TRTLLM_DG_CACHE_DIR:-/tmp/trtllm_dg_cache_$_CACHE_TAG}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/tmp/flashinfer_ws_$_CACHE_TAG}

mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" \
         "$XDG_CACHE_HOME" "$TRTLLM_DG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" \
         2>/dev/null || true
# TORCHINDUCTOR_CACHE_DIR is assigned UNCONDITIONALLY (no :- default): the inherited
# value is a known-bad NFS path, so respecting it would reintroduce the bug.

# Fail loudly and early rather than 8 minutes into a weight load. Writing the real
# thing is the only reliable probe -- a quota'd NFS dir passes `-w` and `mkdir`.
preflight_caches() {
  printf '\n\033[1m== preflight: JIT cache dirs (NFS $HOME is quota-full)\033[0m\n'
  local d bad=0
  for d in "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" \
           "$TILELANG_CACHE_DIR" "$XDG_CACHE_HOME" "$TRTLLM_DG_CACHE_DIR" \
           "$FLASHINFER_WORKSPACE_BASE"; do
    case $d in
      /usr2/*|"$HOME"/*) printf '  \033[31mFAIL\033[0m %s is on NFS $HOME\n' "$d"; bad=1; continue ;;
    esac
    if dd if=/dev/zero of="$d/.probe" bs=1M count=4 status=none 2>/dev/null \
       && rm -f "$d/.probe" 2>/dev/null; then
      printf '  \033[32mok\033[0m   %s\n' "$d"
    else
      printf '  \033[31mFAIL\033[0m %s not writable (quota?)\n' "$d"; bad=1
      rm -f "$d/.probe" 2>/dev/null
    fi
  done
  [[ $bad == 0 ]] || { printf '\n  \033[31mrefusing to launch\033[0m -- a JIT cache dir is unwritable.\n'; return 1; }
  return 0
}

LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$LOGDIR"

# ---------------------------------------------------------------------------
# THE FLAGS, and why each one is not a free choice
# ---------------------------------------------------------------------------
# --enable-expert-parallel
#     MANDATORY, see the TEP note above. Not the optional throughput knob it is
#     on other checkpoints. 512 experts / TP4 = 128 experts per rank.
#
# --moe-backend triton
#     The recipe's Hopper choice. `auto` may select a Blackwell-oriented kernel
#     (flashinfer/deep_gemm paths) that is not the validated one on SM90.
#     Pinning it also makes the run reproducible -- `auto` is free to change its
#     mind between builds, which would silently invalidate a comparison.
#
# --max-num-seqs 256
#     NOT a free choice, and NOT merely a scheduler knob. The recipe: keep it at
#     256 "to avoid a Mamba-cache capacity failure at startup". The reason is
#     structural: 36 of 48 layers are Gated-DeltaNet linear-attention layers
#     holding a CONSTANT recurrent state per sequence --
#         36 layers x 48 v_heads x 128 v_dim x 128 k_dim x 4 B (fp32)
#         = 0.1055 GiB per SEQUENCE, sharded by TP over heads
#     At TP4 that is ~0.0264 GiB/seq/rank, so max_num_seqs=256 reserves ~6.8
#     GiB/rank BEFORE any KV. This state does not grow with context, but it does
#     grow with max_num_seqs -- a memory axis that pure-attention models do not
#     have. Raising this flag trades KV capacity for concurrency slots.
#     bench.sh tops out at concurrency 64, so 256 does not alter our scheduling.
#
# --max-model-len 262144
#     The checkpoint's native window (max_position_embeddings=262144), and it
#     covers the whole bench.sh context sweep (top point ISL 256K). MATCHES the
#     GLM-5.3-Flash arm exactly. 1M is only reachable via static YaRN
#     (--hf-overrides rope_type=yarn factor=4.0) which changes the positional
#     encoding and is therefore a different model, not a longer one.
#
# --limit-mm-per-prompt '{"image":0,"video":0}'
#     Qwen3.8-Flash-Next-FP8 is MULTIMODAL (27-layer ViT, 0.305 B params).
#     Zeroing both modalities makes vLLM SKIP CONSTRUCTING the vision tower, so
#     the comparison stays text-to-text against V4-Flash (text-only).
#     THIS IS A FAIRNESS FLAG, and it matches what the GLM-5.3 arm does.
#
# --enable-prefix-caching
#     In the recipe, and default-on in this build. Stated explicitly so the
#     manifest records it. bench.sh guards the resulting hazard by using a
#     unique seed per point and asserting 0 new cache hits per point.
#
# --reasoning-parser qwen3 / --tool-call-parser qwen3_xml
#     Both verified present in THIS build's lazy registries
#     (reasoning/__init__.py -> "qwen3"; tool_parsers/__init__.py -> "qwen3_xml").
#
# --no-enable-flashinfer-autotune
#     In the recipe. Autotune adds minutes at startup and its kernel choices are
#     not recorded in the manifest, so it silently makes runs non-reproducible.
#
# --enable-mfu-metrics
#     Engine-side FLOP/byte counters. Default OFF -> the /metrics gauges read 0.0
#     and any "achieved bandwidth" number would be analytical-only. This
#     project's rule 5 wants a MEASURED fraction of the 3.35 TB/s H100 peak.
#
# NO --block-size
#     Deliberately UNPINNED, unlike the V4-Flash (256) and GLM-5.3 (128) arms.
#     There is no single hard requirement here, but there IS a real constraint:
#     the QSA side cache asserts
#         block_size % (compress_ratio * ceil((compress_ratio + n_spec)/compress_ratio)) == 0
#     (common/qsa_cache.py:773-790) -- with indexer_compress_ratio=4 that is
#     % 4 == 0 at MTP off and % 8 == 0 at num_speculative_tokens in 1..4.
#     vLLM must also satisfy a FullAttentionSpec (main KV), an MLAAttentionSpec
#     (compressed indexer keys) and MambaSpecs (GDN + PLE conv) at once, and it
#     resolves the LCM itself. Letting it resolve is safer than guessing; the
#     chosen value is printed in the startup log and MUST be read out of there
#     when interpreting KV capacity.
#
# NO --kv-cache-dtype
#     FP8 KV IS NOT AVAILABLE FOR THIS MODEL, on any GPU. Verified in source,
#     not assumed: every QSA backend in this build declares
#         supported_kv_cache_dtypes = ["auto", "bfloat16"]
#     (nvidia/qsa.py:70, common/qsa_cache.py:658, amd/qsa.py:70) and the impl
#     raises NotImplementedError("Qwen3.8-Flash-Next QSA requires a BF16 main KV
#     cache") for anything else (nvidia/qsa.py:~97). The indexer additionally
#     requires BF16 model dtype outright. So "auto" -> BF16 is the only option.
#     This is a KERNEL AVAILABILITY fact to report, not a choice -- see README.md
#     for how it interacts with the V4/GLM KV-dtype asymmetry.
#
# NO --trust-remote-code
#     qwen4_exp is registered NATIVELY in this build (transformers_utils config
#     registry -> Qwen4ExpConfig; ModelRegistry -> Qwen4ExpForConditionalGeneration
#     -> vllm.models.qwen3_8_flash_next). The flag would grant the checkpoint
#     arbitrary code execution for nothing.
#
# NO --speculative-config in run.sh
#     Base model only for headline throughput. run_mtp.sh is the A/B arm.
COMMON_ARGS=(
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --enable-expert-parallel
  --moe-backend triton
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs 256
  --max-model-len 262144
  --enable-prefix-caching
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --no-enable-flashinfer-autotune
  --enable-mfu-metrics
)

# ---------------------------------------------------------------------------
# preflight_gpus -- refuse to launch into a machine that cannot hold the model
# ---------------------------------------------------------------------------
# The image entrypoint (PID 1, `vllm serve` on Qwen/Qwen3-0.6B) parks ~77 GiB on
# GPU 0 via its VLLM::EngineCore child. This function only DETECTS; it never
# kills, because an automated kill of a process this script did not create is
# not a safe default -- and here the unsafe outcome is losing the container:
#
#     PID 1 IS CONTAINER INIT. Killing it kills the session.
#     Killing only the EngineCore CHILD frees GPU 0, but vLLM normally treats
#     EngineCore death as fatal and PID 1 may exit too -- same outcome.
#
# Hence the default GPUS=4,5,6,7 / TP=4, which routes around GPU 0 entirely and
# needs no kill at all.
preflight_gpus() {
  local need=$TP busy=0 idx used
  printf '\n\033[1m== preflight: GPUs\033[0m\n'
  command -v nvidia-smi >/dev/null || { printf '  \033[31mFAIL\033[0m no nvidia-smi -- no GPU on this node\n'; return 1; }

  printf '  host          : %s\n' "$(hostname)"
  printf '  CUDA_VISIBLE_DEVICES: %s  (TP=%s, util=%s)\n' "$GPUS" "$TP" "$GPU_UTIL"

  # nvidia-smi ignores CUDA_VISIBLE_DEVICES, so check exactly the GPUs we asked for.
  local n=0
  for idx in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
    [[ -n $used ]] || { printf '  \033[31mFAIL\033[0m GPU %s not visible on this node\n' "$idx"; return 1; }
    n=$((n+1))
    # >2 GiB in use means something real is resident, not just context overhead.
    if [[ ${used:-0} -gt 2048 ]]; then
      printf '  \033[31mBUSY\033[0m GPU %s holds %s MiB\n' "$idx" "$used"
      busy=1
    else
      printf '  \033[32mok\033[0m   GPU %s free (%s MiB used)\n' "$idx" "$used"
    fi
  done
  [[ $n -eq $need ]] || { printf '  \033[31mFAIL\033[0m GPUS lists %s device(s) but TP=%s\n' "$n" "$need"; return 1; }

  if [[ $busy == 1 ]]; then
    printf '\n  \033[31mrefusing to launch\033[0m -- a requested GPU is not free. Holders:\n'
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | sed 's/^/    /'
    printf '    (PID 1 = image entrypoint `vllm serve` on Qwen/Qwen3-0.6B, ~77 GiB on GPU 0.)\n'
    printf '    Either pick GPUs it does not hold  ->  GPUS=4,5,6,7 TP=4 ./run.sh\n'
    printf '    or free GPU 0 by killing the VLLM::EngineCore CHILD, NEVER PID 1:\n'
    printf '      kill -9 <EngineCore pid>   # risks ending the container -- see header\n'
    return 1
  fi

  # Weights are 172.76 GiB (safetensors index total_size). Sanity-check the budget.
  "$PY_BIN" - "$TP" "$GPU_UTIL" <<'EOP'
import sys
tp, util = int(sys.argv[1]), float(sys.argv[2])
W, CAP = 172.76, 79.6                      # GiB weights, GiB per H100-80GB
per, budget = W/tp, CAP*util
state = 36*48*128*128*4/2**30/tp*256       # GDN recurrent state, 256 seqs, TP-sharded
print(f"  budget: {W:.1f} GiB weights / TP{tp} = {per:.1f} GiB/GPU;"
      f" util {util} of {CAP} = {budget:.1f} GiB/GPU usable")
print(f"          -> {budget-per:.1f} GiB/GPU for KV + GDN state + activations")
print(f"          GDN recurrent state at max_num_seqs=256: {state:.1f} GiB/GPU (constant in ctx)")
print(f"          leaves ~{budget-per-state:.1f} GiB/GPU for KV at 24.75 KiB/token BF16"
      f" = ~{(budget-per-state)*2**30/(24.75*1024)/1e6:.2f} M tokens")
if budget - per - state < 4:
    print("  WARNING: <4 GiB/GPU left for KV -- expect a startup failure or a tiny KV pool")
EOP
  return 0
}

# Print the resolved command, then exec it through tee so the startup log is kept.
# The log is where the RESOLVED block_size, num_gpu_blocks, the chosen attention
# backends and the local/global expert split actually appear -- all of them are
# needed to interpret the numbers later, and block_size is not pinned here.
launch() {
  local tag=$1; shift
  local log="$LOGDIR/serve_${tag}_$(date +%Y%m%d-%H%M%S).log"
  printf '\n\033[1m== launching: %s\033[0m\n' "$tag"
  printf '  engine : %s\n' "$VLLM_BIN"
  printf '  model  : %s\n' "$MODEL"
  printf '  port   : %s\n' "$PORT"
  printf '  gpus   : %s (TP=%s)\n' "$GPUS" "$TP"
  printf '  log    : %s\n\n' "$log"
  printf '  CUDA_VISIBLE_DEVICES=%s %s serve %s \\\n' "$GPUS" "$VLLM_BIN" "$MODEL"
  printf '    %s\n' "$@"
  printf '\n'
  [[ ${DRY_RUN:-0} == 1 ]] && { printf '  DRY_RUN=1 -- not launching\n'; return 0; }

  # Keep stderr merged so tracebacks land in the same log as the startup banner.
  "$VLLM_BIN" serve "$MODEL" "$@" 2>&1 | tee "$log"
}
