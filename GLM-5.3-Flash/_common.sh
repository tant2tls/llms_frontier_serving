#!/usr/bin/env bash
# Shared launch scaffolding for the GLM-5.3-Flash arms. Sourced, not executed.
#
#   run.sh        BF16 KV   <- HEADLINE (the only KV dtype the recipe sanctions on H100)
#   run_fp8kv.sh  fp8_ds_mla KV  <- second dtype arm, requested explicitly
#   run_mtp.sh    BF16 KV + MTP  <- spec-decode A/B, never headline
#
# Every non-default flag below is justified in README.md. The three that are NOT
# free choices -- --block-size 128, the port, and the absent --trust-remote-code --
# are explained inline because getting them wrong fails in confusing ways.

set -uo pipefail

MODEL=${MODEL:-zai-org/GLM-5.3-Flash}

# ---------------------------------------------------------------------------
# 1) THE ENGINE MUST BE THE IMAGE'S, NOT THE CONDA ENV'S.
# ---------------------------------------------------------------------------
# This container (vllm/vllm-openai:glm53-flash) ships a dev build that REGISTERS
# glm5_next; the conda env vllm-py12 does not:
#
#   image  /usr/local/bin/vllm          0.1.dev20051+g487ecf187  -> Glm5Next* present
#   conda  .../vllm-py12/bin/vllm       0.28.0                   -> Glm5Next* ABSENT
#
# The shell profile activates vllm-py12 and puts it FIRST on PATH, so a bare
# `vllm serve` silently resolves to 0.28.0 and dies with "architecture not
# supported". Always use the absolute image path.
VLLM_BIN=${VLLM_BIN:-/usr/local/bin/vllm}
PY_BIN=${PY_BIN:-/usr/bin/python3}

# ---------------------------------------------------------------------------
# 2) PORT 8001, NOT 8000.
# ---------------------------------------------------------------------------
# The image ENTRYPOINT is itself `vllm serve` (PID 1) serving the default
# Qwen/Qwen3-0.6B, and it owns port 8000 + ~77 GiB on GPU 0. We cannot kill PID 1
# (it is container init). See preflight_gpus below.
PORT=${PORT:-8001}

# Utilization is a VARIABLE, not a constant, because the util A/B is an owed
# measurement (CLAUDE.md "REMAINING GLM MEASUREMENTS" task 1). Default 0.82 keeps
# every existing arm reproducible byte-for-byte.
#   GPU_UTIL=0.85 ./run.sh
# ⚠️ This arm SIZES KV, so warm the compile cache first -- a cold torch.compile
# makes vLLM mis-measure peak activation and mis-size the pool (fix_bug.md bug 12).
GPU_UTIL=${GPU_UTIL:-0.82}

# ---------------------------------------------------------------------------
# 3) ABSOLUTE HF_HOME.
# ---------------------------------------------------------------------------
# The inherited env has HF_HOME=cache / HF_HUB_CACHE=cache/hub -- RELATIVE paths
# that resolve against $PWD, so from a $HOME-side directory a 306 GiB download
# lands on the fleet-shared NFS export. GLM-5.3-Flash is ALREADY cached (308 GiB,
# snapshot 03eb5366) under the absolute path below; do not re-download.
export HF_HOME=/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/LLMs_serving_report/cache
export HF_HUB_CACHE=$HF_HOME/hub

# ---------------------------------------------------------------------------
# 4) NODE-LOCAL TILELANG CACHE.
# ---------------------------------------------------------------------------
# TileLang JIT-compiles kernels (mhc_pre/mhc_post for GLM's hyper-connections) and
# caches the resulting CUBINS under $HOME/.tilelang by default. $HOME=/usr2/tanngo
# is a single NFS export mounted identically in EVERY container, so a cu130
# container and a cu129 container silently SHARE compiled binaries -- and a cubin
# built against one CUDA runtime loaded under another segfaults inside
# cuModuleLoadData. Pin the cache to node-local /tmp so each container compiles its
# own.
#
# ⚠️ This is a real hazard but it was NOT the cause of the cu129 MHC segfault:
# forcing a node-local cache made TileLang recompile from scratch and it STILL
# crashed. See the CLAUDE.md GLM-5.3-Flash section. Keep this anyway.
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-/tmp/tilelang_cache_$(id -u)_$(cut -d. -f1 <<<"${CUDA_VERSION:-unknown}")}
mkdir -p "$TILELANG_CACHE_DIR" 2>/dev/null || true

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
# --block-size 128   MANDATORY, and the default (64) HARD-ASSERTS.
#     GLM-5.3 sets index_kpool=4. Glm5NextIndexerCache.get_kv_cache_spec
#     (vllm/models/glm5next/nvidia/attention.py:140-150) requires
#         block_size % index_kpool == 0  AND  (block_size/index_kpool) % 32 == 0
#     i.e. block_size must be a multiple of index_kpool*32 = 128, so DeepGEMM
#     paged-MQA pool pages (32 or 64 entries) tile the storage block. With the
#     default 64 the storage block collapses to 16 and it fails. 128 is the
#     smallest legal value; 256 also works but wastes KV on padding.
#     NOTE: this is a DIFFERENT reason than V4-Flash's --block-size 256
#     (sparse_mla.py:53, per-layer compress_ratios). Same-looking flag, unrelated
#     cause -- do not copy one justification onto the other.
#
# --tensor-parallel-size 8 / --enable-expert-parallel
#     Matched to the V4-Flash run for equal silicon. GLM is MoE (288 experts, k=8).
#     Weights are 305.8 GiB => 38.2 GiB/GPU at TP8, leaving ~27 GiB/GPU for KV at
#     util 0.82. Unlike V4 (which fits in 3 GPUs, making its TP=8 a pessimistic
#     confound) GLM-5.3 genuinely needs ~5 GPUs minimum, so TP8 is closer to a
#     real deployment here. Record that asymmetry when comparing per-GPU numbers.
#
# --gpu-memory-utilization ${GPU_UTIL:-0.82}
#     Matched to V4-Flash (0.9 OOM'd at warmup there). Not independently tuned
#     for GLM; if KV is tight this is the first knob to raise.
#
#     ⚠️ OVERRIDABLE via GPU_UTIL= (added 2026-09-02) because the util A/B is an
#     OWED MEASUREMENT for GLM -- see CLAUDE.md "REMAINING GLM MEASUREMENTS" task 1.
#     Measured on the other two models: the flag is CAPACITY-ONLY (V4: +3.7% util
#     -> +7.0% KV, no throughput change, peak activation 2.32 GiB in both arms).
#     BUT on Qwen a COLD torch.compile made vLLM measure peak activation as
#     17.07 GiB instead of 0.99 GiB, reserving ~14 GiB/GPU of KV it never needed
#     and understating c=4 by 1.58x. GLM's own arms are clean (4.05 GiB), but this
#     arm SIZES KV, so **warm the compile cache before running it**.
#         GPU_UTIL=0.85 GPUS=0,1,2,3,4,5,6,7 TP=8 ./run.sh
#
# --limit-mm-per-prompt '{"image":0,"video":0}'
#     GLM-5.3-Flash is MULTIMODAL (24-layer ViT); V4-Flash is text-only. Zeroing
#     both modalities makes vLLM SKIP CONSTRUCTING the vision tower entirely
#     (interfaces.py:307 "Tower model components are automatically skipped when
#     --limit-mm-per-prompt is set to zero for all of their modalities"), saving
#     HBM and keeping the comparison text-to-text. THIS IS A FAIRNESS FLAG.
#
# --reasoning-parser glm45 / --tool-call-parser glm47
#     Both verified present in this build's registries. glm45 and glm47 map to
#     the SAME underlying glm47_moe_reasoning_parser (reasoning/__init__.py:55-60);
#     glm45 is kept because the recipe names it.
#
# --no-enable-flashinfer-autotune
#     Autotune at startup adds minutes and its kernel choices are not recorded in
#     the manifest, so it silently makes runs non-reproducible.
#
# --max-model-len 262144
#     Covers the whole bench.sh context sweep (top point is ISL 256K). GLM
#     declares 1,048,576; reserving for 1M would crush concurrency. BF16 KV is
#     only ~11.35 KiB/token here so 256K is affordable -- see README.md.
#
# NO --trust-remote-code
#     glm5_next is registered NATIVELY (transformers_utils/config.py:96-98), so
#     the flag would grant the checkpoint arbitrary code execution for nothing.
#
# --max-num-seqs 256   MANDATORY. Without it the engine will not START.
#     ⚠️ THIS IS THE FLAG THE FIRST FOUR LAUNCH ATTEMPTS DIED ON (2026-09-01/02).
#     All four failed in _initialize_kv_caches; the fourth finally printed:
#
#       ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (512).
#       Each decode sequence requires one Mamba cache block, so CUDA graph capture
#       cannot proceed. Please lower max_num_seqs to at most 512 or increase
#       gpu_memory_utilization.
#
#     TWO facts collide here, and BOTH are surprising:
#
#     (a) GLM-5.3-Flash IS A HYBRID, NOT A PURE ATTENTION MODEL. config.json
#         text_config.layer_types is 34x "linear_attention" (KDA -- Kimi Delta
#         Attention, linear_attn_config.kda_layers) + 11x
#         "deepseek_sparse_attention", in a 3:1 repeating pattern. The 34 KDA
#         layers carry a RECURRENT STATE, which vLLM manages as "Mamba" blocks:
#         ONE BLOCK PER DECODE SEQUENCE, allocated up front, and NOT paged the way
#         token KV is. So max concurrency is capped by state blocks, not by KV
#         tokens. This is a structurally different constraint from V4-Flash
#         (pure sparse MLA, no recurrent state) and is a REPORTABLE ARCHITECTURE
#         FINDING, not just a config annoyance.
#
#     (b) --max-num-seqs DOES NOT DEFAULT TO 128 ON THIS HARDWARE. The dataclass
#         default is 128, but arg_utils.get_batch_defaults() (arg_utils.py:2547+)
#         OVERRIDES it by device: any GPU with >= 70 GiB that is not an A100 gets
#         max_num_seqs=1024 (and max_num_batched_tokens=8192 for
#         OPENAI_API_SERVER). H100-80GB hits that branch, so `vllm serve` with no
#         flag runs at 1024. Only 512 Mamba blocks fit at util 0.82 -> hard fail.
#
#     WHY 256 AND NOT 512. 512 is the ceiling the error message reports, but it is
#     the value measured at THIS util on THIS node, and it is what the cudagraph
#     check compares against -- sitting exactly at the boundary means any small
#     change in free HBM (a different node, a driver bump, fragmentation) fails
#     the launch again. 256 also:
#       - covers the whole bench.sh grid, whose top concurrency is 64 (sweep_batch:
#         1/4/16/64), with 4x headroom, so it CANNOT clip any measured point; and
#       - matches the V4-Flash baseline's EFFECTIVE cap, keeping the cross-model
#         comparison honest -- see the note below.
#     Raise it only if a future sweep actually needs concurrency > 256, and if you
#     do, re-record it in the manifest: it changes the memory split.
#
#     ⚠️ FAIRNESS NOTE FOR THE V4 COMPARISON. The V4-Flash baseline
#     (deepseek_v4_flash/results/mtp-off/manifest.txt) passed NO --max-num-seqs on
#     conda vLLM 0.28.0, whose get_batch_defaults has the same H100 branch -- so V4
#     ran at 1024 while GLM is pinned to 256. This does NOT affect any published
#     point (both sweeps top out at concurrency 64, far under either cap), but it
#     IS an engine-config difference between the two arms and must be stated rather
#     than discovered later.
#
# --max-cudagraph-capture-size 256   should EQUAL --max-num-seqs.
#     ⚠️ USE THE DEDICATED FLAG, NOT --compilation-config. Passing
#     `--compilation-config '{"max_cudagraph_capture_size":256}'` REPLACES the whole
#     CompilationConfig object, which silently wiped `pass_config` to `{}` (losing
#     fuse_norm_quant / fuse_act_quant / fuse_allreduce_rms). The dedicated flag is
#     MERGED into the existing config (arg_utils.py:2452-2469) and is mutually
#     exclusive with the compilation_config key, so it cannot clobber siblings.
#
#     WHY CAP IT. The default ladder is 51 entries ending [.. 336, 352, .. 512] and
#     max_cudagraph_capture_size defaults to 512 -- it is NOT derived from
#     max_num_seqs. Any captured graph above max_num_seqs (256) is unreachable by
#     construction, because the decode cudagraph dispatcher caps batch size at
#     max_num_seqs. So sizes 272..512 cost startup time and HBM for shapes that can
#     never be dispatched. Capping to 256 yields a 35-entry ladder.
#
#     ⚠️ THIS WAS NOT THE CAUSE OF THE `CUDA error: invalid argument` CRASH.
#     That was TRTLLM_DG_CACHE_DIR hitting the NFS quota -- see section 5 above.
#     Capping the ladder is kept because it is correct on its own merits (unreachable
#     graphs cost startup time and HBM), NOT as a crash fix.
#
# NO --speculative-config in run.sh / run_fp8kv.sh
#     Base model only for headline throughput. run_mtp.sh is the A/B arm.
COMMON_ARGS=(
  --port "$PORT"
  --max-cudagraph-capture-size 256
  --block-size 128
  --max-num-seqs 256
  --tensor-parallel-size 8
  --enable-expert-parallel
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len 262144
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --no-enable-flashinfer-autotune
  # Engine-side FLOP/byte counters. Default OFF -> the /metrics gauges read 0.0
  # and any "achieved bandwidth" number becomes analytical-only. Rule 5 of this
  # project's eval conventions wants a MEASURED fraction, so turn them on.
  --enable-mfu-metrics
)

# ---------------------------------------------------------------------------
# preflight_gpus -- refuse to launch into a machine that cannot hold the model
# ---------------------------------------------------------------------------
# The image entrypoint (PID 1, `vllm serve` on Qwen/Qwen3-0.6B) parks ~77 GiB on
# GPU 0 via its VLLM::EngineCore child. TP=8 needs all eight GPUs nearly empty,
# so that squatter must go first. PID 1 is container init -- killing it kills the
# container and this session with it. Kill only the EngineCore CHILD:
#
#     nvidia-smi --query-compute-apps=pid,used_memory --format=csv
#     kill -9 <EngineCore pid>     # NOT 1
#
# vLLM normally treats EngineCore death as fatal and PID 1 may exit too, which
# ends the container. If that happens, get rescheduled and run these scripts
# BEFORE anything else touches the GPUs. This function only DETECTS; it never
# kills, because an automated kill of a process this script did not create is
# not a safe default.
preflight_gpus() {
  local need=${1:-8} busy=0 line idx used
  printf '\n\033[1m== preflight: GPUs\033[0m\n'
  command -v nvidia-smi >/dev/null || { printf '  \033[31mFAIL\033[0m no nvidia-smi -- no GPU on this node\n'; return 1; }

  local n; n=$(nvidia-smi -L | wc -l)
  printf '  visible GPUs: %s (need %s)\n' "$n" "$need"
  [[ $n -ge $need ]] || { printf '  \033[31mFAIL\033[0m only %s GPU(s); TP=8 needs %s\n' "$n" "$need"; return 1; }

  # >2 GiB in use means something real is resident, not just context overhead.
  while IFS=, read -r idx used; do
    used=${used// /}; used=${used%% *}
    if [[ ${used:-0} -gt 2048 ]]; then
      printf '  \033[31mBUSY\033[0m GPU %s holds %s MiB\n' "${idx// /}" "$used"
      busy=1
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' | awk -F, '{print $1","$2}')

  if [[ $busy == 1 ]]; then
    printf '\n  \033[31mrefusing to launch\033[0m -- GPUs are not free. Holders:\n'
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | sed 's/^/    /'
    printf '    (PID 1 = image entrypoint `vllm serve` on Qwen/Qwen3-0.6B.)\n'
    printf '    Kill the VLLM::EngineCore CHILD, never PID 1:\n'
    printf '      kill -9 <EngineCore pid>\n'
    printf '    Then wait ~25 s for HBM to drain and re-run.\n'
    return 1
  fi
  printf '  \033[32mok\033[0m   all %s GPUs free\n' "$n"

  # Weights alone are 305.8 GiB; TP8 -> 38.2 GiB/GPU. Sanity-check the budget.
  printf '  budget: 305.8 GiB weights / TP8 = 38.2 GiB/GPU; util 0.82 of 79.6 GiB\n'
  printf '          = 65.3 GiB/GPU usable -> ~27.1 GiB/GPU for KV + activations\n'
  return 0
}

# Print the resolved command, then exec it through tee so the startup log is kept.
# The log is where block_size / num_gpu_blocks / the chosen attention backend
# actually appear -- all three are needed to interpret the numbers later.
launch() {
  local tag=$1; shift
  local log="$LOGDIR/serve_${tag}_$(date +%Y%m%d-%H%M%S).log"
  printf '\n\033[1m== launching: %s\033[0m\n' "$tag"
  printf '  engine : %s\n' "$VLLM_BIN"
  printf '  model  : %s\n' "$MODEL"
  printf '  port   : %s\n' "$PORT"
  printf '  log    : %s\n\n' "$log"
  printf '  %s serve %s \\\n' "$VLLM_BIN" "$MODEL"
  printf '    %s\n' "$@"
  printf '\n'
  [[ ${DRY_RUN:-0} == 1 ]] && { printf '  DRY_RUN=1 -- not launching\n'; return 0; }

  # `vllm serve` needs a tty-free stdin; keep stderr merged so tracebacks land
  # in the same log as the startup banner.
  "$VLLM_BIN" serve "$MODEL" "$@" 2>&1 | tee "$log"
}
