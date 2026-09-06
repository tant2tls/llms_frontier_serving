#!/usr/bin/env bash
# Submit the RunAI workspace that GLM-5.3-Flash needs. **RUN THIS FROM OUTSIDE THE
# CONTAINER** (a login host with a working `runai` CLI), not from inside a workspace.
#
#   ./submit_job.sh              # delete the old workspace if present, then submit
#   DRY_RUN=1 ./submit_job.sh    # print the command, submit nothing
#   NAME=tan-8gpus-glm53b ./submit_job.sh   # submit under a different name
#
# ---------------------------------------------------------------------------
# WHY THIS FILE EXISTS: --command -- sleep infinity
# ---------------------------------------------------------------------------
# The ONE flag that matters here is `--command -- sleep infinity`. Without it this
# workspace cannot run GLM-5.3-Flash at all, for a reason that is invisible until
# you look at PID 1:
#
#   vllm/vllm-openai:glm53-flash declares  ENTRYPOINT ["vllm", "serve"]
#
# With no `--command`, RunAI starts the container on that entrypoint with NO model
# argument. vLLM's serve CLI then falls back to its built-in default
# (entrypoints/cli/serve.py:36 -- "Defaults to Qwen/Qwen3-0.6B if no model is
# specified"), so PID 1 becomes a live server for Qwen/Qwen3-0.6B that:
#
#   - parks ~77 GiB of HBM on GPU 0 via its VLLM::EngineCore child, and
#   - owns port 8000.
#
# GLM-5.3-Flash is 305.8 GiB and needs TP=8, i.e. all eight GPUs nearly empty, so
# that squatter is fatal. MEASURED on tan-8gpus-glm53-0-0 (2026-09-01):
# GPU 0 memory.used = 77,146 MiB, memory.free = 3,934 MiB.
#
# ⚠️ AND YOU CANNOT KILL IT. This was tried, so don't repeat the experiment:
#   `kill -9 <EngineCore pid>` DID free the memory -- and PID 1 died with it.
#   PID 1 is container init, so the container terminated, RunAI rescheduled it
#   (tan-8gpus-glm53-0-0 -> -0-1), and the entrypoint came back with a BRAND NEW
#   EngineCore holding the same 77 GiB. It is a respawn loop, not a fix.
#   vLLM's /sleep endpoint would release weights without killing the process, but
#   it needs VLLM_SERVER_DEV_MODE=1 set AT CONTAINER START and is absent from
#   /openapi.json otherwise -- it cannot be enabled on a running server.
#
# So the squatter has to be prevented at submit time. `sleep infinity` makes PID 1
# an idle process that loads no model and touches no GPU; you then start the real
# server yourself with ./run.sh.
#
# Fallbacks that do NOT work, checked so nobody has to re-check them:
#   TP=7 on the 7 free GPUs -> model.py:669 asserts num_attention_heads % tp == 0;
#                              64 % 7 != 0, fails at init.
#   TP=4 on GPUs 4-7        -> 305.8 GiB / 4 = 76.5 GiB/GPU vs ~65 GiB usable at
#                              --gpu-memory-utilization 0.82. Does not fit.
#   Pipeline parallel       -> glm5next declares SupportsPP, but PP changes the
#                              latency structure and would make TTFT/TPOT
#                              non-comparable to the DeepSeek-V4-Flash baseline,
#                              which is the entire point of the comparison.
#
# ---------------------------------------------------------------------------
# EVERY OTHER FLAG is carried over verbatim from the working submission so the
# environment (mounts, uid mapping, cpu budget) stays identical. Notes on the ones
# that interact with this benchmark:
#
#   -g 8                      GLM-5.3-Flash at TP=8. RUNAI_NUM_OF_GPUS=8 was
#                             already correct before; the GPUs were allocated but
#                             not free. This flag was never the problem.
#   --cpu-core-request/limit 198
#                             Matches the MEASURED cgroup quota on these nodes
#                             (cpu.cfs_quota_us/cfs_period_us = 198), NOT nproc
#                             (256, which is the HOST count and would oversubscribe
#                             into CFS throttling).
#   --large-shm               vLLM's TP workers communicate over /dev/shm. The
#                             default 64 MB shm is too small for 8-way TP.
#   vol22-scratch mount       Holds this repo AND the 308 GiB GLM-5.3-Flash weight
#                             cache. Without it there is nothing to serve.
#   vol11-scratch mount       Holds the conda envs. Kept for the nano-vllm
#                             instrumentation vehicle -- note the report's serving
#                             runs deliberately use the IMAGE's vLLM
#                             (/usr/local/bin/vllm, 0.1.dev20051+g487ecf187),
#                             because the conda env's vLLM 0.28.0 does NOT register
#                             glm5_next. See _common.sh.
#   --preemptible             ⚠️ RunAI CAN EVICT THIS MID-SWEEP. Loading 305.8 GiB
#                             across 62 shards alone takes minutes, and the full
#                             batch+context+prefix sweep is much longer. If you
#                             have non-preemptible quota, spend it here: an
#                             eviction halfway through a context sweep wastes the
#                             whole arm. bench.sh skips existing results, so a
#                             resumed run does recover -- but the server has to
#                             reload from scratch first.
#   --backoff-limit 0         Do not silently retry. A failed launch should be
#                             visible, not quietly restarted into a different
#                             GPU state.
# ---------------------------------------------------------------------------

set -uo pipefail

NAME=${NAME:-tan-8gpus-glm53}

# ⚠️ USE THE CUDA-13 TAG. Do NOT switch this to `:glm53-flash-cu129`.
# MEASURED 2026-09-02 on tan-8gpus-glm53-cu129-0-0: on the cu129 image GLM-5.3-Flash
# loads all 62 shards (38.08 GiB/GPU) and then dies in determine_available_memory --
# the first real forward pass -- with a SEGFAULT inside cuModuleLoadData while
# loading the TileLang kernel `mhc_post_tilelang`:
#
#     !!!!!!! Segfault encountered !!!!!!!
#       cuModuleLoadData / tvm::runtime::CUDAModuleNode::GetFunc
#       __tvm_ffi_mhc_post_tilelang
#
# The image was BUILT for cu130 -- its own /vllm-workspace/torch_lib_versions.txt
# declares torch==2.13.0+cu130 -- and MHC is not optional for this model
# (config.json: mhc=true, hc_mult=4). Confirmed NOT a stale-cubin problem: forcing a
# node-local TILELANG_CACHE_DIR made it recompile from scratch and it still crashed.
# Also NOT avoidable via --compilation-config custom_ops: MHCPostOp.enabled()
# hardcodes `return True` (model_executor/layers/mhc.py:203), so `-mhc_post` is
# silently ignored. See CLAUDE.md.
IMAGE=${IMAGE:-vllm/vllm-openai:glm53-flash}
GPUS=${GPUS:-8}
CPUS=${CPUS:-198}
DRY_RUN=${DRY_RUN:-0}

# Everything after `--` is the container command, so `--command -- sleep infinity`
# MUST be last. Note the chipsets mount is grouped with the other --nfs flags
# rather than trailing at the end, so nothing lands after the `--` separator.
build_cmd() {
  CMD=(
    runai workspace submit "$NAME"
    --image "$IMAGE"
    -e USER=tango
    -e HOME=/usr2/tanngo
    -e REPO_DIR=/prj/corp/airesearch/lasvegas/vol5-scratch/users/phongnh/Long
    -e PORT_NUM=25000
    --nfs path=/usr2/tanngo,mountpath=/usr2/tanngo,server=mudpie,readwrite
    --nfs path=/prj/corp/llm/lasvegas/llm-systems,mountpath=/prj/corp/llm/lasvegas/llm-systems,server=redpill,readwrite
    --nfs path=/prj/corp/llm/lasvegas/llm-systems-scratch,mountpath=/prj/corp/llm/lasvegas/llm-systems-scratch,server=redpill,readwrite
    --nfs path=/prj/corp/crd/morpheus/lasvegas/user_scratch,mountpath=/prj/corp/crd/morpheus/lasvegas/user_scratch,server=redpill,readwrite
    --nfs path=/prj/corp/crd/morpheus/lasvegas/datasets-scratch,mountpath=/prj/corp/crd/morpheus/lasvegas/datasets-scratch,server=redpill,readwrite
    --nfs path=/prj/corp/airesearch/lasvegas/vol5-scratch,mountpath=/prj/corp/airesearch/lasvegas/vol5-scratch,server=rhineheart,readwrite
    --nfs path=/prj/corp/airesearch/lasvegas/vol11-scratch,mountpath=/prj/corp/airesearch/lasvegas/vol11-scratch,server=redpill,readwrite
    --nfs path=/prj/corp/airesearch/lasvegas/vol22-scratch,mountpath=/prj/corp/airesearch/lasvegas/vol22-scratch,server=recall,readwrite
    --nfs path=/prj/neo_lv/user,mountpath=/prj/neo_lv/user,server=stickman,readwrite
    --nfs path=/prj/corp/airesearch/morpheus/lasvegas/chipsets,mountpath=/prj/corp/airesearch/morpheus/lasvegas/chipsets,server=redpill,readwrite
    --run-as-user
    -g "$GPUS"
    --cpu-core-request "$CPUS"
    --cpu-core-limit "$CPUS"
    --large-shm
    --backoff-limit 0
    --port service-type=NodePort,container=25000
    --stdin --tty
    --preemptible
    # THE LOAD-BEARING FLAG -- must stay last. See the header.
    --command -- sleep infinity
  )
}

main() {
  build_cmd

  printf '\033[1mRunAI workspace submit\033[0m  %s\n' "$NAME"
  printf '  image : %s\n' "$IMAGE"
  printf '  gpus  : %s   cpus: %s\n' "$GPUS" "$CPUS"
  printf '  entrypoint override: sleep infinity  <- keeps GPUs free\n\n'

  command -v runai >/dev/null 2>&1 || {
    printf '  \033[31mFAIL\033[0m no `runai` on PATH.\n'
    printf '  Run this from a login host, not from inside a workspace container.\n'
    return 1
  }

  printf '%s\n\n' "${CMD[*]}"

  if [[ $DRY_RUN == 1 ]]; then
    printf '  DRY_RUN=1 -- nothing submitted\n'
    return 0
  fi

  # A workspace with this name already existing makes the submit fail. Deleting it
  # is ALSO how the old container (and its GPU-0 squatter) is released -- there is
  # no way to free that memory from inside the container, so this is the mechanism.
  if runai workspace list 2>/dev/null | grep -q "\b${NAME}\b"; then
    printf '  \033[33mnote\033[0m workspace %s already exists.\n' "$NAME"
    printf '  Deleting it releases the old container AND its GPU-0 squatter.\n'
    read -r -p "  delete and resubmit? [y/N] " ans
    [[ ${ans:-n} =~ ^[Yy]$ ]] || { printf '  aborted\n'; return 1; }
    runai workspace delete "$NAME" || printf '  \033[33mnote\033[0m delete returned nonzero; continuing\n'
    printf '  waiting 20 s for the scheduler to release the GPUs...\n'
    sleep 20
  fi

  "${CMD[@]}"
  local rc=$?

  if [[ $rc == 0 ]]; then
    cat <<'EOS'

  submitted. Once the pod is Running, exec in and VERIFY BEFORE SERVING:

    ps -p 1 -o args=                       # expect: sleep infinity  (NOT vllm serve)
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
                                           # expect: all 8 GPUs near 0 MiB

  then, from this directory:

    ./run.sh                               # BF16 KV, the headline arm
    ./sending.sh --wait                    # smoke test once /health returns 200

  and from the REPO ROOT:

    MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
    VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
    MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/bf16kv \
      ./bench.sh batch context prefix

  The VLLM=/PY= overrides are load-bearing: bench.sh defaults to the conda env's
  vLLM 0.28.0, which does not know glm5_next and fails in preflight even though
  the server is fine.
EOS
  else
    printf '\n  \033[31msubmit failed (rc=%s)\033[0m\n' "$rc"
  fi
  return $rc
}

main "$@"
