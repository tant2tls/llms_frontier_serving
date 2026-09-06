#!/usr/bin/env bash
# Submit the RunAI workspace for Qwen3.8-Flash-Next-FP8. **RUN THIS FROM OUTSIDE THE
# CONTAINER** (a login host with a working `runai` CLI), not from inside a workspace.
#
#   ./submit_job.sh              # delete the old workspace if present, then submit
#   DRY_RUN=1 ./submit_job.sh    # print the command, submit nothing
#   NAME=tan-8gpus-qwen38b ./submit_job.sh
#
# Adapted from GLM-5.3-Flash/submit_job.sh. Same mounts, same uid mapping, same CPU
# budget -- so the ENVIRONMENT is identical and only the image differs. That matters:
# see "the engine confound" below.
#
# ---------------------------------------------------------------------------
# 1) THE IMAGE, AND WHY IT IS A FAIRNESS PROBLEM
# ---------------------------------------------------------------------------
#   vllm/vllm-openai:qwen38-flash-next
#
# Qwen3.8-Flash-Next-FP8 CANNOT run in the glm53-flash image -- verified 2026-09-02:
#
#     config.json     model_type: qwen4_exp / Qwen4ExpForConditionalGeneration
#     glm53 image     vllm 0.1.dev20051+g487ecf187  ->  Qwen4Exp*/Qwen3_8FlashNext*: NONE
#
# ⚠️ SO THIS ARM LANDS ON A DIFFERENT ENGINE BUILD THAN GLM-5.3 AND V4-FLASH, WHICH
# ARE MATCHED TO EACH OTHER. That is a real confound and we already measured how big
# such a thing can be: on GLM, swapping ONLY the attention backend (same model, same
# dtype) cost 27.3% throughput. An engine delta can dwarf the architecture effect you
# are trying to report.
#
# Before quoting ANY Qwen3.8-vs-GLM or Qwen3.8-vs-V4 ratio, do one of:
#   (a) PREFERRED -- check whether GLM-5.3 or V4-Flash also loads in THIS image and
#       rerun that one here as a bridge arm. One rerun buys a matched 3-way compare.
#   (b) FALLBACK  -- publish Qwen3.8 standalone (its own scaling curves, KV structure,
#       MTP A/B) and state plainly that cross-model ratios are engine-confounded.
# NEVER (c): quote a cross-image ratio without saying so. See ../WORKFLOW.md §2.
#
# ---------------------------------------------------------------------------
# 2) THIS IMAGE SHOULD COME UP WITH ALL 8 GPUs FREE -- BUT VERIFY IT
# ---------------------------------------------------------------------------
# The glm53-flash image declares ENTRYPOINT ["vllm","serve"], so with no --command
# its PID 1 became a live Qwen3-0.6B server squatting ~77 GiB on GPU 0 and owning
# port 8000. That is why GLM's scripts default to GPUS=4,5,6,7.
#
# This image is expected to be clean. `--command -- sleep infinity` IS STILL PASSED
# ANYWAY, because it is free insurance: if this image has the same ENTRYPOINT, the
# flag prevents the squatter; if it does not, `sleep infinity` is exactly what we want
# PID 1 to be regardless. Cheap to keep, expensive to omit and be wrong.
#
# ⚠️ VERIFY, DO NOT ASSUME (this project's hardware rule). After the pod is Running:
#
#     ps -p 1 -o args=      # WANT: `sleep infinity`.   BAD: `vllm serve`
#     nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
#                           # WANT: all 8 near 0 MiB
#     nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # WANT: empty
#
# If GPU 0 IS held despite the flag, do NOT `kill -9` PID 1 -- it is container init
# and the container dies with it (tested on GLM: RunAI rescheduled and the squatter
# came back). Fall back to GPUS=4,5,6,7 TP=4 instead.
#
# ---------------------------------------------------------------------------
# 3) WITH 8 CLEAN GPUs, RUN TP8 -- AND SAY WHY
# ---------------------------------------------------------------------------
# `_common.sh` currently defaults to GPUS=4,5,6,7 / TP=4, which was a WORKAROUND for
# the squatter, not a preference. On a clean container prefer TP8:
#
#     TP8 -> 21.6 GiB/GPU of 79.6   (weights 172.76 GiB measured)
#     TP4 -> 43.2 GiB/GPU
#
# TP8 matches the GLM-5.3 and V4-Flash runs (both TP8/EP on 8xH100), which removes one
# more difference from the cross-model comparison. Launch it explicitly:
#
#     cd Qwen3.8-Flash-Next-FP8 && GPUS=0,1,2,3,4,5,6,7 TP=8 ./run.sh
#
# ⚠️ TWO Qwen-SPECIFIC CONSTRAINTS, both verified from config.json:
#   - TP MUST DIVIDE BOTH 24 attention heads AND 512 experts. TP 1/2/4/8 are legal;
#     **TP 3 and TP 6 are NOT** (512 % 3 = 2, 512 % 6 = 2).
#   - --enable-expert-parallel IS MANDATORY, not a tuning knob. The recipe states for
#     Hopper "Plain TP8 is incompatible with the FP8 checkpoint; use TEP8" -- the
#     checkpoint's 128-wide FP8 quantization blocks would be split mid-block by plain
#     TP. `_common.sh` already sets it.
#   - `--gpu-memory-utilization` differs from the other models by design: the recipe
#     sanctions 0.85 at TP8 / 0.90 at TP4, vs 0.82 for GLM and V4. RECORD IT; it
#     changes KV capacity, not the decode cost model.
#
# ---------------------------------------------------------------------------
# 4) EVERY OTHER FLAG is carried over verbatim from the working GLM submission so the
# environment stays identical. The ones that interact with this benchmark:
#
#   -g 8                      All eight GPUs, so TP8 is available. RUNAI_NUM_OF_GPUS=8.
#   --cpu-core-request/limit 198
#                             Matches the MEASURED cgroup quota on these nodes
#                             (cpu.cfs_quota_us/cfs_period_us = 198), NOT nproc (256,
#                             the HOST count, which would oversubscribe into CFS
#                             throttling).
#   --large-shm               vLLM's TP workers talk over /dev/shm; the default 64 MB
#                             is too small for 8-way TP.
#   vol22-scratch mount       Holds this repo AND the 174 GiB Qwen3.8 weight cache.
#                             Without it there is nothing to serve.
#   vol11-scratch mount       Conda envs. NOTE the report's serving runs deliberately
#                             use the IMAGE's vLLM (/usr/local/bin/vllm) -- the conda
#                             env's 0.28.0 cannot parse model_type=qwen4_exp, and that
#                             applies to the `vllm bench serve` CLIENT too.
#   --preemptible             ⚠️ RunAI CAN EVICT MID-SWEEP. `bench.sh` resumes (it skips
#                             existing results) but the server reloads from scratch.
#                             If you have non-preemptible quota, spend it here.
#   --backoff-limit 0         Do not silently retry into a different GPU state.
# ---------------------------------------------------------------------------

set -uo pipefail

NAME=${NAME:-tan-8gpus-qwen38}
IMAGE=${IMAGE:-vllm/vllm-openai:qwen38-flash-next}
GPUS=${GPUS:-8}
CPUS=${CPUS:-198}
DRY_RUN=${DRY_RUN:-0}

# Everything after `--` is the container command, so `--command -- sleep infinity`
# MUST be last. Keep the --nfs flags grouped ABOVE it: anything left after `--` gets
# swallowed into sleep's argv and silently ignored.
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
    # Insurance against an ENTRYPOINT squatter -- must stay last. See the header.
    --command -- sleep infinity
  )
}

main() {
  build_cmd

  printf '\033[1mRunAI workspace submit\033[0m  %s\n' "$NAME"
  printf '  image : %s\n' "$IMAGE"
  printf '  gpus  : %s   cpus: %s\n' "$GPUS" "$CPUS"
  printf '  entrypoint override: sleep infinity  <- insurance; verify PID 1 after start\n\n'

  command -v runai >/dev/null 2>&1 || {
    printf '  \033[31mFAIL\033[0m no `runai` on PATH.\n'
    printf '  Run this from a login host, not from inside a workspace container.\n'
    printf '  (Inside a container the control plane is unreachable:\n'
    printf '   x509: certificate signed by unknown authority.)\n'
    return 1
  }

  printf '%s\n\n' "${CMD[*]}"

  if [[ $DRY_RUN == 1 ]]; then
    printf '  DRY_RUN=1 -- nothing submitted\n'
    return 0
  fi

  # An existing workspace with this name makes the submit fail. Deleting it is ALSO
  # how an old container's GPUs are released -- there is no way to free them from
  # inside.
  if runai workspace list 2>/dev/null | grep -q "\b${NAME}\b"; then
    printf '  \033[33mnote\033[0m workspace %s already exists.\n' "$NAME"
    printf '  Deleting it releases the old container and its GPUs.\n'
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
    nvcc --version                         # record the CUDA major -- caches key on it
    /usr/bin/python3 -c "import vllm; print(vllm.__version__)"   # RECORD THIS:
                                           # it differs from the GLM/V4 arms -> confound

  then, from this directory (TP8 to match the GLM-5.3 / V4-Flash layout):

    GPUS=0,1,2,3,4,5,6,7 TP=8 ./run.sh     # base model, spec decode OFF
    ../sending.sh --wait                   # smoke test: confirm it ANSWERS, not just /health

  and from the REPO ROOT (all five overrides are load-bearing -- see ../REPRODUCE.md §3):

    MODEL=Qwen/Qwen3.8-Flash-Next-FP8 PORT=8001 \
    VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
    MODELDIR=Qwen3.8-Flash-Next-FP8 OUTDIR=Qwen3.8-Flash-Next-FP8/results/base \
    QUANT=fp8 KV_DTYPE=bfloat16 \
      ./bench.sh batch context prefix

  VLLM=/PY= matter because bench.sh defaults to the conda env's vLLM 0.28.0, which
  cannot parse model_type=qwen4_exp and dies in preflight even when the server is
  perfectly healthy. QUANT=/KV_DTYPE= matter because bench.sh's defaults are
  DeepSeek-V4's and would silently mislabel every result JSON.

  BEFORE citing any cross-model ratio, read ../WORKFLOW.md §2 -- this image is a
  DIFFERENT engine build from the GLM-5.3 and V4-Flash arms.

  ALSO: add Qwen3.8 to facts() in ../normalize.sh with param counts MEASURED from
  safetensors shapes. The row is currently a deliberate `unknown` placeholder --
  earlier notes wrongly called this model dense (it is E=512, k=10 under the key
  `num_experts`, not `n_routed_experts`).
EOS
  else
    printf '\n  \033[31msubmit failed (rc=%s)\033[0m\n' "$rc"
  fi
  return $rc
}

main "$@"
