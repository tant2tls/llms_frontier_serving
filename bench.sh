#!/usr/bin/env bash
# Throughput/latency sweep for DeepSeek-V4-Flash on the local vLLM server.
#
#   ./bench.sh                 # preflight + batch sweep + context sweep
#   ./bench.sh quick           # one fast point, proves the harness works
#   ./bench.sh batch           # Q1: bottleneck vs batch size
#   ./bench.sh context         # Q1/Q2: bottleneck vs context length
#   ./bench.sh prefix          # Q2: prefix-cache axis (synthetic control)
#   ./bench.sh sharegpt        # Tier 3: external comparability
#   DRY_RUN=1 ./bench.sh       # print commands, run nothing
#
# Grid points are anchored to MEASURED TraceLab percentiles (UW-SyFI/TraceLab
# v0.0.2, 665,453 rounds): ISL p50=132,092  p90=338,662 / OSL p50=249.
# 16K/64K/128K/256K covers ~84% of real agentic rounds.
#
# Results land in $OUTDIR as one JSON per point plus a manifest recording
# hardware, quantization and server flags -- per this project's rule that every
# performance claim carries its number, its hardware, and its method.

set -uo pipefail

HOST=${HOST:-localhost}
PORT=${PORT:-8000}
BASE="http://${HOST}:${PORT}"
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Flash}

# Results live under the per-model subfolder, one dir per model (deepseek_v4_flash/,
# glm_4_7_flash/, ...). MODELDIR defaults to a slug of the served model name, so
# running this from the repo root against a different server lands in the right place.
MODELDIR=${MODELDIR:-$(printf '%s' "${MODEL##*/}" | tr 'A-Z.-' 'a-z__')}
OUTDIR=${OUTDIR:-$MODELDIR/results/$(date +%Y%m%d-%H%M%S)}
VLLM=${VLLM:-/prj/corp/airesearch/lasvegas/vol11-scratch/tanngo/miniconda3/envs/vllm-py12/bin/vllm}
PY=${PY:-/prj/corp/airesearch/lasvegas/vol11-scratch/tanngo/miniconda3/envs/vllm-py12/bin/python3}

# Cap prefill work per grid point so a 256K-context point cannot run for an
# hour. num_prompts = clamp(2*concurrency, 8, PREFILL_BUDGET/ISL).
PREFILL_BUDGET=${PREFILL_BUDGET:-8000000}

# Provenance stamped into every result JSON's metadata. These USED TO BE HARDCODED
# to DeepSeek-V4-Flash's values, which silently mislabelled every other model's
# results -- GLM-5.3-Flash has FP8 (not MXFP4) experts and BF16 (not fp8) KV, so a
# GLM sweep run with the old defaults produced JSONs claiming
# "quant=fp8-attn-dense+mxfp4-experts kv_cache_dtype=fp8". Override per model:
#
#   GLM-5.3-Flash : QUANT=fp8-attn-dense+fp8-experts KV_DTYPE=bfloat16
#   V4-Flash      : the defaults below
QUANT=${QUANT:-fp8-attn-dense+mxfp4-experts}
KV_DTYPE=${KV_DTYPE:-fp8}
HW=${HW:-8xH100-80GB-HBM3}

DRY_RUN=${DRY_RUN:-0}

pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33mnote\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
head_() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

scrape() { curl -s --max-time 10 "${BASE}/metrics"; }
metric() { printf '%s\n' "$1" | awk -v k="^vllm:$2" '$0 ~ k {v=$2} END{print v+0}'; }

# ---------------------------------------------------------------- preflight
preflight() {
  head_ "preflight"
  for c in curl jq "$VLLM" "$PY"; do
    command -v "$c" >/dev/null 2>&1 || [[ -x $c ]] || { fail "missing: $c"; return 1; }
  done
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "${BASE}/health")
  [[ $code == 200 ]] || { fail "server not healthy at $BASE (got ${code:-nothing})"; return 1; }
  pass "server healthy at $BASE"

  local served
  served=$(curl -s --max-time 10 "${BASE}/v1/models" | jq -r '.data[0].id // empty')
  [[ -n $served ]] || { fail "cannot read /v1/models"; return 1; }
  pass "serving: $served"
  [[ $served == "$MODEL" ]] || warn "MODEL=$MODEL but server reports $served; using the server's id"
  MODEL=$served

  mkdir -p "$OUTDIR"
  pass "output dir: $OUTDIR"

  # Freeze provenance next to the numbers. Without this the results are
  # unciteable six weeks from now.
  {
    echo "timestamp: $(date -Is)"
    echo "host: $(hostname)"
    echo "model: $MODEL"
    echo "base: $BASE"
    echo
    echo "## GPUs"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null
    echo
    echo "## driver / versions"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
    "$PY" -c "import torch,vllm;print('torch',torch.__version__);print('vllm',vllm.__version__)" 2>/dev/null
    echo
    echo "## server command line"
    ps -eo args= 2>/dev/null | grep -m1 '[v]llm serve' || echo "(not found)"
    echo
    echo "## quantization (from served config)"
    echo "QUANT=$QUANT  KV_DTYPE=$KV_DTYPE   (override these per model -- see header)"
    echo "resolved from the LIVE server below, not hardcoded:"
    curl -s --max-time 10 "${BASE}/metrics" | grep -m1 "^vllm:cache_config_info" \
      | tr ',' '\n' | grep -E "block_size|cache_dtype|num_gpu_blocks=|gpu_memory_utilization" \
      | sed 's/^/  /' || echo "  (could not scrape cache_config_info)"
    echo
    echo "## MFU / bandwidth counter coverage  (--enable-mfu-metrics)"
    echo "Which ComponentMetrics vLLM instantiated for THIS model. Any component NOT"
    echo "listed is silently EXCLUDED from estimated_flops/read_bytes, so the achieved"
    echo "numbers are a LOWER BOUND whose gap DIFFERS PER MODEL -- which means the raw"
    echo "counters are NOT comparable across models without this table:"
    if [[ -n ${SERVER_LOG:-} && -f ${SERVER_LOG:-} ]]; then
      grep -h "Instantiated ComponentMetrics" "$SERVER_LOG" 2>/dev/null \
        | sed 's/.*Instantiated/  Instantiated/' || echo "  (none found in $SERVER_LOG)"
    else
      echo "  SERVER_LOG not set -- pass SERVER_LOG=<path to serve_*.log> to record it here."
      echo "  Measured 2026-09-02 by grepping each model's launch log:"
      echo "    GLM-5.3-Flash     : ffn, unembed        (NO attn -- 34 KDA + 11 DSA unmodelled)"
      echo "    DeepSeek-V4-Flash : unembed only        (NO attn, NO ffn)"
      echo "    Qwen3.8-Flash-Next: attn, ffn, unembed  (full coverage)"
    fi
    echo "⚠️ These counters are an ENGINE-SIDE ESTIMATE, not a hardware counter."
    echo "   vLLM computes bytes/FLOPs per scheduler step from CONFIG SHAPES times"
    echo "   that step's real batch composition (perf.py:488-513, :1069-1127), so the"
    echo "   batch/context term is measured and bytes-per-token is modelled. Label [E]."
    echo "   perf.py:428 carries the TODO: 'discern cases where we have mixture of"
    echo "   different attention layer types such as SWA, MLA' -- that TODO is exactly"
    echo "   this report's thesis, which is why hybrid models lose their attn component."
    echo
    echo "## grid provenance"
    echo "ISL points anchored to UW-SyFI/TraceLab v0.0.2 rounds table:"
    echo "  input_tokens_total  p50=132092 p90=338662 p99=856464"
    echo "  output_tokens       p50=249    p90=1332"
    echo "  prefix_tokens/input = 95.6% (mean), 98.8% of rounds have prefix>0"
  } > "$OUTDIR/manifest.txt" 2>&1
  pass "wrote $OUTDIR/manifest.txt"

  # MTP is on unless the server was restarted without --speculative-config.
  local m d
  m=$(scrape); d=$(metric "$m" "spec_decode_num_draft_tokens_total")
  if [[ ${d%.*} -gt 0 ]]; then
    warn "spec-decode (MTP) is ACTIVE on this server."
    warn "plan.md scopes base-model throughput without spec decode -- for those"
    warn "numbers restart run.sh with --speculative-config removed."
  fi
}

# num_prompts for a given ISL and concurrency
nprompts() {
  local isl=$1 conc=$2
  "$PY" -c "
isl,conc,budget=$isl,$conc,$PREFILL_BUDGET
n=max(8, 2*conc)
n=min(n, max(8, budget//max(isl,1)))
print(int(n))"
}

# ------------------------------------------------------------------ one point
# run_point <label> <isl> <osl> <concurrency>
run_point() {
  local label=$1 isl=$2 osl=$3 conc=$4
  local np; np=$(nprompts "$isl" "$conc")
  local out="$OUTDIR/${label}.json"

  if [[ -f $out ]]; then warn "skip $label (result exists)"; return 0; fi

  # UNIQUE SEED PER POINT. A shared seed makes `vllm bench serve` generate the
  # SAME prompts at every point, so later points read the earlier points' prefix
  # cache: measured once as a phantom 640 tok/s at c=4 that collapsed to 214 on a
  # fresh seed. Derive it from the point's shape so it is deterministic but distinct.
  local seed=$(( (isl % 100000) + conc * 7919 + osl * 31 ))

  printf '\n  \033[1m%s\033[0m  ISL=%s OSL=%s conc=%s prompts=%s seed=%s\n' \
         "$label" "$isl" "$osl" "$conc" "$np" "$seed"

  # Cache-hit counter before the run, so we can prove this point was cold.
  local snap0; snap0=$(scrape)
  local hits0; hits0=$(metric "$snap0" "prefix_cache_hits_total")

  # ---- MFU / bandwidth counters, taken as a DELTA across this point ----------
  # `--enable-mfu-metrics` is passed on every arm, but until 2026-09-02 nothing
  # read the gauges it populates, so every achieved-bandwidth number in the
  # report was analytical. These three are prometheus COUNTERS (monotonic
  # totals since server start, per perf.py:1559 `_counter_cls = Counter`), so a
  # single scrape is meaningless -- only (after - before) / elapsed is a rate.
  #
  # ⚠️ READ perf.py BEFORE CALLING THESE "MEASURED HARDWARE BANDWIDTH". They are
  # NOT hardware counters. vLLM computes them per scheduler step from CONFIG
  # SHAPES times the step's ACTUAL batch composition, e.g.
  #     read_bytes["qkv_weight"] = D*(q+2*kv)*d*weight_byte_size*L
  #     read_bytes["routed_down_weights"] = ... num_experts_per_tok ...
  # (perf.py:488-513, :1069-1127). So the batch/context term is measured and the
  # bytes-per-token term is modelled. That still beats our own hand-rolled
  # `active_params x bytes x steps/s` estimate -- it is per-step, it tracks the
  # real prefill/decode mix, and it accounts for activations and the unembed --
  # but it is an ENGINE-SIDE ESTIMATE. Label it [E], not [M].
  local mfu_flops0 mfu_rbytes0 mfu_wbytes0
  mfu_flops0=$(metric "$snap0" "estimated_flops_per_gpu_total")
  mfu_rbytes0=$(metric "$snap0" "estimated_read_bytes_per_gpu_total")
  mfu_wbytes0=$(metric "$snap0" "estimated_write_bytes_per_gpu_total")

  local -a cmd=(
    "$VLLM" bench serve
    --backend openai-chat --endpoint /v1/chat/completions
    --host "$HOST" --port "$PORT" --model "$MODEL"
    --dataset-name random
    --random-input-len "$isl" --random-output-len "$osl"
    --num-prompts "$np" --max-concurrency "$conc"
    --ignore-eos                       # else the model picks its own OSL
    # NO --num-warmups. ⚠️ IT SELF-POISONS THE PREFIX CACHE.
    # `--num-warmups 1` sends a warmup request drawn from the SAME seeded prompt
    # set as the measured run, so with enable_prefix_caching=True (the default) the
    # measured run re-reads the warmup's own blocks. MEASURED on GLM-5.3-Flash
    # (2026-09-02): EVERY batch point reported exactly 16,000 new hits --
    # 25 blocks x block_size 640 = one ISL-16384 prompt, the warmup's, identical at
    # every concurrency. That is not cross-point leakage (rule 2, which unique seeds
    # already fix); it is one point contaminating ITSELF.
    #
    # It biases the grid UNEVENLY, which is the real damage -- the constant 16,000
    # is a large share of a small point and a rounding error on a big one:
    #     c=1  16,000 / 131,072 prompt tokens = 12.2%
    #     c=4  16,000 / 131,072               = 12.2%
    #     c=16 16,000 / 524,288               =  3.1%
    #     c=64 16,000 / 2,097,152             =  0.8%
    # So it inflates exactly the low-concurrency points that anchor the "throughput
    # vs batch size" curve, flattening the measured scaling.
    --seed "$seed"
    --percentile-metrics ttft,tpot,itl,e2el
    --save-result --result-filename "$out"
    --metadata "hw=$HW" "quant=$QUANT"
               "kv_cache_dtype=$KV_DTYPE" "tp=8" "ep=on" "isl=$isl" "osl=$osl" "conc=$conc"
               "seed=$seed"
  )

  if [[ $DRY_RUN == 1 ]]; then printf '    DRY: %s\n' "${cmd[*]}"; return 0; fi

  # Poll KV-cache usage during the run; the peak tells you whether memory or
  # compute was binding, which is the whole point of the sweep.
  local kvf; kvf=$(mktemp)
  ( while :; do scrape | awk '/^vllm:kv_cache_usage_perc/{print $2}' >> "$kvf"; sleep 2; done ) &
  local poller=$!

  # Wall clock around the point, for the MFU rate denominator. `vllm bench serve`
  # reports its own `duration`, which excludes client startup/tokenization; we use
  # that when available (it is the fairer denominator) and fall back to this.
  local t0; t0=$(date +%s.%N)
  "${cmd[@]}" > "$OUTDIR/${label}.log" 2>&1
  local rc=$?
  local t1; t1=$(date +%s.%N)

  kill "$poller" 2>/dev/null; wait "$poller" 2>/dev/null

  local kvmax
  kvmax=$(awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{printf "%.3f", m}' "$kvf" 2>/dev/null)
  rm -f "$kvf"

  if [[ $rc != 0 ]]; then
    fail "$label exited $rc -- tail of log:"
    tail -n 12 "$OUTDIR/${label}.log" | sed 's/^/      /'
    return 1
  fi

  # ⚠️ ZERO-COMPLETION GUARD. `vllm bench serve` EXITS 0 even when every request
  # failed: it warns "All requests failed", prints 0.00 for every metric, and
  # returns success. Measured 2026-09-02 -- the ISL-262144 point 400'd on all 16
  # requests (ISL+OSL was 256 over max_model_len) and produced a summary row of all
  # zeros that looks like a real, terrible datapoint. Fail loudly instead.
  local completed
  completed=$(jq -r '.completed // 0' "$out" 2>/dev/null || echo 0)
  if [[ ${completed%.*} -eq 0 ]]; then
    fail "$label completed 0 of $np requests -- NOT A RESULT, discarding $out"
    grep -m3 -E "Error [0-9]+:|All requests failed" "$OUTDIR/${label}.log" 2>/dev/null | sed 's/^/      /'
    warn "    common cause: ISL+OSL > max_model_len (check the server's max_model_len)"
    mv -f "$out" "$out.failed" 2>/dev/null
    return 1
  fi

  # ⚠️ QUEUEING GUARD. A point can be COLD and COMPLETE and still be wrong by 60%.
  # Measured 2026-09-02 on the V4 image-engine sweep: three batch points passed both
  # other guards yet under-reported throughput by 25-60% because requests sat in the
  # scheduler queue instead of being served. The signal is p99 TTFT >> median TTFT:
  #
  #     point   first run   rerun    p99 TTFT (first)  p99 TTFT (rerun)
  #     c=4      98.7       219.0    13,252 ms          2,633 ms
  #     c=16    132.2       330.0    28,205 ms          9,771 ms
  #     c=64    290.3       389.0    38,973 ms         38,846 ms
  #
  # Using the bad points, the conda-vs-image "engine effect" computed to 0.64x
  # ("the image is 36% slower") -- a headline-grade WRONG conclusion. With reruns it
  # is 1.01x. Cause is most likely first-touch warmup on a freshly started server.
  #
  # ⚠️ THE THRESHOLD MUST BE CONCURRENCY-AWARE. A flat 5x cutoff is USELESS -- it
  # fires on nearly every legitimate c=64 point. Measured p99/median distribution
  # across 67 believed-good points in this repo:
  #
  #     conc=1   n=9   min 1.0  median 1.8  max  3.8
  #     conc=4   n=9   min 1.3  median 1.5  max  2.1
  #     conc=8   n=31  min 1.6  median 3.8  max 39.1
  #     conc=16  n=9   min 2.5  median 3.5  max 10.0
  #     conc=64  n=9   min 3.6  median 10.7 max 19.1
  #
  # At high concurrency a wide spread is REAL (64 requests genuinely queue behind
  # each other), so only low-concurrency points give a clean signal. The corrupted
  # V4 c=4 point had ratio 6.96 where every good c=4 point is <= 2.1 -- unambiguous.
  # Threshold: 4x for conc<=4, 8x for conc<=16, and no check above that (the metric
  # cannot discriminate there; rely on trend-breaks instead).
  local ttft_p50 ttft_p99 lim
  ttft_p50=$(jq -r '.median_ttft_ms // 0' "$out" 2>/dev/null || echo 0)
  ttft_p99=$(jq -r '.p99_ttft_ms // 0' "$out" 2>/dev/null || echo 0)
  # The PREFIX sweep is exempt: its p99 spread is BY DESIGN. The first request must
  # build the 64K shared prefix while the rest wait for it, so a 8-40x p99/median
  # ratio there is the effect being measured, not an artifact. (Verified: the V4 16K
  # context points show ratio 8.3-8.6 on BOTH engines yet agree on throughput to
  # within 4% -- 100.9 vs 104.5 tok/s -- so a high ratio alone is not corruption.)
  if   [[ $label == prefix_* ]]; then lim=0
  elif [[ $conc -le 4  ]]; then lim=4
  elif [[ $conc -le 16 ]]; then lim=8
  else lim=0; fi          # 0 = do not check
  if [[ $lim != 0 ]]; then
    local ratio
    ratio=$("$PY" -c "p50=$ttft_p50; p99=$ttft_p99; print(f'{(p99/p50) if p50>0 else 0:.2f}')" 2>/dev/null || echo 0)
    if "$PY" -c "import sys; sys.exit(0 if $ratio > $lim else 1)" 2>/dev/null; then
      warn "    QUEUEING SUSPECT: p99/median TTFT = ${ratio}x at conc=$conc (limit ${lim}x)."
      warn "    Throughput is probably UNDERSTATED -- rerun before citing. See"
      warn "    deepseek_v4_flash/results/_v4image_firstrun_queued/README.md"
    fi
  fi

  # New prefix-cache hits during this point. Anything > 0 means the prompts were
  # not cold and the throughput number is inflated -- the run is NOT trustworthy.
  local hits1 dhits snap1
  snap1=$(scrape)
  hits1=$(metric "$snap1" "prefix_cache_hits_total")
  dhits=$("$PY" -c "print(int(float('${hits1:-0}') - float('${hits0:-0}')))" 2>/dev/null || echo 0)

  # ---- MFU deltas -> achieved TFLOP/s and GB/s per GPU -----------------------
  # Denominator: prefer the benchmark's own `duration` (excludes client-side
  # tokenization, so it is the honest serving window); fall back to wall clock.
  local mfu_flops1 mfu_rbytes1 mfu_wbytes1 dur
  mfu_flops1=$(metric "$snap1" "estimated_flops_per_gpu_total")
  mfu_rbytes1=$(metric "$snap1" "estimated_read_bytes_per_gpu_total")
  mfu_wbytes1=$(metric "$snap1" "estimated_write_bytes_per_gpu_total")
  dur=$(jq -r '.duration // 0' "$out" 2>/dev/null || echo 0)

  # H100 SXM peaks, for the fraction-of-roofline column rule 5 requires:
  #   HBM3 3.35 TB/s; dense FP8 tensor core 1979 TFLOP/s (no sparsity).
  # Both are per GPU, matching the per-GPU counters.
  local mfu_json
  mfu_json=$("$PY" -c "
f0,f1 = float('${mfu_flops0:-0}'),  float('${mfu_flops1:-0}')
r0,r1 = float('${mfu_rbytes0:-0}'), float('${mfu_rbytes1:-0}')
w0,w1 = float('${mfu_wbytes0:-0}'), float('${mfu_wbytes1:-0}')
d  = float('${dur:-0}') or (float('${t1:-0}') - float('${t0:-0}'))
df, dr, dw = f1-f0, r1-r0, w1-w0
import json
if d <= 0 or (df <= 0 and dr <= 0):
    print(json.dumps({'mfu_available': False}))
else:
    tfps = df/d/1e12
    gbps = (dr+dw)/d/1e9
    print(json.dumps({
        'mfu_available': True,
        'mfu_window_s': round(d, 3),
        'est_flops_per_gpu_delta': df,
        'est_read_bytes_per_gpu_delta': dr,
        'est_write_bytes_per_gpu_delta': dw,
        'achieved_tflops_per_gpu': round(tfps, 4),
        'achieved_gbps_per_gpu': round(gbps, 3),
        'frac_peak_flops_h100_fp8': round(tfps/1979.0, 6),
        'frac_peak_hbm_h100': round(gbps/3350.0, 6),
        'mfu_provenance': 'vllm --enable-mfu-metrics counters; engine-side ESTIMATE '
                          '(config shapes x measured batch composition), not a hardware counter',
    }))
" 2>/dev/null || echo '{"mfu_available":false}')

  # Fold the KV peak, the cache-hit delta and the MFU rates into the saved result
  # so all of them travel with the numbers.
  if [[ -f $out ]]; then
    jq --argjson kv "${kvmax:-0}" --argjson dh "${dhits:-0}" \
       --argjson mfu "$mfu_json" \
       '. + {peak_kv_cache_usage_perc:$kv, new_prefix_cache_hits:$dh,
             cold_run:($dh == 0)} + $mfu' "$out" > "$out.tmp" && mv "$out.tmp" "$out"
  fi

  # Echo the headline numbers so the terminal is readable without jq gymnastics.
  if [[ -f $out ]]; then
    jq -r '
      "    output tok/s   : \(.output_throughput      | tostring | .[0:9])",
      "    total  tok/s   : \(.total_token_throughput | tostring | .[0:9])",
      "    req/s          : \(.request_throughput     | tostring | .[0:8])",
      "    TTFT p50/p99 ms: \(.median_ttft_ms|round) / \(.p99_ttft_ms|round)",
      "    TPOT p50/p99 ms: \(.median_tpot_ms|round) / \(.p99_tpot_ms|round)"' "$out"
    printf '    peak KV usage  : %s\n' "${kvmax:-n/a}"
    # MFU line. Absent => the counters read 0, which for a hybrid model usually
    # means no ComponentMetrics covered its attention stack (see mfu_coverage in
    # the manifest), NOT that the server was idle.
    jq -r 'if .mfu_available then
             "    achieved       : \(.achieved_tflops_per_gpu) TFLOP/s/GPU (\((.frac_peak_flops_h100_fp8*100)|.*100|round|./100)% of FP8 peak)",
             "                     \(.achieved_gbps_per_gpu) GB/s/GPU (\((.frac_peak_hbm_h100*100)|.*100|round|./100)% of 3.35 TB/s) [E, engine estimate]"
           else "    achieved       : mfu counters read 0 -- see mfu_coverage in manifest.txt" end' "$out" 2>/dev/null
    if [[ ${dhits:-0} -gt 0 ]]; then
      fail "    new cache hits : ${dhits} -- NOT A COLD RUN, throughput is inflated"
      warn "    this point reused a previous point's prefix cache; discard it"
    else
      pass "    cold run confirmed (0 new prefix-cache hits)"
    fi
  fi
}

# ------------------------------------------------------------------- sections
# Q1: where is the bottleneck as batch size grows? Fixed modest context so we
# can afford many concurrency points.
sweep_batch() {
  head_ "batch sweep (Q1) -- ISL 16K, OSL 256"
  # CONCS is overridable so the batch axis can be resolved more finely without
  # touching this file:  CONCS="1 2 4 8 16 32 48 64" ./bench.sh batch
  # Each point's seed is derived from (isl, conc, osl) in run_point, so ADDING
  # concurrencies cannot contaminate existing points' prefix cache -- new conc
  # values get new seeds, and existing result files are skipped.
  for conc in ${CONCS:-1 4 16 64}; do
    run_point "batch_isl16k_c${conc}" 16384 256 "$conc" || return 1
  done
}

# Q1/Q2: where is the bottleneck as context grows? Fixed concurrency.
# ISL points bracket the TraceLab median (132K).
sweep_context() {
  head_ "context sweep (Q1/Q2) -- concurrency 8, OSL 256"
  # ⚠️ THE TOP POINT NEEDS REAL HEADROOM, NOT max_model_len - OSL.
  # Measured 2026-09-02 against --max-model-len 262144, OSL 256:
  #     ISL 262,144 -> 400 Bad Request on all 16 requests (obviously: +OSL is over)
  #     ISL 261,888 -> STILL 400, even though 261,888 + 256 == 262,144 exactly
  #     ISL 260,000 -> works (bisected with a 1-prompt probe)
  # So `max_model_len - OSL` is NOT sufficient: `--dataset-name random` does not
  # emit exactly --random-input-len tokens (it jitters, and the chat template adds
  # tokens on top), so a point sitting exactly on the boundary still overflows.
  # 260,000 is the verified-working value and is still 99.2% of max_model_len, which
  # is what the "256K context" claim needs. Re-bisect if max_model_len changes.
  #
  # This mattered because the failure is SILENT: `vllm bench serve` warns "All
  # requests failed", prints 0.00 for every metric, and STILL EXITS 0 -- see the
  # zero-completion guard in run_point.
  local top=${CTX_TOP_ISL:-260000}
  for isl in 16384 65536 131072 "$top"; do
    run_point "ctx_isl${isl}_c8" "$isl" 256 8 || return 1
  done
}

# Q2: prefix-cache axis. Synthetic control -- clean, zero download, and the
# right thing to measure BEFORE replaying a real trace.
sweep_prefix() {
  head_ "prefix-cache sweep (Q2) -- shared prefix 64K + 2K unique suffix"
  for nprefix in 1 4 16; do
    local label="prefix_p64k_n${nprefix}"
    local out="$OUTDIR/${label}.json"
    [[ -f $out ]] && { warn "skip $label (exists)"; continue; }
    printf '\n  \033[1m%s\033[0m  num_prefixes=%s (prompts/prefix = 64/%s)\n' \
           "$label" "$nprefix" "$nprefix"
    local -a cmd=(
      "$VLLM" bench serve
      --backend openai-chat --endpoint /v1/chat/completions
      --host "$HOST" --port "$PORT" --model "$MODEL"
      --dataset-name prefix_repetition
      --prefix-repetition-prefix-len 65536
      --prefix-repetition-suffix-len 2048
      --prefix-repetition-num-prefixes "$nprefix"
      --prefix-repetition-output-len 256
      --num-prompts 64 --max-concurrency 8 --ignore-eos --seed "$((4200 + nprefix))"
      --percentile-metrics ttft,tpot,itl,e2el
      --save-result --result-filename "$out"
      --metadata "hw=$HW" "quant=$QUANT"
                 "kv_cache_dtype=$KV_DTYPE" "num_prefixes=$nprefix" "prefix_len=65536"
    )
    if [[ $DRY_RUN == 1 ]]; then printf '    DRY: %s\n' "${cmd[*]}"; continue; fi
    "${cmd[@]}" > "$OUTDIR/${label}.log" 2>&1 \
      || { fail "$label failed"; tail -n 12 "$OUTDIR/${label}.log" | sed 's/^/      /'; return 1; }
    jq -r '"    output tok/s: \(.output_throughput|tostring|.[0:9])   TTFT p50 ms: \(.median_ttft_ms|round)"' "$out"
  done
  warn "fewer prefixes = more sharing = more cache hits. TTFT should fall as n drops."
}

# Tier 3: one ShareGPT run purely so the harness is comparable to published
# numbers. NOT the primary dataset -- 1K-token prompts hide the KV story.
sweep_sharegpt() {
  head_ "ShareGPT (Tier 3, external comparability)"
  # Shared across models -- keep it at the repo root, not under a model dir.
  local sg=${SHAREGPT_PATH:-datasets/ShareGPT_V3_unfiltered_cleaned_split.json}
  if [[ ! -f $sg ]]; then
    warn "ShareGPT json not found at: $sg"
    warn "one-time download (~1 GB), then re-run this section:"
    cat <<'EOS'
      export HF_HOME=/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/cache
      export HF_HUB_CACHE=$HF_HOME/hub
      huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \
        ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
        --local-dir "$(dirname "$sg")"
EOS
    return 0
  fi
  local out="$OUTDIR/sharegpt_c32.json"
  [[ -f $out ]] && { warn "skip (exists)"; return 0; }
  local -a cmd=(
    "$VLLM" bench serve
    --backend openai-chat --endpoint /v1/chat/completions
    --host "$HOST" --port "$PORT" --model "$MODEL"
    --dataset-name sharegpt --dataset-path "$sg"
    --num-prompts 500 --max-concurrency 32 --seed 0
    --percentile-metrics ttft,tpot,itl,e2el
    --save-result --result-filename "$out"
    --metadata "hw=$HW" "quant=$QUANT" "kv_cache_dtype=$KV_DTYPE" "dataset=sharegpt"
  )
  if [[ $DRY_RUN == 1 ]]; then printf '    DRY: %s\n' "${cmd[*]}"; return 0; fi
  "${cmd[@]}" > "$OUTDIR/sharegpt_c32.log" 2>&1 \
    || { fail "sharegpt failed"; tail -n 12 "$OUTDIR/sharegpt_c32.log" | sed 's/^/      /'; return 1; }
  jq -r '"    output tok/s: \(.output_throughput|tostring|.[0:9])"' "$out"
  warn "no --ignore-eos here on purpose: ShareGPT comparability wants natural OSL."
}

# Smallest possible real run -- use this first to confirm the harness works.
sweep_quick() {
  head_ "quick check -- ISL 1K, OSL 128, conc 4"
  run_point "quick_isl1k_c4" 1024 128 4
}

# --------------------------------------------------------------------- report
summarize() {
  head_ "summary"
  local n=0
  printf '  %-22s %10s %10s %9s %9s %8s\n' label "out tok/s" "tot tok/s" "TTFTp50" "TPOTp50" "KVpeak"
  for f in "$OUTDIR"/*.json; do
    [[ -f $f ]] || continue
    [[ $(basename "$f") == manifest* ]] && continue
    n=$((n+1))
    jq -r --arg l "$(basename "$f" .json)" '
      "  \($l|.[0:22])\t\(.output_throughput//0)\t\(.total_token_throughput//0)\t\(.median_ttft_ms//0)\t\(.median_tpot_ms//0)\t\(.peak_kv_cache_usage_perc//0)"' "$f" 2>/dev/null \
      | awk -F'\t' '{printf "  %-22s %10.1f %10.1f %9.0f %9.1f %8.3f\n",$1,$2,$3,$4,$5,$6}'
  done
  [[ $n == 0 ]] && warn "no results yet"
  printf '\n  results: %s\n' "$OUTDIR"
  cat <<'EOF'

  Reporting reminders (project conventions):
    - every number above is MEASURED; label it as such next to any analytical
      prediction from plan.md
    - carry hardware + quantization with each number (see manifest.txt)
    - MTP/spec-decode state changes throughput; note whether it was on
    - report achieved bandwidth as a FRACTION of 3.35 TB/s (H100) when you
      claim "memory-bound"
EOF
}

main() {
  printf '\033[1mvLLM sweep\033[0m  %s  model=%s\n' "$BASE" "$MODEL"
  [[ $DRY_RUN == 1 ]] && printf '\033[33mDRY_RUN=1 -- no requests will be sent\033[0m\n'

  preflight || exit 1

  local -a secs=("${@:-}")
  [[ -z ${secs[0]:-} ]] && secs=(batch context)

  local rc=0
  for s in "${secs[@]}"; do
    case $s in
      quick)    sweep_quick    || rc=1 ;;
      batch)    sweep_batch    || rc=1 ;;
      context)  sweep_context  || rc=1 ;;
      prefix)   sweep_prefix   || rc=1 ;;
      sharegpt) sweep_sharegpt || rc=1 ;;
      all)      sweep_batch || rc=1; sweep_context || rc=1; sweep_prefix || rc=1 ;;
      *) fail "unknown section: $s"; rc=1 ;;
    esac
  done

  summarize
  exit $rc
}

main "$@"
