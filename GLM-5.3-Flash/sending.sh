#!/usr/bin/env bash
# Quick check: is GLM-5.3-Flash actually hosted and answering?
#
#   ./sending.sh              # all checks
#   ./sending.sh health       # one section (health|models|chat|stream|tools|metrics)
#   ./sending.sh --wait       # poll until the server is up, then run all checks
#
# This is a LIVENESS check, not a benchmark. The timings it prints are single-client
# curl latencies -- do NOT quote them as throughput. Use ../bench.sh for numbers.
#
# Startup is SLOW: 305.8 GiB of weights across 62 shards. DeepSeek-V4-Flash (148.6 GiB)
# took ~255 s to /health 200, so budget several minutes here. `--wait` handles it.

set -uo pipefail

HOST=${HOST:-localhost}
# 8001, not 8000: the image ENTRYPOINT is itself a `vllm serve` (PID 1) holding
# port 8000 with the default Qwen/Qwen3-0.6B. Our arms launch on 8001 -- see
# _common.sh. Hitting 8000 by mistake smoke-tests the WRONG MODEL and passes.
PORT=${PORT:-8001}
BASE="http://${HOST}:${PORT}"
MODEL=${MODEL:-zai-org/GLM-5.3-Flash}

# GLM-5.3 is a reasoning model: it spends tokens in `reasoning` before any `content`.
# Too small a budget truncates mid-thought and you get content:null + finish_reason
# "length" -- which looks like a failure but is not one.
MAX_TOKENS=${MAX_TOKENS:-600}
TIMEOUT=${TIMEOUT:-600}

pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
warn() { printf '  \033[33mnote\033[0m %s\n' "$1"; }
head_() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

post() {
  curl -s --max-time "$TIMEOUT" -w '\n%{http_code} %{time_total}' \
       -H 'Content-Type: application/json' -d "$2" "${BASE}$1"
}
split_resp() {
  CODE=$(printf '%s' "$1" | tail -n1 | awk '{print $1}')
  SECS=$(printf '%s' "$1" | tail -n1 | awk '{print $2}')
  BODY=$(printf '%s' "$1" | sed '$d')
}

# Poll until healthy. Weight loading for a 306 GiB model takes minutes.
wait_ready() {
  local max=${WAIT_SECS:-1800} i=0
  head_ "waiting for server (up to ${max}s)"
  while [[ $i -lt $max ]]; do
    if [[ $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${BASE}/health" 2>/dev/null) == 200 ]]; then
      pass "server healthy after ~${i}s"; return 0
    fi
    # Surface loading progress so a long wait isn't a black box.
    if (( i % 60 == 0 )); then
      local mem
      mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      printf '  ...%4ds  GPU0 %s MiB\n' "$i" "${mem:-?}"
    fi
    sleep 10; i=$((i+10))
  done
  fail "not healthy after ${max}s"; return 1
}

check_health() {
  head_ "health"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "${BASE}/health")
  if [[ $code == 200 ]]; then pass "GET /health -> 200"; return 0; fi
  fail "GET /health -> ${code:-no response}"
  # Distinguish "still loading" from "died", because they look identical from here.
  # Match on the model name only: the image's launcher shows up as
  # "/usr/bin/python3 /usr/local/bin/vllm serve zai-org/GLM-5.3-Flash ...", and its
  # engine child as a bare "VLLM::EngineCore", so a "vllm serve.*GLM" pattern alone
  # would miss the child.
  if pgrep -f "GLM-5.3-Flash" >/dev/null 2>&1; then
    warn "a GLM-5.3 process IS running -- almost certainly still loading weights"
    warn "305.8 GiB across 62 shards; budget several minutes"
    warn "re-run with:  ./sending.sh --wait"
  else
    warn "no GLM-5.3 process found -- server is not running or it crashed"
    warn "check logs/serve_*.log; note port 8000 is the image entrypoint's"
    warn "Qwen/Qwen3-0.6B, NOT our run -- our arms serve on 8001"
  fi
  return 1
}

check_models() {
  head_ "models"
  local body id len
  body=$(curl -s --max-time 10 "${BASE}/v1/models")
  id=$(printf '%s' "$body" | jq -r '.data[0].id // empty' 2>/dev/null)
  len=$(printf '%s' "$body" | jq -r '.data[0].max_model_len // empty' 2>/dev/null)
  if [[ -z $id ]]; then fail "could not parse /v1/models"; return 1; fi
  pass "served model: $id"
  pass "max_model_len: $len"
  [[ $id == "$MODEL" ]] || warn "expected $MODEL; using the server's id instead"
  MODEL=$id
}

check_chat() {
  head_ "chat/completions"
  local req resp
  req=$(jq -nc --arg m "$MODEL" --argjson mt "$MAX_TOKENS" '{
    model:$m, max_tokens:$mt, temperature:0,
    messages:[{role:"user",content:"What is 2+2? Answer briefly."}]
  }')
  resp=$(post /v1/chat/completions "$req"); split_resp "$resp"
  if [[ $CODE != 200 ]]; then
    fail "HTTP $CODE"; printf '%s\n' "$BODY" | head -c 600; return 1
  fi
  pass "HTTP 200 in ${SECS}s"
  printf '%s' "$BODY" | jq -r '
    .choices[0] as $c |
    "  finish_reason : \($c.finish_reason)",
    "  prompt/completion tokens: \(.usage.prompt_tokens) / \(.usage.completion_tokens)",
    "  reasoning     : \(($c.message.reasoning // "<none>")|tostring|.[0:110])",
    "  content       : \(($c.message.content   // "<null>")|tostring|.[0:110])"' 2>/dev/null

  local content finish
  content=$(printf '%s' "$BODY" | jq -r '.choices[0].message.content // "null"' 2>/dev/null)
  finish=$(printf '%s' "$BODY" | jq -r '.choices[0].finish_reason' 2>/dev/null)
  if [[ $content == null && $finish == length ]]; then
    warn "content:null + finish_reason:length = truncated mid-reasoning, not a failure."
    warn "raise MAX_TOKENS (currently $MAX_TOKENS)."
  fi
}

check_stream() {
  head_ "chat/completions (streaming)"
  local req out n
  req=$(jq -nc --arg m "$MODEL" --argjson mt "$MAX_TOKENS" '{
    model:$m, max_tokens:$mt, temperature:0, stream:true,
    messages:[{role:"user",content:"Name three primary colors."}]
  }')
  out=$(curl -s -N --max-time "$TIMEOUT" -H 'Content-Type: application/json' \
             -d "$req" "${BASE}/v1/chat/completions")
  n=$(printf '%s\n' "$out" | grep -c '^data: ' || true)
  if [[ ${n:-0} -lt 2 ]]; then
    fail "expected SSE chunks, got:"; printf '%s\n' "$out" | head -c 400; return 1
  fi
  pass "$n SSE chunks"
  printf '%s\n' "$out" | grep -q 'data: \[DONE\]' && pass "terminates with [DONE]" \
                                                  || warn "no [DONE] sentinel"
}

# run.sh passes --tool-call-parser glm47 --enable-auto-tool-choice.
check_tools() {
  head_ "tool calling (parser: glm47)"
  local req resp n
  req=$(jq -nc --arg m "$MODEL" --argjson mt "$MAX_TOKENS" '{
    model:$m, max_tokens:$mt, temperature:0, tool_choice:"auto",
    messages:[{role:"user",content:"What is the weather in Seattle right now?"}],
    tools:[{type:"function",function:{
      name:"get_weather", description:"Get the current weather for a city",
      parameters:{type:"object",properties:{city:{type:"string"}},required:["city"]}}}]
  }')
  resp=$(post /v1/chat/completions "$req"); split_resp "$resp"
  if [[ $CODE != 200 ]]; then fail "HTTP $CODE"; printf '%s\n' "$BODY" | head -c 500; return 1; fi
  pass "HTTP 200 in ${SECS}s"
  n=$(printf '%s' "$BODY" | jq -r '(.choices[0].message.tool_calls // [])|length' 2>/dev/null)
  if [[ ${n:-0} -gt 0 ]]; then
    pass "emitted $n tool_call(s)"
    printf '%s' "$BODY" | jq -r '.choices[0].message.tool_calls[]
      | "  -> \(.function.name)(\(.function.arguments))"' 2>/dev/null
  else
    warn "no tool_calls; model answered in prose. tool_choice:auto lets it decline,"
    warn "so this is not necessarily a parser failure."
  fi
}

# Config facts worth confirming against the LIVE server rather than the config file.
check_metrics() {
  head_ "server config + metrics"
  local m cc
  m=$(curl -s --max-time 10 "${BASE}/metrics")
  [[ -z $m ]] && { fail "no /metrics"; return 1; }
  pass "GET /metrics"

  cc=$(printf '%s' "$m" | grep -m1 'cache_config_info' | tr ',' '\n')
  printf '%s\n' "$cc" | grep -E '^block_size=|^num_gpu_blocks=|^cache_dtype=|^gpu_memory_utilization=' \
    | sed 's/^/  /'
  warn "block_size MUST be 128 (or a multiple): index_kpool=4 requires"
  warn "block_size % (index_kpool*32) == 0 -- attention.py:140-150. The default 64"
  warn "hard-asserts. This is a DIFFERENT cause than V4's --block-size 256"
  warn "(sparse_mla.py:53, per-layer compress_ratios) -- same flag, unrelated reason."
  warn "cache_dtype: expect bfloat16 for the headline arm. FP8 KV is gated OFF on"
  warn "Hopper here (FlashInfer 0.6.17 < 0.6.18, no ckv_scale_arr)."

  # MTP: run.sh omits --speculative-config, run_mtp.sh adds it. Confirm which is live.
  local d a
  d=$(printf '%s\n' "$m" | awk '/^vllm:spec_decode_num_draft_tokens_total/{v=$2} END{print v+0}')
  a=$(printf '%s\n' "$m" | awk '/^vllm:spec_decode_num_accepted_tokens_total/{v=$2} END{print v+0}')
  if [[ ${d%.*} -gt 0 ]]; then
    printf '  MTP draft/accepted: %s / %s' "$d" "$a"
    printf '  (%.1f%%)\n' "$(echo "100*$a/$d" | bc -l 2>/dev/null || echo 0)"
    warn "spec decode is ACTIVE -- this is the A/B arm (run_mtp.sh), NOT headline numbers."
  else
    pass "spec decode OFF (base model) -- correct for headline throughput"
  fi
  printf '%s\n' "$m" | awk '
    /^vllm:num_requests_running/    {printf "  running          : %s\n", $2}
    /^vllm:prompt_tokens_total/     {printf "  prompt tokens    : %s\n", $2}
    /^vllm:generation_tokens_total/ {printf "  generated tokens : %s\n", $2}
    /^vllm:kv_cache_usage_perc/     {printf "  KV cache usage   : %s\n", $2}'
}

main() {
  local -a secs=()
  for a in "$@"; do
    case $a in
      --wait) DO_WAIT=1 ;;
      *) secs+=("$a") ;;
    esac
  done

  printf '\033[1mGLM-5.3-Flash smoke test\033[0m  %s\n' "$BASE"
  for c in curl jq; do command -v $c >/dev/null || { echo "missing: $c"; exit 1; }; done

  [[ ${DO_WAIT:-0} == 1 ]] && { wait_ready || exit 1; }
  [[ ${#secs[@]} -eq 0 ]] && secs=(health models chat stream tools metrics)

  local rc=0
  for s in "${secs[@]}"; do
    case $s in
      health)  check_health  || { rc=1; break; } ;;   # nothing else can pass if down
      models)  check_models  || rc=1 ;;
      chat)    check_chat    || rc=1 ;;
      stream)  check_stream  || rc=1 ;;
      tools)   check_tools   || rc=1 ;;
      metrics) check_metrics || rc=1 ;;
      *) fail "unknown section: $s"; rc=1 ;;
    esac
  done

  printf '\n'
  if [[ $rc == 0 ]]; then
    printf '\033[32mmodel is hosted and answering\033[0m\n'
    printf 'For throughput numbers run from the repo root:  ./bench.sh batch context prefix\n'
  else
    printf '\033[31msome checks failed\033[0m\n'
  fi
  exit $rc
}

main "$@"
