#!/usr/bin/env bash
# Smoke-test the local vLLM server hosting DeepSeek-V4-Flash.
#
# Usage:
#   ./sending.sh            # run every check
#   ./sending.sh health     # single section (health|models|chat|stream|raw|tools|batch|metrics)
#
# This is a FUNCTIONAL smoke test, not a benchmark. The timings it prints are
# wall-clock curl latencies from one client on a warm server -- do NOT quote
# them as measured throughput. For real numbers use `vllm bench serve`
# (see the note at the bottom of this file).

set -uo pipefail

HOST=${HOST:-localhost}
PORT=${PORT:-8000}
BASE="http://${HOST}:${PORT}"
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Flash}

# DeepSeek-V4 is a reasoning model: it spends tokens in `reasoning` before it
# emits any `content`. Budget generously -- a small max_tokens truncates during
# reasoning and you get content:null with finish_reason:"length".
MAX_TOKENS=${MAX_TOKENS:-600}

pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }
head_() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# POST $1=path $2=json -> body on stdout, "%{http_code} %{time_total}" on fd 3
post() {
  curl -s --max-time "${TIMEOUT:-600}" -w '\n%{http_code} %{time_total}' \
       -H 'Content-Type: application/json' -d "$2" "${BASE}$1"
}

# Split the trailing "code time" line off a post() response.
# Sets: BODY, CODE, SECS
split_resp() {
  local raw=$1
  CODE=$(printf '%s' "$raw" | tail -n1 | awk '{print $1}')
  SECS=$(printf '%s' "$raw" | tail -n1 | awk '{print $2}')
  BODY=$(printf '%s' "$raw" | sed '$d')
}

check_health() {
  head_ "health"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "${BASE}/health")
  if [[ $code == 200 ]]; then pass "GET /health -> 200"; else
    fail "GET /health -> ${code:-no response}"
    echo "     server not reachable at ${BASE} -- is run.sh still loading weights?"
    return 1
  fi
}

check_models() {
  head_ "models"
  local body
  body=$(curl -s --max-time 10 "${BASE}/v1/models")
  local id len
  id=$(printf '%s' "$body" | jq -r '.data[0].id // empty')
  len=$(printf '%s' "$body" | jq -r '.data[0].max_model_len // empty')
  if [[ -n $id ]]; then
    pass "served model: $id"
    pass "max_model_len: $len"
    [[ $len -gt 200000 ]] && \
      printf '  \033[33mnote\033[0m max_model_len=%s is huge; KV cache is sized for it.\n' "$len"
  else
    fail "could not parse /v1/models"; return 1
  fi
}

# Non-streaming chat. This is the endpoint you will use most.
check_chat() {
  head_ "chat/completions (non-streaming)"
  local req resp
  req=$(jq -nc --arg m "$MODEL" --argjson mt "$MAX_TOKENS" '{
    model:$m, max_tokens:$mt, temperature:0,
    messages:[{role:"user",content:"What is 2+2? Answer briefly."}]
  }')
  resp=$(post /v1/chat/completions "$req"); split_resp "$resp"
  if [[ $CODE != 200 ]]; then
    fail "HTTP $CODE"; printf '%s\n' "$BODY" | head -c 500; return 1
  fi
  pass "HTTP 200 in ${SECS}s"
  printf '%s' "$BODY" | jq -r '
    .choices[0] as $c |
    "  finish_reason : \($c.finish_reason)",
    "  prompt_tokens : \(.usage.prompt_tokens)",
    "  completion    : \(.usage.completion_tokens) (reasoning: \(.usage.completion_tokens_details.reasoning_tokens // 0))",
    "  reasoning     : \(($c.message.reasoning // "<none>") | tostring | .[0:110])",
    "  content       : \(($c.message.content  // "<null>")  | tostring | .[0:110])"'

  # The trap worth knowing about, asserted rather than described.
  local content finish
  content=$(printf '%s' "$BODY" | jq -r '.choices[0].message.content // "null"')
  finish=$(printf '%s' "$BODY"  | jq -r '.choices[0].finish_reason')
  if [[ $content == null && $finish == length ]]; then
    printf '  \033[33mnote\033[0m content:null + finish_reason:length = truncated mid-reasoning.\n'
    printf '        Raise MAX_TOKENS (currently %s).\n' "$MAX_TOKENS"
  fi
}

# Streaming: reasoning arrives as delta.reasoning, answer as delta.content.
check_stream() {
  head_ "chat/completions (streaming SSE)"
  local req out t0 t1
  req=$(jq -nc --arg m "$MODEL" --argjson mt "$MAX_TOKENS" '{
    model:$m, max_tokens:$mt, temperature:0, stream:true,
    stream_options:{include_usage:true},
    messages:[{role:"user",content:"Name three primary colors."}]
  }')
  t0=$(date +%s.%N)
  out=$(curl -s -N --max-time "${TIMEOUT:-600}" -H 'Content-Type: application/json' \
             -d "$req" "${BASE}/v1/chat/completions")
  t1=$(date +%s.%N)
  local n
  n=$(printf '%s\n' "$out" | grep -c '^data: ' || true)
  if [[ ${n:-0} -lt 2 ]]; then fail "expected SSE chunks, got:"; printf '%s\n' "$out" | head -c 400; return 1; fi
  pass "$n SSE chunks in $(printf '%.2f' "$(echo "$t1 - $t0" | bc)")s"
  pass "terminates with [DONE]: $(printf '%s\n' "$out" | grep -q 'data: \[DONE\]' && echo yes || echo no)"
  # Reassemble both channels from the deltas.
  printf '%s\n' "$out" | sed -n 's/^data: //p' | grep -v '^\[DONE\]$' \
    | jq -sr '[.[]|.choices[0].delta.reasoning // empty]|add // "<none>"
              | "  reasoning : \(tostring | .[0:110])"'
  printf '%s\n' "$out" | sed -n 's/^data: //p' | grep -v '^\[DONE\]$' \
    | jq -sr '[.[]|.choices[0].delta.content // empty]|add // "<none>"
              | "  content   : \(tostring | .[0:110])"'
}

# Raw completions: no chat template, no reasoning parser. Base-model behavior.
check_raw() {
  head_ "completions (raw, no chat template)"
  local req resp
  req=$(jq -nc --arg m "$MODEL" '{
    model:$m, prompt:"The capital of France is", max_tokens:16, temperature:0
  }')
  resp=$(post /v1/completions "$req"); split_resp "$resp"
  if [[ $CODE != 200 ]]; then fail "HTTP $CODE"; printf '%s\n' "$BODY" | head -c 400; return 1; fi
  pass "HTTP 200 in ${SECS}s"
  printf '%s' "$BODY" | jq -r '"  text: \(.choices[0].text | tostring | .[0:140])"'
  printf '  \033[33mnote\033[0m /v1/completions bypasses the chat template, so this is a\n'
  printf '        pretrained-continuation, not an answer. Expect document-ish drift.\n'
}

# run.sh passes --tool-call-parser deepseek_v4 --enable-auto-tool-choice.
check_tools() {
  head_ "tool calling"
  local req resp
  req=$(jq -nc --arg m "$MODEL" --argjson mt "$MAX_TOKENS" '{
    model:$m, max_tokens:$mt, temperature:0, tool_choice:"auto",
    messages:[{role:"user",content:"What is the weather in Seattle right now?"}],
    tools:[{type:"function",function:{
      name:"get_weather",
      description:"Get the current weather for a city",
      parameters:{type:"object",
        properties:{city:{type:"string",description:"City name"}},
        required:["city"]}}}]
  }')
  resp=$(post /v1/chat/completions "$req"); split_resp "$resp"
  if [[ $CODE != 200 ]]; then fail "HTTP $CODE"; printf '%s\n' "$BODY" | head -c 500; return 1; fi
  pass "HTTP 200 in ${SECS}s"
  local ncalls
  ncalls=$(printf '%s' "$BODY" | jq -r '(.choices[0].message.tool_calls // []) | length')
  if [[ ${ncalls:-0} -gt 0 ]]; then
    pass "model emitted $ncalls tool_call(s)"
    printf '%s' "$BODY" | jq -r '.choices[0].message.tool_calls[]
      | "  -> \(.function.name)(\(.function.arguments))"'
  else
    printf '  \033[33mnote\033[0m no tool_calls; model answered in prose instead.\n'
    printf '        Not necessarily a bug -- tool_choice:"auto" lets it decline.\n'
    printf '%s' "$BODY" | jq -r '"  content: \((.choices[0].message.content // "<null>")|tostring|.[0:110])"'
  fi
}

# Concurrency: the only section that touches batching at all.
check_batch() {
  local n=${CONCURRENCY:-8}
  head_ "concurrent requests (n=$n)"
  local tmp t0 t1
  tmp=$(mktemp -d)
  t0=$(date +%s.%N)
  for i in $(seq 1 "$n"); do
    (
      req=$(jq -nc --arg m "$MODEL" --argjson mt "$MAX_TOKENS" --argjson i "$i" '{
        model:$m, max_tokens:$mt, temperature:0,
        messages:[{role:"user",content:"In one sentence, what is the number \($i) famous for?"}]
      }')
      curl -s --max-time "${TIMEOUT:-600}" -o "$tmp/$i.json" \
           -w '%{http_code} %{time_total}\n' \
           -H 'Content-Type: application/json' -d "$req" \
           "${BASE}/v1/chat/completions" > "$tmp/$i.meta"
    ) &
  done
  wait
  t1=$(date +%s.%N)

  local ok=0 toks=0
  for i in $(seq 1 "$n"); do
    [[ $(awk '{print $1}' "$tmp/$i.meta" 2>/dev/null) == 200 ]] && ok=$((ok+1))
    local c; c=$(jq -r '.usage.completion_tokens // 0' "$tmp/$i.json" 2>/dev/null || echo 0)
    toks=$((toks + c))
  done
  local wall; wall=$(echo "$t1 - $t0" | bc)
  [[ $ok == "$n" ]] && pass "$ok/$n returned 200" || fail "$ok/$n returned 200"
  printf '  wall clock       : %.2fs\n' "$wall"
  printf '  completion tokens: %s\n' "$toks"
  printf '  aggregate        : %.1f tok/s\n' "$(echo "$toks / $wall" | bc -l)"
  printf '  per-request latency (s): %s\n' "$(cat "$tmp"/*.meta | awk '{printf "%s ", $2}')"
  printf '  \033[33mnote\033[0m %s concurrent requests is far below this model'"'"'s saturation\n' "$n"
  printf '        batch. Treat the tok/s above as a liveness signal only.\n'
  rm -rf "$tmp"
}

# Server-side counters. These ARE trustworthy, unlike the curl timings.
check_metrics() {
  head_ "server metrics"
  local m
  m=$(curl -s --max-time 10 "${BASE}/metrics")
  if [[ -z $m ]]; then fail "no /metrics output"; return 1; fi
  pass "GET /metrics"

  # MTP is on via --speculative-config; acceptance rate is the number that
  # tells you whether it is actually paying for itself.
  local d a
  d=$(printf '%s\n' "$m" | awk '/^vllm:spec_decode_num_draft_tokens_total/{print $2}' | tail -1)
  a=$(printf '%s\n' "$m" | awk '/^vllm:spec_decode_num_accepted_tokens_total/{print $2}' | tail -1)
  if [[ -n ${d:-} && ${d%.*} -gt 0 ]]; then
    printf '  MTP draft tokens   : %s\n' "$d"
    printf '  MTP accepted       : %s\n' "$a"
    printf '  MTP acceptance     : %.1f%%\n' "$(echo "100 * $a / $d" | bc -l)"
  fi
  printf '%s\n' "$m" | awk '
    /^vllm:num_requests_running/       {printf "  running            : %s\n", $2}
    /^vllm:num_requests_waiting\{/     {printf "  waiting            : %s\n", $2}
    /^vllm:prompt_tokens_total/        {printf "  prompt tokens      : %s\n", $2}
    /^vllm:generation_tokens_total/    {printf "  generated tokens   : %s\n", $2}'
  # KV cache usage tells you whether 0.82 left you a sane cache.
  printf '%s\n' "$m" | awk '/^vllm:kv_cache_usage_perc|^vllm:gpu_cache_usage_perc/ \
    {printf "  KV cache usage     : %s\n", $2}'
}

main() {
  printf '\033[1mvLLM smoke test\033[0m  %s  model=%s\n' "$BASE" "$MODEL"
  for cmd in curl jq bc; do
    command -v "$cmd" >/dev/null || { echo "missing required tool: $cmd"; exit 1; }
  done

  local sections=("${@:-}")
  [[ -z ${sections[0]:-} ]] && sections=(health models chat stream raw tools batch metrics)

  local rc=0
  for s in "${sections[@]}"; do
    case $s in
      health)  check_health  || rc=1 ;;
      models)  check_models  || rc=1 ;;
      chat)    check_chat    || rc=1 ;;
      stream)  check_stream  || rc=1 ;;
      raw)     check_raw     || rc=1 ;;
      tools)   check_tools   || rc=1 ;;
      batch)   check_batch   || rc=1 ;;
      metrics) check_metrics || rc=1 ;;
      *) echo "unknown section: $s"; rc=1 ;;
    esac
  done

  printf '\n'
  [[ $rc == 0 ]] && printf '\033[32mall checks passed\033[0m\n' \
                 || printf '\033[31msome checks failed\033[0m\n'

  cat <<'EOF'

For actual throughput numbers, do not use this script -- use vLLM's harness,
which drives real concurrency and reports TTFT/TPOT percentiles:

  vllm bench serve --backend openai-chat \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --endpoint /v1/chat/completions \
    --dataset-name random --num-prompts 200 --max-concurrency 32 \
    --random-input-len 1024 --random-output-len 256 \
    --save-result --result-filename bench_v4flash.json

Per this project's conventions, record alongside any number you keep:
hardware (8x H100 80GB HBM3), quantization (FP8 attn/dense + MXFP4 experts,
--kv-cache-dtype fp8), and that MTP spec-decode was ON.
EOF
  exit $rc
}

main "$@"
