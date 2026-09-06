docker run --gpus all \
  --privileged --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:qwen38-flash-next Qwen/Qwen3.8-Flash-Next-FP8 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --tensor-parallel-size 8 \
  --moe-backend triton \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3

# run tensor-parallel-size = 8 first for fair compare, and TP=4, expert parallel 2 as well - consistent evaluation to deepseek and glm