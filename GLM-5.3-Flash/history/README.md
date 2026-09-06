> HISTORICAL MODEL NOTES — use the [current report](../../report.md), [experiment index](../../experiments.md), and [debugging guide](../../fix_bug.md) for corrected conclusions. Original location: `GLM-5.3-Flash/README.md`. Some causal claims and configurations below are superseded.

# GLM-5.3-Flash — served configs and measured results

Per-model directory: the launch configs that actually ran, the raw results, the logs, and the report.
Shared tooling stays at the repo root (`bench.sh`, `sending.sh`, `normalize.sh`).

## Status: measurement COMPLETE (2026-09-02)

**46 measured points across seven arms, all cold (0 new prefix-cache hits) and complete** (+7 quarantined
as evidence). This model is the report's **hybrid-architecture centrepiece**: 34 KDA linear-attention
layers + 11 DSA sparse-attention layers in one stack, which is what makes the heterogeneity thesis
testable rather than asserted. **All four previously-owed measurements landed 2026-09-02** — see
"What session 3 added" below.

| Path | What it is |
|---|---|
| [`report.md`](../report.md) | **The deliverable** — measured throughput, normalization, bottleneck analysis |
| [`run.sh`](../run.sh) | **BF16 KV, base model — produced the HEADLINE numbers** |
| [`run_fp8kv.sh`](../run_fp8kv.sh) | `fp8_ds_mla` attempt — **failed by architecture; that failure is a result** |
| [`run_fp8kv_fi618.sh`](../run_fp8kv_fi618.sh) | FP8 KV via a FlashInfer 0.6.18 overlay — the arm that *worked* |
| [`run_bf16kv_fi618.sh`](../run_bf16kv_fi618.sh) | **The CONTROL** — same overlay, same MoE backend, BF16 KV |
| [`run_mtp.sh`](../run_mtp.sh) | MTP A/B (`NSPEC=1`, `NSPEC=5`) |
| [`_common.sh`](../_common.sh) | Shared flags, `preflight_gpus`, `preflight_caches` |
| [`submit_job.sh`](../submit_job.sh) | RunAI submit with the `--command -- sleep infinity` entrypoint override |
| `results/bf16kv/` | **15 pts — the headline arm**: batch (**8-point curve**), context, prefix |
| `results/fp8kv-fi618/` | 11 pts — FP8 KV, same grid |
| `results/bf16kv-fi618/` | 4 pts — the control that separates dtype from backend |
| `results/bf16kv-mtp-n1,-n5/` | 4 + 4 pts — spec-decode A/B on the batch axis |
| **`results/bf16kv-mtp-n1-context/`** | **4 pts — spec decode vs CONTEXT. Inverts the batch-axis advice** |
| **`results/util085/`** | **4 pts — proves `--gpu-memory-utilization` is capacity-only** |
| `results/fp8kv/` | `RESULT-arm-impossible.md` — the recorded failure, no JSONs |
| `results/_*/` | **Quarantined evidence** (3 dirs, 7 pts), kept deliberately (see below) |

### What session 3 (2026-09-02) added — the four owed measurements

| task | result | writeup |
|---|---|---|
| util A/B 0.82→0.85 | **capacity-only**: +11.2% KV, throughput 0.994–1.000×, peak activation **4.05 GiB in both arms** (the control that makes it clean) | [`results/util085/RESULT-util-ab.md`](../results/util085/RESULT-util-ab.md) |
| 8-point batch curve | **knee at c≈4–8**, not c=64. Matches V4 almost exactly despite a totally different attention stack; **V4's c=48 dip does not reproduce** | [`results/bf16kv/RESULT-8point-batch-curve.md`](../results/bf16kv/RESULT-8point-batch-curve.md) |
| MFU scraping (harness) | **corrected a headline 10×**: "~1% of roofline" → **9.5–18.1%** measured. ⚠️ label **[E]**, and it **excludes attention** for GLM | [`../fix_bug.md`](../../fix_bug.md) bug 13 |
| MTP on the context axis | **inverts the advice**: −4–6% throughput but **−16.7–24.8% TTFT** to 131K. Hypothesis **falsified**; 260K ceiling is **KV-bound**, not accuracy-bound | [`results/bf16kv-mtp-n1-context/RESULT-mtp-context-axis.md`](../results/bf16kv-mtp-n1-context/RESULT-mtp-context-axis.md) |

## The config that ran (headline arm)

```
/usr/bin/python3 /usr/local/bin/vllm serve zai-org/GLM-5.3-Flash --port 8001
  --max-cudagraph-capture-size 256 --block-size 128 --max-num-seqs 256
  --tensor-parallel-size 8 --enable-expert-parallel --gpu-memory-utilization 0.82
  --max-model-len 262144 --limit-mm-per-prompt '{"image":0,"video":0}'
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --no-enable-flashinfer-autotune --enable-mfu-metrics
```

Flags that are **not free choices**:

| Flag | Why it is forced |
|---|---|
| `--block-size 128` | **Mandatory and it is the minimum legal value.** `index_kpool: 4` makes `Glm5NextIndexerCache` (`attention.py:122,140-150`) require **both** `block_size % index_kpool == 0` **and** `(block_size / index_kpool) % 32 == 0` — i.e. a multiple of 128. The default 64 hard-asserts. ⚠️ vLLM then **auto-raises it to 640** so the attention page ≥ the mamba page, and **pads the mamba page by 20.75%**. ⚠️ This is a *different* cause than V4's `--block-size 256` — same-looking flag, unrelated reason. |
| `--max-num-seqs 256` | **Structural.** GLM-5.3 is a **hybrid**: its 34 KDA layers need **one recurrent-state block per decode sequence**, capped at 512. And `--max-num-seqs` does **not** default to 128 on H100 — `arg_utils.py:2547 get_batch_defaults()` gives any non-A100 GPU ≥70 GiB **1024**, which fails at startup. |
| `--max-cudagraph-capture-size 256` | 🛑 **Use this flag, NEVER `--compilation-config '{"max_cudagraph_capture_size":256}'`.** The latter **replaces** the whole `CompilationConfig` and silently wiped `pass_config` to `{}` (losing `fuse_norm_quant`, `fuse_act_quant`, `fuse_allreduce_rms`). The dedicated flag is *merged* (`arg_utils.py:2452`). Caught only by diffing startup banners. |
| `--gpu-memory-utilization 0.82` | Matches the V4-Flash arm so the two are equally provisioned. |
| **no** `--kv-cache-dtype` | Omission is how you ask for BF16 here. `fp8_ds_mla` is **geometrically inapplicable** (needs `pe_dim == 64`; GLM is NoPE, `qk_rope_head_dim: 0`), and the working FP8 path needs FlashInfer ≥ 0.6.18 while the image ships 0.6.17. |
| **no** `--trust-remote-code` | `glm5_next` is **natively registered** (`transformers_utils/config.py:96-98`). Passing it would grant arbitrary code execution for nothing. |
| `--limit-mm-per-prompt '{"image":0,"video":0}'` | **Fairness flag.** Zeroing every modality makes vLLM **skip constructing** the 24-layer ViT (`interfaces.py:307`, 0.56 B params), keeping the comparison text-to-text against text-only V4. Verified in the log. |
| **Port 8001, not 8000** | 8000 belongs to the image entrypoint's Qwen3-0.6B squatter. Hitting 8000 by mistake smoke-tests *that* model **and passes** — the worst kind of failure. |

**TP=8 is more defensible here than for the other two models:** GLM-5.3 needs ≥5 H100s (38.08 GiB/GPU at
TP8), where V4 and Qwen fit in 3. So its per-GPU number is closer to a real deployment, and the per-GPU
column should not be read as if all three were equally handicapped.

## Reproducing

```bash
export HF_HOME=/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/LLMs_serving_report/cache
export HF_HUB_CACHE=$HF_HOME/hub        # 308 GiB already cached; do not re-download

cd GLM-5.3-Flash && ./run.sh            # ~226 s cold / ~24 s warm to /health 200
./sending.sh --wait                     # smoke test, port 8001 (polls; 62 shards load slowly)

# sweeps, from the REPO ROOT -- BOTH client overrides are load-bearing
MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/bf16kv \
QUANT=fp8-attn-dense+fp8-experts KV_DTYPE=bfloat16 \
  ./bench.sh batch context prefix
```

⚠️ **Use the CUDA-13 image `vllm/vllm-openai:glm53-flash`, not `-cu129`.** On cu129 the TileLang
`mhc_post_tilelang` kernel segfaults in `cuModuleLoadData` and the model cannot complete a forward pass.
On cu130: 24 successful compiles, zero segfaults.

⚠️ **`VLLM=`/`PY=` matter for the client too.** The shell profile puts conda `vllm-py12` (0.28.0) first
on PATH, and it has **no `Glm5Next*` entry** — `vllm bench serve` reads the model config to tokenize, so
the sweep dies in preflight **even when the server is perfectly healthy**.

🛑 **Never `pkill -f "vllm serve"`** — PID 1 matches that pattern and killing it restarts the container.
Use `pkill -f "GLM-5.3-Flash"`, then confirm `ps -p 1 -o args=` still says `sleep infinity`.

## Measured startup facts (read from the live server)

| | |
|---|---|
| Weight load | **38.08 GiB/GPU**, all 62 shards (matches the predicted 38.2) |
| KV pool | **2,099,654 tokens**, 23.26 GiB/GPU, **8.01× concurrency at 256K** |
| Resolved block size | **640** (requested 128); `mamba_block_size=128`, mamba page padded 20.75% |
| Expert sharding | `Local/global number of experts: 36/288` at EP8 |
| KV dtype | `auto` → **BF16**; `quantization=fp8` |
| Correctness | ✓ `2+2 = 4` with reasoning parsed into `message.reasoning` |

## Headline numbers (BF16 KV, base model, spec decode off, 8×H100 TP8)

| conc | tok/s | per GPU | per B-active | TTFT p50 | TPOT p50 | KV peak |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 12.04 | 5.54 | 857 ms | 7.1 ms | 1.4% |
| 4 | 225.3 | 28.16 | 12.96 | 1,742 ms | 10.9 ms | 5.1% |
| 16 | 359.2 | 44.90 | 20.67 | 2,685 ms | 33.6 ms | 20.1% |
| 64 | **447.1** | **55.88** | **25.72** | 2,513 ms | 129.5 ms | 80.9% |

**Params [M] from safetensors shapes: 321.34 B total / 17.38 B active.** Routed experts 304.42 B (only
k/E = 8/288 fire) + always-on attn/dense/embed 8.92 B + MTP 7.43 B (excluded) + ViT 0.56 B (excluded).
**97.8% of bytes are native FP8 e4m3** with a 1,509-entry `modules_to_not_convert` list.

> ⚠️ **An earlier *analytical* estimate of "310.96 B / 15.01 B" was wrong** — it under-counted always-on
> attention and mishandled the MTP layer, and it made GLM look **better** per-B-active than it is.
> **Never hand-derive param counts from `config.json`; read tensor shapes.**

## Two results that needed a control arm, and one quarantine

**FP8 KV works on H100** — the vendor recipe says it doesn't. But the arm changed **three** variables at
once (dtype + attention backend + MoE kernel), so `run_bf16kv_fi618.sh` exists to decompose it **[M]**:
**backend −22.0%, dtype −24.9%** (the naive cross-version read would have said −41.4%). Gain:
**1.805× KV capacity**. ⚠️ Not a production config — needs
`FLASHINFER_DISABLE_VERSION_CHECK=1` and `moe_backend=deep_gemm`.

**`--num-warmups` was REMOVED from `bench.sh` — it self-poisons the prefix cache.** It drew warmup
prompts from the **same seeded set** as the measured run, so every batch point reported exactly 16,000
new hits. The bias was **uneven** (12.2% of the c=1 point, 0.8% of c=64), so it inflated low-concurrency
anchors and **flattened measured concurrency scaling**. Contaminated points are kept in
`results/_discarded-warmup-contaminated/` as evidence, not deleted.

See [`report.md`](../report.md) for the full analysis and [`../fix_bug.md`](../../fix_bug.md) for the six
bring-up bugs — four of which named the wrong subsystem.
