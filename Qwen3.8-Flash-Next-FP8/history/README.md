> HISTORICAL MODEL NOTES — use the [current report](../../report.md), [experiment index](../../experiments.md), and [debugging guide](../../fix_bug.md) for corrected conclusions. Original location: `Qwen3.8-Flash-Next-FP8/README.md`. Some causal claims and configurations below are superseded.

# Qwen3.8-Flash-Next-FP8 — served configs and measured results

Per-model directory: the launch configs that actually ran, the raw results they produced, the logs, and
the report. Shared tooling stays at the repo root (`bench.sh`, `sending.sh`, `normalize.sh`).

## Status: base arm MEASURED (2026-09-02)

**11 base-arm points, all cold and complete**, on 8×H100 at TP8/EP. This is the report's **third MoE
data point and its most extreme sparsity point** (E/k = 51.2×).

> ⚠️ **This model is NOT the dense control the plan expected.** `CLAUDE.md`'s architecture table listed
> "Qwen3.8-27B, 64 layers, dense" and designated it the mandatory dense baseline. **That is a different
> model.** This checkpoint reads `n_routed_experts: None` only because Qwen uses a different config key —
> the actual values are `num_experts: 512`, `num_experts_per_tok: 10`, `moe_intermediate_size: 640`,
> verified from **150,528 `.experts.` tensors and zero plain `mlp.{gate,up,down}_proj`**. It is a
> fine-grained MoE. **This report still has no dense baseline; that is stated as a gap, not papered over.**

## Contents

| Path | What it is |
|---|---|
| [`report.md`](../report.md) | **The deliverable** — measured throughput, normalization, bottleneck analysis |
| [`run.sh`](../run.sh) | **Base model, spec decode OFF — produced the PRIMARY numbers** |
| [`run_mtp.sh`](../run_mtp.sh) | Same + `--speculative-config` — the MTP A/B arm only |
| [`_common.sh`](../_common.sh) | Shared flags, `preflight_gpus`, `preflight_caches` |
| [`submit_job.sh`](../submit_job.sh) | The RunAI submit (image + `--command -- sleep infinity`) |
| `results/base/` | 11 points: batch, context, prefix → report §3–§5 |
| `results/base-util082/` | 4 pts — the `--gpu-memory-utilization` A/B; **supersedes `base/` at c=4 and c=16** |
| `results/_base_c1_jitcold/` | **Quarantined, kept as evidence** — the c=1 point from the cold engine, superseded (see below) |
| `logs/` | Server startup logs; source for block-size, KV-pool and backend facts |

## The config that ran

```
/usr/local/bin/vllm serve Qwen/Qwen3.8-Flash-Next-FP8 --port 8001
  --tensor-parallel-size 8 --enable-expert-parallel --moe-backend triton
  --gpu-memory-utilization 0.85 --max-num-seqs 256 --max-model-len 262144
  --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}'
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
  --no-enable-flashinfer-autotune --enable-mfu-metrics
```

Flags that are **not free choices**:

| Flag | Why it is forced |
|---|---|
| `--enable-expert-parallel` | **Mandatory**, not a throughput knob. The recipe states plain TP8 is incompatible with this FP8 checkpoint: its 128-wide quantization blocks would be split mid-block across ranks. At TP8 this gives `Local/global experts 64/512` **[M]**. |
| `--moe-backend triton` | The recipe's Hopper choice. `auto` may pick a Blackwell-oriented kernel; pinning also makes runs reproducible, since `auto` is free to change between builds. Verified resolved: `Using TRITON Fp8 MoE backend` **[M]**. |
| `--max-num-seqs 256` | **Structural, not a scheduler preference.** 36 of 48 layers are Gated-DeltaNet linear-attention layers holding a constant per-sequence recurrent state (~0.1055 GiB/seq over the stack, TP-sharded). The H100 auto-default is **1024** (`get_batch_defaults()` gives any non-A100 GPU ≥70 GiB 1024 — *not* the documented 128), which trips a Mamba-cache capacity failure at startup. |
| `--gpu-memory-utilization 0.85` | Recipe-sanctioned for TP8 (0.90 at TP4). ⚠️ **Differs from the 0.82 used for V4-Flash and GLM-5.3** — it changes KV capacity, not the decode cost model. Recorded, not silently equated. |
| `--limit-mm-per-prompt '{"image":0,"video":0}'` | **Fairness flag.** This checkpoint is multimodal (27-layer ViT, 0.449 B params **[M]**). Zeroing every modality makes vLLM skip constructing the tower, keeping the comparison text-to-text against text-only V4. Verified: *"All limits of multimodal modalities … set to 0, running in text-only mode."* |
| **no** `--kv-cache-dtype` | **FP8 KV is unavailable for this model on any GPU.** Every QSA backend declares `supported_kv_cache_dtypes = ["auto","bfloat16"]` and the impl raises `NotImplementedError` otherwise; the indexer additionally requires BF16 model dtype. `auto` → BF16 is the only option. **This makes Qwen3.8 ↔ GLM-5.3 the one KV-dtype-matched pair in the report.** |
| **no** `--block-size` | Deliberately unpinned, unlike V4 (256) and GLM-5.3 (128). vLLM must satisfy a FullAttentionSpec, an MLAAttentionSpec (compressed indexer keys) and two MambaSpecs (GDN + PLE conv) at once and resolves the LCM itself. **Resolved: `block_size=4`, `mamba_block_size=16` [M]** — read out of the live server, not assumed. |
| **no** `--trust-remote-code` | `qwen4_exp` is natively registered in this build. The flag would grant arbitrary code execution for nothing. |

**TP=8 was chosen to match the GLM-5.3 and V4-Flash layout**, not because the model needs it: weights are
172.76 GiB, so ~3 H100s would hold them. Per-GPU throughput here is therefore **pessimistic** in the same
way V4's is — the same caveat, stated for the same reason.

## ⚠️ The engine confound — read before citing any cross-model ratio

This model **cannot run in the image the other two were measured in**: `model_type: qwen4_exp` is
unregistered there. It requires `vllm/vllm-openai:qwen38-flash-next`.

| model | image | vLLM build |
|---|---|---|
| GLM-5.3-Flash, DeepSeek-V4-Flash | `glm53-flash` | `0.1.dev20051+g487ecf187` |
| **Qwen3.8-Flash-Next-FP8** | **`qwen38-flash-next`** | **`0.1.dev20073+g8e685d198`** |

Both are CUDA 13.0 / torch 2.13.0+cu130, so the CUDA runtime is matched — but the vLLM build is not.
**A backend swap alone was measured to cost 22% on GLM**, so an engine delta can dwarf an architecture
effect. Mitigation available and recommended: **this image registers `DeepseekV4ForCausalLM`**, so
V4-Flash can be re-swept here as a **bridge arm** to measure the `dev20051 → dev20073` effect directly.
Until that arm exists, quote Qwen3.8-vs-GLM ratios **only** with the confound stated. See `report.md` §2.

## One point was quarantined and re-run

`results/_base_c1_jitcold/batch_isl16k_c1.json` is the **first** c=1 measurement. `bench.sh`'s queueing
guard flagged it (p99/median TTFT = **7.81×**, limit 4×). Diagnosis: at concurrency 1 requests are
served strictly sequentially, so **nothing can queue** — the mean (1,264 ms) sitting between median (602)
and p99 (4,702) shows exactly one slow request, the first, paying JIT warmup.

Re-run on the warm engine: **107.7 tok/s, p99/median = 1.07×**, with an **identical 602 ms median TTFT**.
So the guard fired on a real artifact with a wrong label, and the published number is the warm one:
**84.2 → 107.7 tok/s, a 28% understatement** had the flagged point been cited.

## Reproducing

```bash
export HF_HOME=/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/LLMs_serving_report/cache
export HF_HUB_CACHE=$HF_HOME/hub        # 174 GiB already cached; do not re-download

# JIT caches node-local, keyed by CUDA major AND build (this image is a different
# vLLM dev build than the GLM one -- do not let them share compiled artifacts)
TAG="$(id -u)_$(cut -d. -f1 <<<"$CUDA_VERSION")_dev20073"
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$TAG TILELANG_CACHE_DIR=/tmp/tilelang_cache_$TAG

cd Qwen3.8-Flash-Next-FP8 && GPUS=0,1,2,3,4,5,6,7 TP=8 ./run.sh   # ~5 min cold to /health 200

# sweeps, from the REPO ROOT -- all five overrides are load-bearing
MODEL=Qwen/Qwen3.8-Flash-Next-FP8 PORT=8001 \
VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
MODELDIR=Qwen3.8-Flash-Next-FP8 OUTDIR=Qwen3.8-Flash-Next-FP8/results/base \
QUANT=fp8-attn-dense+fp8-experts KV_DTYPE=bfloat16 \
  ./bench.sh batch context prefix
```

`VLLM=`/`PY=` matter for the **client** too: `vllm bench serve` tokenizes prompts and reads
`max_model_len` from the config, so the conda vLLM 0.28.0 client fails in preflight on
`model_type: qwen4_exp` **even when the server is perfectly healthy**.

🛑 **Never `pkill -f "vllm serve"` in these images — PID 1 matches that pattern** and killing it
restarts the container. Match on the model name: `pkill -f "Qwen3.8-Flash-Next-FP8"`. Verify after:
`ps -p 1 -o args=` should still print `sleep infinity`.

## Measured startup facts (read from the live server, not assumed)

| | |
|---|---|
| Weight load | **23.41 GiB/GPU**, 135 s (131 shards, warm page cache) |
| KV pool | **2,048,645 tokens**, 25.1 GiB/GPU, **7.81× concurrency at 262,144** |
| Resolved block size | **4** (unpinned; `mamba_block_size=16`) |
| Expert sharding | `Local/global number of experts: 64/512` |
| MoE backend | TRITON (pinned) |
| KV dtype | `auto` → **BF16** |
| Engine init | 132.05 s (compilation 60.41 s) |

## Headline numbers (base model, spec decode off, 8×H100, TP8)

| conc | tok/s | per GPU | per B-active | TTFT p50 | TPOT p50 | KV peak |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 106.2 | 13.28 | 14.61 | 598 ms | 7.0 ms | 0.6% |
| 4 | **256.8** | 32.10 | 35.32 | 1,496 ms | 9.6 ms | 2.3% |
| 16 | **420.2** | 52.52 | 57.79 | 2,917 ms | 26.3 ms | 9.2% |
| 64 | **517.7** | **64.71** | **71.21** | 2,942 ms | 110.1 ms | 57.5% |

⚠️ **c=4 and c=16 come from `results/base-util082/`, not `results/base/`.** The original arm's KV pool
was mis-sized because vLLM measured peak activation during cold `torch.compile` (17.07 GiB measured vs
0.99 GiB actual), costing ~14 GiB/GPU of KV and understating c=4 by **1.58×**. Diagnosis:
[`results/base-util082/RESULT-util-ab.md`](../results/base-util082/RESULT-util-ab.md) and
[`../fix_bug.md`](../../fix_bug.md) bug 12.

**Params [M], from 152,089 safetensors tensor shapes:** 180.01 B on disk → **176.94 B served**
(− 2.61 B MTP, − 0.449 B ViT) / **7.27 B active on the GEMM path**. Reconciles with the model card
("125B with 6B activated, plus 51B n-gram embedding and 4B MTP"): measured main **125.71 B**, active
excluding embed/lm_head **6.00 B**. ⚠️ Card says 4 B MTP; **measured 2.61 B** — trust the tensors.

⚠️ **The active count is genuinely ambiguous for this model, and the ambiguity is a finding.**
**51.23 B (28.5% of all params) is a PLE n-gram embedding table** — 128 shards × 2,500,012 rows × 160
dim, FP8 — read by `F.embedding(splitmix64_hash(ngram))`, i.e. a **gather of a few rows per token, not a
GEMM**. Reporting 58.5 B "active" would be meaningless (the table is never fully read); reporting 7.27 B
hides a quarter of the model's bytes. See [`report.md`](../report.md) §1.

See [`report.md`](../report.md) for the full analysis and what is *not* measured.
