> HISTORICAL MODEL NOTES — use the [current report](../../report.md), [experiment index](../../experiments.md), and [debugging guide](../../fix_bug.md) for corrected conclusions. Original location: `deepseek_v4_flash/report.md`. Some causal claims and configurations below are superseded.

# Serving DeepSeek-V4-Flash on 8×H100: measured throughput, normalized cost, and where the bottleneck actually is

**Author:** Tan Ngo · **Date:** 2026-09-01 · **For:** Kan Zhu, UW SyFI

**Scope.** First *measured* installment answering [`task.md`](../task.md): **one model** (DeepSeek-V4-Flash),
**one node** (8×H100), **one engine** (vLLM 0.28.0). Answers Q1 (bottleneck by workload/batch), Q2
(prefix-cache challenges), Q3 (spec-decode effect, now with a **measured on/off A/B**), and frames Q4
(cost). The cross-family comparison (GLM, Qwen, Kimi) remains architectural analysis in
[`plan.md`](../plan.md) — §8 states exactly what is missing and what it would cost to close.

Per Kan's email the emphasis is **throughput and serving cost**, not output quality. Nothing here
measures accuracy.

**Headline basis — read this before any number below.** The primary throughput numbers in this report
are **base model, speculative decoding OFF**. Comparing an MTP-accelerated model against a base model
is not a comparison of *models*, it is a comparison of *inference tricks*, and it flatters whichever
vendor shipped a draft head. MTP-on numbers appear only in §5, isolated as a deliberate A/B against the
identical sweep. §1 reports throughput **normalized three ways** — per active parameter, per total
parameter, and per GPU — because raw tok/s says nothing about whether a model is *efficient*.

---

## 0. Provenance

| | |
|---|---|
| **Hardware** | 8× NVIDIA H100 80GB HBM3 (`tan-8gpus-moe-0-0`), driver 575.57.08 |
| **Aggregate HBM** | 26.8 TB/s (8 × 3.35 TB/s vendor spec) |
| **Engine** | vLLM 0.28.0, torch 2.13.0+cu129, transformers 5.16.1 |
| **Model** | `deepseek-ai/DeepSeek-V4-Flash`, commit `60d8d707`, 148.6 GiB, 46 shards |
| **Parallelism** | TP=8, expert-parallel on |
| **Weight dtype** | **FP8 e4m3** (attention/dense, block 128×128, ue8m0 scales) + **MXFP4** (routed experts) |
| **KV dtype** | fp8 |
| **Base runs (primary)** | **spec decode OFF** — `run_nomtp.sh`, `results/mtp-off/` |
| **A/B runs (§5 only)** | MTP on, `num_speculative_tokens=1` — `run.sh`, `results/mtp-on*/` |
| **Harness** | `vllm bench serve`, `--ignore-eos`, unique seed per point |
| **Repro** | `cd deepseek_v4_flash && ./run_nomtp.sh`, then `../bench.sh batch context prefix` |

**Labelling.** **[M]** = measured on this hardware today. **[A]** = analytical, from `config.json`,
safetensors headers, or vendor specs. Never mixed silently.

---

## 1. Parameter accounting, and throughput normalized by it

Kan's likely first question: *what am I actually loading, and is 380 tok/s good for a model this size?*
Raw tok/s cannot answer that. So first, the parameter budget — **[M]**, computed from the 69,187
safetensors tensor headers, with FP4 experts unpacked to logical width (2 params/byte):

| Component | Logical params | GiB on disk | Share |
|---|--:|--:|--:|
| Routed experts (MXFP4) | **283.47 B** | 132.00 | **97.4%** |
| Attention (FP8) | 5.21 B | 5.19 | 1.8% |
| Shared expert | 1.11 B | 1.03 | 0.4% |
| Embed + LM head | 0.53 B | 0.99 | 0.2% |
| MTP head | 0.04 B | 0.03 | — |
| Other (norms, router, indexer) | 0.59 B | 1.15 | 0.2% |
| Quant scales (not params) | — | 8.25 | — |
| **Total** | **290.9 B** | **148.65** | |

**Active per token [A]:** `283.47 × 6/256 + 5.21 + 1.11 + 0.53` = **13.49 B**.

Both reconcile with the model card's *284B total / 13B active* to within 2.4% — the residual is card
rounding plus my counting the MTP head and router. **Sparsity = 290.9 / 13.49 = 21.6×**, while
`E/k = 42.7×`. The gap between those two numbers is the point: **the dense remainder (attention +
shared expert + embeddings = 6.85 B) dominates the active budget**, contributing 51% of active params
while being only 2.4% of total. Expert sparsity does not buy you 42.7× — it buys 21.6×, because
attention does not sparsify.

### Normalized throughput — base model, MTP off

**[M]** ISL 16,384 · OSL 256 · unique seed per point (`results/mtp-off/`):

| Concurrency | Output tok/s | **per B-active** | **per B-total** | **per GPU** | Total tok/s |
|--:|--:|--:|--:|--:|--:|
| 1 | 92 | 6.8 | 0.32 | 11.5 | 6,031 |
| 4 | 222 | 16.5 | 0.76 | 27.8 | 14,504 |
| 16 | 294 | 21.8 | 1.01 | 36.8 | 19,183 |
| 64 | **383** | **28.4** | **1.32** | **47.9** | 25,048 |

These three normalizations answer different questions, and a cross-model comparison needs all three:

- **tok/s per B-active** — efficiency of the compute the model actually does. The right metric for
  "is this architecture's FLOP budget being used well?"
- **tok/s per B-total** — efficiency per byte you had to *buy HBM for*. A 21.6×-sparse model looks
  great per-active and poor per-total; that ratio *is* the MoE bargain, stated numerically.
- **tok/s per GPU** — what a cluster operator pays for. **The only metric that translates to cost.**

**Why this matters for the cross-family comparison.** V4-Flash needs 148.6 GiB → **3 H100s minimum**;
GLM-4.7-Flash is 29.1 GiB → **1 GPU**; Kimi-K3 is 1,453.7 GiB → **~21 GPUs**. A 21× spread in minimum
deployable footprint. Comparing raw tok/s across those is meaningless — one model's number comes from
21× the silicon. **Per-GPU-normalized throughput at matched context is the only fair basis**, and any
model that doesn't fit one node pays an interconnect tax that per-active-param normalization hides
entirely.

---

## 2. Q1 · Bottleneck vs batch size

**[M]** base model, ISL 16,384 · OSL 256 (`results/mtp-off/`):

| Concurrency | Output tok/s | Total tok/s | TTFT p50 | TPOT p50 | Scaling vs c=1 |
|--:|--:|--:|--:|--:|--:|
| 1 | 92 | 6,031 | 714 ms | 8 ms | 1.0× |
| 4 | 222 | 14,504 | 1,914 ms | 11 ms | 2.4× |
| 16 | 294 | 19,183 | 3,994 ms | 43 ms | 3.2× |
| 64 | 383 | 25,048 | 3,687 ms | 151 ms | **4.2×** |

**64× the concurrency buys 4.2× the output throughput**, while TPOT degrades **19×** (8 → 151 ms). Past
c≈16 you buy almost nothing with steeply worse per-token latency — the binding constraint is not the
one batching relieves.

Total token throughput is **25,048 tok/s at c=64**, ~65× the output rate, because input tokens dominate
64:1. **This workload is prefill-bound**, and prefill is compute-bound.

Sanity check a reviewer will run **[A]**: at c=64, per-request time = TTFT + TPOT×OSL = 3.69 + 0.151×256
= 42.4 s, implying 64×256/42.4 ≈ 386 tok/s against 383 measured. The numbers are internally consistent.

### Important qualification: chunked prefill is on, so TTFT is not pure prefill compute

**[M]** From the server log: `Chunked prefill is enabled with max_num_batched_tokens=8192`
(`scheduler.py:242`, vLLM 0.28 default). Every prompt is split into 8,192-token chunks:

| ISL | Chunk-steps to prefill one request |
|--:|--:|
| 16,384 | 2 |
| 131,072 | 16 |
| 262,144 | 32 |

Those chunks **interleave with other requests' decode steps**, so TTFT includes scheduler round-trips and
queueing behind other requests' chunks — at c=64 × ISL 16K there are 128 chunk-steps of prefill work
competing with decode. **TTFT is what a user experiences, but it is not a measurement of prefill FLOPs**,
and "prefill-bound" is more precisely *"wall-clock is dominated by prefill work, scheduled in 8,192-token
chunks"* — a statement about the engine's scheduling regime as much as about the model.

`max_num_batched_tokens` was **not swept**, and it is plausibly the largest server-side lever for this
workload. Listed in §8.

### A measurement error worth reporting

My first batch sweep used a **fixed seed at every concurrency**, so `vllm bench serve` generated
*identical prompts* at every point. c=4 reported **640 tok/s** — a bogus 5.1× "speedup" that was pure
prefix-cache reuse from the preceding c=1 run. Verified by re-running with a fresh seed while watching
`vllm:prefix_cache_hits_total`:

| Run | Output tok/s **[M]** | Duration | New cache hits |
|---|--:|--:|--:|
| seed 0 (after c=1 warmed cache) | **640** | 3 s | > 0 |
| seed 12345 (cold) | **214** | 10 s | **0** |

A 3× phantom speedup. Every point in this report is a unique-seed run with `newcachehits = 0` confirmed.
**Benchmarking a prefix-caching engine with a fixed prompt set measures your cache, not your model.**
The contaminated files are retained in `results/mtp-on/` so the error is auditable, not erased.

---

## 3. Q1 · Bottleneck vs context length

**[M]** base model, concurrency 8 · OSL 256 (`results/mtp-off/`):

| ISL | Output tok/s | Total tok/s | TTFT p50 | TPOT p50 |
|--:|--:|--:|--:|--:|
| 16,384 | 101 | 6,593 | 3.1 s | 23 ms |
| 65,536 | 58 | 14,977 | 12.1 s | 73 ms |
| 131,072 | 42 | 21,433 | **26.0 s** | 89 ms |

**[M]** MTP-on run reached 262,144 (`results/mtp-on/`): 19 tok/s, 19,574 total, **TTFT 51.5 s**, peak KV
**30.2%**.

**Output throughput falls 2.4× as context grows 8×, while total throughput rises 3.3×** — prefill
parallelism is still being exploited up to ~131K; past that the attention term dominates and both fall.

**TTFT is the real casualty: 26.0 s at 131K, 51.5 s at 262K.** At the TraceLab median of 132K, that is
**26 seconds to first token for a single agent turn**. This is the number a serving PI cares about, and
it makes prefix caching existential rather than an optimization (§4).

**KV capacity is not the constraint for this model.** Peak KV was **30.2%** even at 256K × 8 sequences.
DeepSeek-V4's per-layer compression (`compress_ratios` 4/128) plus fp8 KV keeps the cache comfortable —
a direct consequence of the ~50× KV-cost reduction in `plan.md`, and a sharp contrast with the 2025
generation (GLM-4.5 at 368 KiB/tok would be ~30× worse). **[M]** `num_gpu_blocks = 43,197`.

---

## 4. Q2 · Prefix caching: 3.7× on the table, and a hard structural obstacle

### 4.1 The workload justifies it overwhelmingly

**[M]** from **`UW-SyFI/TraceLab` v0.0.2** (665,453 rounds, 8,058 sessions, CC-BY-4.0 — Kan's own lab's
trace, arXiv 2606.30560):

| Metric | p50 | p90 | p99 | mean |
|---|--:|--:|--:|--:|
| `input_tokens_total` | **132,092** | 338,662 | 856,464 | 171,576 |
| `output_tokens` | **249** | 1,332 | 5,542 | 589 |
| `prefix_tokens` | 126,336 | 326,527 | 848,886 | 164,053 |

- **ISL:OSL ≈ 530:1** at the median — agentic serving is a prefill problem, decisively.
- **`prefix_tokens` / `input_tokens_total` = 95.6%** (mean); **98.8%** of rounds have `prefix_tokens > 0`.
- Provider-reported cache reads cover **57.6%** of aggregate input tokens — a *floor*, since the
  360,008 Codex rounds contribute 0 to the numerator while counting in the denominator.

~95% of input tokens are structurally reusable; ~38 points of that reuse are unrealized.

### 4.2 Measured benefit

**[M]** synthetic control, 64K shared prefix + 2K unique suffix, 64 prompts, concurrency 8
(`results/mtp-on/` — MTP on; the sharing *ratio* is the finding, not the absolute level):

| Distinct prefixes | Sharing | Output tok/s | Total tok/s | TTFT p50 |
|--:|---|--:|--:|--:|
| 1 | 64 reqs/prefix | **433** | 114,910 | 752 ms |
| 4 | 16 reqs/prefix | 259 | 68,599 | 548 ms |
| 16 | 4 reqs/prefix | 118 | 31,203 | 3,459 ms |

**3.7× output throughput and 4.6× lower TTFT** from sharing alone, monotonic in sharing degree. Total
throughput reaches **114,910 tok/s** at full sharing — 4.6× the best non-shared number in §2. *This*,
not batching, is the lever for agentic workloads.

### 4.3 The challenge: block size is pinned at 256 by the compression layout

The interesting part of Q2 is not "does it help" but "what makes it hard here."

`--block-size 256` is **mandatory, not tunable**:

```
vllm/models/deepseek_v4/sparse_mla.py:53
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]
```

A single-element list. The reason: V4's per-layer compression makes *storage* block size a derived
quantity (`vllm/v1/kv_cache_interface.py:399,615`):

```python
@property
def storage_block_size(self) -> int:
    return self.block_size // self.compress_ratio
```

With `compress_ratios ∈ {4, 128}` **[A]**:

| `block_size` | storage blocks (ratio 4, 128) | verdict |
|--:|---|---|
| **256** | 64, **2** | ✅ only valid value |
| 128 | 32, **1** | degenerate |
| 64 | 16, **0** | floors to zero |

The compressor hardcodes the dependency (`compressor.py:180`):

> *"Block size is constrained by tensor sharing between compressor states and KV blocks… The KV block
> shape `[256//4, head_dim] = [64, 584]` determines: C4 compressor block shape → block_size = 4,
> C128 → block_size = 8. TODO(yifan): make block size automatically determined and configurable."*

**This is the report's thesis showing up as a vendor TODO.** The uniform-per-layer-cost assumption is
broken *inside a single model*, and the engine's response is to hardcode one layout and pin the user's
block size to it.

Scheduler-visible consequence, from `resolve_kv_cache_block_sizes` (`kv_cache_utils.py:605-666`):
`scheduler_block_size = lcm(group sizes)`, `hash_block_size = gcd(group sizes)`. Observed live **[M]**:

```
block_size="4"   user_specified_block_size="True"   num_gpu_blocks="43197"
```

**We passed 256 and the engine resolved to 4** — `min(g.kv_cache_spec.block_size …)` at
`v1/engine/core.py:322`. Prefix-cache hashing runs at **4-token granularity** across heterogeneous
groups: more hashing work per token, more metadata per cached prefix. On a uniform model this is 16 or
64 and nobody thinks about it.

**Verdict: keep `--block-size 256`. Do not remove or change it.** An earlier draft suggested dropping it;
that was wrong. Removing it is *harmless in effect* (vLLM derives the same layout regardless) but is not
an improvement, and any other explicit value is either rejected or silently degrades the compressed
layers. The flag documents a real constraint.

---

## 5. Q3 · Speculative decoding — measured A/B

The A/B `plan.md` asked for. Identical sweep, identical seeds, `--speculative-config` the only
difference. **[M]**:

| Concurrency | tok/s MTP **off** | tok/s MTP **on** | Speedup | TPOT off | TPOT on | TPOT gain |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 92 | 115 | **1.25×** | 8 ms | 5 ms | 1.60× |
| 4 | 222 | 252 | 1.14× | 11 ms | 9 ms | 1.22× |
| 16 | 294 | 340 | 1.16× | 43 ms | 33 ms | 1.30× |
| 64 | 383 | 380 | **0.99×** | 151 ms | 137 ms | 1.10× |

**The MTP gain decays to nothing as batch grows: 1.25× at c=1 → 0.99× at c=64.** This is the structurally
interesting result, and it is what a systems reader should predict: at c=1 the GPU is idle enough that a
draft head is free parallel work; at c=64 the batch already saturates the machine, so draft compute
*competes* with real tokens instead of filling gaps. **Spec decode buys latency at low load and nothing
at high load** — it trades throughput headroom for TPOT.

Cost, measured: **`num_gpu_blocks` 43,197 → 42,149 = 2.4% of KV capacity** spent on MTP, plus a doubled
fp32 logits buffer that is what caused the original OOM (§9).

**Acceptance rate correction.** An early spot check over ~3,000 draft tokens showed 84.4%. Over the
**full sweep (107,739 draft / 63,868 accepted) the true rate is 59.3%** **[M]**. The small sample came
from short, highly predictable prompts and was not representative — the 84.4% figure should not be
cited. 59.3% on a 1-token draft is respectable but well short of the ceiling, and it is consistent with
the modest speedups above.

**Why this belongs in a throughput report at all:** §2–§4 show this workload is **prefill-bound** at
ISL:OSL = 530:1. MTP accelerates *decode*. It improves the part of the computation that barely matters
here, costs 2.4% of KV, and reaches zero benefit exactly at the concurrency a production server runs at.
**For throughput-oriented serving of long-context agentic traffic, ship the base model.**

Architecturally **[A]**: Kimi-K3 ships `num_nextn_predict_layers: 0` — no MTP at all — while GLM-5.3
sets `index_share_for_mtp_iteration: true`, sharing the sparse-attention indexer across MTP iterations.
Three vendors, three different bets on whether a draft head earns its weights.

---

## 6. Q4 · Serving cost

**[M]** throughput, **[A]** price. Base model (MTP off) except the two rows marked otherwise:

| Regime | Output tok/s | Node-s / 1M tok | $/1M output tok @ $2/GPU-hr | $/1M ÷ B-active |
|---|--:|--:|--:|--:|
| c=1, ISL 16K | 92 | 10,870 | **$48.3** | $3.58 |
| c=64, ISL 16K | 383 | 2,611 | **$11.6** | $0.86 |
| ISL 131K (TraceLab p50), c=8 | 42 | 23,810 | **$105.8** | $7.84 |
| ISL 262K, c=8 *(MTP on)* | 19 | 52,632 | **$233.9** | $17.34 |
| **64K prefix fully shared** *(MTP on)* | **433** | **2,309** | **$10.3** | $0.76 |

*(8 GPUs × $2/GPU-hr = $16/node-hr — a placeholder list-price stand-in, not a quote. Output tokens only;
input tokens are the dominant *work* but are not what APIs typically meter.)*

**The spread is 23×**, driven almost entirely by **context length and prefix sharing — not batch size**.
Serving at TraceLab's median context without prefix caching costs **$105.80/1M output tokens**; with
full prefix sharing, **$10.30**. **Cost engineering for this model is cache engineering.**

The last column is the cross-model-ready form: cost per million tokens per billion active params. It is
what lets you say "model A is 2× more expensive but 3× larger, so per unit of capability it is cheaper"
— a claim raw $/1M cannot support.

Caveat: single-model, single-node, synthetic workload. Real deployments amortize across tenants with
partial sharing, so expect to land between the extremes.

---

## 7. Questions I expect from the lab, and my current answers

Anticipated rather than avoided. Where I lack the measurement, I say so.

| Question | Answer |
|---|---|
| **"Is this BF16 or FP8?"** | Neither, exactly: **FP8 e4m3** attention/dense + **MXFP4** routed experts. `torch_dtype: bfloat16` in `config.json` is the *activation* dtype and is a decoy (§9). |
| **"Are these numbers with spec decode?"** | Primary numbers are **MTP off**. §5 is the isolated A/B. |
| **"380 tok/s — is that good?"** | Unanswerable without normalization. §1 gives per-active-param, per-total-param, per-GPU. |
| **"How many GPUs does it need?"** | Fits **3** H100s on weights (18.6 GiB/GPU at TP=8); we used 8. KV peaked at 30.2%, so 8 is generous. |
| **"Why is per-GPU throughput low?"** | Prefill-bound at ISL:OSL ≥ 64:1, plus achieved HBM < 5% of peak (below). Not yet attributed to a specific kernel — that needs `torch.profiler`. |
| **"Did you validate against the roofline?"** | Partially, and it **failed to confirm**. See below — I am not claiming memory-bound. |
| **"Is prefix caching on?"** | Yes, `enable_prefix_caching=True`, hashing at 4-token granularity (§4.3). All throughput points verified `newcachehits = 0`. |
| **"What about output quality?"** | Not measured, deliberately, per your email. Also unmeasurable on the layer-reduced proxies `plan.md` plans. |

### The roofline claim, and why I am not making it

`plan.md` predicts **[A]** `B*_dense ≈ 148 tok/step` (989.4 TFLOP/s ÷ (2 × 3.35 TB/s)) and
`B*_MoE = B*_dense · E/k ≈ 6,300 tok/step` — i.e. the expert read should bind at any reachable batch.

Implied aggregate HBM traffic from the decode weight read (MTP-off run), against 26.8 TB/s peak:

| Concurrency | Output tok/s **[M]** | Decode steps/s **[M]** | Implied weight read **[A]** | % of peak |
|--:|--:|--:|--:|--:|
| 1 | 92 | 92.0 | 1.24 TB/s | **4.6%** |
| 4 | 222 | 55.5 | 0.75 TB/s | **2.8%** |
| 16 | 294 | 18.4 | 0.25 TB/s | **0.9%** |
| 64 | 383 | 6.0 | 0.08 TB/s | **0.3%** |

*(Weight read modelled as 13.49 B active params × 1 B/param × steps/s — an upper bound, since only k=6
of 256 experts are touched per token.)*

Achieved bandwidth is **under 5% of peak everywhere and falls as batch grows**. A memory-bandwidth-bound
decode sits near peak. Two readings I cannot yet separate:

1. **The workload never enters the decode-bound regime.** At ISL:OSL = 64:1, wall-clock is prefill.
   `B*` is a statement about decode steps; this workload barely does any.
2. **Something other than HBM binds** — and expert-parallel all-to-all is my leading candidate. **[M]**
   the engine reports `[EP Rank 0/8] Expert placement strategy: linear. Local/global number of experts:
   32/256`. With k=6 across 8 ranks, the expected experts touched **per rank per token is 0.75** — most
   ranks contribute 0 or 1 experts, so load imbalance is structural and **all-to-all dispatch/combine
   sits on the critical path of every decode step**. That is a *latency* term that never appears as HBM
   traffic, which is exactly the signature we observe.

**Discriminating experiment (specified, not run):** ISL 512, OSL 4,096, concurrency ≥512. That isolates
decode. Note **`--max-num-seqs` defaults to 128**, so reaching c=512 requires raising it, and a batch of
6,300 tok/step may be **unreachable on one node** — the prediction could be untestable at this scale
rather than merely untested.

**Better instrument, found while writing this and not yet used:** vLLM 0.28 exposes
`vllm:estimated_flops_per_gpu_total` and `vllm:estimated_read_bytes_per_gpu_total` behind
`--enable-mfu-metrics` (default off, so they currently read 0.0). Those are **engine-side FLOP and byte
counters** — a materially better bandwidth measurement than the analytical model above, and one flag away.

---

## 8. What is not measured, and why

| Gap | Why it matters | Cost to close |
|---|---|---|
| **Achieved bandwidth is modelled, not measured** | §7's <5% figure is **[A]**, an upper bound ignoring KV traffic and all-to-all. | `--enable-mfu-metrics` — **one flag** |
| **`max_num_batched_tokens` unswept** | Chunked prefill at 8,192 shapes every TTFT number (§2); plausibly the biggest server-side lever. | 4-point sweep at ISL 131K |
| **TP=8 unjustified** | Model fits on 3 GPUs; TP=8 adds collectives and inflates per-GPU cost, so **47.9 tok/s/GPU is pessimistic**. | TP sweep 2/4/8 |
| **No repeats, no error bars** | Single run per point, 8–128 prompts. Only an internal consistency check (386 predicted vs 383 measured). | 3 seeds per point |
| **No cross-model comparison** | Q4 asks for it; `model_list.md` names GLM-5.3-Flash and Qwen3.8-Flash-Next-FP8. Neither served. | GLM-4.7-Flash: 29 GiB, 1 GPU |
| **No dense control** | Without a dense model on this harness, "MoE is memory-bound" is unfalsifiable. Qwen3.8 is designated. | 52 GiB + one sweep |
| **Decode-bound regime untested** | Everything ran ISL:OSL ≥ 64:1; `B*_MoE` lives at low ISL / high OSL, and may be unreachable on one node. | short-ISL/long-OSL sweep, raise `--max-num-seqs` |
| **256K point is MTP-on only** | The base-model context sweep stops at 131K. | one run |
| **Prefix sweep is MTP-on only** | The 3.7× *ratio* should be spec-decode-neutral, but that is unverified. | one run |
| **TraceLab not replayed** | Distributions *anchor* the grid; the trace itself is not replayed. vLLM's `timed_trace` loader consumes it via `hash_ids`. | `bench/tracelab_to_timedtrace.py` |
| **No operator attribution** | NanoFlow's dense/attention/**network**/other taxonomy needs `torch.profiler`; the network bucket is §7's hypothesis. | one profiled run |
| **Synthetic prompts route uniformly** | Random tokens spread across experts — best case for coverage, so real-text expert-read cost may be *lower*. | trained-router hit counts |
| **Single engine** | vLLM only; `plan.md` wants one SGLang cross-check. | SGLang install |
| **No quality measurement** | Deliberate, per Kan's email. | out of scope |

TraceLab caveats: token counts come from **Claude/GPT tokenizers** (top models `gpt-5.5`,
`claude-opus-4-8`), not DeepSeek's, so lengths are approximations; prompt text is sanitized away, so only
*structure* is real; documented `prefix_tokens` over-reporting (TraceLab issue #22) means session-local
replay should cap reusable prefix at `previous.input_tokens_total + previous.output_tokens`. Sessions are
extremely skewed (rounds/session p50 = 16, max = 21,351) — **subsample by session, not by round**.

---

## 9. Operational findings worth keeping

**The OOM that started this was a warmup budget overshoot, not insufficient capacity.** Weights are
148.6 GiB ÷ 8 = **18.6 GiB/GPU** against a 71.3 GiB budget. At `gpu_memory_utilization=0.9` the engine
allocates KV *first* (`_initialize_kv_caches`), then `compile_or_warm_up_model()` had nothing left. The
failing allocation was **506 MiB = 1,024 rows × 129,536 cols × 4 B** — the fp32 logits copy at
`sampler.py:196` (vocab 129,280 padded to 129,536). **MTP doubles the sampled rows, doubling that
buffer** — the spec-decode feature caused the OOM. `0.82` fixed it.

**`torch_dtype: bfloat16` is a decoy.** It is the activation dtype. Weights are FP8 + MXFP4 via
`DeepseekV4FP8Config` (`quant_config.py:29`), which resolves `expert_dtype` lazily and logs
`DeepSeek V4 expert_dtype resolved to 'fp4'`. Reporting this run as "BF16" would be wrong — and this is
precisely the error that makes cross-model comparisons meaningless.

**FP4 halves bytes but does not move the ridge point.** `fp4_gemm_kernel` unpacks FP4→FP8 in shared
memory and runs H100's FP8 tensor cores. Byte traffic halves; FLOP/byte at the tensor core is unchanged.
Confirms `plan.md`'s correction: **do not** claim FP4 doubles `B*_dense` to 296.

**vLLM 0.28 supports V4 natively** — a full `vllm/models/deepseek_v4/` package with `nvidia`/`amd`/`xpu`
backends. No `tilelang`, no `fast_hadamard_transform`. Supersedes the CLAUDE.md note predicting an
SGLang-only FP8 route.

**Reasoning output is a separate field.** With `--reasoning-parser deepseek_v4`, text lands in
`message.reasoning`, only the final answer in `message.content`; too small a `max_tokens` yields
`content: null` + `finish_reason: "length"`. Budget ≥600 output tokens.

**Server startup is ~255 s** from launch to `/health` 200 — worth knowing before scheduling sweeps.

---

## 10. Recommended next step

**GLM-4.7-Flash on 1 GPU** (29.1 GiB, full L=47) through the same `bench.sh`, base model, no spec decode.
It is the first cross-architecture data point (MLA-all-layers vs V4's DSA + compression), the cheapest
complete architecture in the 2026 set, and it validates the harness on a second model before anything is
extrapolated. Report it in §1's normalized form so the comparison is defensible: **per-GPU and
per-active-param, not raw tok/s.**

Second: the **decode-bound sweep** (§7) — the only experiment that can confirm or kill `B*_MoE`, the
central analytical claim.

### Artifacts

This model's files live in `deepseek_v4_flash/`; shared tooling stays at the repo root. See
[`README.md`](../README.md) for the full rationale behind each non-default flag.

| File | Contents |
|---|---|
| `deepseek_v4_flash/run_nomtp.sh` | **Base-model control** — the primary numbers (§1–§3) |
| `deepseek_v4_flash/run.sh` | Server launch, MTP on — §5 A/B arm. **Keep `--block-size 256`** (§4.3); `0.82` utilization (§9) |
| `deepseek_v4_flash/results/mtp-off/` | **Base-model batch + context sweeps — the primary numbers (§1–§3)** |
| `deepseek_v4_flash/results/mtp-on-noreuse/` | MTP-on batch sweep, unique seeds — the §5 A/B |
| `deepseek_v4_flash/results/mtp-on/` | Context + prefix sweeps, MTP on; batch points seed-contaminated (§2), kept for audit |
| `deepseek_v4_flash/logs/` | Server startup log (source for the chunked-prefill and EP-sharding facts) + sweep console output |
| `sending.sh` *(root)* | 8-section functional smoke test (health/models/chat/stream/raw/tools/batch/metrics) |
| `bench.sh` *(root)* | Sweep harness — writes `manifest.txt` with hardware + quantization + server cmdline |
| `plan.md` *(root)* | Architectural analysis, roofline derivation, cross-family config tables |
| `potential_questions.md` *(root)* | Anticipated questions from the lab, with answers and named gaps |
