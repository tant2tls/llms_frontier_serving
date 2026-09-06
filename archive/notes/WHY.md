> ARCHIVED SESSION RECORD — not current instructions or conclusions. Original location: `WHY.md`. See [current report](../../report.md), [debugging guide](../../fix_bug.md), and [session handoff](../../WORKFLOW.md). Historical claims may be superseded.

# WHY.md — the mechanism behind every number in this report

Companion to `REPRODUCE.md` (how to re-measure) and `fix_bug.md` (what broke).
This file answers the question a serving-systems PI will actually ask: **why is the
number what it is, and what would change it?**

Written to be attacked. Every claim is labelled **[M]** measured, **[A]** analytical,
or **[H]** hypothesis-not-yet-tested. Where a prediction and a measurement disagree,
both are shown.

---

## 0. The one-paragraph answer

> On 8×H100, neither GLM-5.3-Flash nor DeepSeek-V4-Flash is anywhere near the memory
> roofline during decode — **6.4% and 2.3% of it respectively at batch=1 [M]**. So the
> textbook story ("decode is memory-bound, throughput ∝ 1/active_bytes") **does not
> describe this hardware regime at all**, and any ranking derived from active-parameter
> counts is measuring the wrong thing. What actually binds is a **serial latency chain**:
> 45 sequential layers, each with an EP all-to-all on its critical path, and a per-step
> cost that barely falls when you read fewer bytes. That is why GLM reads **2.6× more
> bytes per token [A]** than V4 and is still **1.04–1.22× faster [M]**. The architectural
> features that matter here are the ones that shorten or widen that chain — recurrent KDA
> layers, sparse-attention top-k, and expert routing — not the ones that shrink bytes.

---

## 1. Establish the regime before explaining anything

Everything below depends on this, so it goes first.

| quantity | GLM-5.3-Flash | DeepSeek-V4-Flash |
|---|--:|--:|
| active params | **17.38 B [M]** (safetensors shapes) | 13.49 B [A] |
| expert weight dtype | FP8 e4m3, ~1 B/param [M] | MXFP4, ~0.5 B/param [A] |
| **bytes read / token** | **~17.75 GB [A]** | ~6.75 GB [A] |
| roofline ceiling @8×H100 (3.35 TB/s ea.) | 1,510 tok/s [A] | 3,973 tok/s [A] |
| **measured @ conc=1** | **96.3 tok/s [M]** | 92.3 tok/s [M] |
| **% of roofline achieved** | **6.4%** | **2.3%** |

**Read that last row twice.** Both models are ~1.5–2 orders of magnitude below their
bandwidth ceiling at batch=1. Two consequences:

1. **Active-parameter accounting cannot rank these models here.** It predicts V4 should
   be 2.6× faster. Measured: V4 is *slower*. The prediction fails because the premise
   (bandwidth-saturated) is false.
2. **The bound is latency, not bandwidth.** At batch=1 a decode step is 45 dependent
   layer-steps; each is a small GEMM plus collectives. The GPU is idle most of the step.

> **Anticipated question — "isn't <5% of peak just bad kernels?"**
> Partly, but the *structure* dominates. A decode step at batch=1 issues hundreds of
> small kernels with a serial dependency between layers, and (at EP8, k=8, E=288) an
> all-to-all whose latency is set by the slowest rank. Little's law: with almost no work
> in flight, achieved bandwidth ≈ (bytes per step) / (latency per step), and latency is
> floored by launch + collective overhead, not by HBM. This is why throughput rises
> **4.64×** from conc 1→64 [M] while per-token bytes are unchanged — you are filling
> idle time, not buying bandwidth.

---

## 2. Why GLM wins despite reading 2.6× more bytes

Measured, matched grid (ISL 16,384 · OSL 256 · TP8 · base model, both cold) [M]:

| conc | GLM t/s | V4 t/s | GLM/V4 |
|--:|--:|--:|--:|
| 1 | 96.3 | 92.3 | 1.04× |
| 4 | 225.3 | 222.0 | 1.01× |
| 16 | 359.2 | 293.7 | **1.22×** |
| 64 | 447.1 | 383.4 | **1.17×** |

Three mechanisms, in order of estimated importance:

**(a) 34 of GLM's 45 layers are recurrent (KDA) and carry NO per-token KV.**
A KDA layer reads a **fixed-size state** — O(1) per token, independent of context.
Only the 11 DSA layers pay O(ctx). Per-token KV [A]:

```
GLM:  11 DSA × (512-dim latent × 2 B + ~64 B kpool indexer) = 11.69 KiB/token
      if GLM were pure attention (all 45 layers DSA):        47.81 KiB/token  (4.1× more)
```

This is the single biggest architectural lever in the model. It is why a **306 GiB**
model holds **2,099,654 KV tokens in 23.26 GiB [M]**, and why its attention cost grows
far more slowly with context than a pure-MLA model's.

**(b) GLM's per-layer serial chain is shorter where it counts.** 45 layers vs V4's 43 —
comparable — but only 11 of GLM's do sparse top-k index selection, versus all 43 in V4.
Sparse-attention indexing is not free: it is a top-k over the context per layer per step.
GLM pays it 11 times per token; V4 pays it 43 times. **[H]** — this is a strong candidate
for the c=16/c=64 crossover and is directly testable with a layer-wise NVTX trace
(`--enable-layerwise-nvtx-tracing`), which I did not run.

**(c) Batch amortizes GLM's larger weight read better than V4's.** GLM's extra bytes are
almost entirely **routed experts** (304.42 of 321.34 B [M]). Expert reads amortize across
a batch: at k=8 of E=288, a batch of B tokens touches ≈ min(B·8, 288) distinct experts, so
per-token expert cost *falls* as B grows until all 288 are resident in the step. V4 has
E=256, k=6 → saturates sooner. GLM has more headroom to amortize, which is consistent with
the gap **opening** at c=16 (1.22×) rather than at c=1 (1.04×) [M].

### 2b. ✅ THE ENGINE CONFOUND IS NOW CLOSED [M]

V4-Flash was rerun **in the GLM image** — same engine (`0.1.dev20051+g487ecf187`), same
CUDA 13.0/torch 2.13.0+cu130, same node, same TP8/EP, same `--max-num-seqs 256`, same
`bench.sh` grid, spec decode off. `deepseek_v4_flash/results/mtp-off-image/`, 11 points, all cold.

**★ THE DEFENSIBLE HEADLINE — same engine, same node, same grid, both base models:**

| conc | GLM t/s | V4 t/s | GLM/V4 | GLM /GPU | V4 /GPU | GLM /B-act [M] | V4 /B-act [M] |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 85.0 | **1.13×** | 12.04 | 10.63 | 5.54 | 6.04 |
| 4 | 225.3 | 219.8 | 1.02× | 28.16 | 27.48 | 12.96 | 15.61 |
| 16 | 359.2 | 330.9 | **1.09×** | 44.90 | 41.36 | 20.67 | 23.50 |
| 64 | 447.1 | 389.4 | **1.15×** | 55.89 | 48.67 | 25.72 | 27.65 |

Concurrency scaling 1→64: GLM **4.64×**, V4 **4.58×** — nearly identical.

**How large was the engine effect?** Measured per-concurrency (conda 0.28.0/cu129 →
image dev/cu130, identical model and flags): **0.92× / 0.99× / 1.13× / 1.02×, mean 1.01×.**
**Essentially neutral.** So my original cross-engine ratios (1.04–1.22×) were *approximately
right by luck*, and the corrected same-engine ratios (1.02–1.15×) are slightly **tighter**.
The confound was worth closing precisely because we could not have known that in advance.

**Both param counts are now MEASURED [M]** — GLM 321.34 B/17.38 B, V4 290.91 B/14.08 B — so
**per-B-active is finally a fair column.** ⚠️ V4's needs care: its routed experts are stored
as `I8` tensors packing **two MXFP4 values per byte**, so naive shape-summing gives
158.07 B/11.01 B, which is **wrong by ~2× on the expert term**. Correct: 141.734 B I8
elements × 2 = 283.468 B logical + 7.438 B non-expert = 290.91 B; active = 7.438 +
283.468·6/256 = **14.08 B**. That reconciles with the card's 284 B/13 B.

> **Anticipated question — "so is GLM just better?"**
> **On raw and per-GPU throughput at matched settings, yes — by 1.02–1.15×, now
> engine-matched.** But it **loses on per-B-active at every concurrency** (5.54 vs 6.04 at
> c=1; 25.72 vs 27.65 at c=64): GLM needs *more* active parameters to deliver that
> throughput, so **V4 uses its FLOP budget more efficiently**. Which model is "better"
> depends on whether you are paying for GPUs (GLM) or for parameter efficiency (V4).
> Remaining unmatched: the **forced-opposite KV dtypes** (V4 fp8_ds_mla only, GLM BF16
> only — each can only run what the other cannot) and **different tokenizers**.

---

## 3. The linearity test, stated as a falsifiable claim

If decode were bandwidth-bound, `tok/s × bytes_per_token` would be ≈ constant across
models on identical hardware (both would equal aggregate HBM bandwidth). Test [M/A]:

Now computed on **engine-matched** data [M]:

| conc | GLM tok/s × 17.75 GB | V4 tok/s × 7.04 GB | ratio | GLM × B-act | V4 × B-act | spread |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1,709 GB/s | 598 GB/s | 2.86× | 1,674 | 1,197 | **+39.8%** |
| 64 | 7,936 GB/s | 2,741 GB/s | 2.90× | 7,771 | 5,477 | **+41.9%** |

Aggregate HBM available: **26,800 GB/s**. GLM reaches 6.4% → 29.6%; V4 2.2% → 10.2%.
**Neither is constant and neither approaches peak** — the memory-bound model is rejected
for this regime, on same-engine data. The `tok/s × B-active` spread is **+40% to +42%**,
i.e. GLM extracts ~40% more throughput per active-parameter than V4 despite reading
**2.5× more bytes per token** (V4's MXFP4 experts are 0.5 B/param vs GLM's FP8 1.0 B/param).

> **The strongest single statement in this report:** two 2026-generation sparse MoE models,
> on identical hardware and an identical engine, sit at **6.4% and 2.2% of their memory
> roofline**, and their throughput ordering is the **inverse** of what byte-counting predicts.
> Any capacity-planning model built on active parameters × bytes will mis-rank them.

**What this predicts, and how to falsify it:** if the bound is serial latency + collectives,
then (i) throughput should keep climbing with concurrency until the expert read saturates,
(ii) TPOT should degrade roughly linearly with batch once saturated, and (iii) a TP sweep
should *change the ranking* because TP alters the collective count. Measured (i) ✓ (4.64×
over 64×) and (ii) ✓ (TPOT 7.1 → 129.5 ms [M]). **(iii) is untested** — the honest gap.

---

## 4. Context scaling: where the architecture is most visible

GLM, conc=8, OSL 256 [M]:

| ISL | out tok/s | TTFT p50 | TPOT p50 | KV peak |
|--:|--:|--:|--:|--:|
| 16,384 | 295.9 | 2.6 s | 16.7 ms | 0.102 |
| 65,536 | 104.3 | 7.2 s | 48.0 ms | 0.366 |
| 131,072 | 47.2 | 12.9 s | 116.7 ms | 0.725 |
| 260,000 | 26.5 | 38.5 s | 147.6 ms | 0.895 |

16× the context costs **11.2×** the throughput — **sublinear**, which is the KDA
signature: 34 layers' cost is flat in context, so only 11/45 of the attention work grows.

**But KV capacity becomes binding at max context for GLM (89.5% at 260K [M]) where it
never did for V4 (30.2% at 256K [M]).** That inverts the usual story and is a direct
consequence of the forced KV dtype: GLM pays BF16 (11.69 KiB/tok) where V4 pays fp8. It is
a *kernel-availability* artifact, not an architecture verdict — see §5.

**⚠️ TTFT is not prefill compute.** Chunked prefill is on with
`max_num_batched_tokens=8192`, so a 260K prompt is split into **~32 chunks** interleaved
with other requests' decode steps. TTFT includes scheduler round-trips and queueing.
Never quote it as a prefill-FLOPs measurement.

---

## 5. FP8 KV: I was wrong, and the corrected result is a real finding

**What I claimed earlier:** FP8 KV is "impossible by architecture" for GLM-5.3 because
`fp8_ds_mla` requires `pe_dim == 64` and GLM is NoPE (`qk_rope_head_dim = 0`).

**What is actually true:** that was the *wrong route*. vLLM has a **second** FP8-KV path
built for exactly this case — `FLASHINFER_MLA_SPARSE_SM90` — whose requirements
(`flashinfer_mla_sparse_sm90.py:150-158`) GLM **fully satisfies**:

| requirement | GLM-5.3 | |
|---|--:|:--|
| `kv_lora_rank == 512` | 512 | ✅ |
| `qk_rope_head_dim in (0, 64)` | **0** | ✅ NoPE *explicitly* allowed |
| `hasattr(hf, "index_topk")` | 2048 | ✅ |

`cuda.py:150-157` even **prefers** this backend when `qk_rope_head_dim == 0`. The only
gate was a FlashInfer feature probe for the `ckv_scale_arr` kwarg (**≥ 0.6.18**); the
image ships **0.6.17**. So the recipe's *"Hopper does not support FP8 KV cache for this
model"* is **true of the shipped image, not of Hopper or of the model.**

**Measured with FlashInfer 0.6.18 overlaid [M]:**

| | BF16 KV (0.6.17) | FP8 KV (0.6.18) | gain |
|---|--:|--:|--:|
| KV pool | 23.26 GiB | 22.05 GiB | — |
| KV tokens | 2,099,654 | **3,790,580** | **1.805×** |
| concurrency @256K | 8.01× | **14.46×** | **1.805×** |
| attention backend | FLASH_ATTN_MLA_SPARSE | FLASHINFER_MLA_SPARSE_SM90 | (swapped) |
| correctness | ✓ | ✓ (`2+2 = 4`, coherent reasoning) | |

**Why 1.805× and not the naive 2.0× — the interesting part [A]:**
1. Only the **11 DSA layers'** latent KV is quantized. The **34 KDA layers' recurrent
   state stays BF16** (`mamba_cache_dtype=auto`) and is allocated **one block per
   sequence regardless of dtype** — a fixed floor FP8 cannot shrink.
2. The pool itself shrank (23.26 → 22.05 GiB) for the new backend's workspace.

Predicted from per-token KV arithmetic: **1.889×**. Measured: **1.805×**. The residual is
the workspace. *A theory that lands within 5% is worth more than the measurement alone.*

### 5b. The control arm — and why it changed the headline [M]

The FP8 arm changed **three** things at once (KV dtype, attention backend, MoE backend). So I ran
`run_bf16kv_fi618.sh`: **same overlay, same `moe_backend=deep_gemm`, BF16 KV** — verified to select
the *same* `FLASHINFER_MLA_SPARSE_SM90` backend. That isolates the dtype:

| conc | A: bf16 / 0.6.17 / FLASH_ATTN_MLA_SPARSE | B: bf16 / 0.6.18 / FI_SM90 **(control)** | C: fp8 / 0.6.18 / FI_SM90 | backend A→B | **dtype B→C** | naive A→C |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 71.5 | 68.9 | 0.74× | **0.96×** | 0.72× |
| 4 | 225.3 | 147.5 | 128.7 | 0.65× | **0.87×** | 0.57× |
| 16 | 359.2 | 263.0 | 208.8 | 0.73× | **0.79×** | 0.58× |
| 64 | 447.1 | 348.8 | 262.0 | 0.78× | **0.75×** | 0.59× |
| **mean** | | | | **0.73×** | **0.85×** | 0.61× |

> **The backend swap costs 27.3%. The KV dtype costs 15.5%.** Reporting the naive A→C delta would
> have produced the headline *"FP8 KV costs 39%"* — **and most of that is a kernel artifact, not the
> dtype.** This is the single most important methodological result in the report: a two-arm
> comparison across a library upgrade is not a dtype experiment.

So the defensible statement is: **FP8 KV costs ~15% throughput and buys 1.805× KV capacity** on this
backend. Whether that trade is good depends entirely on whether you are KV-bound — at ISL 16K
(KV peak 10.2%) it is a pure loss; at 260K (KV peak 89.5%) it is what lets you raise concurrency at
all. Note the dtype cost also *grows* with concurrency (0.96× → 0.75×), consistent with FP8
dequant work scaling with the number of KV reads per step.

**⚠️ Remaining confound.** FlashInfer 0.6.18's Python runs against the image's
pinned **0.6.17 compiled artifacts** (no cubin > 0.6.13 in this mirror), needing
`FLASHINFER_DISABLE_VERSION_CHECK=1` and `moe_backend=deep_gemm` to dodge an ABI
mismatch in FlashInfer's fused-MoE (`Expected 8 but got 9 arguments`). Correctness was
verified, but this is **not a production configuration**.

---

## 6. TP4 and prefill/decode disaggregation: disproven, with numbers

The recipe's single-node example is **TP4**. On 8×H100-80GB it **cannot run**, and the
recipe says so itself: *"306 GiB ... alone exceeds 4×H100-80GB."*

**Measured [M]:** TP4 at `--gpu-memory-utilization 0.95`, `--max-model-len 32768`:
```
Model loading took 75.36 GiB     (= 305.78 / 4)
torch.OutOfMemoryError: ... GPU 0 has 79.18 GiB total, 1.60 GiB free
```
Weights alone leave 1.6 GiB — no room for KV, activations, or cudagraphs.

**Therefore TP4+TP4 prefill/decode disaggregation is also impossible on one node**, and
for a structural reason worth stating: **PD disaggregation needs two full weight copies**
(one per pool). At 306 GiB that is 612 GiB against 640 GiB of node HBM — before any KV.
The recipe's PD example is a **GB200 tray** (much larger per-GPU HBM), not an H100 node.

| layout | GPUs | GiB/GPU | fits @0.95 |
|---|--:|--:|:--|
| TP8 (this report's baseline) | 8 | 38.2 | ✅ **[M]** |
| TP4 single pool | 4 | 76.4 | ❌ **[M] OOM** |
| PD disagg TP4+TP4 | 4+4 | 76.4 | ❌ [A] |

**The generalizable lesson:** PD disaggregation is a *latency-structure* optimization
(it stops long prefills from blocking decode steps), and it costs a **2× weight
footprint**. For a 306 GiB model that trade is unavailable on 640 GiB of HBM. It becomes
attractive exactly when weights are small relative to node memory — the opposite of this
model. **This is why "the recipe says TP4" cannot be followed on H100**, and it is worth
saying to a PI as a memory-hierarchy argument rather than a config complaint.

---

## 7. Speculative decoding: the question worth being precise about

**Can a low-throughput model be rescued by spec decode?** Yes at low concurrency, almost
never at high, and the reason is the regime from §1.

Spec decode amortizes **one weight read over k+1 candidate tokens**. Bound:

```
speedup ≲ (1 + n_accepted) / (1 + n_spec · cost_draft/cost_target)
```

The decisive term is not in that formula: **it only helps if the machine is idle.** At
batch=1 the GPU sits at 6.4% of roofline, so verifying 6 tokens costs nearly the same
wall-clock as verifying 1 — near-free parallelism. At batch=64 the machine is already
busy; verify FLOPs now **compete** with real requests and every rejected draft is waste.

**Measured on V4-Flash (this repo, MTP n=1) [M]: 1.25× at c=1 → 0.99× at c=64**, and it
cost 2.4% of KV capacity. That shape is the law, not a V4 quirk:

> Speculative decoding is a **latency** optimization that only *looks* like a throughput
> win in the regime where throughput was already being wasted.

**So the models with the most to gain are the ones with the worst low-batch numbers** —
and GLM at c=1 (96.3 tok/s, 6.4% of roofline) is exactly that profile.

### How GLM's MTP differs from V4's — three concrete differences

1. **Draft depth.** GLM's recipe specifies `num_speculative_tokens: 5`; the V4 A/B used
   **1**. GLM ships **one** MTP layer (`num_nextn_predict_layers: 1`), so 5 tokens means
   running that head **recurrently 5×**. Deeper drafts compound acceptance: if per-token
   acceptance is p, expected accepted length is ≈ (1−p⁵)/(1−p) rather than p — much more
   upside, but also more wasted verify when p is low.
2. **`index_share_for_mtp_iteration: true` — GLM shares the DSA sparse-attention indexer
   across MTP iterations. V4 does not.** The indexer is a top-k(2048) over context per DSA
   layer; sharing it makes each *additional* draft token materially cheaper for GLM. **[H]
   Prediction: GLM's break-even concurrency should be HIGHER than V4's** — i.e. GLM should
   still show gain at concurrencies where V4 has already crossed below 1.0×.
3. **Verify cost is asymmetric.** GLM verifies across 34 KDA + 11 DSA layers. The KDA
   layers' recurrent state must be advanced per accepted token, which does **not**
   parallelize across draft positions the way attention does. **[H] This partially offsets
   (2)** and is the mechanism I would probe first if GLM's MTP underperforms the prediction.

### 7b. MEASURED — and my prediction was WRONG [M]

Both depths run, matched grid (ISL 16K · OSL 256 · TP8 · cold):

| conc | base | MTP n=1 | gain | MTP n=5 (recipe) | gain | V4 n=1 gain |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 119.7 | **1.24×** | 120.8 | **1.25×** | 1.25× |
| 4 | 225.3 | 239.8 | 1.06× | 230.7 | 1.02× | — |
| 16 | 359.2 | 344.7 | 0.96× | 342.1 | 0.95× | — |
| 64 | 447.1 | 441.9 | 0.99× | 408.3 | **0.91×** | 0.99× |

**Acceptance per draft token [M]: n=1 → 71.8% · n=5 → 30.6%.**

**Result 1 — the regime theory (§7) is CONFIRMED.** The gain decays monotonically with
concurrency and crosses break-even between c=4 and c=16, exactly as the idle-machine
argument predicts. 1.24× where the GPU sits at 6.4% of roofline; ≤1.0× once batching has
filled the pipeline.

**Result 2 — my `index_share_for_mtp_iteration` prediction is NOT SUPPORTED.** I predicted
GLM's shared indexer would push its break-even concurrency *higher* than V4's. Measured:
GLM 1.24× → 0.99× versus V4 1.25× → 0.99× — **nearly identical**. The indexer sharing does
not move the crossover at matched draft depth. Either the indexer is not a large enough
share of per-token cost to matter, or the saving is offset by mechanism (3) below. **The
hypothesis is falsified; I am not going to quietly restate it as a win.**

**Result 3 — deeper drafting does NOT pay off, and the acceptance data says why.** n=5 is
no better than n=1 at low concurrency (1.25× vs 1.24×) and **worse** at c=64 (0.91× vs
0.99×). Mechanism: GLM ships **one** MTP layer (`num_nextn_predict_layers: 1`) run
**recurrently** 5×, so drafts 2–5 are conditioned on the draft head's *own* predictions and
error compounds. Per-token acceptance collapses **71.8% → 30.6%**, so expected accepted
length is:

```
n=1:  Σ p^i, p=0.718, i=1..1  =  0.72 accepted per 1 draft   -> 72% of verify work useful
n=5:  Σ p^i, p=0.306, i=1..5  =  0.44 accepted per 5 drafts  ->  9% useful, 4.6 wasted
```

So n=5 does **5× the verify work for less accepted output than n=1**. It also costs KV:
concurrency @256K falls 8.01× (base) → 6.34× (n=1) → 6.16× (n=5).

> **This is the answer to "can spec decode dramatically rescue a low-throughput model?"**
> Only at low concurrency, and **only at a draft depth its acceptance rate can sustain.**
> The recipe's `num_speculative_tokens: 5` is *worse than n=1 on this model on H100* — a
> concrete case where following vendor defaults without measuring costs you throughput.
> **[H]** The recipe targets GB200, where the compute/bandwidth ratio and larger HBM shift
> the break-even; the setting may well be right there. Untested here.

**Honest caveat:** the first c=4 n=1 run measured 94.7 tok/s (0.42×) — an outlier with p99
TTFT 12.3 s vs median 1.7 s, i.e. transient queueing. Rerun gave 239.8 tok/s with p99 2.4 s.
Both were "cold" and completed 8/8; **the cold-run guard does not catch scheduling
artifacts**, so a single anomalous point should be reproduced before it is believed.

---

## 8. What binds throughput, ranked — the summary a PI can challenge

| rank | mechanism | evidence | lever |
|--:|---|---|---|
| 1 | **Serial per-step latency** (45 dependent layers, small GEMMs, launch overhead) | <7% of roofline at c=1 [M]; 4.64× gain from batching alone [M] | batching, cudagraphs, spec decode |
| 2 | **EP all-to-all on the decode critical path** | achieved HBM <5% of peak with EP8 [M]; k=8/E=288 → ~0.22 experts/rank/token [A] | TP/EP layout, DP+EP, larger batch |
| 3 | **Expert weight read** (304.42 of 321.34 B [M]) | per-token bytes 2.6× V4's [A]; gap opens at c=16 as it amortizes [M] | FP4/FP8 experts, batching |
| 4 | **KV capacity** — binding *only* at extreme context | 89.5% at 260K vs 10.2% at 16K [M] | FP8 KV (**1.805× capacity, −15.5% throughput** [M]), KDA layers |
| 5 | **Sparse-attention indexing** (top-k 2048/layer) | GLM pays it 11× vs V4 43× per token [A] | **[H] untested** — needs NVTX trace |

**Highest-leverage architectural features, from this data:**
1. **Recurrent (KDA) layers** — remove O(ctx) KV from 34/45 layers. Turns a 47.81 KiB/token
   model into an 11.69 KiB/token one [A]. Biggest single win.
2. **Sparse attention (DSA/top-k)** — bounds the remaining 11 layers' attention work.
3. **Native FP8 weights** — halves expert bytes vs BF16 (though V4's MXFP4 halves them
   again; **state bytes, not the label "FP8"**).
4. **MTP** — but only in the low-concurrency regime AND only at n=1, per §7b. At the
   recipe's n=5 acceptance collapses to 30.6% and it becomes a net loss at c=64.

**Features that did NOT matter as much as expected:**
- **`index_share_for_mtp_iteration`** — predicted to raise GLM's spec-decode break-even vs
  V4; measured no difference (§7b). A reminder that a config flag naming a real
  optimization does not imply a measurable system-level effect.
- **FP8 KV** — a 1.805× capacity win that costs 15.5% throughput, so it is only correct
  when you are actually KV-bound (ISL ≥ ~130K here).

---

## 9. Open questions I would raise before anyone else does

1. ~~The V4 comparison is engine-confounded~~ — **CLOSED** (§2b). V4 rerun in the GLM
   image; engine effect mean **1.01×**, so the ratios are now architecture results.
2. ~~V4's active count is [A]~~ — **CLOSED**. Both measured from safetensors: GLM
   321.34/17.38 B, V4 290.91/14.08 B (correcting for V4's I8-packed MXFP4 experts).
3. **⚠️ A GUARD GAP, FOUND THE HARD WAY.** Three V4 image points passed *both* existing
   guards (cold + completed) yet under-reported throughput by **25–60%** from scheduler
   queueing on a freshly-started server. Using them, the "engine effect" computed to
   **0.64× ("the image is 36% slower")** — a headline-grade wrong conclusion. Reruns gave
   **1.01×**. Diagnostic: **p99 TTFT ≫ median TTFT**. A concurrency-aware guard is now in
   `bench.sh` (4× for conc≤4, 8× for conc≤16, prefix sweep exempt — a flat 5× threshold was
   useless, firing on nearly every legitimate c=64 point). Evidence:
   `deepseek_v4_flash/results/_v4image_firstrun_queued/`.
3. **No TP sweep.** TP=8 on a model that fits in 5 GPUs adds collectives a smaller TP
   wouldn't. Per-GPU numbers are pessimistic for both, and §3's prediction (iii) is untested.
4. **Achieved-bandwidth numbers are [A]**, derived from active params × dtype. `--enable-mfu-metrics`
   is on but I did not scrape `estimated_read_bytes_per_gpu_total` per point to get a
   **measured** fraction. That would upgrade §1 from analytical to measured and is cheap.
5. **Tokenizers differ** (GLM vocab 154,880 vs V4 ~129k), so tok/s is not strictly
   commensurable across families. Nothing here corrects for it; bytes/s on a fixed corpus
   would.
6. ~~The FP8-KV arm swaps the attention backend~~ — **RESOLVED** via the control arm (§5b).
   Backend = −27.3%, dtype = −15.5%. The naive two-arm read would have been −39%.
7. **No PD-disaggregation measurement** — impossible on this node (§6), so the recipe's
   latency-structure claim is untested rather than refuted. It needs a GB200 tray or a
   smaller model.
8. **MTP was measured on the batch axis only.** Spec decode should also interact with
   *context* (longer ctx → more verify cost per token) and that is unrun.
7. **Synthetic random prompts route ~uniformly across experts** — best case for expert
   coverage. Real text has skewed routing, so measured expert cost may be *pessimistic*
   and real-world locality better. Untested here.
