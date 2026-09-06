> HISTORICAL MODEL NOTES — use the [current report](../../report.md), [experiment index](../../experiments.md), and [debugging guide](../../fix_bug.md) for corrected conclusions. Original location: `GLM-5.3-Flash/report.md`. Some causal claims and configurations below are superseded.

# Serving GLM-5.3-Flash on 8×H100: a hybrid KDA/DSA stack measured against DeepSeek-V4-Flash

**Author:** Tan Ngo · **Date:** 2026-09-02 · **For:** Kan Zhu, UW SyFI

**Scope.** The second measured model in this report, and the first one that lets the central claim be
tested rather than asserted. GLM-5.3-Flash is **34 KDA linear-attention layers + 11 DSA sparse-attention
layers in one stack** — so a single decode step of this model touches two layer classes with
*different memory-growth laws*. Everything below is 8×H100, vLLM image dev build, TP8/EP, base model.

**46 measured points**, all cold (0 new prefix-cache hits) and all complete. Seven arms:

| arm | dir | what it isolates |
|---|---|---|
| **BF16 KV — headline** | `results/bf16kv/` | **15 pts**: batch (**8-point curve**), context, prefix |
| FP8 KV (FlashInfer 0.6.18) | `results/fp8kv-fi618/` | 11 pts: the capacity/throughput trade |
| BF16 KV **control** on 0.6.18 | `results/bf16kv-fi618/` | 4 pts: **separates dtype from backend** |
| MTP A/B, n=1 and n=5 | `results/bf16kv-mtp-n1,-n5/` | 8 pts: spec decode on the batch axis |
| **MTP n=1, context axis** | `results/bf16kv-mtp-n1-context/` | **4 pts: spec decode vs CONTEXT — inverts the batch-axis advice (§6b)** |
| **util 0.85 A/B** | `results/util085/` | **4 pts: proves `--gpu-memory-utilization` is capacity-only (§10)** |
| quarantined | `results/_*/` | 7 pts kept as evidence of measurement errors caught, never deleted |

**Labelling.** **[M]** measured on this hardware today · **[A]** analytical (config, tensor shapes,
vendor spec) · never mixed silently. Nothing here measures accuracy — per Kan's email the emphasis is
throughput and serving cost.

---

## 0. Provenance

| | |
|---|---|
| **Hardware** | 8× NVIDIA H100 80GB HBM3 (`tan-8gpus-glm53-0-0`), driver 575.57.08 |
| **Aggregate HBM** | 26.8 TB/s (8 × 3.35 TB/s vendor spec) |
| **Engine** | vLLM `0.1.dev20051+g487ecf187`, torch 2.13.0+cu130, transformers 5.15.1 |
| **Image** | `vllm/vllm-openai:glm53-flash` (**CUDA 13.0** — the `-cu129` tag segfaults, see §7) |
| **Model** | `zai-org/GLM-5.3-Flash`, snapshot `03eb5366`, 305.8 GiB, 62 shards |
| **Parallelism** | TP=8, expert-parallel on (`Local/global experts 36/288`) |
| **Weight dtype** | **FP8 e4m3 for 97.8% of params** on disk; 6.93 B BF16 remainder (§1) |
| **KV dtype** | **BF16** on the headline arm — forced, not chosen (§5) |
| **Resolved block size** | **640** (requested 128, vLLM auto-raised); `mamba_block_size=128` |
| **KV pool** | **2,099,654 tokens**, 23.26 GiB/GPU, 8.01× concurrency at 256K (util 0.82; the 0.85 arm gives 2,131,562 — §6c) |
| **MFU coverage** | ⚠️ `ffn` + `unembed` only — **no `attn`** ComponentMetrics for this hybrid stack, so every bandwidth figure is a **lower bound** (§8 item 5) |
| **Spec decode** | **OFF** on all headline numbers |
| **Harness** | `vllm bench serve`, `--ignore-eos`, **unique seed per point** |
| **Repro** | `cd GLM-5.3-Flash && ./run.sh`, then see `README.md` §Repro |

---

## 1. Parameter accounting — measured, and a correction to an earlier estimate

**[M], from safetensors tensor shapes.** These buckets are mutually exclusive and partition the total
exactly:

| Component | Params | Share | Fires per token? |
|---|--:|--:|---|
| Routed experts (main) | **304.42 B** | 94.7% | only **k/E = 8/288** |
| Always-on attn / dense / embed | **8.92 B** | 2.8% | yes |
| MTP layer | 7.43 B | 2.3% | **excluded** — base model |
| Vision tower (ViT) | 0.56 B | 0.2% | **excluded** — `--limit-mm-per-prompt 0` |
| **Total** | **321.34 B** | | |

**active = 8.92 + 304.42 × 8/288 = 17.38 B** **[M]**

> ⚠️ **This corrects an earlier analytical estimate of 310.96 B / 15.01 B, which was wrong.** It
> under-counted always-on attention (GLM keeps q/k/v/o in BF16 and adds a kpool indexer) and mishandled
> the MTP layer. The error made GLM look **better** per-B-active than it is — i.e. it flattered the
> conclusion I was testing. The model card's 18 B active / 320 B total are the correct figures.
> **Lesson, now a project rule: never hand-derive param counts from `config.json`; read tensor shapes.**

**"Both models are FP8" is too coarse for the term this report is about.** GLM ships **native FP8
weights**: 314.40 of 321.34 B (97.8%) are `F8_E4M3` on disk, with a **1,509-entry
`modules_to_not_convert`** list. The 6.93 B BF16 remainder is q/k/v/o_proj (1.14 B each), embeddings +
lm_head (0.63 B each), `kv_b_proj`, the kpool indexer, the MoE gate, and the ViT. DeepSeek-V4-Flash uses
**MXFP4** experts. So per expert-parameter **GLM reads ~1 byte and V4 ~0.5** — a 2× difference in exactly
the expert-read term the thesis concerns. **Report the bytes, not the label.**

---

## 2. ★ The headline: engine-matched GLM vs DeepSeek-V4-Flash

Both models, **same engine build, same CUDA, same node, same grid, both base, all points cold.** V4 was
re-swept inside the GLM image specifically to close the engine gap (`results/mtp-off-image/`).
ISL 16,384 · OSL 256 · TP8/EP · `--max-num-seqs 256`. All cells **[M]**:

| conc | GLM t/s | V4 t/s | GLM/V4 | GLM /GPU | V4 /GPU | GLM /B-act | V4 /B-act |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 85.0 | **1.13×** | 12.04 | 10.63 | 5.54 | 6.04 |
| 4 | 225.3 | 219.8 | 1.02× | 28.16 | 27.48 | 12.96 | 15.61 |
| 16 | 359.2 | 330.9 | **1.09×** | 44.90 | 41.36 | 20.67 | 23.50 |
| 64 | 447.1 | 389.4 | **1.15×** | 55.89 | 48.67 | 25.72 | 27.65 |

Concurrency scaling 1→64: GLM **4.64×**, V4 **4.58×**.

**The engine effect is measured, not assumed: mean 1.01×** (0.92 / 0.99 / 1.13 / 1.02 per conc) against
the earlier cross-engine V4 baseline. So the ratios above are architecture, not tooling — **but we could
not have known that without running it.**

**The honest reading — and the tension is the result.** GLM wins raw and per-GPU throughput at every
concurrency (1.02–1.15×), yet **loses per-B-active at every concurrency**. It needs *more* active
parameters to get that throughput, so V4 uses its FLOP budget better. Which model is "better" depends
entirely on whether you pay for **GPUs** or for **parameters** — and those two answers disagree here.

### 2b. Throughput is NOT linear in active params, and that deviation is a finding

If decode were purely memory-bound, `tok/s × B-active` would be roughly constant across models on
identical hardware. Measured spread: **+30.8% at c=4 → +57.6% at c=16** (GLM higher). GLM carries **29%
more active params** and reads **~2.5× more expert bytes** (FP8 vs MXFP4) — and is still **faster** at
high concurrency.

So the roofline cost model **mis-ranks these two models**. Consistent with:

- **34 of GLM's 45 layers are recurrent KDA** reading a **fixed-size state** (~0.017 GiB/seq/GPU at TP8),
  not O(ctx) KV. Their cost does not grow with context.
- **Achieved HBM is far below peak — GLM measures 9.5–18.1% of 3.35 TB/s [E]** (⚠️ corrected 2026-09-02
  from an earlier `<5% [A]` estimate; see §8 item 5) — neither model sits at the memory roofline, so
  active-param × bytes is the wrong cost model for this regime.

**EP all-to-all on the decode critical path remains the leading hypothesis for the real bound.** At EP8
with k=8 across 8 ranks, expected experts touched per rank per token is **1.0** — most ranks contribute
0 or 1, so load imbalance is *structural*, and it is a **network** term that bandwidth accounting cannot
see. ⚠️ **Untested:** no TP sweep was run, so this hypothesis is motivated but not confirmed. §8.

---

## 3. Batch axis — full table

ISL 16,384 · OSL 256 · `results/bf16kv/`. All **[M]** (bandwidth column **[E]**):

**Now an 8-point curve** (c=2, 8, 32, 48 added 2026-09-02), so the knee is measured, not assumed:

| conc | out t/s | per B-active | per B-total | per GPU | TTFT p50 | TTFT p99 | TPOT p50 | KV peak | GB/s/GPU [E] |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 5.54 | 0.30 | 12.04 | 857 ms | 886 ms | 7.1 ms | 1.4% | — |
| **2** | **122.2** | 7.03 | 0.38 | 15.28 | 1,341 ms | 4,615 ms | 8.5 ms | 2.7% | **318.6** |
| 4 | 225.3 | 12.96 | 0.70 | 28.16 | 1,742 ms | 2,319 ms | 10.9 ms | 5.1% | — |
| **8** | **296.9** | 17.08 | 0.92 | 37.11 | 2,191 ms | 4,273 ms | 18.1 ms | 10.1% | **542.5** |
| 16 | 359.2 | 20.67 | 1.12 | 44.90 | 2,685 ms | 8,322 ms | 33.6 ms | 20.1% | — |
| **32** | **406.4** | 23.38 | 1.26 | 50.80 | 2,240 ms | 16,375 ms | 68.6 ms | 40.4% | **608.0** |
| **48** | **432.6** | 24.89 | 1.35 | 54.08 | 1,992 ms | 24,406 ms | 100.5 ms | 60.7% | **519.4** |
| 64 | 447.1 | 25.72 | 1.39 | 55.89 | 2,513 ms | 32,606 ms | 129.5 ms | 80.9% | — |

⚠️ **The blank bandwidth cells are deliberate, not missing data.** The MFU harness fix
([`../fix_bug.md`](../../fix_bug.md) bug 13) landed *after* c=1/4/16/64 were collected. The util-0.85 arm
(§6c) does cover exactly those four concurrencies — 347 / 471 / 594 / 448 GB/s — but at a different
`gpu_memory_utilization`, and splicing two arms into one curve is the confound rule 1 exists to prevent.
Blank is the honest entry; re-polling this arm at 1/4/16/64 would fill it.

**64× the concurrency buys 4.64× the throughput and costs 18× the TPOT.** That is the operator's actual
trade, and it is why "tok/s" alone is not a serving number: at c=64 the p99 TTFT is **32.6 s**.

**The knee is at c≈4–8** — 1→8 buys **3.08×**, then 8→64 buys only **1.51×** for 8× the concurrency.
Scaling efficiency `(tok/s ÷ tok/s@c1) ÷ conc` falls 1.00 → 0.385 (c=8) → **0.073** (c=64). On the old
1/4/16/64 grid this trade was invisible and the default conclusion was "use c=64."

**Two things only the finer grid shows:**
- **TTFT is non-monotonic** — it rises to 2,685 ms at c=16 then **falls to 1,992 ms at c=48**. Chunked
  prefill (`max_num_batched_tokens=8192`) packs prefill chunks better at moderate concurrency, so **TTFT
  is not a prefill-compute measurement.**
- **Bandwidth peaks at c=32 (608 GB/s/GPU, 18.1% of 3.35 TB/s) then falls to 519 at c=48, while
  throughput keeps rising** to c=64. **Whatever binds at high concurrency is not HBM.**

Full writeup incl. the V4 comparison: `results/bf16kv/RESULT-8point-batch-curve.md`.

**Per-B-total (0.30 → 1.39) is 18× worse than per-B-active** — that column is the honest cost of
sparsity: you bought 305.8 GiB of HBM to use 17.38 B of it per token.

---

## 4. Context axis — where the hybrid architecture is most visible

conc 8 · OSL 256. All **[M]**:

| ISL | out t/s | TTFT p50 | TPOT p50 | KV peak | t/s vs 16K |
|--:|--:|--:|--:|--:|--:|
| 16,384 | 295.9 | 2.6 s | 16.7 ms | 10.2% | 1.00× |
| 65,536 | 104.3 | 7.2 s | 48.0 ms | 36.6% | 0.35× |
| 131,072 | 47.2 | 12.9 s | 116.7 ms | 72.5% | 0.16× |
| 260,000 | 26.5 | 38.5 s | 147.6 ms | **89.5%** | **0.11×** |

**16× the context costs 9.1× the throughput and 14.7× the TTFT.** Context, not batch size, is the
dominant cost axis — which matters because TraceLab's median real round is **132,092 input tokens**.

**KV capacity DOES become binding for GLM at max context — unlike V4-Flash.** GLM peaks at **89.5%** KV
at 260K×8 where V4 peaked at **30.2%** even at 256K×8. Cause is dtype, not architecture: GLM pays
**~11.35 KiB/tok** BF16 (11 DSA layers × 512 latent × 2 B = 11.00, + kpool indexer ≈ 0.35) where V4 pays
fp8. This is the one place in the comparison where GLM's forced KV dtype has a *capacity* consequence,
not just a throughput one.

⚠️ **Top context point is ISL 260,000, not 262,144.** `max_model_len - OSL` (261,888) still returned
400s: `--dataset-name random` does not emit exactly `--random-input-len` tokens. Bisected to 260,000 =
99.2% of `max_model_len`. See `../fix_bug.md` bug 7.

**Prefix sharing** (64K shared prefix, `results/bf16kv/`): n=1 **265.3** · n=4 **365.9** · n=16 **199.4**
tok/s. The non-monotonicity is real and reproducible — n=4 is the optimum because n=1 serializes on
building the single shared prefix while n=16 dilutes reuse.

---

## 5. FP8 KV: the vendor recipe says it's impossible on Hopper. It isn't.

The vLLM recipe states *"Hopper does not support FP8 KV cache for this model and must run BF16 KV."*
**That statement is true of the shipped image, not of Hopper or of the model.**

The obvious route does fail, for a *geometric* reason: `--kv-cache-dtype fp8_ds_mla` aborts with
`pe_dim must be 64` because that is DeepSeek-V3.2's decoupled-RoPE layout and **GLM is NoPE**
(`qk_rope_head_dim: 0`). No flag can change that. But vLLM ships a **second FP8-KV path built for NoPE
sparse models**: `FLASHINFER_MLA_SPARSE_SM90`, whose requirements GLM satisfies exactly
(`kv_lora_rank==512` ✅, `qk_rope_head_dim in (0,64)` — **0 explicitly allowed** ✅, `index_topk` ✅).
`cuda.py:150-157` even *prefers* it when `qk_rope_head_dim==0`. The only gate is a FlashInfer feature
probe for `ckv_scale_arr`, added in **≥ 0.6.18**; the image ships **0.6.17**.

**MEASURED with 0.6.18 overlaid [M]:**

| | BF16 KV (0.6.17) | FP8 KV (0.6.18) | Δ |
|---|--:|--:|--:|
| KV pool tokens | 2,099,654 | **3,790,580** | **1.805×** |
| concurrency @256K | 8.01× | **14.46×** | **1.805×** |
| out t/s @c=1 | 96.3 | 68.9 | 0.72× |
| out t/s @c=16 | 359.2 | 208.8 | 0.58× |
| out t/s @c=64 | 447.1 | 262.0 | 0.59× |
| correctness | ✓ | ✓ (`2+2=4`, coherent) | |

**FP8 KV trades throughput for 1.81× KV capacity — it is not a free win, and that is the finding.** The
recipe's advice to avoid it on Hopper is *directionally right for throughput* even though its stated
reason (unsupported) is wrong.

**Why 1.805× and not 2.0× [A].** Only the **11 DSA layers'** latent KV is quantized. The **34 KDA
layers' recurrent state stays BF16** (`mamba_cache_dtype=auto`) and is allocated **one block per sequence
regardless of dtype** — a floor FP8 cannot shrink. Per-token KV arithmetic predicts **1.889×**; measured
**1.805×**; the residual is the new backend's larger workspace (23.26 → 22.05 GiB).

### 5b. The control arm — and why it changed the headline

⚠️ **The FP8 arm changed three things at once**: KV dtype, attention backend
(FLASH_ATTN_MLA_SPARSE → FLASHINFER_MLA_SPARSE_SM90), and `moe_backend=deep_gemm`. So the raw
cross-version delta **cannot** be attributed to the dtype. `run_bf16kv_fi618.sh` is the control: same
overlay, same MoE backend, **BF16** KV — verified to select the *same* new backend (1,990,167 KV tokens).

Decomposed at c=64 **[M]**:

| comparison | what it isolates | Δ |
|---|---|--:|
| `bf16kv` → `bf16kv-fi618` | backend + version, dtype held | **−22.0%** (447.1 → 348.8) |
| `bf16kv-fi618` → `fp8kv-fi618` | **dtype alone** | **−24.9%** (348.8 → 262.0) |
| `bf16kv` → `fp8kv-fi618` | naive cross-version read | −41.4% |

**Reported as "FP8 KV costs 41%" this would have been wrong.** The dtype costs ~25%; the backend swap
costs ~22% independently. This is the report's clearest illustration of its own rule 1 — one variable per
arm, always run the control.

⚠️ **Not a production config.** FlashInfer 0.6.18's Python runs against the image's pinned 0.6.17
compiled artifacts (this mirror has no `flashinfer-cubin` > 0.6.13), so
`FLASHINFER_DISABLE_VERSION_CHECK=1` is required, and `moe_backend=deep_gemm` is required to dodge an ABI
mismatch in FlashInfer's fused-MoE. **A capability demonstration, not a deployment recommendation.**

---

## 6. Speculative decoding (MTP): the gain decays to nothing, and n=5 is worse than n=1

GLM-5.3 ships `num_nextn_predict_layers: 1` (a 7.43 B MTP layer) and
`index_share_for_mtp_iteration: true` — the sparse-attention indexer is **shared across MTP iterations**,
a genuine sparse-attn/spec-decode co-design. The recipe suggests `n=5`. Measured, ISL 16,384, all **[M]**:

| conc | base t/s | MTP n=1 | n=1 gain | MTP n=5 | n=5 gain |
|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 119.7 | **1.24×** | 120.8 | **1.25×** |
| 4 | 225.3 | 239.8 | 1.06× | 230.7 | 1.02× |
| 16 | 359.2 | 344.7 | **0.96×** | 342.1 | 0.95× |
| 64 | 447.1 | 441.9 | **0.99×** | 408.3 | **0.91×** |

**Two results a serving system should care about:**

1. **The gain decays to nothing as the batch saturates the machine** — 1.24× at c=1 → 0.99× at c=64.
   Spec decode converts *idle* parallel compute into tokens; at c=64 there is no idle compute to convert,
   and the draft overhead becomes pure cost. **Any single-number "MTP gives 1.2×" claim is a statement
   about concurrency, not about the model.**
2. **The recipe's `n=5` is worse than `n=1`** at every concurrency ≥4, and materially worse at c=64
   (0.91×). Acceptance collapses **71.8% → 30.6%** as n grows, so the extra drafts are computed and
   thrown away. It also costs KV: peak KV at c=64 goes 80.9% (base) → 92.8% (n=1) → **99.2% (n=5)** — at
   n=5 KV is nearly exhausted, which is a *capacity* risk on top of the throughput loss.

This is why the report's rule 1 exists: **headline throughput is base-model only.** Reporting MTP-on
numbers as a model comparison would compare inference tricks, not architectures.

### 6b. ⚠️ On the CONTEXT axis the recommendation INVERTS — MTP buys TTFT, not throughput

Everything above is the **batch axis**. Measured on the **context axis** (2026-09-02, conc 8, MTP n=1,
`results/bf16kv-mtp-n1-context/`) — the first such measurement for any model in this report. The two
metrics move in **opposite directions**:

| ISL | base t/s | MTP t/s | throughput | TTFT base | TTFT MTP | **TTFT change** | KV peak MTP |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 16,384 | 295.9 | **314.8** | **1.064×** | 2,629 ms | 2,101 ms | **−20.1%** | 11.5% |
| 65,536 | 104.3 | 97.6 | 0.936× | 7,212 ms | 5,427 ms | **−24.8%** | 40.6% |
| 131,072 | 47.2 | 45.3 | 0.958× | 12,891 ms | 10,734 ms | **−16.7%** | 78.9% |
| 260,000 | 26.5 | 22.2 | **0.839×** | 38,476 ms | 41,067 ms | +6.7% | **96.9%** |

**MTP makes first-token latency 16.7–24.8% better at every context up to 131K while making sustained
throughput 4–6% worse.** Not a contradiction: chunked prefill interleaves prefill chunks with decode
steps, so a draft head that resolves decode in fewer scheduler iterations lets prefill chunks land
sooner. Aggregate output rate falls; first tokens arrive earlier.

**This flips the advice for agentic serving.** TraceLab's measured median is ISL 132,092 / OSL 249 — an
**ISL:OSL of ~530:1** — so user-visible latency is prefill-dominated. At ISL 131K, MTP costs **4.2% of
throughput** and buys **16.7% of TTFT**: a good trade on that workload, and the opposite of what §6's
batch-axis table implies. **A spec-decode decision made on batch-axis data alone is made on the wrong
axis for this workload.**

**A prediction of mine was falsified.** `index_share_for_mtp_iteration: true` (§0) predicted GLM's MTP
would degrade *less* with context than Qwen's full-attention draft head. On throughput it degrades
**more** as context grows (1.064× → 0.839×). Recorded as a refuted prediction, not a null result.

**The 260K ceiling is KV-bound, not accuracy-bound.** Acceptance held at **69.1%** across the whole arm,
while KV hit **96.9%**. The draft head costs **13.3% of KV capacity** (pool 1,916,967 → 1,662,741 tokens;
max concurrency 7.31× → 6.34×). At 16K–131K there is headroom to absorb that; at 260K there is not.
⚠️ **13.3% is far more than the 2.4% reported for V4's MTP** — do not carry V4's figure over to GLM.

Full writeup: `results/bf16kv-mtp-n1-context/RESULT-mtp-context-axis.md`.

---

## 6c. `--gpu-memory-utilization` 0.82 → 0.85: capacity-only [M]

The last unmeasured provisioning variable, and GLM was the decisive test — it has **the least KV headroom
of the three models** (89.5% peak KV at 260K×8 vs V4's 36.9%), so if util ever moved throughput by
relieving scheduler pressure, it would show here.

| conc | util 0.82 | util 0.85 | ratio | TPOT 0.82 | TPOT 0.85 |
|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 96.3 | **1.000×** | 7.1 ms | 7.0 ms |
| 4 | 225.3 | 223.9 | 0.994× | 10.9 ms | 11.0 ms |
| 16 | 359.2 | 357.6 | 0.996× | 33.6 ms | 35.6 ms |
| 64 | 447.1 | 445.6 | 0.997× | 129.5 ms | 132.0 ms |

**+11.2% KV capacity (1,916,967 → 2,131,562 tokens; max concurrency 7.31× → 8.13×) and no throughput
change** — every point within 0.6%, no trend, TPOT flat. **The control that makes this clean:** peak
activation measured **4.05 GiB in both arms**, with the whole memory accounting identical (39.64 GiB
weights / 4.05 activation / 1.41 cudagraph). Third model to confirm the flag is capacity-only, and the
two clean ones (GLM, V4) agree; Qwen's apparent 1.58× was a cold-compile artifact, not the flag.

⚠️ **The cold-compile hazard reproduced here and was avoided by construction.** `/tmp` caches start empty
on a freshly scheduled container; the session's first launch measured **4.88 GiB** peak activation and
sized KV at 1,838,761 tokens (−4% vs warm). Both A/B arms were therefore run **warm**. Full detail:
`results/util085/RESULT-util-ab.md`.

---

## 7. What did not run, and why (each is a result)

| attempt | outcome | cause |
|---|---|---|
| **TP4 single pool** (the recipe's own example) | ❌ **[M]** OOM | weights = 75.36 GiB/GPU; **1.60 GiB free** at util 0.95. Nothing left for KV |
| **PD-disaggregation TP4+TP4** (recipe example) | ❌ **[A]** | needs **two** weight copies = 612 GiB of 640 GiB node HBM before any KV. The recipe's PD example targets a **GB200 tray** |
| TP=7 on 7 GPUs | ❌ | `num_attention_heads % tp != 0` (64 % 7) |
| `--kv-cache-dtype fp8_ds_mla` | ❌ **[M]** | `pe_dim must be 64`; GLM is NoPE. Unreachable on **any** hardware — §5 |
| `-cu129` image tag | ❌ | TileLang `mhc_post_tilelang` segfaults in `cuModuleLoadData`. **cu130 fixes it** (24 successful compiles, 0 segfaults) |
| `--compilation-config '{"max_cudagraph_capture_size":N}'` | ❌ | **replaces** the whole `CompilationConfig`, silently wiping `pass_config`. Use the dedicated flag |

**Say the PD result to a PI as a memory-hierarchy argument, not a config complaint.** PD disaggregation
is a *latency-structure* optimization — long prefills stop blocking decode steps — bought with a **2×
weight footprint**. That trade is simply unavailable when weights are ~48% of node HBM. It pays off when
weights are small relative to node memory, which is the opposite of this model.

**Four bring-up bugs had to be fixed before any number was trustworthy** — full diagnosis in
[`../fix_bug.md`](../../fix_bug.md). The two most transferable:

- **A `CUDA error: invalid argument` at 69–78% of cudagraph capture was an NFS quota write failure**
  (DeepGEMM JITs into `$HOME/.tensorrt_llm`; `df` shows 622 G free because it reports the volume, not the
  quota). Three plausible CUDA hypotheses were tried and all failed — `--enforce-eager` failing was the
  tell: *if disabling capture doesn't help, the bug was never in capture.*
- **`--num-warmups` self-poisoned the prefix cache.** It draws warmup prompts from the same seeded set as
  the measured run, so every batch point reported exactly 16,000 new hits. The bias is **uneven** — 12.2%
  of the c=1 point but 0.8% of c=64 — so it inflated low-concurrency anchors and **flattened measured
  concurrency scaling**. Flag removed; contaminated points quarantined in
  `results/_discarded-warmup-contaminated/` as evidence rather than deleted.

---

## 8. Caveats that normalization cannot fix

State these with the numbers; they are not fixable by arithmetic.

1. **KV dtype is forced and opposite** on H100. V4 can only run `fp8_ds_mla` (its BF16 path is gated to
   Blackwell); GLM can only run BF16 in the shipped image. Each model can run only the dtype the other
   cannot. **Kernel availability, not a design choice.**
2. **TP=8 handicaps the two models unequally.** V4-Flash fits in 3 GPUs but was measured at TP=8, so its
   per-GPU number is **pessimistic**; GLM genuinely needs ≥5, so TP=8 is closer to a real deployment for
   it. Do not read the per-GPU column as if both were equally handicapped.
3. **Tokenizers differ** (GLM vocab 154,880), so tok/s is not strictly commensurable across families. For
   strict cross-family claims use bytes/s or a fixed corpus.
4. **`max_num_seqs` differs from the original V4 baseline** (GLM pinned 256; V4 ran the H100 default
   1024). **No published point is affected** — both grids top out at conc 64 — but it is an engine-config
   difference and must travel with the numbers.
5. ✅ **Achieved bandwidth is now [E]** — and the earlier estimate was **wrong by an order of magnitude**.
   `--enable-mfu-metrics` was passed on every arm but nothing read its counters until 2026-09-02;
   `bench.sh` now scrapes them per point. The hand-rolled `active_params × bytes × steps/s` estimate said
   **<5%**; vLLM's per-step accounting measures **9.5–18.1% of 3.35 TB/s**. Two reasons this is **[E]**,
   not [M]: (a) it is an **engine-side estimate** — config shapes × measured batch composition, not a
   hardware counter; (b) ⚠️ **for GLM it excludes attention entirely.** vLLM instantiated only `ffn` and
   `unembed` ComponentMetrics for this model — **no `attn`** — because both of its attention estimators
   gate on the whole-model `is_deepseek_mla` boolean and a hybrid stack satisfies neither
   (`perf.py:403-408`, `:551-554`). **So 18.1% is a lower bound, and it is not comparable to V4's or
   Qwen's raw counters, whose coverage differs.** `fix_bug.md` bug 13.
   *That vLLM cannot cost-model a hybrid attention stack — `perf.py:428`: "TODO: discern cases where we
   have mixture of different attention layer types" — is itself evidence for §9.*
6. **No TP sweep**, so the EP-all-to-all hypothesis in §2b is motivated but **unconfirmed**.
7. **Synthetic random prompts route ~uniformly across experts** — best case for expert coverage. Real
   text has correlated routing, so measured expert-read cost may *exceed* what a trained router produces.

---

## 9. What this model contributes to the report's thesis

The thesis: *2026 models are internally heterogeneous within one transformer stack, so the uniform
per-layer cost model every scheduler and paged-KV allocator assumes is no longer true of a single model.*

GLM-5.3-Flash is the cleanest single-model evidence for it:

- **One decode step touches two memory-growth laws.** 34 KDA layers read a fixed-size recurrent state
  (O(1) in context); 11 DSA layers read O(ctx) latent KV. The engine must satisfy a `MambaSpec` and an
  `MLAAttentionSpec` **simultaneously** — and its resolution is visible in the logs: it auto-raised
  `block_size` 128 → **640** so the attention page ≥ the mamba page, then **padded the mamba page by
  20.75%**. That padding is the uniform-block-size assumption failing, quantitatively, in production
  code.
- **The recurrent state adds a memory axis pure-attention models don't have.** It scales with
  `max_num_seqs`, not context — which is why `--max-num-seqs 1024` (the H100 auto-default) fails at
  startup with `exceeds available Mamba cache blocks (512)`. A scheduler tuned on attention-only models
  gets this wrong by default.
- **FP8 KV could only shrink 11 of 45 layers**, giving 1.805× instead of 2× (§5). A per-model "KV
  bytes/token" scalar cannot express that; you need per-layer-class accounting.
- **And the cost model mis-ranks the models** (§2b): GLM reads ~2.5× more expert bytes and carries 29%
  more active params, yet is faster at high concurrency. Whatever binds decode here, it is **not** HBM
  bandwidth — which is precisely why heterogeneity is a *scheduling and memory-management* problem rather
  than a kernel problem.

**Ranked, what binds throughput in these runs [M] except where noted:**

1. **Context length** — 9.1× throughput spread 16K→260K, the largest single lever measured.
2. **Prefix sharing** — up to 1.38× at the n=4 optimum; TraceLab says 95.6% of real input tokens are
   prefix, so this is under-exploited in practice.
3. **Concurrency** — 4.64× for 64×, bought with 18× TPOT.
4. **KV dtype** — 25% throughput for 1.81× capacity (§5b, decomposed).
5. **Spec decode** — 1.24× at c=1, nothing at c=64.
6. **HBM bandwidth** — **[E]** 9.5–18.1% of peak (attention excluded, so a lower bound). Not the binding
   constraint at any measured point, and it **peaks at c≈16–32 then FALLS while throughput rises to c=64.**
