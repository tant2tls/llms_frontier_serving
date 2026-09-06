> ARCHIVED SESSION RECORD — not current instructions or conclusions. Original location: `report_previous.md`. See [current report](../../report.md), [debugging guide](../../fix_bug.md), and [session handoff](../../WORKFLOW.md). Historical claims may be superseded.

# Previous report — retained for provenance; superseded by report.md

**Measured comparison of GLM-5.3-Flash, DeepSeek-V4-Flash, and Qwen3.8-Flash-Next-FP8 on 8×H100**

**Author:** Tan Ngo · **Date:** 2026-09-02 · **For:** Kan Zhu, UW SyFI

---

## The claim

> **The uniform per-layer cost model that every scheduler and paged-KV allocator assumes — one block
> size, one bytes-per-token, one bottleneck per decode step — is no longer true of a *single* model.**
> A 2026 decode step touches layers with different memory-growth laws, different bytes-per-parameter,
> and different cost classes, inside one transformer stack. Heterogeneity moved *inside* the layer
> stack, which makes it a **scheduling and memory-management problem, not a kernel problem**.

M\* (Kasikci & Wang, June 2026) argues serving systems break because they model inference as *"a single
autoregressive loop"* on a *"flat DAG"*, and answers with the Walk Graph at the **inter-component**
level. **This report's finding is that M\*'s premise now holds one level lower than M\* addresses it.**

**115 measured points**, three models, one node, all cold and complete (some are superseded pre-bridge
V4 arms, kept for the record). Plus **12 quarantined** as evidence of measurement errors caught (§7).

**Labelling on every number: [M]** measured on this hardware · **[A]** analytical (config, tensor
shapes, vendor spec) · **[H]** hypothesis, not yet tested. Never mixed silently. **Nothing here
measures accuracy** — per the assignment, the emphasis is throughput and serving cost.

---

## 0. What was measured

| | GLM-5.3-Flash | DeepSeek-V4-Flash | Qwen3.8-Flash-Next-FP8 |
|---|---|---|---|
| **Layer stack** [A] | 34 KDA + 11 DSA | 43 DSA (+compression) | **36 GDN + 12 QSA** |
| **E / k / (E/k)** [A] | 288 / 8 / 36× | 256 / 6 / 42.7× | **512 / 10 / 51.2×** |
| **Total params** | **321.34 B [M]** | **290.91 B [M]** | **176.94 B [M]** served |
| **Active (GEMM path)** | **17.38 B [M]** | **14.08 B [M]** | **7.27 B [M]** |
| **Expert dtype** | FP8 e4m3 (~1 B/param) | **MXFP4 (~0.5 B/param)** | FP8 e4m3 (~1 B/param) |
| **KV KiB/token** [A] | 11.35 | **~7.4** | **24.75** |
| **KV dtype on SM90** | BF16 only | **fp8_ds_mla only** | BF16 only |
| **Weights on disk** | 305.8 GiB | 148.6 GiB | 172.8 GiB |
| **Min H100s** [A] | 5 | 3 | 3 |
| **Points measured** | **46** | 50 | 19 |

All runs: 8× H100 80GB HBM3, driver 575.57.08, TP8 + expert parallel, `vllm bench serve`,
`--ignore-eos`, **unique seed per point**, spec decode **off** on all headline numbers, vision towers
disabled for text-to-text fairness. **Engine matched across all three** — see §7.

Per-model detail: [`GLM-5.3-Flash/report.md`](../../GLM-5.3-Flash/report.md) ·
[`deepseek_v4_flash/report.md`](../../deepseek_v4_flash/report.md) ·
[`Qwen3.8-Flash-Next-FP8/report.md`](../../Qwen3.8-Flash-Next-FP8/report.md) ·
mechanism: [`WHY.md`](../../archive/notes/WHY.md) · reproduction: [`REPRODUCE.md`](../../archive/notes/REPRODUCE.md) ·
what broke: [`fix_bug.md`](../../fix_bug.md)

---

## 1. Q1 — "For various workloads and at different batch sizes, which part of the model is the bottleneck?"

### The short answer: not the part anyone predicts, and it changes with the axis

**Batch axis** (ISL 16,384 · OSL 256), all **[M]**:

| conc | Qwen3.8 | GLM-5.3 | V4-Flash | Qwen TPOT | GLM TPOT | V4 TPOT |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | **106.2** | 96.3 | 85.0 | 7.0 ms | 7.1 ms | 7.9 ms |
| 4 | **256.8** | 225.3 | 219.8 | **9.6 ms** | 10.9 ms | 10.7 ms |
| 16 | **420.2** | 359.2 | 330.9 | **26.3 ms** | 33.6 ms | 32.4 ms |
| 64 | **517.7** | 447.1 | 389.4 | 110.1 ms | 129.5 ms | 148.5 ms |

Concurrency scaling 1→64: **Qwen 4.87× · GLM 4.64× · V4 4.58×**, all costing **15–19× worse TPOT**.
**Qwen wins at every concurrency** (1.10–1.17× over GLM, 1.17–1.33× over V4).

### The bottleneck is a serial latency chain, not memory bandwidth

**Achieved HBM bandwidth as a fraction of the 26.8 TB/s aggregate peak [A]** — batched accounting (at
concurrency *c* one weight read serves the whole batch, so `steps/s = tok/s ÷ c`):

| model | weight bytes/step | c=1 | c=4 | c=16 | c=64 |
|---|--:|--:|--:|--:|--:|
| Qwen3.8 | 6.77 GiB | 2.88% | 1.74% | 0.71% | **0.22%** |
| GLM-5.3 | 16.19 GiB | 6.25% | 3.65% | 1.46% | **0.45%** |
| V4-Flash | 7.21 GiB | 2.46% | 1.59% | 0.60% | **0.18%** |

Adding the KV re-read term at c=64 brings Qwen to ~1.0% and GLM to ~0.8%. **All three models decode at
roughly 1% of the memory roofline** *by this hand-rolled estimate* — the textbook story
(*decode is memory-bound, throughput ∝ 1/active_bytes*) **does not describe this hardware regime at all.**

> ⚠️ **CORRECTED 2026-09-02 (session 3): the ~1% figure is too low by an order of magnitude.** The table
> above is a hand-rolled `active_params × bytes × steps/s` estimate. vLLM's own per-step accounting
> (`--enable-mfu-metrics`, finally scraped into the result JSONs — `fix_bug.md` bug 13) measures
> **GLM-5.3-Flash at 9.5%–18.1% of 3.35 TB/s** across the batch curve, peaking mid-curve at c=16–32:
>
> | conc | 2 | 8 | 32 | 48 | | 1 | 4 | 16 | 64 |
> |---|--:|--:|--:|--:|:--|--:|--:|--:|--:|
> | | *util 0.82 arm* ||||| *util 0.85 arm* ||||
> | GB/s/GPU **[E]** | 319 | 543 | **608** | 519 | | 347 | 471 | 594 | 448 |
> | % of 3.35 TB/s | 9.5 | 16.2 | **18.1** | 15.5 | | 10.4 | 14.1 | 17.7 | 13.4 |
>
> ⚠️ **The two halves are different arms and must not be spliced into one curve.** The MFU harness fix
> landed mid-session, so within the published `bf16kv` arm only the four *newly measured* concurrencies
> (2/8/32/48) carry counters; 1/4/16/64 predate the fix and read `-`. The util-0.85 arm was measured
> after the fix, so it covers 1/4/16/64 — but at a **different `gpu_memory_utilization`**. Since util is
> measured to be capacity-only for throughput (§4b), the two are *comparable in shape*, but quoting them
> as a single 8-point curve would violate the one-variable-per-arm rule this report enforces elsewhere.
> **Re-poll `bf16kv` at 1/4/16/64 to get a true single-arm curve.**
>
> And these figures **exclude attention entirely** for GLM — vLLM instantiated only `ffn` + `unembed`
> ComponentMetrics for this model, so 18.1% is a **lower bound**. Label these **[E]** (engine-side
> estimate: config shapes × measured batch composition), not [M].
>
> **The conclusion below is unchanged and the correction strengthens one part of it:** 18% is still far
> from the roofline, decode is still not bandwidth-bound, and critically **bandwidth peaks at c≈16–32 and
> then FALLS while throughput keeps rising to c=64** — so whatever binds at high concurrency is
> demonstrably not HBM (if it were, tok/s could not climb while bytes/s declines). What changes is that "1%" overstated how much headroom is idle. Do not quote ~1% anywhere.

**The decisive evidence is that the ranking inverts the prediction** — and note this evidence is
independent of the absolute percentage above. Qwen3.8 reads **6.77 GiB/step**
against GLM's **16.19** — 2.4× fewer bytes — and is **16% faster at c=64**. If the expert read were
binding, that is impossible. Meanwhile GLM carries **29% more active params** than V4 and reads ~2.5×
more expert bytes, and is **1.15× faster**. `tok/s × B-active` should be constant if decode were
bandwidth-bound; measured spread is **+31% to +58%**.

**What actually binds:** a decode step is 43–48 *dependent* layer-steps, each a small GEMM plus an EP
all-to-all whose latency is set by the slowest rank. Per Little's law, achieved bandwidth ≈ (bytes per
step) ÷ (latency per step), and that latency is floored by kernel-launch and collective overhead, not
HBM. This is why throughput rises 4.6–4.8× from c=1→64 while per-token bytes are unchanged — **you are
filling idle time, not buying bandwidth.** The measured plateau at c≈16–32 above is direct support:
bytes/s stops improving while tok/s does not.

**Qwen3.8 sharpens the mechanism because its sparsity is fine-grained [H].** With
`moe_intermediate_size: 640` and 512 experts over 8 EP ranks, k=10 routed experts per token means
**~1.25 experts per rank per token** — most ranks do near-zero useful work, so all-to-all sits on the
critical path of every step. Consistent with its elevated p99/median TTFT (2.98× at c=4 vs GLM's 1.33×).
⚠️ **This remains [H]:** confirming it needs a TP/EP sweep or an Nsight trace attributing time in
NanoFlow's dense/attention/network/other taxonomy. Neither was run (§8).

### The bottleneck moves as the workload changes

**Context axis** (conc 8 · OSL 256), all **[M]**:

| ISL | Qwen3.8 | Qwen KV | GLM-5.3 | GLM KV | V4-Flash | V4 KV |
|--:|--:|--:|--:|--:|--:|--:|
| 16,384 | **322.7** | 7.2% | 295.9 | 10.2% | 104.5 | 18.1% |
| 65,536 | **114.5** | 28.2% | 104.3 | 36.6% | 47.9 | 21.9% |
| 131,072 | **56.5** | 56.1% | 47.2 | 72.5% | 34.7 | 27.0% |
| 260,000 | 25.1 | **97.3%** | 26.5 | **89.5%** | 16.7 | 36.9% |

**Context is the dominant cost axis — 12.9× / 11.2× / 6.3× throughput decay for 16× context** — and it
is where the three architectures separate most. At max context the bottleneck has *moved* to KV
capacity for two of the three models, for **two different reasons**:

- **Qwen3.8 (97.3%)** — its 12 full-attention layers are *uncompressed* real attention (2 KV heads ×
  256 head_dim), the highest KV cost in the set.
- **GLM-5.3 (89.5%)** — its KV dtype is *forced* to BF16 by kernel availability.
- **V4-Flash (36.9%)** — the only one with headroom, via **per-layer** `compress_ratios` {4,128} + fp8.

So: **at short context the bottleneck is per-step latency; at long context it becomes KV capacity; and
which model hits the wall first depends on a mechanism (compression vs dtype vs uncompressed heads) that
a single "KV bytes/token" scalar cannot express.**

**Prefix axis** (64K shared prefix + 2K unique suffix), all **[M]**:

| n prefixes | Qwen3.8 | GLM-5.3 | V4-Flash |
|--:|--:|--:|--:|
| 1 (max sharing) | **453.9** | 265.3 | 198.2 |
| 4 | 398.1 | 365.9 | 391.6 |
| 16 | **262.2** | 199.4 | 114.6 |

**Prefix sharing is the single largest lever available to an operator** — up to **2.29×** (Qwen vs V4 at
n=1) and **3.7× within V4 itself**. Given that TraceLab (Kan's own lab's trace) shows **95.6% of real
input tokens are prefix** with only **57.6% provider-reported cache coverage**, roughly **38 points of
reuse are unrealized in production**.

---

## 2. Q2 — "If implementing prefix cache, what are the challenges?"

Five challenges, four of them measured here.

### (a) The allocator has to satisfy contradictory page-size constraints — and it shows

vLLM must satisfy a `FullAttentionSpec`, an `MLAAttentionSpec` (compressed indexer keys) **and**
`MambaSpec`s (recurrent state) simultaneously. Left unpinned, the *same allocator in the same engine
family* resolved **[M]**:

| model | resolved `block_size` | mamba page | why |
|---|--:|--:|---|
| **Qwen3.8-Flash-Next** | **4** | 16 | LCM across 4 cache kinds; QSA ring needs `% 4 == 0` |
| **GLM-5.3-Flash** | **640** | 128 | auto-raised from the requested 128 so attention page ≥ mamba page, **then padded the mamba page by 20.75%** |
| DeepSeek-V4-Flash | 256 (**mandatory**) | — | `sparse_mla.py:53` returns a single-element list `[256]`; `storage_block_size = block_size // compress_ratio` floors to 0 at 64 |

**Two hybrid models of the same generation force the same allocator to page sizes 160× apart, and one of
them pays 20.75% padding waste.** That padding *is* the uniform-block-size assumption failing,
quantitatively, in production code. A prefix cache keyed on fixed-size blocks cannot be simultaneously
efficient for both.

### (b) Recurrent state does not participate in prefix caching at all

36 of Qwen3.8's 48 layers and 34 of GLM-5.3's 45 hold a **fixed-size recurrent state** instead of
per-token KV. That state is a *function of the whole prefix*, not a per-token array, so **it cannot be
sliced, shared, or partially reused** the way KV blocks can. Reusing a cached prefix still requires
either re-running the recurrent layers or storing their state per unique prefix.

It is also a **memory axis that scales with `max_num_seqs`, not context** [A]:

| model | recurrent state | at 256 seqs, TP8 |
|---|--:|--:|
| Qwen3.8 (36 GDN) | 0.1055 GiB/seq | **3.38 GiB/GPU** |
| GLM-5.3 (34 KDA) | 0.0166 GiB/seq/GPU | ~4.2 GiB/GPU |

**This breaks schedulers tuned on attention-only models, by default, on this hardware [M]:** the H100
auto-default is `max_num_seqs=1024` (not the documented 128 — `get_batch_defaults()` gives any non-A100
GPU ≥70 GiB 1024), and both hybrid models **fail at startup** with a Mamba-cache capacity error. Both
needed `--max-num-seqs 256` pinned.

### (c) Benchmarking a prefix-caching engine is itself a trap — measured twice

With `enable_prefix_caching=True` (the default), **a fixed seed makes every concurrency point replay the
earlier points' prompts.** Measured artifact: a **phantom 640 tok/s at c=4** that collapsed to 214 with a
fresh seed — a **3× overstatement**. Separately, `--num-warmups` drew warmup prompts from the *same*
seeded set, so every batch point reported exactly 16,000 new hits; the bias was **uneven** (12.2% of the
c=1 point but 0.8% of c=64), so it **inflated low-concurrency anchors and flattened measured concurrency
scaling**. **Benchmarking a prefix-caching engine with a fixed prompt set measures your cache, not your
model.** Every point in this report asserts `newcachehits == 0`.

### (d) Sharing is not monotone in reuse

Qwen3.8 peaks at n=1 (453.9) but GLM and V4 both peak at **n=4** (365.9, 391.6), not n=1. With a single
shared prefix the first request must *build* it while everything else waits — head-of-line blocking on
cache fill. So maximum theoretical reuse is not maximum throughput, and the optimum depends on the
engine's fill behaviour, not just the workload.

### (e) TTFT is not a clean prefill measurement — so cache-hit accounting is muddied

Chunked prefill is **on by default with `max_num_batched_tokens=8192`**, so a 131K prompt is split into
16 chunks that **interleave with other requests' decode steps**. TTFT therefore includes scheduler
round-trips and queueing, and cannot be read as prefill compute. Sweeping this knob is untested and is
plausibly the largest server-side lever for long-context work (§8).

---

## 3. Q3 — "How does speculative decoding affect the performance?"

**Measured A/B, base vs MTP, identical grid, all cold [M].** Gains as ratio to each model's own base:

| conc | Qwen3.8 n=1 | GLM-5.3 n=1 | GLM-5.3 n=5 | V4-Flash n=1 |
|--:|--:|--:|--:|--:|
| 1 | **1.17×** | **1.24×** | **1.25×** | **1.25×** |
| 4 | **0.92×** | 1.06× | 1.02× | 1.09× |
| 16 | **0.87×** | 0.96× | 0.95× | 1.03× |
| 64 | **0.93×** | 0.99× | **0.91×** | 0.99× |

### Five findings

**1. The gain decays to nothing — or below — as the batch saturates the machine.** Speculative decoding
converts *idle* parallel capacity into tokens. At c=64 there is no idle capacity to convert and the draft
becomes pure overhead. **"MTP gives 1.2×" is not a property of a model; it is a property of a
*(model, concurrency)* pair.** At c=1 it is a genuine **latency** feature (Qwen TPOT 7.0 → 5.6 ms, −20%);
on a loaded server it is a cost.

**2. More draft tokens is worse, not better.** GLM's recipe suggests n=5; measured, n=5 is worse than
n=1 at every concurrency ≥ 4 and materially worse at c=64 (0.91×) because **acceptance collapses
71.8% → 30.6%** — the extra drafts are computed and discarded.

**3. Draft-head architecture decides where the crossover lands, and Qwen's is mismatched.** Qwen3.8 is
the only model that goes *negative* by c=4, and the config says why: its draft is
`mtp: {hybrid: true, layer_types: ["full_attention"], num_hidden_layers: 1}` — **a full-attention draft
head over a target whose 36 of 48 layers are linear-attention.** The draft pays O(ctx) KV to predict for
layers that pay O(1) state. GLM avoids exactly this with `index_share_for_mtp_iteration: true` (indexer
shared across MTP iterations — genuine sparse-attn/spec-decode co-design). Qwen has no such sharing.
**Measured acceptance: 57.6% [M].**

**4. It is not free in memory.** MTP costs KV capacity: **Qwen −12.0%** (2,048,645 → 1,803,660 tokens),
V4 −2.4%, **GLM −13.3%** (1,916,967 → 1,662,741 tokens). At GLM n=5, peak KV at c=64 reaches **99.2%** —
nearly exhausted, a *capacity* risk stacked on a throughput loss.

### ⚠️ 5. On the CONTEXT axis the recommendation INVERTS — MTP buys TTFT, not throughput

All of the above is the **batch axis**. Measured on the **context axis** for the first time
(GLM-5.3-Flash, conc 8, MTP n=1, `GLM-5.3-Flash/results/bf16kv-mtp-n1-context/`) — and the two metrics
move in **opposite directions**:

| ISL | base t/s | MTP t/s | throughput | TTFT base | TTFT MTP | **TTFT change** |
|--:|--:|--:|--:|--:|--:|--:|
| 16,384 | 295.9 | 314.8 | **1.064×** | 2,629 ms | 2,101 ms | **−20.1%** |
| 65,536 | 104.3 | 97.6 | 0.936× | 7,212 ms | 5,427 ms | **−24.8%** |
| 131,072 | 47.2 | 45.3 | 0.958× | 12,891 ms | 10,734 ms | **−16.7%** |
| 260,000 | 26.5 | 22.2 | 0.839× | 38,476 ms | 41,067 ms | +6.7% |

**MTP makes first-token latency 16.7–24.8% better at every context up to 131K while making sustained
throughput 4–6% worse.** Not a contradiction: because chunked prefill interleaves prefill chunks with
decode steps, a draft head that resolves decode in fewer scheduler iterations lets prefill chunks land
sooner. Aggregate output rate falls; first tokens arrive earlier.

**This flips the advice for agentic serving.** TraceLab's measured median is ISL 132,092 / OSL 249 — an
**ISL:OSL of ~530:1**, so user-visible latency is dominated by prefill. At ISL 131K, MTP costs **4.2% of
throughput** and buys **16.7% of TTFT**. On a 530:1 workload that is a *good* trade — the opposite of what
the batch-axis A/B alone implies. **A spec-decode decision made on batch-axis data alone is made on the
wrong axis for this workload.**

**A predicted mechanism was falsified.** `index_share_for_mtp_iteration: true` predicted GLM's MTP would
degrade *less* with context than Qwen's full-attention draft head. On throughput it degrades *more* as
context grows (1.064× → 0.839×). The 260K point is **KV-bound, not accuracy-bound**: it hits **96.9% peak
KV** while acceptance held at **69.1%** across the whole arm. **MTP's context ceiling is set by KV
capacity**, which is exactly where its −13.3% capacity cost stops being amortizable.

⚠️ The context axis is measured for **GLM only**; V4 and Qwen have batch-axis MTP arms only.

**Why headline numbers are base-model only:** comparing an MTP-accelerated model to a base model compares
*inference tricks*, not architectures, and flatters whichever vendor shipped a draft head. Had headline
numbers been taken MTP-on, Qwen would have looked worse than it is at every concurrency but one.

---

## 4. Q4 — "What is the serving cost comparison between these models?"

**Raw tok/s is never a valid cross-model comparison** — minimum deployable footprint across this
generation spans **21×** (GLM-4.7-Flash 1 GPU → Kimi-K3 ~21 GPUs), so raw throughput mostly reports how
much silicon was used. All three normalizations, at c=64 / ISL 16K, all **[M]**:

| model | tok/s | **per GPU** | **per B-active** | **per B-total** | min H100s [A] |
|---|--:|--:|--:|--:|--:|
| **Qwen3.8-Flash-Next** | **517.7** | **64.71** | **71.21** | **2.93** | 3 |
| GLM-5.3-Flash | 447.1 | 55.88 | 25.72 | 1.39 | 5 |
| DeepSeek-V4-Flash | 389.4 | 48.67 | 27.65 | 1.34 | 3 |

**At $2.50/GPU-hour and TP8: Qwen $10.73 · GLM $12.43 · V4 $14.27 per million output tokens [A].**

### The three metrics disagree, and that disagreement is the answer

- **Per GPU** is the only number that maps to cost. Qwen wins (1.16× GLM, 1.33× V4).
- **Per B-active** asks whether the architecture's FLOP budget is used well. **Qwen wins by 2.6–2.8×**
  — E/k = 51.2× with a 7.27 B active budget genuinely converts fine-grained sparsity into efficiency.
- **Per B-total** penalizes sparsity by design. GLM and V4 are within 4% of each other; Qwen leads only
  because it is the smallest checkpoint.

**Between GLM and V4 the metrics invert outright:** GLM wins raw and per-GPU at every concurrency
(1.02–1.15×) but **loses per-B-active at every concurrency**. Which is "better" depends entirely on
whether you pay for **GPUs** or for **parameters** — and no single scalar resolves it.

### Cost at realistic context is a different ranking again

TraceLab's median real round is **132,092 input tokens**. At ISL 131,072 / conc 8 **[M]**:

| model | tok/s | TTFT p50 | KV peak |
|---|--:|--:|--:|
| Qwen3.8 | **56.5** | 19.8 s | 56.1% |
| GLM-5.3 | 47.2 | **12.9 s** | 72.5% |
| V4-Flash | 34.7 | 26.7 s | **27.0%** |

**A 23× cost spread is driven by context and prefix sharing, not by batch size or model choice.** The
operator lever with the largest measured effect is not which model you pick — it is whether you exploit
the 95.6% prefix structure that agentic workloads already have.

### What the FP4-vs-FP8 difference actually buys

⚠️ **"Both models are FP8" is too coarse for the term that matters.** GLM is **97.8% native FP8 e4m3**
(~1 byte/expert-param) with a 1,509-entry `modules_to_not_convert` list; V4 uses **MXFP4** experts (~0.5
byte). That is a **2× difference in exactly the expert-read term** the sparsity thesis is about — and it
still does not determine the ranking (§1). **FP4 halves byte traffic but does not move the roofline
ridge point:** `fp4_gemm_kernel` unpacks FP4→FP8 in shared memory and runs H100's FP8 tensor cores.
There is no FP4 hardware on Hopper. **Report the bytes, not the label.**

---

## 4b. `--gpu-memory-utilization`: capacity-only — measured on ALL THREE models

**Question: does this flag affect performance?** Measured on all three, and the answer has two parts.

### On DeepSeek-V4-Flash and GLM-5.3-Flash, with a warm compile cache: the clean controls [M]

util 0.82 → 0.85, everything else byte-identical:

| conc | V4 0.82 | V4 0.85 | ratio | **GLM 0.82** | **GLM 0.85** | **ratio** |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 84.9 | 92.9 | 1.09× | **96.3** | **96.3** | **1.000×** |
| 4 | 219.4 | 214.8 | 0.98× | **225.3** | **223.9** | **0.994×** |
| 16 | 330.0 | 323.4 | 0.98× | **359.2** | **357.6** | **0.996×** |
| 64 | 386.4 | 384.9 | 1.00× | **447.1** | **445.6** | **0.997×** |

**V4:** +3.7% utilization → +7.0% KV capacity (1,487,070 → 1,590,723 tokens) → no throughput change
(0.98–1.09×, scattered around 1.0, no trend). Peak activation **2.32 GiB in both arms**.

**GLM (added 2026-09-02):** → **+11.2% KV capacity** (1,916,967 → 2,131,562 tokens; max concurrency
7.31× → 8.13×) → **no throughput change**, and tighter than V4's: every point within **0.6%**, no trend,
TPOT flat. Peak activation **4.05 GiB in both arms**, and the *entire* memory accounting identical
(39.64 GiB weights / 4.05 activation / 1.41 cudagraph). **GLM was the decisive test** because it has the
least KV headroom of the three (89.5% peak KV at 260K×8 vs V4's 36.9%) — if util ever relieved scheduler
pressure enough to move throughput, it would show here. It does not.

**The flag buys KV capacity and nothing else.** Two independent models, one of them KV-constrained.

### On Qwen3.8, with a cold cache: it looked like a 1.58× throughput knob — and that was a bug

| conc | util 0.85 | util 0.82 | ratio |
|--:|--:|--:|--:|
| 1 | 107.7 | 106.2 | 0.99× |
| 4 | 162.2 | **256.8** | **1.58×** |
| 16 | 371.0 | **420.2** | 1.13× |
| 64 | 517.8 | 517.7 | 1.00× |

**Lowering utilization produced 1.56× MORE KV** — the inversion that proves the flag wasn't the variable.
vLLM sizes the pool as `util × total − weights − peak_activation − cudagraph` and **measures
`peak_activation` at startup**; here that overlapped a cold `torch.compile`:

| | Qwen cold compile | Qwen warm | V4 (both arms) |
|---|--:|--:|--:|
| **peak activation measured** | **17.07 GiB** | **0.99 GiB** | 2.32 GiB |
| compilation time | 60.41 s | 1.03 s | ~1 s |

**17× overestimate on identical weights**, so vLLM reserved ~14 GiB/GPU of KV it never needed and its own
startup log said so (*"Replace gpu_memory_utilization config with `--kv-cache-memory=…` to fully utilize
gpu memory"*). Only **mid-concurrency** is KV-pressure-sensitive — c=1 has nothing to schedule, c=64
saturates either way — which is exactly the regime where a capacity artifact impersonates an architecture
effect. **Audited all 16 GLM and V4 startup logs: peak activation 2.03–4.27 GiB, all sane. The artifact
was isolated to that one Qwen arm.**

**For an operator:** don't tune this flag for throughput. Set it high enough that KV isn't binding, prefer
**`--kv-cache-memory`** to take profiling off the critical path, and **warm the compile cache before any
run that sizes KV** — otherwise the number you publish is a function of your cache state.

⚠️ **This corrected a published finding.** The old Qwen c=4 number made Qwen look 0.72× GLM, and **I had
already written an architectural explanation for it** (fine-grained routing overhead unamortized at low
concurrency). Coherent, and wrong. Corrected, Qwen wins at every concurrency. See §7 row 5.

---

## 4c. Resolving the batch axis: the 4-point grid hid the knee

`1/4/16/64` is too coarse to locate saturation. Full 8-point curve on V4-Flash, all **[M]**, all cold:

| conc | t/s | per GPU | TPOT p50 | scaling vs c=1 | marginal gain |
|--:|--:|--:|--:|--:|--:|
| 1 | 92.9 | 11.61 | 7.9 ms | 1.00× | — |
| 2 | 151.9 | 18.99 | 8.6 ms | 1.64× | **+64%** |
| 4 | 214.8 | 26.85 | 11.1 ms | 2.31× | **+41%** |
| **8** | **281.1** | **35.14** | **16.2 ms** | **3.03×** | **+31%** |
| 16 | 323.4 | 40.42 | 33.7 ms | 3.48× | +15% |
| 32 | 363.1 | 45.39 | 72.7 ms | 3.91× | +12% |
| 48 | 357.5 | 44.69 | 120.2 ms | 3.85× | **−2%** |
| 64 | 384.9 | 48.11 | 152.1 ms | 4.14× | +8% |

**Three things only the finer grid shows:**

1. **Saturation begins at c≈16, not c=64.** Marginal gain runs +64% → +41% → +31% through c=8, then
   collapses to +15%/+12% and goes **negative at c=48**. **The knee is between 8 and 16.**
2. **c=8 is the efficiency sweet spot** — it captures **3.03× of the total 4.14×** at only **16.2 ms** TPOT.
   Going 8 → 64 buys the last **1.37×** for **9.4× worse TPOT**. On a 1/4/16/64 grid that trade is
   invisible and the default conclusion is "use c=64."
3. **c=48 is a reproducible local regression** (−2% and worse TPOT than c=32), surviving a re-run.
   Consistent with the cudagraph capture ladder and EP imbalance at non-power-of-two batch — **not yet
   explained**, and worth an Nsight trace.

**This also strengthens §1's conclusion.** Throughput saturates at ~4× while per-token weight bytes are
constant, so the plateau is not a bandwidth wall — it is the point where added concurrency stops filling
idle latency and starts queueing. Per-GPU throughput plateaus at **45–48 tok/s/GPU from c=32 on**.

### ✅ Replicated on GLM-5.3-Flash (2026-09-02) — same knee, no c=48 dip

The 8-point curve now exists for **two** models, so the knee is measured rather than assumed:

| conc | GLM t/s | ×prev | GLM efficiency¹ | V4 t/s | ×prev | V4 efficiency¹ |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | — | 1.000 | 92.9 | — | 1.000 |
| 2 | 122.2 | 1.269× | 0.634 | 151.9 | 1.635× | 0.818 |
| 4 | 225.3 | 1.844× | 0.585 | 214.8 | 1.414× | 0.578 |
| **8** | **296.9** | 1.318× | **0.385** | **281.1** | 1.309× | **0.378** |
| 16 | 359.2 | 1.210× | 0.233 | 323.4 | 1.150× | 0.218 |
| 32 | 406.4 | 1.132× | 0.132 | 363.1 | 1.123× | 0.122 |
| 48 | **432.6** | **1.064×** | 0.094 | **357.5** | **0.985× ← dip** | 0.080 |
| 64 | 447.1 | 1.033× | 0.073 | 384.9 | 1.077× | 0.065 |

¹ `(tok/s ÷ tok/s@c=1) ÷ conc`; 1.0 = perfect linear scaling.

**Two architecturally different hybrids saturate almost identically** — 1→8 buys 3.08× (GLM) vs 3.03×
(V4); 8→64 buys only 1.51× vs 1.37×; efficiency at c=8 is 0.385 vs 0.378 and at c=64 is 0.073 vs 0.065.
GLM is 34 KDA + 11 DSA layers, V4 is 43 uniform DSA layers. **That the saturation *shape* is shared while
the attention design is not is itself evidence for §1:** the knee is set by what the two models have in
common — the engine's scheduling and collective structure — not by their attention mechanism.

**But V4's c=48 regression does NOT generalize.** GLM gives **1.064×** where V4 gives 0.985×. So the dip
is specific to V4's configuration, not a property of the engine at c=48. ⚠️ The two runs differ in
`--max-num-seqs` (GLM pinned 256, V4 at the H100 default 1024), which is the leading candidate and is
**not controlled** — so this says "GLM does not show it," not "V4's dip is caused by X."

**Also visible only on the finer grid: TTFT is non-monotonic.** GLM's TTFT p50 rises 857 → 2,685 ms
(c=1→16) then **falls to 1,992 ms at c=48** before rising again. Chunked prefill packs prefill chunks more
efficiently at moderate concurrency. One more reason TTFT is not a prefill-compute measurement (§2e).

⚠️ **Remaining coverage gap:** Qwen3.8 still has only a 4-point grid, so its knee is assumed, not measured.

---

## 5. The thesis, measured: four cost classes in one forward pass

Qwen3.8-Flash-Next is the clearest single-model evidence. **One decode step touches four per-layer cost
classes with three different memory-growth laws**, all **[A]** from config + tensor shapes:

| class | count | state per token | growth law |
|---|--:|---|---|
| GDN linear attention | **36 layers** | fixed recurrent state, 0.1055 GiB/seq | **O(1) in context**, O(max_num_seqs) in memory |
| QSA full attention | **12 layers** | 24.00 KiB/tok KV | **O(ctx)** |
| QSA compressed indexer | 12 caches | 0.75 KiB/tok (`compress_ratio 4`) | **O(ctx/4)** |
| **PLE n-gram embedding** | 1 layer | ~5 KiB/tok gathered from a **51.23 B** table | **O(1)**, and a *gather*, not a GEMM |

### The PLE table breaks "active parameters" as a metric

**28.5% of Qwen3.8's parameters are a hash-indexed embedding table**, not a weight matrix: 128 shards ×
2,500,012 rows × 160 dim, FP8, indexed by a **splitmix64 hash** of the token n-gram. Verified from
source, not inferred — `nvidia/ple_layer.py:184` is literally `F.embedding(input_, layer.weight)`. So
"active params" has three defensible values differing by **8×**:

| definition | value | verdict |
|---|--:|---|
| GEMM path only | **7.27 B** | used here — comparable to GLM/V4 |
| + entire PLE table | 58.51 B | meaningless: never fully read |
| PLE bytes actually touched | ~5 KiB/token [A] | the honest cost |

The vendor is explicit that this is the point — the card calls n-gram embedding *"a unique axis for
parameter scaling that requires less computation and is more amenable to offloading than MoE."* **Any
tool reporting "active parameters" as a single scalar mis-costs this model.** That is the heterogeneity
thesis appearing in the *parameter budget*, before the layer stack.

**And the engine has already conceded the point in code.** vLLM ships a **CPU-offload worker for this one
layer class** — `VLLM_PLE_CPU_OFFLOAD` ("Run n-gram PLE lookup in a dedicated CPU offload worker",
`envs.py:2036`), with its own readiness timeout and a `PleOffloadLayer` base class. **No other layer in
the model gets an offload path**, because no other layer is a pure gather. An engine that treats
parameters as one undifferentiated pool cannot express *"these 51.23 B can live on the host; those 120.80 B
must be in HBM"* — so it grew a per-layer-class special case instead. That is the uniform cost model
failing in production code, not in a paper. ⚠️ **[A]** — the offload path was not exercised in these runs.

### Cross-vendor convergence, and cross-vendor divergence

**Converged independently — ~25% full-attention layers at interval 4** [A]: Kimi-K3 24/93, Qwen3.8
12/48, GLM-5.3 11/45. Three vendors, same structural answer.

**Diverged completely — KV cost spans 50× in one generation** [A]: 368 KiB/tok (GLM-4.5, 2025) → **~7.4**
(V4-Flash, 2026), and **3.3× *within* the three models measured here** (24.75 / 11.35 / 7.4). Two
independent mechanisms reach it: **per-layer compression** (DeepSeek) vs **hybrid O(1)-state layers**
(Kimi/GLM/Qwen). A cost model with one bytes-per-token constant cannot describe either.

### And a mixed-precision fact inside one model

DeepSeek-V4-Flash reads **MXFP4 experts and FP8 attention weights** under **per-layer KV compression
ratios differing 32×** (`compress_ratios` {4,128}) — verified from safetensors headers, not just the
config. **Different bytes-per-parameter in different layers of the same forward pass.**

---

## 6. Where the E/k prediction survives, and where it dies

The report's Revision-1 supporting result was `B*_MoE = B*_dense · E/k`, from `D(B) ≈ min(Bk, E)`
distinct experts per step with `B*_dense = 148` tok/step (H100 FP8) [A]. With E/k now 36–51×, that
predicts **B\* = 5,300–7,600 tok/step** — i.e. the expert read should bind at any reachable batch.

**Measured: it does not, and the prediction fails for a reason worth stating.** All three models sit far
below the roofline — **9–18% measured for GLM [E]**, and ~1–3% by the hand-rolled estimate (§1) — so the
premise (bandwidth-saturated) is false either way. Two corrections:

1. **Sparsity does not buy E/k, because attention does not sparsify.** V4's dense remainder (attention
   5.21 + shared 1.11 + embed 0.53 = **6.85 B**) is **51% of its active budget** while being 2.4% of
   total. **Measured sparsity is 21.6×, not 42.7×.** Any `E/k`-based decode-cost prediction overestimates
   the MoE contribution; the dense half sets a floor.
2. **`--max-num-seqs` defaults cap the reachable batch anyway**, so `B*` may be **untestable on one
   node**, not merely untested.

**What survives:** E/k does predict **per-B-active efficiency** well — Qwen at E/k=51.2× is 2.6–2.8×
better per active parameter than the other two. **What dies:** using E/k to predict *where throughput
saturates*. Sparsity is a good model of the FLOP budget and a bad model of the bottleneck.

---

## 7. Trust: what nearly went wrong, and the guards that exist because of it

**A PI should discount a report that cannot say how its numbers could have been wrong.** Six errors were
caught, all preserved as evidence rather than deleted.

| # | error | magnitude if published | how it was caught |
|--:|---|---|---|
| 1 | Fixed seed + prefix caching | **3× overstatement** (640 vs 214 tok/s) | `newcachehits` assert per point |
| 2 | `--num-warmups` self-poisoning | uneven bias: 12.2% at c=1, 0.8% at c=64 → **flattened scaling** | constant 16,000 hits/point |
| 3 | Cold-engine scheduler queueing | **engine effect read 0.64× instead of 1.01×**; and 0.73× instead of 0.997× on the bridge arm | **p99 TTFT ≫ median** guard |
| 4 | Hand-derived param counts | GLM 15.01 → **17.38 B**; V4 11.01 → **14.08 B** (I8 packs 2 MXFP4/byte) | read tensor shapes |
| 5 | **KV pool mis-sized by cold-compile profiling** | **Qwen c=4 understated 1.58×**, and I had already written an architectural explanation for the artifact | `--gpu-memory-utilization` A/B (§4b) |
| 6 | **`--enable-mfu-metrics` passed on every arm, read by nothing** | the headline **"~1% of roofline" was low by an order of magnitude** (GLM measures 9.5–18.1%) | grepped for a *consumer* of the metric, not the flag |

Error 3 has now recurred **four times** (twice more this session, on the util-0.85 and MTP-context arms),
and its lesson is general: **`/health 200` is not a readiness signal for
benchmarking.** The `dev20073` build adds a DeepGEMM warmup pass (1,261 kernels, engine init 538.8 s) that
completes *after* the server reports healthy. Points that are cold and complete can still be **25–62%
low**. Error 4's lesson: **never hand-derive parameter counts from `config.json`; read tensor shapes** —
and note that both errors flattered the conclusion being tested.

**Error 6 is the newest and the most uncomfortable**, because nothing failed: the flag was set correctly on
every arm of all three models for two sessions, and no error, warning, or missing field ever appeared. The
counters were populated and simply never read. Worse, vLLM's coverage is **silently partial and differs per
model** — it instantiates `attn`+`ffn`+`unembed` for Qwen, only `ffn`+`unembed` for GLM, and only `unembed`
for V4, discarding the rest at `debug` level. So the corrected bandwidth numbers are **lower bounds that
are not comparable across models**. Lesson: **a flag you set is not a measurement you made**, and when an
API silently skips optional components, **publish which components were active next to the number**. Full
diagnosis: `fix_bug.md` bug 13.

⚠️ **Why the attention components drop out is itself evidence for this report's thesis.** Both of vLLM's
attention estimators gate on a single whole-model boolean — `AttentionMetrics` raises if
`is_deepseek_mla` is true, `MLAAttentionMetrics` raises if it is false (`perf.py:403-408`, `:551-554`) — so
a **hybrid stack satisfies neither and loses attention accounting entirely.** The source says so directly
at `perf.py:428`: *"TODO: discern cases where we have mixture of different attention layer types such as
SWA, MLA, etc."* The engine already models per-layer heterogeneity in its **allocator**
(`model_arch.py:119-122` explicitly prevents that flag from collapsing across layers, or `use_mla` would
go true model-wide) but **not in its performance accounting.** That asymmetry — heterogeneity handled for
correctness, ignored for cost — is precisely the gap §5 argues about, found in the engine rather than in a
config file.

### ✅ The engine is matched across all three models — measured, not assumed

Two separate bridge arms were run, because two different images were needed:

| bridge | comparison | measured effect |
|---|---|--:|
| V4-Flash re-swept in `glm53-flash` | conda 0.28.0 → `dev20051` | **1.01×** |
| V4-Flash re-swept in `qwen38-flash-next` | `dev20051` → `dev20073` | **0.997×** |

Both are negligible — **but neither was knowable without running it**, and on GLM a backend swap alone
cost **22%**. So the cross-model ratios in §1–§4 are architecture, not tooling.

### The FP8-KV arm: a corrected claim, and why the control mattered

The vendor recipe says *"Hopper does not support FP8 KV cache for this model."* **That is true of the
shipped image, not of Hopper.** GLM's FP8 path exists (`FLASHINFER_MLA_SPARSE_SM90` explicitly allows
NoPE) and is gated only by a FlashInfer version probe. Measured with 0.6.18 overlaid: **+1.805× KV
capacity** at a throughput cost.

**But the arm changed three things at once** (dtype + attention backend + MoE kernel), so the naive read
"FP8 KV costs 41%" would have been wrong. The control arm decomposes it **[M]**: **backend −22.0%, dtype
−24.9%**. *One variable per arm, always run the control* — this is the report's cleanest illustration of
its own rule.

---

## 8. What is NOT measured — stated rather than hidden

1. **No dense control anywhere.** Qwen3.8 was the designated dense baseline and is a **512-expert MoE**
   (`num_experts: 512`, verified from 150,528 expert tensors and zero plain `mlp.*_proj`; it reads
   `n_routed_experts: None` only because Qwen uses a different config key). So *"MoE decode is
   memory-bound"* has **not been falsified against a dense model on this harness** — though §1 suggests
   the premise is wrong for all three MoEs anyway. **This is the largest single gap.**
2. **No TP/EP sweep**, so the all-to-all/launch-overhead explanation for §1 is **[H]**, not [M] — and it
   is the report's central mechanistic claim.
3. ✅ **Achieved bandwidth is now [E] for GLM-5.3-Flash, still [A] for the other two.**
   `--enable-mfu-metrics` was passed on every arm but nothing read its counters until 2026-09-02;
   `bench.sh` now scrapes them per point (`fix_bug.md` bug 13). GLM measures **9.5–18.1% of 3.35 TB/s**,
   an order of magnitude above the hand-rolled `active_params × bytes × steps/s` estimate that produced
   the "~1%" figure. Two caveats keep this from being [M]: it is an **engine-side estimate** (config
   shapes × measured batch composition), and its **component coverage differs per model** — GLM gets
   `ffn`+`unembed`, V4 only `unembed`, Qwen all three — so the raw numbers are **not comparable across
   models**, and re-polling V4/Qwen is still owed.
4. **Kimi-K3 and DeepSeek-V4-Pro were not measured** — 1,453.7 GiB (~21 H100s) and ~740 GiB (~11) exceed
   one node. Weight offloading was deliberately *not* used: the numbers would be PCIe-dominated and
   architecturally meaningless. The layer-reduction proxy remains unbuilt, and its linearity gate still
   gates any extrapolation.
5. ✅ **MTP now measured on the context axis for GLM-5.3-Flash** (§3 finding 5) — and it **inverted the
   recommendation**: MTP costs 4–6% throughput but buys **~17–25% TTFT** up to 131K, which is the metric
   that matters on TraceLab's 530:1 ISL:OSL workload. **Still owed: the context axis for V4 and Qwen**
   (batch-axis only), and **no prefix-axis MTP points for any model.** Since one model's context axis
   reversed the conclusion drawn from four models' batch axes, the missing two are a real gap, not a
   completeness nicety.
6. **`max_num_batched_tokens` (chunked prefill) never swept** — plausibly the largest untested
   server-side lever for long context.
7. **Tokenizers differ** (Qwen 248,320 · GLM 154,880 · V4 129,536), so cross-family tok/s is not strictly
   commensurable; a denser tokenizer does more work per token. Use bytes/s for strict claims.
8. **TP=8 handicaps the models unequally.** V4 and Qwen fit in 3 GPUs but were measured at TP=8, so their
   per-GPU numbers are **pessimistic**; GLM genuinely needs ~5. `gpu_memory_utilization` also differs
   (Qwen 0.85 recipe-sanctioned vs 0.82) — affects KV capacity, not the decode cost model.
9. **Synthetic random prompts route ~uniformly across experts** — best case for expert coverage and load
   balance. Real text has correlated routing, so measured expert cost may understate imbalance. This
   matters most for Qwen (512 experts over 8 ranks).
10. **TraceLab was used for grid design, not replay.** Its token counts come from Claude/GPT tokenizers,
    and it documents `prefix_tokens` over-reporting (issue #22).
11. **One clean impossibility, recorded as a result:** TP4 OOMs for GLM (75.36 GiB/GPU, 1.6 GiB free) and
    PD-disaggregation needs two weight copies = 612 of 640 GiB. **PD is a latency-structure optimization
    bought with a 2× weight footprint — unavailable when weights are ~48% of node HBM.** The recipe's PD
    example targets a GB200 tray.
12. **The Qwen3.8 configuration is an *adaptation* of the vendor recipe, not a validated one.** The
    recipe publishes no H100 config — its Hopper block targets **8× H200** — and it notes the FP8 weights
    are 172.78 GiB, "relevant if adapting to 80 GB H100s." Every flag used matches that block, and the
    run fits comfortably (23.41 GiB/GPU), but no vendor number exists for this hardware. The recipe also
    prescribes **no throughput benchmark suite**, so there is no published baseline to check against —
    which is why the *relative* numbers here (same node, same harness, engine-bridged) are more
    trustworthy than any absolute comparison would be.

---

## 9. What I would build next

Ranked by what would change a serving system's design, not by effort.

1. **A per-layer-class cost model in the scheduler.** Every §5 finding says one number per model is
   wrong. A scheduler that knows *this* layer holds O(1) state and *that* one holds O(ctx) KV can
   admit sequences the current uniform allocator refuses — and stop padding mamba pages by 20.75%.
2. **Confirm or kill the latency-chain hypothesis (§1).** A TP/EP sweep plus an Nsight trace in
   NanoFlow's §2.2 taxonomy. If all-to-all is the bound, the leverage is in routing and placement, not
   quantization — and the whole industry's focus on bytes is misdirected for this regime.
3. **Prefix caching for recurrent state.** 36–70% of layers in these models hold state that current
   prefix caches simply cannot reuse (§2b), while 95.6% of real input tokens are prefix. This is the
   largest gap between what the workload offers and what the engine exploits.
4. **A dense control** — the one measurement that would let §1's claim be falsified rather than merely
   supported.

---

### One-paragraph summary

On 8×H100, none of GLM-5.3-Flash, DeepSeek-V4-Flash, or Qwen3.8-Flash-Next-FP8 decodes anywhere near the
memory roofline — GLM measures **9–18% of 3.35 TB/s at best [E]**, and bandwidth **peaks at c≈16–32 and then FALLS,
while throughput keeps rising to c=64** — so the textbook "decode is memory-bound, throughput ∝
1/active_bytes" story does not describe this regime, and rankings derived from active-parameter counts
measure the wrong thing. Qwen3.8 reads **2.4× fewer weight bytes per step than GLM and is 16% faster**;
GLM carries **29% more active params than V4 and is 15% faster**. What binds is a serial latency chain —
43–48 dependent layer-steps, each with an EP all-to-all on its critical path — and the features that
matter are the ones that shorten or widen that chain, not the ones that shrink bytes. Meanwhile the
models have become **internally heterogeneous**: one Qwen3.8 decode step touches four cost classes with
three memory-growth laws, 28.5% of its parameters are a hash-indexed table that a roofline model should
ignore, KV cost spans **3.3× across three models of the same generation**, and the same allocator resolves
page sizes **160× apart** for two of them — padding one by 20.75%. **The uniform per-layer cost model is
no longer true of a single model, which makes 2026 serving a scheduling and memory-management problem
rather than a kernel problem.**
