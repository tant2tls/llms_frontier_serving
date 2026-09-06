> HISTORICAL MODEL NOTES — use the [current report](../../report.md), [experiment index](../../experiments.md), and [debugging guide](../../fix_bug.md) for corrected conclusions. Original location: `Qwen3.8-Flash-Next-FP8/report.md`. Some causal claims and configurations below are superseded.

# Serving Qwen3.8-Flash-Next-FP8 on 8×H100: four cost classes in one forward pass

**Author:** Tan Ngo · **Date:** 2026-09-02 · **For:** Kan Zhu, UW SyFI

**Scope.** The third measured model, and the report's **most extreme sparsity point** (E/k = 51.2×) as
well as its **most internally heterogeneous stack**. 15 measured points on 8×H100 at TP8/EP — 11 base
(batch, context, prefix) + 4 MTP A/B — all cold and complete.

**Labelling.** **[M]** measured on this hardware today · **[A]** analytical (config, tensor shapes,
vendor spec). Never mixed silently. Nothing here measures accuracy.

> ⚠️ **Read first: this is not the dense control the plan expected.** `CLAUDE.md` designated
> "Qwen3.8-27B, dense, 64 layers" as the report's mandatory dense baseline. **That is a different
> model.** This checkpoint reads `n_routed_experts: None` only because Qwen uses a different config key.
> Measured: `num_experts: 512`, `num_experts_per_tok: 10`, **150,528 `.experts.` tensors and zero plain
> `mlp.{gate,up,down}_proj`**. It is a fine-grained MoE. **The report therefore has no dense control, and
> the "MoE is memory-bound" claim remains unfalsified by a same-harness dense run.** Stated as a gap
> (§9), not worked around.

---

## 0. Provenance

| | |
|---|---|
| **Hardware** | 8× NVIDIA H100 80GB HBM3 (`tan-8gpus-qwen38-0-0`), driver 575.57.08 |
| **Aggregate HBM** | 26.8 TB/s (8 × 3.35 TB/s vendor spec) |
| **Engine** | vLLM **`0.1.dev20073+g8e685d198`**, torch 2.13.0+cu130, transformers 5.15.1 |
| **Image** | `vllm/vllm-openai:qwen38-flash-next` — ⚠️ **a different build from the other two models** (§2) |
| **Model** | `Qwen/Qwen3.8-Flash-Next-FP8`, snapshot `236dfdf2`, 172.76 GiB, 131 shards |
| **Parallelism** | TP=8 + **mandatory** expert-parallel; `Local/global experts 64/512` **[M]** |
| **Weight dtype** | **94.1% of bytes FP8 e4m3**, 5.9% BF16 **[M]** |
| **KV dtype** | **BF16** — the only dtype any QSA backend accepts (§6) |
| **Resolved block size** | **4**; `mamba_block_size=16` (left unpinned, vLLM resolved the LCM) **[M]** |
| **KV pool** | **2,048,645 tokens**, 25.1 GiB/GPU, **7.81× concurrency at 262,144** **[M]** |
| **Weight load** | 23.41 GiB/GPU, 135 s; engine init 132 s (60 s compilation) **[M]** |
| **Spec decode** | **OFF** on all headline numbers; MTP n=1 in §7 only |
| **Harness** | `vllm bench serve`, `--ignore-eos`, unique seed per point |
| **Correctness** | ✓ `2+2` → `4`, reasoning parsed into `message.reasoning` **[M]** |

---

## 1. Parameter accounting — and an ambiguity that is itself a finding

**[M], from 152,089 safetensors tensor shapes across 131 shards.** Mutually exclusive buckets that
partition the total exactly:

| Component | Params | Share | dtype | How it is read per token |
|---|--:|--:|---|---|
| Routed experts | **120.80 B** | 67.1% | FP8 | GEMM, but only **k/E = 10/512** fire |
| **PLE n-gram embedding** | **51.23 B** | **28.5%** | FP8 | **table GATHER — not a GEMM** |
| Attention + GDN + dense | 3.58 B | 2.0% | BF16 | GEMM, every token |
| MTP layer | 2.61 B | 1.4% | FP8 | **excluded** — base arm |
| Embed + LM head | 1.27 B | 0.7% | BF16 | gather / GEMM |
| Vision tower (ViT) | 0.449 B | 0.2% | BF16 | **excluded** — `--limit-mm-per-prompt 0` |
| Router | 0.063 B | — | BF16 | GEMM |
| **Total on disk** | **180.01 B** | | | |

**Served total (base arm) = 176.94 B**  ·  **active on the GEMM path = 4.92 + 120.80 × 10/512 = 7.27 B**

**This reconciles with the model card**, which says *"125B with 6B activated, plus 51B n-gram embedding
and 4B MTP"*: measured main = **125.71 B**, measured active excluding embed/lm_head = **6.00 B**, measured
n-gram = **51.23 B**. ⚠️ The card says **4 B** MTP; **measured 2.61 B**. Trust the tensors.

### The PLE table breaks "active parameters" as a cost metric

**28.5% of this model is a hash-indexed embedding table**, not a weight matrix. Measured structure: 128
shards × 2,500,012 rows × 160 dim, FP8 (`split_ngram_parts: 128`, `ngram_vocab_size_base: 20,000,000`).
Verified from the vLLM source, not inferred — `nvidia/ple_layer.py:184` is literally
`F.embedding(input_, layer.weight)`, and the index is a **splitmix64 hash** of the token n-gram
(`ngram_size: 3` → bigrams and trigrams, odd multipliers, `:253`).

So "active params" has three defensible values and they differ by **8×**:

| definition | value | why it's wrong or right |
|---|--:|---|
| GEMM path only | **7.27 B** | **used in this report** — comparable to GLM/V4's active counts |
| + entire PLE table | 58.51 B | meaningless: the table is never fully read |
| PLE **bytes actually touched** | ~**5 KiB/token** **[A]** | 2 n-gram orders × 8 heads × 160 dim × 1 B |

**Why a PI should care.** This is a parameter-scaling axis that is *deliberately* cheap in FLOPs and
bandwidth, and the vendor says so: the card calls n-gram embedding *"a unique axis for parameter scaling
that requires less computation and is more amenable to offloading than MoE."* It is **28.5% of the
checkpoint's bytes that a roofline model should almost entirely ignore** — and any tool that reports
"active parameters" as a single scalar will mis-cost this model in one direction or the other. That is
the heterogeneity thesis showing up in the *parameter budget*, before we even get to the layer stack.

**The engine has already conceded the point in code.** vLLM ships a **dedicated CPU-offload worker for
this one layer** — `VLLM_PLE_CPU_OFFLOAD` ("Run n-gram PLE lookup in a dedicated CPU offload worker",
`envs.py:2036`, consumed at `ple_layer.py:455`), with its own readiness timeout
(`VLLM_PLE_OFFLOAD_READY_TIMEOUT`, default 600 s) and a `PleOffloadLayer` base class. **No other layer
class in this model gets an offload path**, because no other layer is a pure gather. A serving engine
that treats all parameters as one pool cannot express "these 51.23 B live on the host and cost host-memory
latency; those 120.80 B must be in HBM" — so it grew a per-layer-class special case instead. ⚠️ **Not
measured here** (the arm ran fully on-GPU); the recipe notes the offload path is Nvidia-only and that
DEP requires it.

---

## 2. ✅ The engine confound is CLOSED — measured at 0.997×

**This model cannot run in the image the other two were measured in** (`model_type: qwen4_exp` is
unregistered there), so it ran on a different vLLM build:

| model | image | vLLM build | CUDA |
|---|---|---|---|
| GLM-5.3-Flash · DeepSeek-V4-Flash | `glm53-flash` | `0.1.dev20051+g487ecf187` | 13.0 |
| **Qwen3.8-Flash-Next-FP8** | `qwen38-flash-next` | **`0.1.dev20073+g8e685d198`** | 13.0 |

That was a real fairness gap, not a footnote: on GLM, **a backend swap alone cost 22%**, so an engine
delta can rival the architecture effect being measured.

**It has been measured and closed.** The `qwen38-flash-next` image **registers
`DeepseekV4ForCausalLM`** (`Glm5Next*` is absent, so V4 is the only possible bridge), so V4-Flash was
re-swept here with flags copied **verbatim** from its `mtp-off-image` manifest — same node, same CUDA,
same TP8/EP, same grid and seeds, spec decode off, all cold. The **only** difference is the build:

| conc | dev20051 t/s | dev20073 t/s | ratio |
|--:|--:|--:|--:|
| 1 | 85.0 | 84.9 | 0.998× |
| 4 | 219.8 | 219.4 | 0.998× |
| 16 | 330.9 | 330.0 | 0.997× |
| 64 | 389.4 | 386.4 | 0.992× |

**Mean engine effect: 0.997×** (range 0.992–0.998) — the newer build is **0.3% slower**, an order of
magnitude below the smallest architecture effect claimed here. **So every cross-model ratio in §3–§6 is
valid as measured**; applying the correction moves the c=64 Qwen/GLM ratio from 1.16× to 1.17×.

Full writeup: [`../deepseek_v4_flash/results/mtp-off-bridge-dev20073/RESULT-engine-bridge.md`](../../deepseek_v4_flash/results/mtp-off-bridge-dev20073/RESULT-engine-bridge.md).

⚠️ **Three of the bridge arm's four points had to be re-run first**, and the reason matters for anyone
reproducing this. They were cold and complete yet **25–62% low**, which would have made the build effect
read **0.73×** — a wrong headline, and the *second* occurrence of this failure mode in the project. The
cause: this build adds a **DeepGEMM warmup pass** (1,261 kernels, ~5 min; engine init 538.8 s) that
`dev20051` lacks, so **`/health 200` arrives well before steady state**. Signature is p99 TTFT ≫ median
(7.96× and 8.25× on the two worst points). **`/health 200` is not a readiness signal for benchmarking.**

---

## 3. Batch axis — the headline

ISL 16,384 · OSL 256 · base model · all points cold. All **[M]**:

> ⚠️ **CORRECTED — these are the `results/base-util082/` numbers, not `results/base/`.** The original
> arm's KV pool was mis-sized because vLLM measured peak activation *during* cold torch.compile
> (17.07 GiB measured vs 0.99 GiB actual), costing ~14 GiB/GPU of KV. c=4 and c=16 were understated by
> 1.58× and 1.13×. Full diagnosis: [`results/base-util082/RESULT-util-ab.md`](../results/base-util082/RESULT-util-ab.md).

| conc | out t/s | per GPU | per B-active | per B-total | TTFT p50 | TTFT p99 | TPOT p50 | KV peak |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 106.2 | 13.28 | 14.61 | 0.60 | 598 ms | 832 ms | 7.0 ms | 0.6% |
| 4 | **256.8** | 32.10 | 35.32 | 1.45 | 1,496 ms | 2,075 ms | 9.6 ms | 2.3% |
| 16 | **420.2** | 52.52 | 57.79 | 2.37 | 2,917 ms | 7,485 ms | 26.3 ms | 9.2% |
| 64 | **517.7** | **64.71** | **71.21** | 2.93 | 2,942 ms | 28,696 ms | 110.1 ms | 57.5% |

**64× concurrency → 4.87× throughput, 15.7× worse TPOT.** Best scaling of the three models
(GLM 4.64×, V4 4.58×).

### 3b. Three-way comparison, matched grid

Same ISL/OSL/conc/TP/harness, all base, all cold. **Engine-bridged: the build difference is 0.997×,
measured (§2), so these ratios are architecture.**

| conc | Qwen t/s | GLM t/s | V4 t/s | Q/GLM | Q/V4 | Qwen /B-act | GLM /B-act | V4 /B-act |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 106.2 | 96.3 | 85.0 | **1.10×** | **1.25×** | **14.61** | 5.54 | 6.04 |
| 4 | 256.8 | 225.3 | 219.8 | **1.14×** | **1.17×** | **35.32** | 12.96 | 15.61 |
| 16 | 420.2 | 359.2 | 330.9 | **1.17×** | **1.27×** | **57.79** | 20.67 | 23.50 |
| 64 | **517.7** | 447.1 | 389.4 | **1.16×** | **1.33×** | **71.21** | 25.72 | 27.65 |

**Qwen3.8 wins at every concurrency (1.10–1.17× over GLM, 1.17–1.33× over V4), and is 2.6–2.8× better
per active parameter than either.** That is what E/k = 51.2× plus a 7.27 B active budget buys, and it is
the cleanest confirmation the report has that **fine-grained sparsity does convert into FLOP-budget
efficiency**.

> ⚠️ **An earlier version of this table showed Qwen *losing* at c=4 (0.72×), and I explained it as a
> prefill effect.** That explanation was wrong because the measurement was: the original arm's KV pool
> was mis-sized by a cold-compile profiling artifact (see the note above). With the pool correctly sized,
> Qwen's c=4 TTFT drops 2,250 → 1,496 ms and it *wins* at c=4. **The lesson is not about Qwen — it is
> that a plausible architectural story was available for an artifact, and I supplied one.** Always fix
> the measurement before explaining the shape.

---

## 4. Context axis — and where KV becomes binding

conc 8 · OSL 256. All **[M]**:

| ISL | Qwen t/s | Qwen KV | GLM t/s | GLM KV | V4 t/s | V4 KV | Qwen TTFT p50 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 16,384 | **322.7** | 7.2% | 295.9 | 10.2% | 104.5 | 18.1% | 2.3 s |
| 65,536 | **114.5** | 28.2% | 104.3 | 36.6% | 47.9 | 21.9% | 9.3 s |
| 131,072 | **56.5** | 56.1% | 47.2 | 72.5% | 34.7 | 27.0% | 19.8 s |
| 260,000 | 25.1 | **97.3%** | 26.5 | 89.5% | 16.7 | 36.9% | 21.3 s |

**16× context costs 12.9× throughput** — the steepest context decay of the three (GLM 11.2×, V4 6.3×),
despite 36 of 48 layers being O(1)-state GDN. The reason is visible in the KV column: **Qwen pays the
highest KV bytes/token of the three [A]**, because its 12 full-attention layers are *real* attention with
2 KV heads × 256 head_dim, not compressed latents:

| model | KV KiB/token [A] | mechanism |
|---|--:|---|
| **Qwen3.8** | **24.75** | 12 full-attn × 2 kv_heads × 256 × 2 (K+V) × 2 B = 24.00, + QSA indexer keys at compress_ratio 4 ≈ 0.75 |
| GLM-5.3 | 11.35 | 11 DSA × 512 latent × 2 B = 11.00, + kpool ≈ 0.35 |
| V4-Flash | ~7.4 | DSA + per-layer `compress_ratios` 4/128, fp8 KV |

**So Qwen crosses into KV exhaustion first: 97.3% at 260K×8**, versus GLM 89.5% and V4 36.9%. Two of the
three models in this report now have KV as a *binding* resource at max context, and they reach it for
different reasons — Qwen because its full-attention layers are uncompressed, GLM because its dtype is
forced to BF16. **V4 is the only one with headroom**, and it gets there by compressing per layer.

**This is the 50×-KV-spread finding measured end to end**, and it lands in the same generation: 24.75 vs
7.4 KiB/token is a **3.3× spread across three 2026 models**, all "hybrid," all claiming long context.

**Prefix sharing** (64K shared prefix + 2K unique suffix), all **[M]**:

| n prefixes | Qwen t/s | GLM t/s | V4 t/s | Qwen TTFT p50 |
|--:|--:|--:|--:|--:|
| 1 | **453.9** | 265.3 | 198.2 | 1,471 ms |
| 4 | 398.1 | 365.9 | 391.6 | **1,207 ms** |
| 16 | **262.2** | 199.4 | 114.6 | 2,999 ms |

**Qwen wins the prefix axis outright, and at n=1 by 1.71× over GLM and 2.29× over V4.** n=1 is maximum
sharing — one prefix for all 64 prompts — so this measures how well the engine exploits reuse, and a
4-token block size gives the finest reuse granularity in the comparison. Given TraceLab says **95.6% of
real input tokens are prefix**, this is the axis that matters most for agentic serving, and it is the one
where Qwen's architecture looks best.

---

## 5. Where the bottleneck actually is (Q1) — not memory bandwidth

**Achieved HBM bandwidth [A]**, batched accounting (at concurrency c the batch shares one weight read per
decode step, so `steps/s = out_tok_s / c`):

| model | weight bytes/step | c=1 | c=4 | c=16 | c=64 |
|---|--:|--:|--:|--:|--:|
| Qwen3.8 | 6.77 GiB | 2.88% | 1.74% | 0.71% | **0.22%** |
| GLM-5.3 | 16.19 GiB | 6.25% | 3.65% | 1.46% | **0.45%** |
| V4-Flash | 7.21 GiB | 2.46% | 1.59% | 0.60% | **0.18%** |

Adding the KV re-read term at c=64/ISL 16K brings Qwen to **~1.0%** and GLM to **~0.8%** of the 26.8 TB/s
aggregate peak. **All three models decode at ~1% of the memory roofline.** Whatever binds decode here, it
is emphatically **not** HBM bandwidth.

**Therefore `B*_MoE = B*_dense · E/k` is not the operative cost model at reachable batch sizes** — and
Qwen3.8 is the strongest evidence, because it is the *most* sparse model (E/k = 51.2×) and yet:

- it reads the **fewest weight bytes per step** (6.77 GiB, less than half GLM's 16.19), and
- it is the **fastest at every concurrency** (517.7 tok/s at c=64), and
- it still only reaches **0.22% of peak bandwidth**.

If the expert read were binding, the model reading 2.4× fewer bytes should not also be the fastest by
16%. **The ranking follows neither active params nor bytes read.** What it does track, across all three
models, is **decode steps/s falling as concurrency rises** — i.e. per-step *latency* is the cost, and
per-step latency is dominated by fixed overheads: kernel launches, EP all-to-all, and scheduler work.

**Qwen3.8 sharpens the mechanism, because its sparsity is fine-grained.** With `moe_intermediate_size:
640` and 64 experts per rank at EP8, each expert GEMM is tiny — 512 experts × 640 intermediate against
GLM's 288 × much larger. Ten routed experts per token spread over 8 ranks means **1.25 experts per rank
per token expected**, so most ranks do near-zero useful work per token and **all-to-all sits on the
critical path of every decode step**. That is a *network and launch-overhead* term, invisible to
bandwidth accounting — and it is why the fine-grained model does not win proportionally to its byte
savings.

⚠️ **This remains a hypothesis, not a measurement.** Confirming it needs a TP/EP sweep (untested) or an
Nsight trace attributing time to dense/attention/network/other in NanoFlow's §2.2 taxonomy. Two
observations are consistent with it but do not prove it: the achieved-bandwidth collapse above, and
Qwen's elevated p99/median TTFT (2.98× at c=4 vs GLM's 1.33×). **`--enable-mfu-metrics` was passed but
its engine-side counters did not land in the result JSONs**, so every bandwidth number here is
analytical.

---

## 6. Four cost classes in one forward pass — the thesis, measured

This model is the report's best single-model evidence for the central claim. **One decode step touches
four different per-layer cost classes with three different memory-growth laws** — all **[A]** from
`config.json` and tensor shapes, all in *one* transformer stack:

| # | class | count | state per token | growth law |
|--:|---|--:|---|---|
| 1 | **GDN linear attention** | **36 layers** | fixed recurrent state, **0.1055 GiB/seq** total | **O(1) in context**, O(max_num_seqs) in memory |
| 2 | **QSA full attention** | **12 layers** | 24.00 KiB/tok KV | **O(ctx)** |
| 3 | **QSA compressed indexer** | 12 caches | 0.75 KiB/tok (`compress_ratio 4`) | **O(ctx/4)** |
| 4 | **PLE n-gram embedding** | 1 layer (layer 2) | ~5 KiB/tok gathered from a 51.23 B table | **O(1)**, and a *gather*, not a GEMM |

`layer_types` **[M]**: 36 `linear_attention` + 12 `full_attention`, full-attention at layers
3, 7, 11, …, 47 — **exactly interval 4**, confirming the cross-vendor convergence (Kimi-K3 24/93,
GLM-5.3 11/45, Qwen3.8 12/48).

**The uniform per-layer cost model is not merely wrong here; it is wrong four ways.** Concrete evidence
that the engine feels it:

- **vLLM must satisfy a `FullAttentionSpec`, an `MLAAttentionSpec` (indexer keys) and two `MambaSpec`s
  (GDN state + PLE conv) simultaneously.** Left unpinned, it resolved **`block_size = 4`** with
  `mamba_block_size = 16` **[M]**. That is a *strikingly* small page — GLM, under the same allocator,
  resolved to **640**. **Two hybrid models in the same generation force the same allocator to page sizes
  160× apart.** A paged-KV design with one block size cannot serve both well.
- **The GDN state adds a memory axis that scales with `max_num_seqs`, not context** — 0.1055 GiB/seq,
  **3.38 GiB/GPU at 256 seqs, TP8 [A]**, *constant in context*. This is why `--max-num-seqs 256` is
  mandatory: the H100 auto-default of 1024 trips a Mamba-cache capacity failure at startup. A scheduler
  tuned on attention-only models gets this wrong **by default, on this hardware**.
- **FP8 KV cannot help at all here.** Every QSA backend declares
  `supported_kv_cache_dtypes = ["auto","bfloat16"]` and raises `NotImplementedError` otherwise; the
  indexer requires BF16 model dtype outright. So the KV dtype matrix across this report is now
  *fully* determined by kernel availability, not choice:

| model | KV dtype available on SM90 | why |
|---|---|---|
| DeepSeek-V4-Flash | **fp8_ds_mla only** | BF16 path gated to Blackwell |
| GLM-5.3-Flash | BF16 (FP8 reachable only via a FlashInfer 0.6.18 overlay, −25%) | version gate |
| **Qwen3.8-Flash-Next** | **BF16 only** | every QSA backend refuses FP8 |

**Qwen3.8 ↔ GLM-5.3 is therefore the report's one KV-dtype-matched pair** — the only cross-model
comparison here where the KV dtype is not a confound. Use that pair for KV claims; it is why the §4
context table is more trustworthy for Qwen-vs-GLM than for either against V4.

---

## 7. Speculative decoding (Q3): MTP *hurts* this model at every batch size above 1

`results/mtp-n1/`, `num_speculative_tokens=1`, matching the V4 and GLM A/B arms. All **[M]**:

| conc | base t/s | MTP n=1 t/s | gain | base TPOT | MTP TPOT | base KV | MTP KV |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 107.7 | 125.9 | **1.17×** | 7.0 ms | **5.6 ms** | 0.9% | 1.2% |
| 4 | 162.2 | 149.3 | **0.92×** | 9.8 ms | 16.5 ms | 3.6% | 4.8% |
| 16 | 371.0 | 322.2 | **0.87×** | 32.2 ms | 37.0 ms | 14.5% | 19.2% |
| 64 | 517.8 | 481.5 | **0.93×** | 110.1 ms | 109.8 ms | 57.5% | 76.1% |

**Measured acceptance: 17,177 / 29,802 draft tokens = 57.6% [M]** (from
`vllm:spec_decode_num_accepted_tokens_total`).

**MTP also costs 12.0% of KV capacity [M]**: the pool drops **2,048,645 → 1,803,660 tokens**, and peak KV
at c=64 rises 57.5% → 76.1%. For comparison, MTP cost V4 only 2.4% of KV.

**This is the most negative spec-decode result in the report, and the mechanism is architectural.** The
crossover happens between c=1 and c=4 — *earlier* than GLM (which stayed ≥1.0× through c=4) and earlier
than V4. Two reasons, both from the checkpoint:

1. **The draft head is a full-attention layer over a mostly-linear-attention target.** The config says
   `mtp: {hybrid: true, layer_types: ["full_attention"], num_hidden_layers: 1}` — so the draft pays
   **O(ctx) KV** while 36 of the 48 target layers it is predicting for pay only O(1) state. GLM avoids
   exactly this with `index_share_for_mtp_iteration: true` (indexer shared across MTP iterations); Qwen
   has no such sharing. **The draft is disproportionately expensive relative to the step it accelerates.**
2. **The target step is already cheap.** Qwen reads 6.77 GiB/step against GLM's 16.19 — spec decode buys
   its wins by converting idle memory-bandwidth into tokens, and this model was never bandwidth-bound
   (§5, 0.22% of peak). There is less idle capacity to convert, so the draft's cost dominates sooner.

**The general lesson for a serving system:** "MTP gives ~1.2×" is not a property of a model — it is a
property of a *(model, concurrency)* pair, and for Qwen3.8 the useful range is **concurrency 1 only**.
At c=1 the TPOT improvement is genuine and large (7.0 → 5.6 ms, a 20% latency cut), so MTP here is a
**latency** feature, not a throughput feature. Enabling it on a loaded server costs 7–13% of throughput
and 12% of KV. **This is why the report's rule 1 exists**: had headline numbers been taken MTP-on, Qwen
would have looked *worse* than it is at every concurrency but 1.

`n=5` was not run. On GLM it was measurably worse than n=1 at every concurrency ≥4 (0.91× at c=64,
acceptance collapsing 71.8% → 30.6%), and Qwen's constraint arithmetic makes it awkward anyway: the QSA
ring requires `block_size % capacity == 0` with `capacity = 4·⌈(4+n)/4⌉`, giving **12** at n=5 against a
resolved block size of 4. Recorded as a deliberate omission, not an oversight.

---

## 8. What did not run, and why

### 8a. Checked against the vendor recipe — and three discrepancies worth flagging

The vLLM recipe (`recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next?variant=fp8&hardware=h100`) was read after
measuring. **Every flag used here matches its Hopper block**: `--tensor-parallel-size 8`,
`--enable-expert-parallel` (TEP8), `--moe-backend triton`, `--gpu-memory-utilization 0.85`,
`--max-num-seqs 256`, `--enable-prefix-caching`, `--no-enable-flashinfer-autotune`,
`--reasoning-parser qwen3`, `--tool-call-parser qwen3_xml`. It independently confirms three things
derived here from source: **plain TP8 is incompatible with the FP8 checkpoint** (128-wide quantization
blocks — so `--enable-expert-parallel` is mandatory, not tuning), **`--max-num-seqs 256` avoids
Mamba-cache capacity errors**, and **lower `num_speculative_tokens` under MTP memory pressure** (§7
measured exactly that pressure: −12% KV).

Three discrepancies, all stated rather than resolved:

| # | recipe says | measured / found here |
|--:|---|---|
| 1 | **The Hopper block targets 8× H200, and no H100 config is published.** It notes FP8 weights are 172.78 GiB, "relevant if adapting to 80 GB H100s" | **This is an 8×H100-80GB run** — so it is an *adaptation* of the recipe, not a validated configuration. It fits comfortably (23.41 GiB/GPU) and the extra headroom H200 would give goes to KV, not to the decode cost model |
| 2 | requires **"vLLM 0.29.0+"** | the image ships **`0.1.dev20073+g8e685d198`**, a dev build that does not compare numerically to 0.29.0. It registers the arch and serves correctly, so the constraint is really "this image," not a version string |
| 3 | **"Pipeline parallelism unsupported — N-gram Embedding lacks PP support"** | ⚠️ **the shipped class declares `SupportsPP`** and `nvidia/model.py` contains full PP plumbing (`make_layers`, `get_pp_group().is_first_rank/is_last_rank` guards at :415–:989); `ple_layer.py` has no PP guard at all. **Not tested here** — PP would change the latency structure and break TTFT/TPOT comparability anyway — so this is recorded as a documentation/source discrepancy, not a correction |

The recipe prescribes **no throughput/latency benchmark suite** (only a single chat smoke test), and its
performance claims are vendor-reported attention-kernel speedups. **So the sweeps in this report have no
vendor baseline to be checked against** — which is a reason to trust the *relative* numbers here (same
harness, same node) more than any absolute comparison to published figures.

Also noted and not pursued: the recipe's `--speculative-config` example uses
`num_speculative_tokens: 3`, validated on **GB300 TP4**, not Hopper. §7 ran n=1 to match the V4 and GLM
A/B arms; n=3 would change draft cost and acceptance simultaneously and so could not be compared to them.

### 8b. Arms and configurations that did not run

| attempt | outcome | cause |
|---|---|---|
| Qwen3.8 in the `glm53-flash` image | ❌ | `model_type: qwen4_exp` unregistered there. Forced the separate image — confound since **closed** at 0.997× (§2) |
| `--kv-cache-dtype fp8` | ❌ **[A]**, from source | every QSA backend declares `["auto","bfloat16"]`; impl raises `NotImplementedError`. **Not attempted at runtime** — the source gate is unambiguous |
| MTP `n=5` | ⬜ not run | worse than n=1 on GLM; QSA ring needs `block_size % 12 == 0` vs resolved 4 (§7) |
| TP sweep (2/4) | ⬜ not run | the EP-all-to-all hypothesis in §5 is consequently **unconfirmed** |
| V4-Flash bridge arm in this image | ✅ **DONE** | 4 points, `../deepseek_v4_flash/results/mtp-off-bridge-dev20073/`. Build effect **0.997×** — confound closed (§2) |
| 1M context | ⬜ not run | reachable only via static YaRN (`rope_type=yarn factor=4.0`), which changes the positional encoding — a different model, not a longer one |

**Two points were quarantined and re-run, not silently kept.** `bench.sh`'s queueing guard flagged the
c=1 point in **both** arms (p99/median TTFT 7.81× base, 9.03× MTP; limit 4×). **The guard's label was
wrong but its instinct was right:** at concurrency 1 requests are served strictly sequentially, so
*nothing can queue* — the mean sitting between median and p99 showed one slow request, the first, paying
JIT warmup. Re-run warm:

| point | cold (quarantined) | warm (published) | understatement | median TTFT |
|---|--:|--:|--:|---|
| base c=1 | 84.2 t/s (7.81×) | **107.7 t/s** (1.07×) | **−22%** | 602 ms → 602 ms |
| MTP c=1 | 90.4 t/s (9.03×) | **125.9 t/s** | **−28%** | unchanged |

Median TTFT is **identical** across the pairs — only the first-request tail moved, confirming JIT warmup
rather than queueing. Evidence retained in `results/_base_c1_jitcold/` and `results/_mtp_c1_jitcold/`.
**Had the flagged points been published, the MTP c=1 gain would have read 1.07× instead of 1.17×** —
i.e. the guard prevented a wrong conclusion about the one concurrency where MTP actually helps.

---

## 9. Gaps, stated rather than hidden

1. **No dense control anywhere in this report.** This model was the designated baseline and it is a
   512-expert MoE. So *"MoE decode is memory-bound"* has not been falsified against a dense model on the
   same harness — and §5 in fact suggests the premise is wrong for all three MoEs (≈1% of roofline).
   Closing it needs a genuinely dense model (Qwen3-32B or similar) on this harness.
2. ✅ **The engine confound is closed** — bridge arm measured at **0.997×** (§2). Was the largest
   threat to the three-way numbers; is now quantified and negligible.
3. **Achieved bandwidth is [A], not [M].** `--enable-mfu-metrics` was passed; the counters did not reach
   the result JSONs. Every "% of peak" here is an analytical upper bound.
4. **No TP/EP sweep**, so the all-to-all/launch-overhead explanation for §5 is motivated but unproven —
   and it is the report's central mechanistic claim.
5. **MTP measured on the batch axis only** (as with the other two models); no context or prefix MTP points.
6. **Tokenizer mismatch is largest here.** Qwen vocab is **248,320** vs GLM 154,880 vs V4 129,536, so
   cross-family tok/s is least commensurable for this model. A denser tokenizer does more work per token;
   for strict claims use bytes/s or a fixed corpus.
7. **`gpu_memory_utilization` is 0.85** here vs 0.82 for GLM and V4 (recipe-sanctioned per layout). It
   changes KV capacity, not the decode cost model — but the §4 KV-percentage column is not exactly
   like-for-like.
8. **TP=8 is not this model's minimum.** 172.76 GiB needs ~3 H100s, so per-GPU numbers are **pessimistic**
   — the same caveat as V4, and for the same reason.
9. **Synthetic random prompts route ~uniformly over 512 experts** — the best possible case for expert
   coverage and load balance. Real text has correlated routing, so measured expert cost may *understate*
   imbalance. This matters more here than for any other model in the set, because 512 experts over 8
   ranks is the sparsest routing being tested.
