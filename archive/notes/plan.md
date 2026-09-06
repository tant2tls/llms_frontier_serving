> ARCHIVED SESSION RECORD — not current instructions or conclusions. Original location: `plan.md`. See [current report](../../report.md), [debugging guide](../../fix_bug.md), and [session handoff](../../WORKFLOW.md). Historical claims may be superseded.

# plan.md — Architecture & Serving Analysis of the 2026 Open-Weight Frontier

**Task source:** [`task.md`](../../task.md) · **Assigned by:** Kan Zhu (UW, **SyFI** lab — Baris Kasikci & Stephanie Wang)
**Author:** Tan Ngo · **Budget:** ≤ 8× H100 SXM · **Deliverable:** `report.md` + reproducible `bench/` + `figs/`
**Revision 2 · 2026-09-01** — rewritten for the 2026 model generation (DeepSeek-V4, Qwen3.8, Kimi-K3,
GLM-4.7/5.3) after M\* discussion. **Phase 1 = architecture report, delivered first.** Evaluation
design deliberately deferred to §6 and kept coarse.

---

## 0 · What changed in Revision 2, and why it matters

Revision 1 built its thesis on the **2025** generation (GLM-4.5, Qwen3-235B-A22B, DeepSeek-V3,
Kimi-K2): all **fine-grained MoE + full attention at every layer**, differing mainly in GQA vs MLA.
The thesis was that fine-grained MoE inflates the compute-bound batch threshold by the expert
dilution factor `E/k`.

**That thesis is still correct but is no longer the interesting story**, because the 2026 generation
changed the *attention* layer, not just the FFN. Verified from HuggingFace `config.json` on
2026-09-01:

| | 2025 gen | **2026 gen** |
|---|---|---|
| Attention | full attention, **every layer** | **hybrid** — linear/KDA on 75% of layers, full/sparse on ~25% |
| KV growth | O(ctx) on all layers | **O(ctx) on ~25% of layers, O(1) on the rest** |
| Sparsity | dense attention | **DSA / indexer** (`index_topk` 512–2048), sliding windows |
| Expert dtype | FP8 | **FP4** (`expert_dtype: "fp4"` in DeepSeek-V4) |
| KV compression | MLA latent (512+64) | **per-layer `compress_ratios`** alternating 4 / 128 |
| Modality | text | **multimodal by default** (Kimi-K3, Qwen3.8, GLM-5.3 all have `vision_config`) |

**The revised thesis is stronger and lands directly on his current work.** M\* (Kasikci & Wang, June
2026) argues serving systems break because they assume inference is *"a single autoregressive loop"*
and model requests as a *"flat DAG"*. Its answer is the **Walk Graph** — component nodes, tensor
edges, named **Walks** ("a labeled subgraph for one phase of behavior") — aimed at *inter-component*
heterogeneity (BAGEL's `prefill_vit`/`prefill_vae`, Qwen3-Omni's Thinker→Talker→Code2Wav).

> ### The thesis
> **M\*'s premise now holds one level lower than M\* addresses it. The 2026 frontier models are
> internally heterogeneous *within a single transformer stack*: a Kimi-K3 decode step touches 69 KDA
> layers with O(1) state and 24 MLA layers with O(ctx) KV; a DeepSeek-V4-Flash step reads FP4 experts
> and FP8 attention weights under per-layer KV compression ratios that differ by 32×. The uniform
> per-layer cost model that every serving system's scheduler and memory manager assumes — one KV
> block size, one bytes-per-token, one bottleneck per step — is no longer true of a single model, let
> alone across models. Heterogeneity has moved inside the layer stack, and that is a scheduling and
> memory-management problem, not a kernel problem.**

That is a claim about **architecture**, which is what `task.md` line 2 actually asks for
("similarities, differences, and, most importantly, performance implications"), it is derivable from
configs alone (**no GPU needed**), and it extends M\*'s own abstraction inward. Say it in one
sentence, derive it in a page, and let evaluation confirm it later.

---

## 1 · The architecture report (Phase 1 — the deliverable to hand over first)

This is what goes to the lab first. **All of it is GPU-free**, derived from `config.json` +
`model.safetensors.index.json`, so it can be finished now and is independently checkable by anyone.

### 1.1 The model set — verified 2026-09-01

Every field below was read from the live HF config. **Weight sizes are `metadata.total_size` from the
safetensors index — actual bytes on disk, not estimates.**

| Model | Arch class | L | Attention pattern | E | k | E/k | Weights | Multimodal |
|---|---|--:|---|--:|--:|--:|--:|:--:|
| **GLM-4.7-Flash** | `Glm4MoeLiteForCausalLM` | 47 | **MLA all layers** (kv_lora 512, v_hd 256) | 64 | 4 | 16× | **29.1 GiB** | – |
| **Qwen3.8-27B** | `Qwen3_5ForConditionalGeneration` | 64 | **hybrid**: 48 linear + 16 full (interval 4) | **dense** | – | – | **51.7 GiB** | ✓ vision |
| **DeepSeek-V4-Flash** | `DeepseekV4ForCausalLM` | 43 | **DSA** + `compress_ratios` 4/128 + swin 128 | 256 | **6** | 42.7× | **148.6 GiB** | – |
| **GLM-5.3-Flash** | `Glm5NextForConditionalGeneration` | 45 | **hybrid**: 34 KDA + 11 DSA (`index_topk` 2048) | 288 | 8 | 36× | **305.8 GiB** | ✓ |
| **DeepSeek-V4-Pro** | `DeepseekV4ForCausalLM` | 61 | DSA + compress + swin | 384 | **6** | **64×** | ~740 GiB | – |
| **Kimi-K3** | `KimiK3ForConditionalGeneration` | 93 | **hybrid**: 69 KDA + 24 MLA (interval 4) | **896** | **16** | **56×** | **1453.7 GiB** | ✓ vision |
| *2025 baselines* | | | | | | | | |
| GLM-4.5 | `Glm4MoeForCausalLM` | 92 | GQA 96:8, all layers | 160 | 8 | 20× | 657 GiB BF16 | – |
| Qwen3-235B-A22B | `Qwen3MoeForCausalLM` | 94 | GQA 64:4, all layers | 128 | 8 | 16× | 438 GiB BF16 | – |
| DeepSeek-V3 | `DeepseekV3ForCausalLM` | 61 | MLA, all layers | 256 | 8 | 32× | 1250 GiB BF16 | – |
| Kimi-K2 | `DeepseekV3ForCausalLM` | 61 | MLA, all layers | 384 | 8 | 48× | 1912 GiB BF16 | – |

**Four observations that are the report's substance:**

1. **"Flash" is a new product tier, not a smaller checkpoint.** GLM-4.7-Flash is 29 GiB with **full
   MLA** — MLA has migrated from 671B-class models down to 30B-class. DeepSeek-V4-**Flash** (148.6 GiB)
   fits 8×H100 while V4-**Pro** (~740 GiB) does not. The vendors have split their lineup along the
   single-node boundary. **This is the single most serving-relevant fact in the table.**
2. **Expert dilution kept climbing:** Kimi-K2 48× → **Kimi-K3 56×** (E=896, k=16), DeepSeek-V3 32× →
   **V4-Pro 64×** (E=384 at k=**6**, down from 8). Revision 1's `E/k` thesis is *confirmed as a trend*,
   not refuted — but see §1.3 for why FP4 experts partly offset it.
3. **Attention is now hybrid and the fraction matters more than the mechanism.** Kimi-K3 runs
   full attention on **24 of 93 layers (26%)**; Qwen3.8 on **16 of 64 (25%)**; GLM-5.3-Flash on
   **11 of 45 (24%)**. Three vendors independently converged on **~25% full-attention layers at
   interval 4.** That convergence is a finding worth stating plainly.
4. ⚠️ **CORRECTED 2026-09-02 — Qwen3.8-Flash-Next is NOT dense, and there is no control.** This
   plan claimed "no `n_routed_experts` at all → the natural control for the whole study." **That was
   wrong.** Qwen uses a *different config key*: measured from the served checkpoint,
   **`num_experts: 512`, `num_experts_per_tok: 10`, `moe_intermediate_size: 640`** — verified from
   **150,528 `.experts.` tensors and zero plain `mlp.{gate,up,down}_proj`**. It is a **fine-grained MoE
   at E/k = 51.2×**, the *most* sparse model in the report, not its dense baseline.
   **Consequence: the study has NO dense control**, so "MoE decode is memory-bound" is unfalsified by a
   same-harness dense run — see `report.md` §8.1. (Measured results also revise the layer counts above
   for this checkpoint: **12 full-attention of 48**, not 16 of 64.)

### 1.2 KV / state cost per token — the number that reorders everything

Computed per-layer from configs, counting only layers that actually grow with context:

| Model | Full-attn layers | **KV KiB/token** | vs GLM-4.5 | Constant state |
|---|--:|--:|--:|---|
| GLM-4.5 (2025, GQA) | 92/92 | **368.0** | 1.0× | – |
| Qwen3-235B (2025, GQA) | 94/94 | 188.0 | 2.0× cheaper | – |
| DeepSeek-V3 / Kimi-K2 (2025, MLA) | 61/61 | 68.6 | 5.4× | – |
| **Qwen3.8-27B** | 16/64 | **64.0** | 5.8× | 0.023 GiB/seq (48 linear layers) |
| **GLM-4.7-Flash** | 47/47 | **52.9** | 7.0× | – |
| **Kimi-K3** | 24/93 | **27.0** | **13.6×** | KDA state on 69 layers |
| **GLM-5.3-Flash** | 11/45 | **12.4** | **29.7×** | KDA state on 34 layers |
| **DeepSeek-V4-Flash** | 43/43 | **~7.4** | **~50×** | – (compression, not hybrid) |

> ⚠️ **Every number in §1.1–1.3 is analytical, computed from published config fields by
> `bench/params.py`. None is measured.** They are *predictions*. Label them as such in `report.md`
> and never let one appear styled as a measurement. (Inherited from `../PhD/ray_learning/CLAUDE.md`.)
> The DeepSeek-V4 figure additionally depends on interpreting `compress_ratios` as a per-layer KV
> divisor — **flag this as an inference from the field name, pending the modeling code**, which is
> exactly the kind of caveat that makes the rest trustworthy.

**Two orders of magnitude of spread in KV cost across one generation.** That is the headline number.
And note the two mechanisms are *independent*: DeepSeek-V4-Flash reaches ~7 KiB/tok with **compression
on every layer**; GLM-5.3-Flash reaches 12 KiB/tok with **hybrid layers**. Same destination, different
route, different implications for a paged-KV allocator.

### 1.3 The FP4 expert wrinkle — don't miss this

DeepSeek-V4 sets **`expert_dtype: "fp4"`** while keeping FP8 elsewhere (`quantization_config` is
FP8 e4m3 with `weight_block_size [128,128]`, `scale_fmt: ue8m0`). So within one model, weight classes
have **different bytes-per-parameter**.

This changes the Revision-1 arithmetic directly. The dense crossover is
`B*_dense = (Compute/2·MemBW)·bytes_per_param`, so **halving expert bytes doubles the batch at which
experts become compute-bound**:

| | bytes/param | B\*_dense | × E/k | **B\*_MoE** |
|---|--:|--:|--:|--:|
| FP8 experts (2025 gen) | 1.0 | 148 | 32× (V3) | 4,736 |
| **FP4 experts (V4-Flash)** | 0.5 | **296** | 42.7× | **12,637** |
| **FP4 experts (V4-Pro)** | 0.5 | **296** | 64× | **18,944** |
| **Kimi-K3** (FP8, E=896 k=16) | 1.0 | 148 | 56× | **8,288** |

So the `E/k` story from Revision 1 **holds and intensifies** — B\*_MoE is now 8K–19K tokens/step,
far outside any reachable serving batch. **The expert-weight read is unambiguously the binding
constraint for the entire 2026 MoE generation at any realistic batch size.** FP4 is the vendors'
response: they can't reduce `E/k`, so they reduce bytes-per-expert instead. That's a system-level
read of a model-architecture decision, and it's the kind of connection the report should make.

### 1.4 The similarity/difference table `task.md` line 2 asks for

Organize the architecture section by **mechanism**, not by vendor — that's what makes it analysis
rather than a catalogue:

| Axis | Converged on (similarity) | Diverged on (difference) | Performance implication |
|---|---|---|---|
| **Attention topology** | ~25% full-attn layers, interval 4 | KDA (Kimi/GLM) vs gated-DeltaNet-style linear (Qwen) vs all-layer compression (DeepSeek) | KV bytes/token spread **50×**; breaks uniform block accounting |
| **KV reduction** | everyone reduces it hard | latent (MLA) · hybrid (O(1) layers) · per-layer compress_ratios · DSA top-k | different *shape* of allocator pressure, not just magnitude |
| **Sparse selection** | indexer-based top-k | `index_topk` 512 (V4-Flash) / 1024 (V4-Pro) / 2048 (GLM-5.3) | selection is itself a kernel with its own bottleneck |
| **MoE granularity** | fine-grained, growing | E: 64 → 896; k: 4 → 16 | `E/k` = 16–64× → expert read binds |
| **Expert precision** | quantized | **FP4** (DeepSeek-V4) vs FP8 (others) | doubles B\*; mixed dtype *within* a model |
| **MTP** | mostly present | K3 = **0**, V4 = 1, Qwen3.8 = `mtp_num_hidden_layers: 1` | speculation is free for some, external draft for others |
| **Modality** | multimodal default | K3/Qwen3.8/GLM-5.3 have `vision_config`; DeepSeek-V4 text-only | this is where **M\*'s Walk Graph** becomes necessary |
| **Context** | ≥256K | **1,048,576** (V4, K3, GLM-5.3) vs 262K (Qwen3.8) | 1M ctx is only viable *because* of the KV work above |
| **Positional/scoring** | RoPE + YaRN | `scoring_func: sqrtsoftplus` (V4) vs sigmoid (GLM-5.3); `hidden_act: situ` (K3) | minor perf, notable novelty |

**The synthesis sentence:** *every vendor independently concluded that KV cost, not parameter count,
is the binding constraint on 2026 serving — and each attacked it with a different mechanism, so
"KV cache" is no longer one thing a serving system can manage uniformly.*

### 1.5 Where this meets M\* explicitly

Make this connection in the report — it is the reason the architecture report is interesting *to him*:

- M\* observes prior systems assume *"a single autoregressive loop"* and use a *"Flat DAG"*. True
  between components. **Also now true between *layers* of one model.**
- M\*'s Walks are *"a labeled subgraph for one phase of behavior."* A Kimi-K3 decode step has **two
  cost regimes inside one Walk** (KDA layers: O(1) state, compute-bound; MLA layers: O(ctx) KV,
  memory-bound). The Walk is the right abstraction but the wrong *granularity* for this.
- M\* already has *"sharded MoE/KV cache"* and per-Walk placement in the YAML. **The natural
  extension is per-layer-*class* KV policy** — different block sizes, eviction, and precision for
  KDA vs MLA vs DSA layers. That's a concrete "what I'd build next" that lands inside his codebase.
- The 12.5× V-JEPA-2 rollout win came from *"a persistent KV cache across steps."* Hybrid models make
  KV persistence **per-layer-class**, so the same trick needs a heterogeneity-aware allocator.

---

## 2 · Hardware reality: what 8× H100 can hold

**Detected this session** (`tan-cpu-0-0`, **no GPU**, CFS quota 198, 1.5 TiB RAM, HF+PyPI reachable,
`runai` project `gen-opt-vn` quota 24). **Re-detect on every node** — per global `CLAUDE.md`,
container hardware is not stable across sessions.

576 GiB usable at `gpu_memory_utilization=0.9`, against **real weight bytes**:

| Model | Weights | Fits 8×H100? | Min H100s |
|---|--:|:--|--:|
| **GLM-4.7-Flash** | 29.1 GiB | ✅ **1 GPU** | 1 |
| **Qwen3.8-27B** | 51.7 GiB | ✅ **1 GPU** | 1 |
| **DeepSeek-V4-Flash** | 148.6 GiB | ✅ **3 GPUs** | 3 |
| **GLM-5.3-Flash** | 305.8 GiB | ✅ (tight) | 5 |
| DeepSeek-V4-Pro | ~740 GiB | ❌ | ~11 |
| **Kimi-K3** | **1453.7 GiB** | ❌ | **~21** |

**This is much better than Revision 1.** The "Flash" tier means **four of six 2026 models fit**, and
three fit comfortably. Revision 1 could measure no MLA model at frontier scale; Revision 2 can
measure a real DSA + FP4-expert + compressed-KV model (V4-Flash) on 3 GPUs.

Still out of reach: **Kimi-K3 (1.45 TiB) and V4-Pro**. Handle exactly as before — state it on page 1
as a finding (*the frontier now spans a 50× weight range within one generation*), then use proxies:

| Axis to measure | Proxy that fits | Extrapolates to |
|---|---|---|
| Hybrid KDA + full-attn interleave | **GLM-5.3-Flash** (34 KDA + 11 DSA) | Kimi-K3 (69 KDA + 24 MLA) |
| DSA + FP4 experts + compress_ratios | **DeepSeek-V4-Flash** | DeepSeek-V4-Pro |
| MLA at small scale | **GLM-4.7-Flash** (1 GPU!) | any MLA model |
| Hybrid linear attention, **dense** | **Qwen3.8-27B** | isolates attention from MoE |
| Fine-grained MoE, 2025 baseline | Qwen3-30B-A3B (28 GiB FP8) | the previous generation |

**Do not attempt Kimi-K3 or V4-Pro with weight offload** — PCIe-dominated numbers say nothing about
architecture and would consume the entire GPU budget.

---

## 3 · Layer reduction as a throughput proxy (your idea — adopted, with guardrails)

**Your proposal:** without speculative decoding, using base models only, cut `num_hidden_layers` while
keeping the architecture, to get throughput numbers cheaply.

**This is the right instrument for this study, and it is a better fit here than it would have been for
Revision 1** — because the quantity under test (`KV bytes/token`, `E/k`, per-layer-class cost mix) is
a **per-layer property**. Cutting layers scales the model down while *preserving exactly the thing
being measured*. Concretely it makes Kimi-K3's 69-KDA/24-MLA interleave measurable on hardware that
could never hold the real thing.

### 3.1 What it preserves and what it destroys

| Preserved (safe to measure) | Destroyed (never claim) |
|---|---|
| KV bytes/token **per layer** | model quality / accuracy — **weights are random** |
| Attention pattern & interleave period | anything about acceptance rates or output text |
| `E/k` dilution and routing behavior | absolute end-to-end throughput of the real model |
| Per-layer-class cost mix (KDA vs MLA vs DSA) | emergent load-balance of *trained* routers |
| Kernel shapes, GEMM dims, grouped-GEMM widths | total memory footprint of the real model |
| Scaling *slope* vs. layer count | |

### 3.2 The interleave constraint — the one real trap

Hybrid models place full-attention layers at **interval 4**. A naive truncation breaks the ratio:

| Model | Real L | Pattern | ✅ Valid reduced L | ❌ Invalid |
|---|--:|---|---|---|
| Kimi-K3 | 93 | full at 4,8,…,92,93 | **12, 24, 48** (keep 1:3) | 23, 46 (breaks period) |
| Qwen3.8-27B | 64 | `full_attention_interval: 4` | **16, 32** | 25% of 64 = 16 ✅ but verify |
| GLM-5.3-Flash | 45 | 34 KDA + 11 DSA, interval 4 | **12, 24** | 11, 22 |
| DeepSeek-V4-Flash | 43 | `compress_ratios` alternate 4/128 | **12, 20** (even, keep both) | odd L |

**Rule: reduced `L` must be a multiple of the interleave period (4), and you must verify the
resulting layer-type histogram matches the original ratio.** `bench/reduce.py` should *assert* this
and print the before/after histogram — a silent ratio change would invalidate every number
downstream. Also truncate `compress_ratios` and `layer_types`/`full_attn_layers` lists consistently,
and keep `first_k_dense_replace` proportional.

### 3.3 Validation gate (do not skip)

The proxy is only trustworthy if throughput scales predictably with `L`. Establish that on a model you
can also run at full size:

1. Take **GLM-4.7-Flash** (29 GiB, 1 GPU, runs at full L=47).
2. Measure throughput at L = 12, 24, 36, 47.
3. Fit; confirm per-layer cost is ~linear with a constant offset (embeddings, sampling, launch).
4. **Report the fit quality.** If it's linear, layer-reduced numbers extrapolate with a stated error
   bar. If not, the proxy is only good for *comparing architectures at equal L* — still useful, but
   say so.

Then all cross-architecture comparisons run at **equal reduced L** (e.g. L=24 for everything), which
is the cleanest form of the comparison anyway: same depth, same layer budget, different architecture.
**That is arguably a better experiment than comparing real models of wildly different sizes.**

### 3.4 Random weights are fine here — say why

Throughput at fixed shapes is **independent of weight values** (barring MoE routing collapse, see
below), so `from_config` + random init avoids downloading 1.45 TiB. State this explicitly in the
report so no reader thinks quality was measured.

**One real caveat:** random routers route ~uniformly, which is the *best case* for `D(B) ≈ min(Bk,E)`
expert coverage. A trained router is more skewed, so **real expert-read cost may be lower than the
proxy predicts.** Cross-check on one small real model with trained weights (Qwen3-30B-A3B) by logging
actual expert-hit counts vs. the uniform prediction. This turns a limitation into a measurement.

---

## 4 · Dataset / evaluation sketch (deliberately coarse — to be planned properly later)

Per your instruction: **plan evaluation later.** This section fixes only what's needed so Phase 1
doesn't paint us into a corner. **No speculative decoding; base models only.**

**What we're measuring is throughput and latency, not quality.** So the "dataset" only needs to
supply *realistic token-length distributions and prefix-sharing structure* — not correct answers.
That makes it cheap.

Candidate small sources (decide in Phase 4, verify each still exists before use):

| Purpose | Candidate | Why | Size |
|---|---|---|---|
| Realistic mixed lengths | **ShareGPT** (the vLLM-standard benchmark input) | what every serving paper uses → comparable | ~1 GB, subsample 500 reqs |
| Long-context / 1M-ctx claims | **LongBench** or synthetic needle-style padding | exercises the KV story where it matters | small subsets |
| Prefix sharing / agentic | **TraceLab** (his lab's own 2026 trace: ~4,300 coding-agent sessions) | *the* justification for agentic shapes, and it's theirs | check availability |
| Pure scaling sweeps | **synthetic** fixed (in,out) pairs | full control, zero download | 0 |

**Recommendation:** do the systematic sweeps on **synthetic** shapes (total control of the batch ×
context grid, which is what the regime map needs), then a **single ShareGPT run** for external
comparability, and **TraceLab if obtainable** for the prefix-cache section. Keep total download under
a few GB. **Do not** pull an accuracy benchmark suite — we are not measuring accuracy, and random-init
proxies could not measure it anyway.

Fixed now so Phase 1 stays consistent: report **throughput (tok/s)**, **TTFT**, **TPOT**, and
**achieved memory bandwidth / FLOP fraction** — the last one is what makes a bottleneck claim
falsifiable rather than asserted.

---

## 5 · The four `task.md` questions, updated

### Q1 · Bottleneck by workload and batch size

**Revision 2 answer: there is no single bottleneck per step, because layer classes within one model
bind on different resources simultaneously.** The regime map becomes 3-D — batch × context ×
**layer class**:

| Layer class | Binding at small B | Binding at large B | Scales with ctx? |
|---|---|---|---|
| **KDA / linear** | weight read | compute | **No** — O(1) state |
| **Full attn (MLA)** | KV read | KV read | **Yes** — O(ctx) |
| **DSA / sparse** | index/selection kernel | compute | Sub-linear (top-k) |
| **MoE FFN** | **expert read** (B\*=8K–19K) | expert read | No |
| **Dense/shared FFN** | weight read | compute (B\*≈148–296) | No |

Consequence worth stating: **as context grows, the *same model* shifts which of its own layers
dominates.** At 4K ctx a Kimi-K3 step is expert-read-bound; at 1M ctx the 24 MLA layers dominate
everything. **The crossover context length is computable per model** and is a genuinely useful number
for anyone sizing a deployment. Compute it in Phase 1; measure it later.

Method (when we get to it): attribute per-operator time with `torch.profiler`/Nsight using
**NanoFlow's §2.2 taxonomy** — *dense / attention / network / other* — extended with a **layer-class
dimension**, so results stay comparable to his Fig. 4 while showing the new axis.

### Q2 · Prefix cache challenges

Revision 1's answers hold. The 2026 generation adds three sharper ones:

1. **Hybrid models break prefix caching's core assumption.** Prefix caching replays KV blocks. **A
   linear/KDA layer has no KV to cache — it has a recurrent state that must be *rolled forward*.**
   You cannot random-access into a linear-attention state the way you index a KV block. So a cached
   prefix restores only the ~25% full-attention layers; the 75% KDA layers must either be recomputed
   or have their state snapshotted (0.023 GiB/seq for Qwen3.8 — cheap to store, but a *different
   mechanism* with different eviction semantics). **This is the single most interesting prefix-cache
   question in the report** and it is new with this generation.
2. **Per-layer `compress_ratios` break uniform block sizing.** DeepSeek-V4-Flash layers differ 32× in
   KV bytes (ratio 4 vs 128). One `kvcache_block_size` (nano-vllm asserts `% 256 == 0`) is either
   wasteful on compressed layers or too coarse on uncompressed ones.
3. **DSA `index_topk` selection isn't cacheable the way KV is.** A resumed sequence must re-run
   selection; a cached prefix restores keys, not *which* keys the indexer would pick.

Plus the Revision-1 point that still stands: prefix-cache hits **don't reduce expert-read cost**, only
attention/prefill work — so at `E/k` = 42–64× the win is far smaller than dense-model intuition says.

### Q3 · Speculative decoding

**Excluded from the experimental plan per your instruction** (base models, no spec decode). Keep it as
**architecture analysis only** in Phase 1, because the configs make a clean point:

| Model | MTP field | Speculation story |
|---|---|---|
| DeepSeek-V4-Flash / Pro | `num_nextn_predict_layers: 1` | native MTP head |
| GLM-4.7-Flash / GLM-5.3-Flash | `num_nextn_predict_layers: 1` | native MTP head |
| Qwen3.8-27B | `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` | native, shares embeddings |
| **Kimi-K3** | `num_nextn_predict_layers: 0` | **none** — needs external draft |
| GLM-5.3-Flash | + `index_share_for_mtp_iteration: true` | **indexer shared across MTP iterations** |

That last field is a nice catch: GLM-5.3 explicitly co-designs sparse-attention indexing with MTP, so
the draft doesn't re-pay selection cost. And the `E/k` argument predicts speculation should be *more*
valuable this generation (B\*_MoE = 8K–19K keeps these models memory-bound at any real batch) — state
the prediction, note it's untested here.

### Q4 · Serving cost

`$/Mtok = (N_GPU × $_GPU-hr) / (throughput × 3600) × 10⁶`, reported measured / modeled /
SLO-conditioned (the last per **PolyServe**'s framing).

Revision 2 changes the punchline. The **min-GPU** column from §2 is now a 21× spread within one
generation (1 GPU for GLM-4.7-Flash → ~21 for Kimi-K3). Combined with the 50× KV spread, cost per
token is now dominated by **architecture choices, not parameter count** — GLM-4.7-Flash at 29 GiB with
full MLA may beat much larger models on $/Mtok at long context by a wide margin. That's the
counterintuitive result to build toward, and it's now *measurable* rather than modeled.

---

## 6 · Phasing (Phase 1 first, evaluation later)

| # | Phase | GPU | Output | Status |
|---|---|:--:|---|---|
| **1** | **Architecture report** | **none** | `report.md` §1–2 + `bench/params.py` + `figs/arch_*.pdf` | **Do now — hand to lab first** |
| 2 | Cost/roofline model | none | `bench/roofline.py`, crossover-ctx per model | Do now |
| 3 | `bench/reduce.py` + interleave validator | none | layer-reduced configs, ratio assertions | Do now |
| 4 | **Evaluation design** | none | dataset choice, sweep grid | **Plan later (§4 is a sketch)** |
| 5 | Layer-reduction validation gate | 1 | GLM-4.7-Flash L∈{12,24,36,47} linearity fit | Gates 6–7 |
| 6 | Cross-arch sweeps at equal L | 1–8 | regime map, per-layer-class attribution | After 5 |
| 7 | Prefix cache / hybrid state | 1–8 | the KDA-state question (Q2 #1) | After 5 |
| 8 | Cost synthesis + writeup | none | `report.md` complete | Continuous |

**Phases 1–3 need no GPU and are the agreed deliverable.** This session's node has none — so start
there regardless of allocation.

---

## 7 · Tooling

Unchanged from Revision 1 (vLLM primary, one SGLang cross-check, nano-vllm as instrumentation
vehicle, CUDA events + `torch.profiler` + Nsight, conda env `nano-vllm`, models to
`/prj/.../vol22-scratch` never `$HOME`) with three additions:

- **`transformers` 4.57.1 is installed; several 2026 configs declare `transformers_version: 5.0.0rc0`**
  (GLM-4.7-Flash) and Kimi-K3/Qwen3.8 use `auto_map` with **remote code**. Verify loadability in
  Phase 3 — this is the most likely blocker and it costs nothing to check now.
- **FP4 support** (`expert_dtype: fp4`) needs a recent vLLM; H100 has no native FP4 (that's Blackwell),
  so expect emulation or a required FP8 fallback. **Check before planning any V4-Flash run**, and if
  FP4 falls back to FP8 the §1.3 B\* arithmetic changes — record which path actually ran.
- **`bench/reduce.py`** — build reduced configs via `AutoConfig` → truncate `layer_types`,
  `full_attn_layers`, `compress_ratios`, `kda_layers` consistently → `from_config` with random init →
  assert layer-type histogram ratio preserved.

---

## 8 · Risks

| Risk | Mitigation |
|---|---|
| **Remote-code / transformers-version mismatch** | Check in Phase 3 (no GPU needed). Most likely blocker. |
| **No FP4 on H100** | Expect FP8 fallback; record which ran; adjust §1.3 arithmetic accordingly. |
| **`compress_ratios` semantics guessed** | Flagged in §1.2. Read `modeling_deepseek_v4.py` from the repo to confirm before publishing the ~7.4 KiB/tok number. |
| **Layer reduction non-linear** | §3.3 gate detects it; fall back to equal-L comparison only. |
| **Random routing overstates expert cost** | Cross-check trained-router hit counts on Qwen3-30B-A3B (§3.4). |
| Kimi-K3 / V4-Pro unmeasurable | By design: proxy ladder §2 + labelled modeling. |
| Configs change under us | All numbers dated 2026-09-01; `bench/params.py` re-derives from live configs. |
| Scope creep across 10 models | Phase 1 is config-only, so breadth is cheap. Restrict *measurement* to the 4 that fit. |

---

## 9 · Definition of done — Phase 1 (the near-term deliverable)

- [ ] Architecture table (§1.1) with every field traceable to a config key, dated, re-derivable by script.
- [ ] KV-bytes/token derivation (§1.2) per model, **labelled analytical**, with the `compress_ratios`
      assumption either confirmed from modeling code or explicitly flagged.
- [ ] `E/k` + FP4 B\* analysis (§1.3) — the 2025→2026 trend quantified.
- [ ] Mechanism-organized similarity/difference table (§1.4) + the one-sentence synthesis.
- [ ] The M\* connection (§1.5) stated as a concrete extension, not a compliment.
- [ ] Crossover context length per model (where MLA layers overtake expert read).
- [ ] Weight sizes from safetensors index + min-GPU count; the 21× / 50× spreads called out on page 1.
- [ ] `bench/params.py` regenerates every number from live configs.
- [ ] Scope limits (no measurements yet; random-init proxy planned) stated **on page 1**.

---

## Appendix A · Provenance

All architecture fields read from HuggingFace `config.json` and `model.safetensors.index.json` on
**2026-09-01**: `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4-Pro`, `moonshotai/Kimi-K3`,
`Qwen/Qwen3.8-27B`, `zai-org/GLM-4.7-Flash`, `zai-org/GLM-5.3-Flash`, plus 2025 baselines
`zai-org/GLM-4.5`, `zai-org/GLM-4.5-Air`, `Qwen/Qwen3-235B-A22B`, `Qwen/Qwen3-30B-A3B`,
`deepseek-ai/DeepSeek-V3`, `moonshotai/Kimi-K2-Instruct`, `deepseek-ai/DeepSeek-V2-Lite`.

**Also on HF and worth a line in the report** (shows the tier structure is industry-wide):
`Qwen/Qwen3.8-Flash-Next` (`Qwen4ExpForConditionalGeneration`, `hc_count`/`hc_lowrank`,
`indexer_budget: 2048`), `deepseek-ai/DeepSeek-V4-Flash-Base` (base ckpt — **use this one** for
layer-reduction work), `DeepSeek-V4-Flash-DSpark`, `nvidia/*-NVFP4`, `sgl-project/DeepSeek-V4-Flash-FP8`.

**Unverified / to confirm:** `compress_ratios` semantics · whether Kimi-K3's `num_experts: 896` counts
per-layer or globally · exact DeepSeek-V4 attention formulation (`o_lora_rank`, `o_groups`,
`num_hash_layers`, `hc_*` fields have no published paper I've read) · M\* arXiv 2606.12688 contents
beyond the project page.

## Appendix B · Reading list

**Primary — his own work:**

| Work | Venue | Why |
|---|---|---|
| **M\*** — "Modular, Extensible Serving System for Multimodal Models" | 2026, arXiv 2606.12688 | **The Walk Graph. The thesis extends this inward.** Read the paper, not just the page. |
| **NanoFlow** (Kan Zhu, first author) | OSDI 2025 | Cost model, `T_R`, §2.2 op taxonomy, compute-bound claim |
| **DynaFlow** | MLSys 2026 | Programmable operator scheduling — where layer-class-aware scheduling would live |
| **Fiddler** | ICLR 2025 | MoE serving under memory pressure |
| **Tactic** / **Quest** | ICLR 2026 / ICML 2024 | Sparse attention — now *in* the models (DSA), not bolted on |
| **BlendServe** | ASPLOS 2026 | Resource-aware prefix tree |
| **TraceLab** | 2026 | Agentic traces — workload shapes + possible dataset |
| **PolyServe** | 2025 | SLO-conditioned cost framing |
| **Atom** | MLSys 2024 | Quantization — context for FP4 experts |
| **FlashInfer** | MLSys 2025 (best paper) | The attention kernel layer; M\* uses its paged attention |

**Model-side:** DeepSeek-V3.2-Exp report (DSA/indexer origin) · Kimi Linear / KDA · Qwen3-Next
(hybrid linear attention lineage) · GLM-4.5/4.6 reports.

> **People note:** Kan Zhu is co-advised by **Baris Kasikci and Arvind Krishnamurthy**. **SyFI**
> ("Systems for Future Intelligence") is co-directed by **Kasikci and Stephanie Wang** — both are M\*
> authors, and Stephanie Wang is already PI #6 on `../PhD/report.md`. M\* is a Stanford–UW
> collaboration (Jha, Sagan, Kamahori, Meng, Sanda, Zettlemoyer, Hsu, Leskovec, Kasikci, Wang).
