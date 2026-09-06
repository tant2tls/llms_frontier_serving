> ARCHIVED SESSION RECORD — not current instructions or conclusions. Original location: `potential_questions.md`. See [current report](../../report.md), [debugging guide](../../fix_bug.md), and [session handoff](../../WORKFLOW.md). Historical claims may be superseded.

# Potential questions from Kan and the SyFI lab

**Purpose.** Rehearsal sheet for presenting [`deepseek_v4_flash/report.md`](../../deepseek_v4_flash/report.md). The audience
is **ML systems experts** — Kan Zhu (NanoFlow author), Baris Kasikci, Arvind Krishnamurthy, and their
students. They will not ask "what is MoE." They will attack **methodology, normalization, and whether
each number means what I claim**.

**How to use this.** For each question: the answer I can defend today, and — where I can't — the honest
"not measured, here's the experiment." **Saying "I don't know, here's how I'd find out" is a better
answer than a confident guess.** These people can tell the difference instantly, and a wrong confident
answer costs more credibility than the gap it papers over.

**Ranked by likelihood × damage if I fumble it.** §1 items are near-certain and go to the core of the
report's validity.

---

## 1. Tier 1 — near-certain, and they undermine the report if I miss them

### Q1.1 "Is chunked prefill on? What's `max_num_batched_tokens`?"

**This is the question most likely to puncture my "prefill-bound" framing, and I should raise it myself
before they do.**

**Answer:** Yes. **[M]** From the server log: `Chunked prefill is enabled with max_num_batched_tokens=8192`
(`scheduler.py:242`), vLLM 0.28 default. So:

| ISL | Chunk-steps to prefill ONE request |
|--:|--:|
| 16,384 | 2 |
| 65,536 | 8 |
| 131,072 | 16 |
| 262,144 | 32 |

**Consequence I must state plainly:** my TTFT numbers are **not** pure prefill-compute measurements.
A 131K prompt is split into 16 chunks that interleave with other requests' decode steps, so TTFT includes
scheduler round-trips and queueing behind other requests' chunks. At c=64 × ISL 16K there are 128
chunk-steps of prefill work competing with decode.

**What survives:** the *ranking* and *order of magnitude* (TTFT 26 s at 131K) are real and are what a
user experiences. **What does not survive:** any claim that TTFT measures prefill FLOPs. And "this
workload is prefill-bound" is better stated as **"wall-clock is dominated by prefill *work*, scheduled in
8192-token chunks"** — which is a statement about the engine's scheduling regime as much as the model.

**Follow-up they'll ask: "did you sweep `max_num_batched_tokens`?"** No. That's a real gap — it is
plausibly the single highest-leverage server knob for this workload, and it's a one-line sweep
(4096/8192/16384/32768). I should run it before presenting.

### Q1.2 "Are these numbers with speculative decoding on or off?"

**Answer:** Primary numbers are **base model, MTP OFF** (`deepseek_v4_flash/run_nomtp.sh`,
`deepseek_v4_flash/results/mtp-off/`). This was a
deliberate correction — comparing an MTP-accelerated model to a base model compares *inference tricks*,
not architectures. §5 has the isolated A/B, identical seeds:

| Concurrency | off | on | speedup |
|--:|--:|--:|--:|
| 1 | 92 | 115 | **1.25×** |
| 64 | 383 | 380 | **0.99×** |

**The gain decays to nothing as batch grows** — at c=1 the draft head is free parallel work; at c=64 it
competes with real tokens. Cost: **2.4% of KV capacity** (`num_gpu_blocks` 43,197 → 42,149).

### Q1.3 "What dtype? Don't tell me BF16 because the config says so."

**Answer:** **FP8 e4m3** (attention/dense, block 128×128, ue8m0 scales) + **MXFP4** (routed experts).
`torch_dtype: bfloat16` in `config.json` is the *activation* dtype and is a decoy. Engine confirms:
`quantization=deepseek_v4_fp8`, and `quant_config.py:75` logs `expert_dtype resolved to 'fp4'`.

**Why I care:** comparing an FP8+FP4 model against a BF16 model and attributing the delta to architecture
is the easiest way to publish a wrong result. Every run records dtype in `manifest.txt`.

### Q1.4 "380 tok/s — is that good? Compared to what?"

**Answer:** Unanswerable as raw tok/s; that's why §1 normalizes three ways. At c=64, base model:

| Metric | Value |
|---|--:|
| Output tok/s | 383 |
| per B-active (13.49 B) | 28.4 |
| per B-total (290.9 B) | 1.32 |
| **per GPU (8×H100)** | **47.9** |

**Per-GPU is the only one that translates to cost.** And it's the only defensible cross-model basis:
V4-Flash needs ≥3 H100s, GLM-4.7-Flash 1, Kimi-K3 ~21 — a **21× spread in minimum footprint**. Comparing
raw tok/s across that is comparing silicon budgets.

### Q1.5 "You claim it's not memory-bound. Prove the bandwidth number."

**Answer, and I must lead with the weakness:** my bandwidth figures are **[A] analytical**, not measured —
`active_params × 1 B/param × decode_steps/s`, an upper bound. They come out **under 5% of the 26.8 TB/s
aggregate peak and *falling* with batch**, which is inconsistent with memory-bound decode.

**But I did not measure achieved bandwidth**, and my model ignores KV traffic, activations, and all-to-all.
So the honest claim is: **"the roofline prediction failed to confirm, and I have two competing
explanations I cannot yet separate"** — (a) the workload never enters the decode-bound regime at
ISL:OSL ≥ 64:1, (b) something other than HBM binds.

**What I found while preparing this, and should run before presenting:** vLLM 0.28 exposes
`vllm:estimated_flops_per_gpu_total` and `vllm:estimated_read_bytes_per_gpu_total` behind
`--enable-mfu-metrics` (currently `False`, so they read 0.0). **That gives engine-side FLOP and byte
counters — a far better bandwidth measurement than my analytical model.** Turning it on is one flag.

### Q1.6 "`B*_MoE ≈ 6,300` — did you validate it?"

**Answer: no, and I will not present it as validated.** It's an **[A]** prediction. Every run was at
ISL:OSL ≥ 64:1; the prediction lives at *high OSL, low ISL, huge batch*. The discriminating experiment is
specified but not run: **ISL 512, OSL 4,096, concurrency ≥512.**

Also note **`--max-num-seqs` defaults to 128**, so c=512 needs that raised — a batch of 6,300 tok/step is
not reachable without it, and possibly not reachable at all on one node. Worth saying that the prediction
may be **untestable** at this scale rather than merely untested.

---

## 2. Tier 2 — methodology and fair comparison

### Q2.1 "Expert parallelism is on. How are experts sharded, and is all-to-all on the critical path?"

**Answer:** **[M]** `[EP Rank 0/8] Expert placement strategy: linear. Local/global number of experts:
32/256`. So each GPU holds 32 experts; routed weights shard EP8 (**35.4 B params/GPU**), dense parts
shard TP8 (0.93 B/GPU).

**The systems consequence, and this is my leading hypothesis for Q1.5:** with k=6 and 8 ranks, the
expected number of experts touched *per rank per token* is **0.75**. Most ranks contribute 0 or 1 experts
per token, so (a) load imbalance across ranks is structural, and (b) **all-to-all dispatch/combine sits on
the critical path of every decode step**. That is a latency term that doesn't show up as HBM traffic —
consistent with <5% achieved bandwidth.

**Not measured.** Confirming it needs Nsight Systems or `torch.profiler` with NanoFlow's
dense/attention/**network**/other taxonomy. The "network" bucket is exactly the hypothesis.

### Q2.2 "TP=8 for a 148 GiB model that fits on 3 GPUs. Isn't that wasteful, and doesn't it distort throughput?"

**Answer: yes, and it's a real confound.** Weights are 18.6 GiB/GPU at TP=8; peak KV was 30.2%. The model
fits on 3 H100s. **TP=8 on an oversized allocation inflates per-GPU cost and adds all-to-all/all-reduce
that TP=4 wouldn't.** So my **47.9 tok/s per GPU is pessimistic** — a TP=4 or TP=2 config would likely
show better per-GPU throughput.

**Not measured.** A TP sweep (2/4/8) at matched context is the clean experiment and would strengthen every
cost number. I'd flag this as a limitation rather than defend TP=8 as optimal.

### Q2.3 "Your prefix-cache and 256K numbers are MTP-on but the rest is MTP-off. Is that a fair mix?"

**Answer: no, and it's labelled inline as such.** The 3.7× prefix-sharing *ratio* should be
spec-decode-neutral (both arms have the same MTP state), so the finding holds. But the **absolute** levels
in that table aren't comparable to §2's base-model numbers. Same for the 262K context point. Two one-off
reruns close it; listed in §8.

### Q2.4 "How do you compare models with different tokenizers?"

**Answer:** Carefully, and this bites twice.

1. **Throughput in tok/s isn't comparable across tokenizers** — a model with a denser tokenizer does more
   work per token. Cross-model comparison should be on **tokens-per-fixed-corpus** or bytes/s, not raw
   tok/s.
2. **TraceLab's token counts come from Claude/GPT tokenizers** (top models `gpt-5.5`,
   `claude-opus-4-8`), not DeepSeek's. So ISL p50 = 132,092 is an approximation under a *different*
   tokenizer. It's the right anchor for *grid design*, wrong for precise claims.

### Q2.5 "You used synthetic random tokens. Doesn't that misrepresent MoE routing?"

**Answer: yes, and in a specific direction.** Random tokens route ~uniformly across experts, which is
**best-case for expert coverage** — so measured expert-read cost is likely *higher* than real text, where
routing is skewed and hot experts stay cached. This is the same trap `CLAUDE.md` flags for random-init
routers.

It also means **prefill FLOPs are realistic but MoE routing is not**. Cross-check: trained-router expert
hit counts, or replay TraceLab via `timed_trace`. Not done.

### Q2.6 "Was the KV cache cold? Prefix caching is on — did you contaminate your own numbers?"

**Answer: I did, caught it, and fixed it.** First sweep used a fixed seed across concurrency points →
identical prompts → c=4 reported **640 tok/s**, a 5.1× phantom. Fresh seed: **214 tok/s**, `newcachehits=0`.

Every point in the report is unique-seed with `newcachehits = 0` verified against
`vllm:prefix_cache_hits_total`. Contaminated files retained in `deepseek_v4_flash/results/mtp-on/` for audit.

**The general lesson, which is worth stating to this audience:** benchmarking a prefix-caching engine with
a fixed prompt set measures your cache, not your model.

### Q2.7 "`--ignore-eos` — doesn't that make outputs unrealistic?"

**Answer:** Yes, deliberately. Without it this reasoning model picks its own output length and OSL stops
being a controlled variable. Trade-off: the decode phase never sees natural early-stopping, so the OSL=256
column is exactly 256 tokens for every request. For realism-oriented runs (the planned ShareGPT
comparability run) I drop the flag.

### Q2.8 "Only 8 or 16 prompts per point. Where are your error bars?"

**Answer: I have none, and that's a genuine weakness.** `num_prompts` was clamped by a prefill budget so
the 256K point wouldn't run for an hour, giving 8–128 prompts per point and **single runs, no repeats**.

What I can offer instead: an **internal consistency check**. At c=64, TTFT + TPOT×OSL = 3.69 + 0.151×256
= 42.4 s → 386 tok/s predicted vs **383 measured [M]** (0.8%). That's self-consistency, not variance.
Fix: 3 repeats per point with a different seed, report median and spread.

---

## 3. Tier 3 — architecture and the thesis

### Q3.1 "Your `E/k = 42.7×` thesis — but you measured sparsity of 21.6×. Which is it?"

**Answer: both, and the gap is a finding.** **[M]** from safetensors headers: 290.9 B total, 13.49 B
active → **21.6× sparsity**, against `E/k = 42.7×`.

The reconciliation: the **dense remainder** — attention 5.21 B + shared expert 1.11 B + embed 0.53 B =
**6.85 B — is 51% of the active budget** while being 2.4% of total. Expert sparsity doesn't buy 42.7×
because **attention doesn't sparsify**. Any `E/k`-based prediction of decode cost is therefore an
overestimate of the MoE contribution; the dense half sets a floor.

This *supports* the report's thesis (heterogeneity within one stack) while *correcting* the arithmetic.

### Q3.2 "Why is `--block-size 256` mandatory? Why not tune it?"

**Answer:** `sparse_mla.py:53` returns `[256]` — a single-element list. Because
`storage_block_size = block_size // compress_ratio` with `compress_ratios ∈ {4,128}`:

| `block_size` | storage blocks | verdict |
|--:|---|---|
| **256** | 64, **2** | only valid |
| 128 | 32, **1** | degenerate |
| 64 | 16, **0** | floors to zero |

`compressor.py:180` hardcodes it and ends `TODO(yifan): make block size automatically determined and
configurable`.

**The punchline for this audience:** we pass 256 and the engine resolves `block_size="4"` **[M]** —
`min()` across KV groups at `core.py:322`, with `hash_block_size = gcd(...)`. Prefix-cache hashing runs at
**4-token granularity**. That's the report's thesis as a vendor TODO: heterogeneous layers can't agree
what a block *is*, so the engine hardcodes one layout and pins the user's knob to it.

### Q3.3 "FP4 — doesn't that double your compute roofline?"

**Answer: no, and this was a correction to my own earlier draft.** `fp4_gemm_kernel` unpacks FP4→FP8 in
shared memory and runs H100's **FP8** tensor cores. **Byte traffic halves; FLOP/byte at the tensor core is
unchanged.** So do *not* claim FP4 doubles `B*_dense` to 296. H100 has no FP4 hardware; FP4 here is a
storage format.

### Q3.4 "Why is KV usage only 30% at 256K context? That seems too good."

**Answer:** V4's per-layer compression (`compress_ratios` 4/128 — a 32× spread *within one model*) plus
`--kv-cache-dtype fp8`. **[M]** `num_gpu_blocks = 43,197`, peak KV 30.2% at 256K × 8 seqs. Consistent with
the ~50× KV-cost reduction vs the 2025 generation (GLM-4.5: 368 KiB/tok). **KV capacity is not the
constraint for this model** — which is itself a notable 2026-generation finding, since paged-KV capacity
was *the* constraint in 2024–25.

### Q3.5 "How does DSA/CSA/HCA actually work? Did you verify `compress_ratios` semantics?"

**Answer: partially, and I should be candid.** I verified the *engine's* treatment
(`storage_block_size = block_size // compress_ratio`, the compressor's hardcoded 4/8 block shapes) and
observed `attn.indexer.*` / `attn.compressor.*` / `hc_attn_base` tensors in the checkpoint.

**Not verified:** the mathematical semantics of CSA/HCA from the paper (**arXiv 2606.19348**), and whether
the ~7.4 KiB/tok figure in `plan.md` is right. I'm treating vendor claims (V4-Pro: *"27% of single-token
inference FLOPs and 10% of KV cache"* vs V3.2 at 1M) as **vendor claims**, cited not endorsed.

---

## 4. Tier 4 — the questions I most want to be asked, because the answer is "not yet"

These are the report's real gaps. Better to name them first than be caught.

| Question | Honest answer |
|---|---|
| **"Where's the cross-model comparison? `task.md` Q4 asks for it."** | Not done. One model served. GLM-4.7-Flash (29 GiB, 1 GPU) is the cheapest next run and is the recommended next step. |
| **"Where's the dense control?"** | Not done. Without it, "MoE is memory-bound" is unfalsifiable — it could be the setup. Qwen3.8-27B (dense, BF16-native) is designated. |
| **"Did you replay TraceLab or just read its percentiles?"** | Only percentiles, to anchor the grid. vLLM's `timed_trace` loader can replay it via `hash_ids` (chunk-hash → seeded tokens → real cache hits). Converter not written. |
| **"Operator breakdown? NanoFlow's taxonomy?"** | Not done. Needs `torch.profiler`. This is precisely what would resolve Q1.5/Q2.1 — the network bucket is my hypothesis. |
| **"Second engine?"** | vLLM only. `plan.md` wants one SGLang cross-check. |
| **"Multi-node? Kimi-K3 needs ~21 GPUs."** | Out of reach (24-GPU quota, single node measured). Proxy ladder in `plan.md`; **never weight-offload** — numbers would be PCIe-dominated and architecturally meaningless. |
| **"Quality/accuracy?"** | Not measured, deliberately, per your email. Also unmeasurable on layer-reduced random-init proxies. |

---

## 5. Questions about *me*, not the numbers

Likely in a PhD-application evaluation context.

- **"What surprised you?"** That the MTP gain vanishes at high concurrency (1.25× → 0.99×). I expected a
  roughly constant speedup and got a regime-dependent one. Also that vLLM silently resolves
  `block_size` 256 → 4.
- **"What did you get wrong?"** Three things, all in the report: (1) advised removing `--block-size 256`
  before reading `sparse_mla.py`; (2) cited 84.4% MTP acceptance from a ~3K-token sample when the
  full-sweep figure is **59.3%**; (3) shipped a seed-contaminated 640 tok/s point. All corrected with the
  evidence retained.
- **"What would you do with a month?"** Order: (1) `--enable-mfu-metrics` + `torch.profiler` to settle
  what actually binds; (2) `max_num_batched_tokens` and TP sweeps — likely the biggest real wins;
  (3) GLM-4.7-Flash + Qwen3.8 dense control for a real cross-model table; (4) TraceLab replay.
- **"What's the one-sentence takeaway?"** For long-context agentic serving of this model, **prefix-cache
  engineering dominates every other lever** — 3.7× throughput and 10× cost, versus 4.2× from 64× the
  batch and ~0× from speculative decoding at production concurrency.

---

## 6. Pre-presentation checklist

Cheap things that close the most likely attacks. Ordered by value per hour.

- [ ] **`--enable-mfu-metrics`** and re-run one point → replaces analytical bandwidth with engine-side
      FLOP/byte counters. Directly answers Q1.5. *One flag.*
- [ ] **Sweep `max_num_batched_tokens`** (4096/8192/16384/32768) at ISL 131K → answers Q1.1, plausibly the
      biggest server-side win.
- [ ] **TP sweep** (2/4/8) at matched context → answers Q2.2, fixes the pessimistic per-GPU number.
- [ ] **3 repeats per point**, report median + spread → answers Q2.8.
- [ ] Rerun 262K + prefix sweep **MTP-off** → removes the mixed-basis caveat (Q2.3).
- [ ] **GLM-4.7-Flash** on 1 GPU → first cross-architecture point (Tier 4).
- [ ] Read **arXiv 2606.19348** to move CSA/HCA from vendor claim to verified (Q3.5).
