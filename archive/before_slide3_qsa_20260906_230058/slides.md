---
marp: true
theme: default
paginate: true
size: 16:9
---

> Two parts: frontier architecture (slides 2–7), then measurements and research analysis (slides 8–16). 16 main slides (18 minutes) + 8 hidden backups. [Full speaking script](script.md). Editable charts and diagrams are in the PowerPoint. The builder synchronizes visible content; chart tables retain three decimals for review. Edit spoken text in script.md and rehearsal cues in the comments below.

## 1. Three frontier models, one comparison framework

DeepSeek-V4-Flash · GLM-5.3-Flash · Qwen3.8-Flash-Next-FP8

Throughput, latency  
and memory lead to  
different serving choices.

389.4

DeepSeek output tok/s · 16K, c64

447.1

GLM output tok/s · 16K, c64

517.7

Qwen output tok/s · 16K, c64

8 × H100 80GB  |  Three measured deployments  |  Architecture + runtime + workload

Tan Ngo · Professor Kan Zhu and UW SyFI · 6 September 2026

<!--
SLIDE 1 — 0:40; cumulative 0:40.

FULL SPOKEN SCRIPT
Today I will connect frontier model design to measured serving behavior on eight H100 GPUs.

The talk has two parts. First, I will explain how sparse attention, hybrid state, Muon and native speculation pursue efficiency. Then I will test the serving implications using DeepSeek, GLM and Qwen.

My central question is: when does less model work become better performance for the user? The results show why throughput, latency and capacity can lead to different answers.
-->

---

## 2. Two parts: architecture, then evidence

Research question: when does reduced model work become useful serving performance?

PART I / FRONTIER DESIGN

Where efficiency comes from

Sparse attention, hybrid state, MoE, Muon and native MTP.

PART II / LOCAL MEASUREMENTS

Where the savings survive

Concurrency, context, prefix sharing, speculation and cost.

METHOD / RESEARCH REASONING

Prediction → test → correction

Separate published mechanisms, local observations and hypotheses.

OUTCOME / A TESTABLE AGENDA

Optimize useful completions

Match runtime state; explain phase costs; test a serving policy.

Compare in the same order throughout: DeepSeek → GLM → Qwen. Same questions, explicit controls.

Source: task.md; report.md §§1–8

<!--
SLIDE 2 — 0:40; cumulative 1:20.

FULL SPOKEN SCRIPT
Part one asks where efficiency comes from: which work is removed, what new overhead appears, and whether the change affects training or inference.

Part two asks where those savings survive in a deployed system. I vary concurrency, context and prefix sharing, and examine MTP and allocation cost.

My approach is to move from a mechanism to a prediction, then to evidence and a possible correction. Across comparisons, DeepSeek comes first in orange, GLM second in purple, and Qwen third in teal. The measured contribution is their tradeoff map and the debugging evidence behind it.
-->

---

## 3. Sparse attention reduces reads; selection has a cost

A query needs useful history; the system must find, gather and process it.

Dense history

query × history

All available  
positions  
  
Core prefill:  
O(S²)

Sparse selection

query × history

Selected entries  
plus index cost  
  
Core attention:  
O(SK)

Compressed history

query × history

Summaries  
plus local tail  
  
Fewer entries;  
extra state

Sparse path = indexer + top-k + gather + selected attention + state management

Prediction: the crossover depends on context length, selection overhead, kernel efficiency and quality.

**Boundary:** Conceptual operator diagram. Sparse kernel savings and vendor comparisons are not local end-to-end speedups.

Sources: DeepSeek-V4 report §2.3; Qwen3.8-Next report §2.1.2; report.md §2

<!--
SLIDE 3 — 1:20; cumulative 2:40.

FULL SPOKEN SCRIPT
[Point across the three designs.] Dense attention considers the full available history. Sparse attention selects a subset, while compression changes how history is represented. These are related but distinct ways to reduce work.

For a sequence of length S, dense prefill attention-score work grows quadratically. If each query attends to K selected entries, that attention component is roughly proportional to S times K. But that expression does not include the indexer, top-k selection, gathers or state management. Index scoring can still grow with the history.

DeepSeek V4 combines sparse selection with compressed representations. Qwen QSA also has a compressed indexer, while its recurrent layers supply a different memory mechanism.

The prediction is a context-dependent crossover: savings become useful when they exceed selection and kernel overhead. Short contexts may not amortize that overhead. Selection quality also matters.

Our local measurements can characterize the context regime. They do not isolate a sparse-versus-dense speedup because we have no matched dense control.

PRIMARY REFERENCES (checked 6 September 2026)
https://arxiv.org/html/2606.19348v1
https://arxiv.org/html/2608.30320v1
-->

---

## 4. Compare model size, active parameters and layer mix

Text-only base model · MTP and vision excluded · B = billion parameters

DeepSeek

GLM

Qwen

Base-model params*

≈290.9B

≈313.3B

176.94B

Active params / token†

14.08B

17.38B

7.27B

Routed experts / selected

256 / 6

288 / 8

512 / 10

Attention layers

43 sparse;  
compressed history

34 recurrent KDA  
+ 11 sparse

36 recurrent GDN  
+ 12 QSA

*DeepSeek/GLM: derived from rounded buckets. †Active GEMM path; Qwen’s 51.23B lookup is included only in base total.

Sparse expert use reduces active work; parameter counts alone do not rank serving speed.

| Dimension | DeepSeek | GLM | Qwen |
| --- | --- | --- | --- |
| Base-model params* | ≈290.9B | ≈313.3B | 176.94B |
| Active params / token† | 14.08B | 17.38B | 7.27B |
| Routed experts / selected | 256 / 6 | 288 / 8 | 512 / 10 |
| Attention layers | 43 sparse;<br>compressed history | 34 recurrent KDA<br>+ 11 sparse | 36 recurrent GDN<br>+ 12 QSA |

**Boundary:** Prior tensor accounting, not a new checkpoint recount. Base totals and active-GEMM estimates are not vendor headline definitions.

Source: report.md §2; per-model history/report.md parameter buckets and recorded configurations

<!--
SLIDE 4 — 1:10; cumulative 3:50.

FULL SPOKEN SCRIPT
[Read across the first row.] These are text-only base-model parameter counts, excluding MTP and vision: approximately 291 billion for DeepSeek, 313 billion for GLM, and 177 billion for Qwen. They come from prior tensor accounting, not a new checkpoint recount.

[Point to the second row.] The active GEMM estimates are much smaller: 14.08, 17.38 and 7.27 billion parameters per token. Qwen's base total includes a 51.23-billion-parameter lookup table. A lookup touches selected entries, so that full table is not counted as active matrix multiplication.

The routed expert counts are 256 with six selected for DeepSeek, 288 with eight selected for GLM, and 512 with ten selected for Qwen. More selected experts does not automatically mean more compute because expert dimensions differ.

Finally, DeepSeek has 43 sparse-attention layers. GLM combines 34 recurrent KDA layers with 11 sparse layers; Qwen combines 36 recurrent GDN layers with 12 QSA layers. These dimensions help explain what to measure, but parameter counts alone do not rank serving speed.
-->

---

## 5. Muon: 50% less optimizer state, matrix geometry

For matrix parameters optimized with Muon · equal state precision · training updates, not inference

Muon uses matrix products to couple entries; AdamW’s adaptive scaling treats coordinates separately.

One state buffer instead of two; matrix geometry instead of coordinate-wise rescaling.

| Comparison | AdamW | Muon |
| --- | --- | --- |
| Stored state / parameter | 2 values: first moment m<br>+ second moment v | 1 value: momentum |
| FP32 state / parameter | 8 bytes | 4 bytes → 50% less |
| Update geometry | Elementwise rescaling;<br>no matrix-direction normalization | Joint matrix transformation;<br>approximate orthogonalization |

**Boundary:** 50% covers persistent optimizer-state tensors only; excludes weights, gradients, activations and workspace. No local training A/B.

Sources: Keller Jordan, Muon explanation + reference code; PyTorch / DeepSpeed Muon; AdamW paper

<!--
SLIDE 5 — 1:15; cumulative 5:05.

FULL SPOKEN SCRIPT
[Point to the first two rows.] Muon has two useful distinctions from AdamW. First, AdamW stores a first moment and a second moment for every parameter. Standard Muon stores one momentum value. With both states in FP32, that is eight bytes versus four bytes per parameter: fifty percent less persistent optimizer-state memory on the parameters assigned to Muon.

This does not halve total training memory. Weights, gradients, activations, master weights if used, and temporary workspace remain. Parameters still using AdamW retain its two buffers.

[Point to the geometry row.] AdamW rescales each gradient coordinate using that coordinate's moment history. Muon transforms the matrix update jointly through approximate orthogonalization, usually using Newton-Schulz iterations. Its matrix products couple entries and reshape the update's singular-value spectrum. AdamW's elementwise rescaling does not explicitly normalize these matrix directions. This is matrix geometry, not a claim that Muon computes the full curvature of the loss.

DeepSeek and Qwen use mixed Muon and AdamW parameter groups; GLM's recipe is not established here. These are training properties, not a direct explanation of our inference throughput results.

PRIMARY REFERENCES (checked 6 September 2026)
https://kellerjordan.github.io/posts/muon/
https://arxiv.org/abs/1711.05101
https://github.com/KellerJordan/Muon/blob/master/muon.py
https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/
-->

---

## 6. Day-0 speculation: ship a draft, still pay for verification

Here “zero-day” means native drafting and runtime support at release, not zero overhead.

01 / DRAFT

Propose future tokens

Native MTP can avoid waiting for a separately trained draft model.

02 / VERIFY

Target checks candidates

Batch candidate verification; keep valid progress and resample as required.

03 / COMMIT

Restore consistent state

Discard rejected suffix state; commit KV, indices and recurrent updates.

Break-even:  round time / expected committed tokens  <  ordinary time / token

Availability does not guarantee acceleration: acceptance, draft cost, verification and load set the payoff.

**Boundary:** Runtime support is version-specific. Target-distribution preservation requires correct acceptance/resampling and state handling.

Sources: DeepSeek-V3 MTP; Leviathan et al., speculative decoding; SGLang Qwen day-0 support

<!--
SLIDE 6 — 1:10; cumulative 6:15.

FULL SPOKEN SCRIPT
I use zero-day here to mean that native drafting and runtime support are available around a model's release. It does not mean zero computation, and it is not a separate guarantee of speed.

[Point across the three stages.] A native MTP module proposes future tokens. The target verifies the candidates. The runtime then commits valid progress and restores state for any rejected suffix. Proper rejection sampling is needed for the usual target-distribution guarantee; merely having a draft head is insufficient.

Native MTP can avoid waiting for a separately trained draft model. Runtime engineering can reduce repeated indexing and synchronization, but those optimizations are version-specific.

The break-even condition is simple: time per speculative round divided by expected committed tokens must beat ordinary time per token. Acceptance alone cannot answer that question, because drafting, verification and state handling all have costs.

This predicts a load-dependent payoff. We will test that prediction with the controlled GLM comparison in part two.

PRIMARY REFERENCES (checked 6 September 2026)
https://github.com/deepseek-ai/DeepSeek-V3
https://arxiv.org/abs/2211.17192
https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/
-->

---

## 7. An architectural saving must survive the whole request

Analytical lens: phase-local improvements compete with input work, state, communication and scheduling.

If attention is 30% of runtime,  
a 10× attention speedup gives…

1 / (0.70 + 0.30 / 10) = 1.37×

TESTABLE PREDICTIONS

Measure what moved

Sparse state → context scaling  
MoE → load / communication  
MTP → accepted progress / round  
Pool sizing → capacity / preemption

Part II tests the operating regimes; a causal speedup claim still needs a matched intervention.

**Boundary:** Illustrative Amdahl calculation, not a measured time breakdown. No dense control or local operator attribution is available.

Source: report.md §2 analytical framework; §8 proposed experiments

<!--
SLIDE 7 — 1:05; cumulative 7:20.

FULL SPOKEN SCRIPT
Before showing results, I want to make the causal reasoning explicit.

Suppose attention accounts for thirty percent of a request's runtime and a new kernel makes attention ten times faster. If everything else stays unchanged, the whole request improves by only 1.37 times. This is an illustrative calculation, not our measured time breakdown.

[Point to the right.] Each design therefore needs a discriminating prediction. Sparse or compressed state should affect context scaling. MoE should change the interaction between selected compute and communication. MTP should change committed progress per round. Memory-pool sizing should affect capacity and possibly scheduling.

Part two maps these regimes using local measurements. Where we lack an operator trace or matched intervention, I will explain the plausible mechanism and identify what would falsify it instead of assigning the whole ranking to architecture.
-->

---

## 8. A fixed GPU budget; three workload axes

8 × H100 80GB · TP8 + expert parallelism · 256 output tokens · synthetic text prompts

BATCH

16K input

c = 1, 4, 16, 64  
8, 8, 32, 128 requests

CONTEXT

16K → 260K input

c = 8  
16 requests per point

PREFIX

64K shared + 2K suffix

1, 4, 16 distinct prefixes  
c = 8 · 64 requests

ARM BOUNDARY

Qwen batch: base-util082   |   Qwen context/prefix: earlier base pool

Concurrency caps client requests; output tok/s includes input work; TTFT includes queueing.

**Boundary:** Mostly one retained run per point. GPU class/count match; builds, backends and some startup pools differ.

Source: bench.sh; experiments.md; report.md §1

<!--
SLIDE 8 — 1:05; cumulative 8:25.

FULL SPOKEN SCRIPT
Each deployment uses eight H100 80-gigabyte GPUs, TP8 and expert parallelism. Main comparisons have MTP off, synthetic text prompts and 256 target output tokens.

The batch axis changes client concurrency at 16K input. The context axis changes input length at concurrency eight. The prefix axis changes sharing among sixty-four requests.

Concurrency is a client cap, not the instantaneous server batch. Output throughput includes input work, and first-token latency includes queueing and prefill.

[Point to the arm boundary.] Corrected Qwen batch results use base-util082. Its context and prefix data retain the earlier pool. Builds, precision and backends also differ across deployments. Most points have one retained run, with no repeat-based confidence intervals.

These are named deployment comparisons, with the limits stated before we interpret the curves.
-->

---

## 9. Higher concurrency buys throughput at a latency cost

16,384 input / 256 output · 8 × H100 · MTP off · c = client concurrency cap, not fixed GPU batch size

c64: GLM gives the earliest first token; Qwen leads output rate and token spacing.

c1 → c64: output rate grows 4.6–4.9×, while median time per output token grows 16–19×.

| Metric / concurrency | DeepSeek | GLM | Qwen |
| --- | --- | --- | --- |
| Output tok/s ↑ · c1 | 85.0 | 96.3 | 106.2 |
| Output tok/s ↑ · c4 | 219.8 | 225.3 | 256.8 |
| Output tok/s ↑ · c16 | 330.9 | 359.2 | 420.2 |
| Output tok/s ↑ · c64 | 389.4 | 447.1 | 517.7 |
| Median TTFT (s) ↓ · c64 | 3.64 | 2.51 | 2.94 |
| Median TPOT (ms/token) ↓ · c64 | 148.5 | 129.5 | 110.1 |

**Boundary:** Output rate includes input processing; TTFT includes queueing. Different deployed configurations; no quality or phase-only comparison.

Source: throughput.md; main JSONs — DeepSeek mtp-off-image; GLM bf16kv; Qwen base-util082

<!--
SLIDE 9 — 1:30; cumulative 9:55.

FULL SPOKEN SCRIPT
[Point to the first four rows.] To answer Kan's question about throughput at different batch sizes, we vary the client concurrency cap while keeping input at sixteen thousand tokens and output at 256. The server continuously batches requests, so this cap is not a fixed GPU batch size.

Qwen leads output throughput at every tested concurrency. At sixty-four, DeepSeek, GLM and Qwen produce about 389, 447 and 518 output tokens per second. These are whole-deployment rates across eight H100s, with input processing included in elapsed time.

[Point to the last two rows.] The latency winner depends on the metric. GLM gives the earliest median first token at concurrency sixty-four: 2.51 seconds versus 3.64 for DeepSeek and 2.94 for Qwen. Qwen has the lowest median time per output token.

Across all three models, increasing concurrency from one to sixty-four yields only 4.6 to 4.9 times more output, while median token latency grows about sixteen to nineteen times. The deployment serves more total work each second, but individual requests receive tokens more slowly.

These results compare deployed configurations. They do not isolate prefill computation, identify the dominant bottleneck, or establish equal answer quality. The next slide shows the full token-latency curves.
-->

---

## 10. All three buy throughput with higher token latency

Median time per output token, milliseconds ↓ · same 16K batch arms as the preceding slide

c1 → c64 growth

Output rate / median TPOT

DeepSeek

4.58× / 18.70×

GLM

4.64× / 18.36×

Qwen

4.87× / 15.79×

Client concurrency cap

Read throughput and latency together: less than 5× more output, about 16–19× more token latency.

| Category | DeepSeek | GLM | Qwen |
| --- | --- | --- | --- |
| 1 | 7.938 | 7.056 | 6.970 |
| 4 | 10.721 | 10.932 | 9.616 |
| 16 | 32.401 | 33.570 | 26.320 |
| 64 | 148.470 | 129.517 | 110.080 |

| Model | Output-rate growth | Median-TPOT growth |
| --- | --- | --- |
| DeepSeek | 4.579 | 18.703 |
| GLM | 4.642 | 18.357 |
| Qwen | 4.873 | 15.793 |

**Boundary:** Median TPOT describes output spacing, not first-token latency or an SLO pass rate. Concurrency positions are categories.

Source: main batch JSONs — DeepSeek mtp-off-image; GLM bf16kv; Qwen base-util082

<!--
SLIDE 10 — 1:05; cumulative 11:00.

FULL SPOKEN SCRIPT
This chart uses exactly the same three main batch arms as the throughput slide. It shows median time per output token, so lower is better.

[Read the c64 endpoints in model order.] DeepSeek is about 149 milliseconds, GLM about 130, and Qwen about 110. Qwen has the lowest observed token latency as well as the highest output throughput at that point.

[Point to the growth summary.] For every model, moving from concurrency one to sixty-four buys less than five times the output rate while increasing token latency by about sixteen to nineteen times.

Batching may improve utilization while requests compete for resources, but we still need phase measurements to identify the mechanism. The practical choice must satisfy a latency objective. A high aggregate rate does not guarantee a good interactive experience.
-->

---

## 11. Long input changes which metric wins

131,072 input / 256 output · c8 · 16 requests

Output tokens/s ↑

Median first-token latency, seconds ↓

Qwen has the highest observed output rate; GLM has the lowest observed median TTFT.

| Category | Observed |
| --- | --- |
| DeepSeek | 34.666 |
| GLM | 47.243 |
| Qwen | 56.468 |

| Category | Observed |
| --- | --- |
| DeepSeek | 26.665 |
| GLM | 12.891 |
| Qwen | 19.836 |

**Boundary:** Qwen uses the earlier smaller pool. Rerun matched startup before interpreting long-context capacity as architectural.

Source: ctx_isl131072_c8.json — GLM bf16kv; DeepSeek mtp-off-image; Qwen base

<!--
SLIDE 11 — 1:05; cumulative 12:05.

FULL SPOKEN SCRIPT
[Read both charts in the same model order.] At 131K input and concurrency eight, output throughput is about 34.7 for DeepSeek, 47.2 for GLM, and 56.5 for Qwen. First-token latency is 26.66, 12.89, and 19.84 seconds respectively. Qwen leads output rate, while GLM gives the earliest median first token.

The architecture lesson is that retained state, indexing and input processing influence different parts of a request. A smaller history representation does not automatically produce the earliest first token. Output throughput also includes input work, so this is not a decode-only curve.

Qwen's context arm retains the smaller-pool confound. DeepSeek also has an unresolved discrepancy between its 16K context anchor and a later concurrency-eight batch point.

These results identify what to investigate next. Phase-specific timing and matched startup would distinguish an input-work effect from scheduling or memory pressure.
-->

---

## 12. The best sharing pattern differs across models

64 requests · 64K shared + 2K unique · c8 · prefix caching enabled throughout

Output tokens/s ↑

Distinct shared prefixes (categories)

Restore the same prefix boundary

Retained KV / compressed history

Recurrent-state checkpoint

Then branch safely; copy or restore mutable state.

DeepSeek and GLM peak at four prefixes; Qwen peaks at one in its qualified arm.

| Category | DeepSeek | GLM | Qwen |
| --- | --- | --- | --- |
| 1 | 198.232 | 265.283 | 453.927 |
| 4 | 391.648 | 365.875 | 398.057 |
| 16 | 114.608 | 199.440 | 262.236 |

**Boundary:** Sharing patterns, not cache-on/off speedups. Qwen retains the earlier smaller pool; mechanisms remain hypotheses.

Source: DeepSeek mtp-off-image; GLM bf16kv; Qwen base prefix_p64k_n*.json; report.md §4

<!--
SLIDE 12 — 1:05; cumulative 13:10.

FULL SPOKEN SCRIPT
Prefix caching is enabled in every condition here. We vary one, four or sixteen shared prefixes across sixty-four requests.

[Compare all three curves.] DeepSeek and GLM peak at four prefixes, while Qwen peaks at one. There is no single best sharing pattern across the retained arms. Qwen still uses the earlier smaller pool. Fill order, eviction and scheduling are candidate explanations, but none is isolated by a trace.

This is not a cache-on versus cache-off speedup.

[Point to the state diagram.] The architectural challenge is restoring a consistent boundary. KV or compressed history must agree with recurrent checkpoints. A recurrent final state cannot be sliced backward to an arbitrary earlier prefix, but suitable checkpoints can be reused.

Branching and speculative rejection then require safe copies or rollback. Finer checkpoints trade memory and copying for less recomputation. That makes prefix reuse a state-placement and scheduling problem as well as a lookup problem.
-->

---

## 13. MTP gains fade as concurrency rises

One draft token · 16K input / 256 output · throughput change = (MTP / paired off − 1) × 100%

GLM: 71.7% of drafts accepted at c64, yet output throughput falls 1.2%.

Light-load gains; no recorded gain at c64. High acceptance alone does not ensure a speedup.

| Metric / concurrency | DeepSeek | GLM | Qwen |
| --- | --- | --- | --- |
| Throughput change · c1 | +24.9% | +24.3% | +16.9% |
| Throughput change · c4 | +13.4% | +6.4% | -8.0% |
| Throughput change · c16 | +15.8% | -4.0% | -13.2% |
| Throughput change · c64 | -0.9% | -1.2% | -7.0% |
| Draft acceptance · c64 | 62.1% | 71.7% | 56.8% |

**Boundary:** DeepSeek: historical runtime, missing cold telemetry. GLM: strongest control. Qwen: startup/pool confounded. No repeat-based uncertainty.

Source: speculative.md; DeepSeek off/on-noreuse; GLM bf16kv/n1; Qwen original base/mtp-n1

<!--
SLIDE 13 — 1:10; cumulative 14:20.

FULL SPOKEN SCRIPT
[Point to the first row.] One-token MTP improves recorded output throughput at concurrency one by about twenty-five percent for DeepSeek, twenty-four percent for GLM and seventeen percent for Qwen. At concurrency sixty-four, none records a gain. DeepSeek and GLM are close to flat; Qwen loses about seven percent.

[Point to the acceptance row.] GLM still accepts about seventy-two percent of draft tokens at concurrency sixty-four, yet throughput falls slightly. Acceptance alone cannot predict speedup: committed progress must pay for drafting, verification and state handling. We have not measured which overhead dominates.

Each column uses its own paired baseline. GLM provides the strongest controls. DeepSeek uses historical runs without newer cold telemetry, and Qwen retains startup and pool differences. Small changes have no established statistical significance.

The result motivates testing a load-aware draft budget. It does not establish that such a policy already works, or that MTP should always be enabled.
-->

---

## 14. Lower token cost is purchased with higher latency

Illustrative $2.50/GPU-hour × 8 GPUs = $20/node-hour · 16K input / 256 output

c1: dollars / million output tokens ↓

c64: dollars / million output tokens ↓

GLM: $57.68 → $12.43 per million outputs, but median TPOT rises from 7.1 → 129.5 ms.

| Category | Allocation cost |
| --- | --- |
| DeepSeek | 65.333 |
| GLM | 57.684 |
| Qwen | 52.293 |

| Category | Allocation cost |
| --- | --- |
| DeepSeek | 14.269 |
| GLM | 12.427 |
| Qwen | 10.730 |

**Boundary:** Includes input work. No quality adjustment, idle-time model, price quote or measured cost per successful task.

Source: exact main batch JSONs; report.md §6. Cost = 20 × 1,000,000 / (3,600 × tok/s)

<!--
SLIDE 14 — 1:00; cumulative 15:20.

FULL SPOKEN SCRIPT
I assume two dollars fifty per GPU-hour, or twenty dollars for the eight-GPU node. Dividing that allocation cost by measured output throughput gives dollars per million output tokens, including all input work.

At concurrency sixty-four, the costs in model order are about fourteen dollars twenty-seven for DeepSeek, twelve dollars forty-three for GLM, and ten dollars seventy-three for Qwen.

[Point to GLM.] Raising concurrency reduces calculated token cost from about fifty-eight to twelve dollars per million, while median token latency rises from roughly seven to one hundred and thirty milliseconds.

That is an allocation-efficiency result, not production cost per successful task. Quality, arrivals, idle time and the best GPU count remain unmeasured. My research objective would be useful completions within explicit latency targets, with task quality held comparable.
-->

---

## 15. Startup state can impersonate an architecture effect

Qwen c4: 162.2 tok/s in the earlier arm → 256.8 tok/s in the corrected batch arm.

STARTUP ACCOUNTING

17.07 → 0.99 GiB

Recorded peak activation  
Cold → warm compile cache

REPORTED CACHE POOL

2.05M → 3.20M

Token capacity  
Utilization 0.85 → 0.82

OBSERVED C4 RESULT

162.2 → 256.8

Output tokens/s  
Checkpoint unchanged

Inspect runtime state before assigning an architectural cause to a changed ranking.

**Boundary:** Startup and utilization changed together. Preemption causality is unmeasured; the batch correction does not transfer to other axes.

Source: Qwen base-util082/RESULT-util-ab.md, both c4 JSONs; fix_bug.md Bug 12

<!--
SLIDE 15 — 1:10; cumulative 16:30.

FULL SPOKEN SCRIPT
The Qwen startup case is a concrete example of revising a hypothesis when the evidence changes.

[Point across the cards.] Recorded peak activation changed from 17.07 to 0.99 gigabytes. The reported cache pool increased from 2.05 million to 3.20 million tokens even though utilization was lowered. Concurrency-four throughput changed from 162.2 to 256.8 tokens per second.

An architecture-only explanation for the earlier low throughput is therefore insufficient: the checkpoint stayed the same, while runtime conditions changed.

This is not a clean utilization-only experiment, because startup changed too. Specific preemption causality is unmeasured. I use the corrected batch arm and retain the context, prefix and MTP caveats rather than transferring a correction factor across axes.

For me, this is the methodological contribution: preserve the failed explanation, locate the confound, correct the comparison, and design the next experiment to separate causes.
-->

---

## 16. Research direction: budget state and speculation together

Hypothesis: load-aware draft budgets can improve goodput when hybrid-state capacity is constrained.

MY EVIDENCE

Operating points matter

GLM MTP helps at light load.  
Qwen startup changes the ranking.  
Latency and token cost diverge.

FIRST / ESTABLISH

Match and profile

Repair gates; repeat matched arms.  
Separate prefill / decode costs.  
Measure acceptance and state work.

THEN / TEST

Adaptive draft budget

Compare off, fixed, adaptive.  
Useful completions under SLOs.  
Quality, overhead and tails.

Falsifier: no useful gain over fixed policies after matching quality, latency targets and runtime state.

Source: report.md §8; fix_bug.md §6. Research hypothesis and evaluation below are proposed.

<!--
SLIDE 16 — 1:30; cumulative 18:00.

FULL SPOKEN SCRIPT
The findings motivate a concrete research direction: can a serving policy budget speculation and hybrid state together to improve useful throughput under latency constraints?

The hypothesis follows from two observations. GLM's MTP payoff changes with load, and Qwen's startup state changes the resources available to the scheduler. A fixed draft budget may therefore be a poor choice across changing request and memory conditions.

I would first repair the pending measurement gates, match startup conditions, and repeat the qualified comparisons. Then I would collect accepted progress, draft and verification time, state-copy work, preemption and per-request latency.

The intervention would compare MTP off, fixed draft budgets and a simple adaptive budget under the same workload, quality checks and latency targets. I would report useful completions, tails, policy overhead and failures, not just aggregate tokens.

A clear negative result would be no useful gain over fixed policies after those controls. That would tell us to investigate another source of cost rather than defend the policy.

This is the kind of work I want to pursue at SyFI: connect architectural mechanisms to measurements, correct weak explanations, and turn the remaining uncertainty into a falsifiable systems experiment. Thank you.
-->

---

## Backup B1. What is controlled, and what remains different?

DeepSeek’s build bridge applies to its batch grid only; all models are not engine-matched.

| Held or documented | Different / missing |
| --- | --- |
| 8 × H100 80GB; TP8 + EP | Physical node/topology equivalence unestablished |
| Synthetic input/output length targets | Tokenizers, actual counts and task quality |
| MTP off in main comparison | Weight/KV precision, builds and backends |
| Named arm and per-point evidence | Some startup pools; session/repeat coverage |

Source: report.md §1; experiments.md

<!--
BACKUP B1 — use only for relevant questions.

FULL SPOKEN SCRIPT
The comparison holds the GPU class and count, parallelism arrangement, and workload targets in common. MTP is off for the main cross-model comparison, and each point is tied to a named arm.

However, precisions, tokenizers, backends, builds and some startup pools differ. We have not established identical physical topology or equal answer quality. So I describe this as a deployment comparison under a fixed GPU budget.

The DeepSeek build bridge observed less than one percent throughput difference on its batch grid. That narrows one concern for that model and grid; it does not make every model and workload engine-matched. Repeated matched-startup comparisons are still needed.
-->

---

## Backup B2. A workload curve does not identify a dominant kernel

Enabled → implemented → emitted → scraped → aligned → saved: audit the metric chain.

| Candidate mechanism | Discriminating evidence needed |
| --- | --- |
| Small GEMMs / expert routing | Operator durations, shapes and per-rank expert load |
| Collectives / launch overhead | Timeline and controlled TP/EP or graph intervention |
| HBM traffic | Hardware counters within prefill/decode intervals |
| Scheduler / cache pressure | Per-step batches, preemption and matched-pool controls |

**Boundary:** GLM engine estimates omit attention. Whole-request averages can hide bandwidth-bound individual kernels.

Source: report.md §3; fix_bug.md Bug 13

<!--
BACKUP B2 — use only for relevant questions.

FULL SPOKEN SCRIPT
The current request metrics identify regimes, not a dominant operator. Small GEMMs, expert dispatch, collectives, memory traffic and scheduling are all candidates, depending on the phase and load.

To discriminate between them, I would first separate prefill and decode, collect operator durations and shapes, and inspect per-rank load. Hardware memory counters and a controlled TP or EP intervention would then test specific explanations.

The engine estimates cannot substitute for this. GLM's recorded estimates omit attention, and the models have different estimator coverage. A low average modeled bandwidth over mixed request work cannot rule out individual bandwidth-bound kernels. Unknown telemetry is also not a zero measurement.
-->

---

## Backup B3. A reusable prefix needs a complete state checkpoint

IDENTITY

Same token prefix

Model revision, adapter, positions and relevant execution state must match.

BOUNDARY

All layers agree

KV/history and recurrent state resume at the same compatible position.

BRANCH / ROLLBACK

Protect mutable state

Share immutable history; copy or restore state when requests diverge.

Finer checkpoints trade additional memory/copy work for less recomputation after partial matches.

Source: report.md §4; fix_bug.md Bug 6

<!--
BACKUP B3 — use only for relevant questions.

FULL SPOKEN SCRIPT
Yes, recurrent state can support prefix reuse through checkpoints. The limitation is that a state at the end of a long sequence does not reconstruct arbitrary earlier states.

The cache key needs the same token prefix and relevant execution identity. At a hit, every layer must resume at a compatible boundary: retained history and recurrent checkpoints must agree on position.

Immutable history can be shared, but requests that branch need protection for mutable state. Speculative rejection also requires a valid rollback point. The design tradeoff is checkpoint granularity: more checkpoints cost memory and copying, while fewer checkpoints can require more recomputation after a partial match. Our sharing sweep does not isolate those costs.
-->

---

## Backup B4. Resident parameters are not per-token arithmetic

DeepSeek

GLM

Qwen

Served total / GEMM

290.91B / 14.08B

321.34B / 17.38B

176.94B / 7.27B

Recorded disk size

148.6 GiB

305.8 GiB

172.8 GiB

Main weight format

MXFP4 experts;  
FP8 attention/dense

FP8 weights

FP8 weights

Main cache precision

FP8 KV

BF16 KV

BF16 KV

Qwen’s 51.23B lookup table adds resident capacity without a full-table GEMM per token.

MoE reduces selected compute; weights, communication, lookup and state still need accounting.

| Dimension | DeepSeek | GLM | Qwen |
| --- | --- | --- | --- |
| Served total / GEMM | 290.91B / 14.08B | 321.34B / 17.38B | 176.94B / 7.27B |
| Recorded disk size | 148.6 GiB | 305.8 GiB | 172.8 GiB |
| Main weight format | MXFP4 experts;<br>FP8 attention/dense | FP8 weights | FP8 weights |
| Main cache precision | FP8 KV | BF16 KV | BF16 KV |

**Boundary:** Historical tensor analyses, not a new recount. Disk size is not runtime HBM or a validated deployment lower bound.

Source: report.md §2; prior tensor analyses

<!--
BACKUP B4 — use only for relevant questions.

FULL SPOKEN SCRIPT
Resident weights and selected per-token arithmetic are different quantities. With MoE, different tokens can choose different experts, so a batch may touch many more experts than one token does.

Qwen also has a large n-gram lookup table. Its prior analysis counts about 51 billion parameters in that table, but a lookup does not perform a dense multiplication over the entire table each token. Communication, indexing and state management add further costs.

The counts shown are prior tensor analyses, not a new recount or a measured timing breakdown.

The three columns also show a precision difference: DeepSeek uses MXFP4 experts with FP8 attention and dense weights, while GLM and Qwen use FP8 weights. DeepSeek's main KV is FP8; the other two use BF16 KV. Therefore even the same GPU count does not create a pure architecture comparison. Recorded disk size is not runtime HBM.
-->

---

## Backup B5. More capacity and more drafts are not automatic wins

Reported FP8 capacity gain = 1.805× in its startup comparison; preserve session boundaries.

| GLM option | Measured tradeoff | Decision supported |
| --- | --- | --- |
| MTP n1 / c1 | +24% throughput; TPOT 7.06 → 5.05 ms | Candidate for light-load testing |
| MTP n5 / c64 | −9% throughput; 99.2% cache occupancy | More drafts do not ensure benefit |
| Alternate FP8 KV / c64 | −25% versus paired alternate BF16 | Price capacity against speed |

**Boundary:** Alternate FP8 stack bypasses a version check; numerical parity and production suitability remain unverified.

Source: GLM paired arms; report.md §§5, 7

<!--
BACKUP B5 — use only for relevant questions.

FULL SPOKEN SCRIPT
I would begin with the workload objective rather than enable every feature. GLM with one draft token is a promising light-load candidate: throughput improves and median token latency falls. At concurrency sixty-four, five drafts reduce throughput and recorded cache occupancy reaches 99.2 percent. Occupancy alone does not prove the mechanism.

For FP8 KV, the alternate stack provides more reported capacity but loses about twenty-five percent throughput against its paired BF16 control at concurrency sixty-four. Its reported capacity comparison spans specific startup records, which must not be mixed with a later main pool.

The alternate stack also bypasses a version check. Successful synthetic execution does not establish numerical parity or production suitability. Those checks remain necessary before making a deployment choice.
-->

---

## Backup B6. Answer directly, then name the missing experiment

| Question | Defensible answer |
| --- | --- |
| Where are error bars? | Most cells have one retained run; randomized repeats are pending. |
| Cheapest in production? | Unknown: no quality, real arrival trace, idle-time or GPU-count sweep. |
| How fast is prefix caching? | Only sharing patterns were measured; a cache-off control is missing. |
| Why not name the bottleneck? | Request metrics do not isolate operators; capture phase-specific traces. |
| What could change the ranking? | Matched startup/pools, repeats, different workloads and quality. |

Expanded answers: report.md §9; fix_bug.md §7

<!--
BACKUP B6 — use only for relevant questions.

FULL SPOKEN SCRIPT
[If asked about error bars.] Most cells have one retained run. Within-run latency percentiles do not measure uncertainty across repeated runs. Randomized repeats are pending.

[If asked about production cost.] We measured allocation cost for synthetic workloads. Quality, realistic arrivals, idle time and the best feasible GPU count remain unmeasured.

[If asked about cache speedup.] We compared sharing patterns with caching enabled. We need an identical cache-off control to measure cache-on versus cache-off speedup.

[If asked what might change the ranking.] Matched startup and pools, repeated runs, different workload lengths and quality requirements could all change the decision. I would test those explicitly rather than extend the ranking beyond the measured grid.
-->

---

## Backup B7. Read the earliest cause, not the loudest CUDA symptom

SYMPTOM

CUDA invalid argument

Eager execution did not resolve startup; inspect shared dependencies.

EARLIER EVIDENCE

Cache write fails

TensorRT/DeepGEMM preparation hits a separate filesystem path.

OBSERVED RESOLUTION

Redirect the caches

Redirect backend-specific caches and add write preflight; graph-enabled startup succeeds.

A failed intervention narrows the cause; eager execution can still use the same compiler cache.

**Boundary:** Logs support the failure chain; they do not establish every internal kernel-handle transition.

Source: fix_bug.md Bugs 2–3; GLM cu130 run6_eager and run7_trtllmfix logs

<!--
BACKUP B7 — use only for relevant questions.

FULL SPOKEN SCRIPT
The visible symptom was a CUDA invalid-argument error during startup. Disabling graph-related paths and trying eager execution did not resolve it. That made a graph-only explanation insufficient.

Reading earlier errors showed a compiler-cache write failure involving the TensorRT and DeepGEMM path. Eager and graph execution still shared that preparation dependency. Redirecting the additional backend-specific caches and adding a write preflight allowed graph-enabled startup.

The lesson is to find the earliest relevant exception and test a hypothesis that distinguishes causes. Free space on a volume also does not establish available user quota. The logs support this failure chain, but they do not reveal every internal transition between the failed preparation and the CUDA error.
-->

---

## Backup B8. Separate the backend change from the dtype change

Output tokens/s ↑ · concurrency 64

PAIRED COMPARISONS

−22% then −25%

Combined: −41%, not −47%.  
  
0.780 × 0.751 ≈ 0.586  
of original throughput

Geometry, layout, kernels and versions determine whether a dtype has a working execution path.

| Category | GLM |
| --- | --- |
| Original BF16 | 447.067 |
| Alternate BF16 | 348.776 |
| Alternate FP8 | 262.004 |

**Boundary:** Failed layout required 64 positional dimensions; GLM NoPE has zero. Alternate execution does not validate numerical parity.

Source: GLM c64 JSONs — bf16kv, bf16kv-fi618, fp8kv-fi618; fix_bug.md Bug 8

<!--
BACKUP B8 — use only for relevant questions.

FULL SPOKEN SCRIPT
The original FP8 route failed because its layout required sixty-four positional dimensions while GLM's NoPE configuration had zero. That establishes an incompatible route, not a universal inability to use FP8 KV.

The alternate stack ran, so the next task is to separate its effects. Original BF16 gives about 447 tokens per second. Alternate BF16 gives 349, a twenty-two percent loss associated with the software and backend change. Alternate FP8 gives 262, another twenty-five percent loss relative to that paired BF16 control.

The ratios multiply: 0.780 times 0.751 is about 0.586. The combined reduction is about forty-one percent, not forty-seven percent, and it should not all be attributed to dtype.

The successful route remains experimental. Geometry, layout, kernels, versions and numerical validation all matter. The result establishes a measured capacity–speed tradeoff, not a production recommendation.
-->
