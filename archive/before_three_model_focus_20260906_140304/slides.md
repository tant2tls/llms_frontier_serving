---
marp: true
theme: default
paginate: true
size: 16:9
---

> Two parts: frontier architecture (slides 2–7), then measurements and research analysis (slides 8–16). 16 main slides (18 minutes) + 8 hidden backups. [Full speaking script](script.md). Editable charts and diagrams are in the PowerPoint. The builder synchronizes visible content; chart tables retain three decimals for review. Edit spoken text in script.md and rehearsal cues in the comments below.

## 1. Frontier MoE serving: the operating point matters

GLM-5.3-Flash · DeepSeek-V4-Flash · Qwen3.8-Flash-Next-FP8

Throughput, latency  
and memory lead to  
different serving choices.

517.7

Qwen output tok/s · 16K, c64

12.89 s

GLM median TTFT · 131K, c8

+24%

GLM MTP n1 throughput · 16K, c1

8 × H100 80GB  |  Three measured deployments  |  Architecture + runtime + workload

Tan Ngo · Professor Kan Zhu and UW SyFI · 6 September 2026

<!--
SLIDE 1 — 0:40; cumulative 0:40.

FULL SPOKEN SCRIPT
Today I will connect frontier model design to measured serving behavior on eight H100 GPUs.

The talk has two parts. First, I will explain how sparse attention, hybrid state, Muon and native speculation pursue efficiency. Then I will test the serving implications using GLM, DeepSeek and Qwen.

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

Kimi K3 is architecture-only; no local performance or cost is extrapolated.

Source: task.md; report.md §§1–8

<!--
SLIDE 2 — 0:40; cumulative 1:20.

FULL SPOKEN SCRIPT
Part one asks where efficiency comes from: which work is removed, what new overhead appears, and whether the change affects training or inference.

Part two asks where those savings survive in a deployed system. I vary concurrency, context and prefix sharing, and examine MTP and allocation cost.

My approach is to move from a mechanism to a prediction, then to evidence and a possible correction. Kimi is architecture-only here. The measured contribution is the three-model tradeoff map and the debugging evidence behind it.
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

## 4. Sparse expert compute; different memory obligations

MoE selects a few experts per token, while the deployment retains a large weight footprint.

ATTENTION LAYER COMPOSITION

EXPERTS / SELECTED

GLM

34 recurrent

11 sparse

288 / 8

DeepSeek

43 sparse · compressed

256 / 6

Qwen

36 recurrent

12 QSA

512 / 10

Kimi K3*

69 recurrent

24 gated MLA

896 / 16

Recurrent: fixed state / active sequence

Retained history: grows with sequence length

Budget weights + workspace + graphs + history + recurrent state.   *Kimi unmeasured

**Boundary:** Layer-count segments are schematic by type, not execution order or proportions of runtime.

Source: report.md §2; recorded model analyses/configurations

<!--
SLIDE 4 — 1:10; cumulative 3:50.

FULL SPOKEN SCRIPT
All four families use mixture-of-experts computation. Each token selects a few experts, but the deployment still holds many more weights and must dispatch and combine work across them.

[Point to the segments.] GLM and Qwen combine recurrent layers with history-retaining attention. DeepSeek uses compressed sparse history. Kimi also has a hybrid structure at a much larger weight footprint. These segment lengths show layer counts, not time spent.

Recurrent state stays fixed as one sequence gets longer, but it still scales with active sequences and saved checkpoints. Attention history grows with sequence length. Sparse reads do not necessarily remove all stored history.

This gives two separate capacity constraints: tokens of retained history and active sequence state. It also creates a cache-consistency problem when requests reuse a prefix or reject speculative tokens.

So a serving budget must include weights, workspace, graph buffers, retained history and recurrent state. Active parameter count alone cannot choose the fastest deployment.
-->

---

## 5. Muon changes training updates; AdamW still has a role

A matrix-aware optimizer is a training design choice, not a decode kernel.

ADAMW

Coordinate-wise scaling

Momentum + second-moment scaling  
Decoupled weight decay  
Embeddings and other selected groups

MUON

Matrix update geometry

Momentum → approximate orthogonalization  
Newton–Schulz matrix operations  
Matrix-aware partitioning and batching

Qwen and DeepSeek use mixed recipes; Kimi K2 reports MuonClip for training stability.

Evaluate quality reached per training GPU-hour; the optimizer update is absent from serving.

**Boundary:** Published optimizer recipes; no local training comparison. Do not attribute measured inference throughput to Muon.

Sources: Jordan, Muon; Loshchilov & Hutter, AdamW; Qwen/DeepSeek reports; Kimi K2 report

<!--
SLIDE 5 — 1:15; cumulative 5:05.

FULL SPOKEN SCRIPT
Muon belongs in a frontier architecture talk because training efficiency influences which models are practical to build. But it is important to locate that effect correctly.

[Point left.] AdamW uses coordinate-wise moment estimates and decoupled weight decay. [Point right.] Muon transforms a matrix momentum update through approximate orthogonalization, commonly using Newton-Schulz iterations. This changes update geometry and introduces matrix-operation and partitioning costs.

It is not a blanket replacement. The published Qwen and DeepSeek recipes use Muon for selected matrix groups and retain AdamW for other parameters. Kimi K2 reports MuonClip as part of its training-stability work; I am not presenting that as a local Kimi experiment.

The right evaluation is the quality reached for training time and resources, including optimizer overhead and stability. Muon is not executed in the serving forward pass. Therefore, it cannot directly explain our tokens-per-second ranking. We need to keep training savings separate from inference savings.

PRIMARY REFERENCES (checked 6 September 2026)
https://kellerjordan.github.io/posts/muon/
https://arxiv.org/abs/1711.05101
https://github.com/KellerJordan/Muon
https://moonshotai.github.io/Kimi-K2/
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

## 9. Qwen leads the corrected 16K throughput grid

Output tokens/s ↑ · 16,384 input / 256 output · MTP off

Client concurrency cap

AT CONCURRENCY 64

517.7 tok/s

Qwen / GLM       1.16×  
Qwen / DeepSeek 1.33×  
  
Observed throughput lead

64× more concurrency yields only 4.6–4.9× more output throughput.

| Category | Qwen | GLM | DeepSeek |
| --- | --- | --- | --- |
| 1 | 106.240 | 96.309 | 85.034 |
| 4 | 256.838 | 225.297 | 219.805 |
| 16 | 420.161 | 359.172 | 330.879 |
| 64 | 517.746 | 447.067 | 389.357 |

**Boundary:** Deployment comparison; quality unmeasured. Lines connect measured concurrency categories, not evenly spaced numeric intervals.

Source: main batch JSONs — Qwen base-util082; GLM bf16kv; DeepSeek mtp-off-image

<!--
SLIDE 9 — 1:30; cumulative 9:55.

FULL SPOKEN SCRIPT
[Trace the teal curve.] Qwen leads every measured concurrency in the corrected 16K throughput grid. At concurrency sixty-four it produces about 518 output tokens per second, versus 447 for GLM and 389 for DeepSeek: roughly sixteen and thirty-three percent higher.

The second result is diminishing return. Sixty-four times more concurrency produces less than five times more aggregate output.

Why might Qwen be fast? Its smaller selected GEMM path and hybrid attention are plausible contributors. They can reduce work. But the result also depends on kernels, precision, runtime state and scheduling. We did not ablate those components, so this chart does not measure their individual contributions.

That caution matters: the earlier Qwen concurrency-four result was 162 tokens per second, versus 257 in the corrected arm after the startup investigation. The checkpoint was unchanged.

The defensible conclusion is a throughput lead for this grid, not a universal architecture or quality ranking. Next I will show the latency price of moving along these curves.
-->

---

## 10. Past c8, latency grows much faster than throughput

Each metric normalized to its own c8 value = 1.0× · median TPOT = time per output token

GLM: 1.51× throughput / 7.15× TPOT

DeepSeek: 1.37× throughput / 9.41× TPOT

Choose concurrency against a latency objective; aggregate token rate alone cannot choose it.

| Category | Throughput | Median TPOT |
| --- | --- | --- |
| 8 | 1.000 | 1.000 |
| 16 | 1.210 | 1.853 |
| 32 | 1.369 | 3.785 |
| 48 | 1.457 | 5.549 |
| 64 | 1.506 | 7.149 |

| Category | Throughput | Median TPOT |
| --- | --- | --- |
| 8 | 1.000 | 1.000 |
| 16 | 1.150 | 2.089 |
| 32 | 1.292 | 4.499 |
| 48 | 1.272 | 7.438 |
| 64 | 1.369 | 9.413 |

**Boundary:** DeepSeek is a separate build/utilization arm. c8 is an anchor, not a proven optimum. Horizontal positions are concurrency categories.

Source: GLM bf16kv; DeepSeek util085-dev20073 finer batch grids

<!--
SLIDE 10 — 1:05; cumulative 11:00.

FULL SPOKEN SCRIPT
Each metric here is divided by its own value at concurrency eight, allowing us to compare relative growth.

[Point to both charts.] Moving to concurrency sixty-four gives GLM 1.51 times the throughput but 7.15 times the median token latency. DeepSeek gains 1.37 times the throughput at 9.41 times the latency.

Batching can improve utilization of model work, but more concurrent requests also compete for resources and scheduler time. That is a plausible interpretation, not proof of which kernel or collective dominates.

For an interactive service, the useful setting depends on a latency target. Concurrency eight is an anchor, not a proven optimum. DeepSeek's finer sweep is also a separate arm from the previous chart.

This is the systems distinction between making the device produce more and making each user wait less.
-->

---

## 11. Long input changes which metric wins

131,072 input / 256 output · c8 · 16 requests

Output tokens/s ↑

Median first-token latency, seconds ↓

Qwen has the highest observed output rate; GLM has the lowest observed median TTFT.

| Category | Observed |
| --- | --- |
| Qwen | 56.468 |
| GLM | 47.243 |
| DeepSeek | 34.666 |

| Category | Observed |
| --- | --- |
| Qwen | 19.836 |
| GLM | 12.891 |
| DeepSeek | 26.665 |

**Boundary:** Qwen uses the earlier smaller pool. Rerun matched startup before interpreting long-context capacity as architectural.

Source: ctx_isl131072_c8.json — GLM bf16kv; DeepSeek mtp-off-image; Qwen base

<!--
SLIDE 11 — 1:05; cumulative 12:05.

FULL SPOKEN SCRIPT
[Point left, then right.] At 131K input and concurrency eight, Qwen has the highest output throughput: about 56.5 tokens per second. But GLM has the lowest median first-token latency: 12.89 seconds, compared with 19.84 for Qwen and 26.66 for DeepSeek.

The architecture lesson is that retained state, indexing and input processing influence different parts of a request. A smaller history representation does not automatically produce the earliest first token. Output throughput also includes input work, so this is not a decode-only curve.

Qwen's context arm retains the smaller-pool confound. DeepSeek also has an unresolved discrepancy between its 16K context anchor and a later concurrency-eight batch point.

These results identify what to investigate next. Phase-specific timing and matched startup would distinguish an input-work effect from scheduling or memory pressure.
-->

---

## 12. Prefix reuse is a state-consistency problem

64 requests · 64K shared + 2K unique · c8 · prefix caching enabled throughout

Output tokens/s ↑

Distinct shared prefixes (categories)

Restore the same prefix boundary

Retained KV / compressed history

Recurrent-state checkpoint

Then branch safely; copy or restore mutable state.

GLM and DeepSeek peak at four prefixes; maximum sharing did not maximize throughput.

| Category | GLM | DeepSeek |
| --- | --- | --- |
| 1 | 265.283 | 198.232 |
| 4 | 365.875 | 391.648 |
| 16 | 199.440 | 114.608 |

**Boundary:** Sharing patterns, not cache-on/off speedups. Fill order, eviction and scheduling explanations remain hypotheses.

Source: GLM/DeepSeek prefix_p64k_n*.json; report.md §4

<!--
SLIDE 12 — 1:05; cumulative 13:10.

FULL SPOKEN SCRIPT
Prefix caching is enabled in every condition here. We vary one, four or sixteen shared prefixes across sixty-four requests.

[Point to the middle.] GLM and DeepSeek peak at four prefixes. Maximum sharing did not produce maximum throughput in these retained runs. Fill order, eviction and scheduling are candidate explanations, but none is isolated by a trace.

This is not a cache-on versus cache-off speedup.

[Point to the state diagram.] The architectural challenge is restoring a consistent boundary. KV or compressed history must agree with recurrent checkpoints. A recurrent final state cannot be sliced backward to an arbitrary earlier prefix, but suitable checkpoints can be reused.

Branching and speculative rejection then require safe copies or rollback. Finer checkpoints trade memory and copying for less recomputation. That makes prefix reuse a state-placement and scheduling problem as well as a lookup problem.
-->

---

## 13. Speculation helps at light load; the gain disappears

GLM throughput with MTP ÷ base throughput · 16K input / 256 output

131K CONTEXT / N1

−17% TTFT

But −4% output rate.  
  
Latency and allocation cost move in opposite directions.

At c1: +24% with one draft. At c64: approximately flat with one draft; −9% with five.

| Category | 1 draft token | 5 draft tokens | Base = 1× |
| --- | --- | --- | --- |
| 1 | 1.243 | 1.254 | 1.000 |
| 4 | 1.064 | 1.024 | 1.000 |
| 16 | 0.960 | 0.953 | 1.000 |
| 64 | 0.988 | 0.913 | 1.000 |

**Boundary:** GLM supplies the strongest controls. Real-text acceptance and quality are untested; scheduling attribution remains open.

Source: GLM bf16kv / bf16kv-mtp-n1 / bf16kv-mtp-n5; paired context arms

<!--
SLIDE 13 — 1:10; cumulative 14:20.

FULL SPOKEN SCRIPT
This is the local test of the speculative break-even argument from part one. The vertical axis is MTP throughput divided by base throughput.

[Point left to right.] GLM with one draft gains about twenty-four percent at concurrency one. At sixty-four, one draft is approximately flat and five drafts lose about nine percent. Ordinary batching and speculation compete for resources, so adding draft work may stop paying off as load increases. That mechanism remains a hypothesis until we measure the phases.

At 131K input, one draft lowers median first-token latency by about seventeen percent while reducing output rate by four percent. The desirable policy depends on the objective.

GLM supplies the strongest controls. Other models have qualified comparisons in the report. We have no real-text acceptance or quality study, so the next step is to measure committed progress, phase costs and correctness under representative traffic.
-->

---

## 14. Lower token cost is purchased with higher latency

Illustrative $2.50/GPU-hour × 8 GPUs = $20/node-hour · 16K input / 256 output

Dollars per million output tokens ↓

GLM / C1 → C64

$57.68 → $12.43

Cost / million outputs  
  
Median TPOT:  
7.1 → 129.5 ms

A serving objective needs useful completions within latency targets; token cost is only one input.

| Category | c1 | c64 |
| --- | --- | --- |
| Qwen | 52.293 | 10.730 |
| GLM | 57.684 | 12.427 |
| DeepSeek | 65.333 | 14.269 |

**Boundary:** Includes input work. No quality adjustment, idle-time model, price quote or measured cost per successful task.

Source: exact main batch JSONs; report.md §6. Cost = 20 × 1,000,000 / (3,600 × tok/s)

<!--
SLIDE 14 — 1:00; cumulative 15:20.

FULL SPOKEN SCRIPT
I assume two dollars fifty per GPU-hour, or twenty dollars for the eight-GPU node. Dividing that allocation cost by measured output throughput gives dollars per million output tokens, including all input work.

At concurrency sixty-four, Qwen is about ten dollars seventy-three, GLM twelve dollars forty-three, and DeepSeek fourteen dollars twenty-seven.

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

Kimi K3: 2.8T × 0.5 bytes ≈ 1,304 GiB of ideal four-bit weights

MoE reduces selected compute; weights, communication, lookup and state still need accounting.

| Model | Total / active GEMM parameters | Additional obligation |
| --- | --- | --- |
| GLM | 321.34B / 17.38B | Recurrent state + retained history |
| DeepSeek | 290.91B / 14.08B | Compression and sparse indexing |
| Qwen | 176.94B served / 7.27B | 51.23B n-gram lookup table |

**Boundary:** Historical tensor analyses, not a new recount. Weight payload lower bounds are not validated deployments.

Source: report.md §2; prior tensor analyses; recorded Kimi official-card reference

<!--
BACKUP B4 — use only for relevant questions.

FULL SPOKEN SCRIPT
Resident weights and selected per-token arithmetic are different quantities. With MoE, different tokens can choose different experts, so a batch may touch many more experts than one token does.

Qwen also has a large n-gram lookup table. Its prior analysis counts about 51 billion parameters in that table, but a lookup does not perform a dense multiplication over the entire table each token. Communication, indexing and state management add further costs.

The counts shown are prior tensor analyses, not a new recount or a measured timing breakdown.

For Kimi K3, even an ideal four-bit payload for 2.8 trillion parameters is about 1,304 gibibytes, before scales and runtime state. That exceeds this node's memory budget. I therefore cover its architecture without extrapolating throughput or claiming a validated deployment.
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
