# Speaking script — frontier architecture and efficient serving

Tan Ngo · Professor Kan Zhu and UW SyFI · 6 September 2026

Use with [the PowerPoint](SyFI_ML_Serving_refined.pptx) and [storyboard](slides.md). **Two parts, 16 main slides, 18 minutes + 2 minutes Q&A.** Part I covers frontier design (slides 2–7); Part II covers measurements, interpretation and a research proposal (8–16). Eight hidden backups follow. Bracketed directions are not spoken. Timings are rehearsal targets, including brief visual pauses.

Edit spoken text between the SCRIPT markers. The builder embeds it verbatim in PowerPoint notes and synchronizes slides.md. Sources: [report.md](report.md), especially its architecture sections; debugging evidence: [fix_bug.md](fix_bug.md). Published mechanisms, local results and proposed research are kept distinct.

## Slide 1 — Three frontier models, one comparison framework

**0:40 · cumulative 0:40**

<!-- SCRIPT 1 -->
Today I will connect frontier model design to measured serving behavior on eight H100 GPUs.

The talk has two parts. First, I will explain how sparse attention, hybrid state, Muon and native speculation pursue efficiency. Then I will test the serving implications using DeepSeek, GLM and Qwen.

My central question is: when does less model work become better performance for the user? The results show why throughput, latency and capacity can lead to different answers.
<!-- END SCRIPT -->

## Slide 2 — Two parts: architecture, then evidence

**0:40 · cumulative 1:20**

<!-- SCRIPT 2 -->
Part one asks where efficiency comes from: which work is removed, what new overhead appears, and whether the change affects training or inference.

Part two asks where those savings survive in a deployed system. I vary concurrency, context and prefix sharing, and examine MTP and allocation cost.

My approach is to move from a mechanism to a prediction, then to evidence and a possible correction. Across comparisons, DeepSeek comes first in orange, GLM second in purple, and Qwen third in teal. The measured contribution is their tradeoff map and the debugging evidence behind it.
<!-- END SCRIPT -->

## Slide 3 — Sparse attention reduces reads; selection has a cost

**1:20 · cumulative 2:40**

<!-- SCRIPT 3 -->
[Point across the three designs.] Dense attention considers the full available history. Sparse attention selects a subset, while compression changes how history is represented. These are related but distinct ways to reduce work.

For a sequence of length S, dense prefill attention-score work grows quadratically. If each query attends to K selected entries, that attention component is roughly proportional to S times K. But that expression does not include the indexer, top-k selection, gathers or state management. Index scoring can still grow with the history.

DeepSeek V4 combines sparse selection with compressed representations. Qwen QSA also has a compressed indexer, while its recurrent layers supply a different memory mechanism.

The prediction is a context-dependent crossover: savings become useful when they exceed selection and kernel overhead. Short contexts may not amortize that overhead. Selection quality also matters.

Our local measurements can characterize the context regime. They do not isolate a sparse-versus-dense speedup because we have no matched dense control.
<!-- END SCRIPT -->

## Slide 4 — Compare model size, active parameters and layer mix

**1:10 · cumulative 3:50**

<!-- SCRIPT 4 -->
[Read across the first row.] These are text-only base-model parameter counts, excluding MTP and vision: approximately 291 billion for DeepSeek, 313 billion for GLM, and 177 billion for Qwen. They come from prior tensor accounting, not a new checkpoint recount.

[Point to the second row.] The active GEMM estimates are much smaller: 14.08, 17.38 and 7.27 billion parameters per token. Qwen's base total includes a 51.23-billion-parameter lookup table. A lookup touches selected entries, so that full table is not counted as active matrix multiplication.

The routed expert counts are 256 with six selected for DeepSeek, 288 with eight selected for GLM, and 512 with ten selected for Qwen. More selected experts does not automatically mean more compute because expert dimensions differ.

Finally, DeepSeek has 43 sparse-attention layers. GLM combines 34 recurrent KDA layers with 11 sparse layers; Qwen combines 36 recurrent GDN layers with 12 QSA layers. These dimensions help explain what to measure, but parameter counts alone do not rank serving speed.
<!-- END SCRIPT -->

Parameter-accounting reference for slide 4 (not spoken): DeepSeek's current 290.91B total minus the historical 0.04B MTP bucket is approximately 290.9B. GLM's rounded main buckets (304.42B + 8.92B) give approximately 313.3B. Subtracting 7.43B MTP and 0.56B vision from its 321.34B checkpoint total gives 313.35B; the 0.01B difference reflects bucket rounding, so this is an approximate base count. Qwen's prior analysis explicitly records 176.94B served base parameters, including 51.23B n-gram lookup. Active GEMM estimates remain the current report's 14.08B / 17.38B / 7.27B. Sources: [DeepSeek buckets](deepseek_v4_flash/history/report.md), [GLM buckets](GLM-5.3-Flash/history/report.md), [Qwen buckets](Qwen3.8-Flash-Next-FP8/history/report.md), [current definitions](report.md).

## Slide 5 — Muon changes training updates; AdamW still has a role

**1:15 · cumulative 5:05**

<!-- SCRIPT 5 -->
Muon belongs in a frontier architecture talk because training efficiency influences which models are practical to build. But it is important to locate that effect correctly.

[Point left.] AdamW uses coordinate-wise moment estimates and decoupled weight decay. [Point right.] Muon transforms a matrix momentum update through approximate orthogonalization, commonly using Newton-Schulz iterations. This changes update geometry and introduces matrix-operation and partitioning costs.

It is not a blanket replacement. DeepSeek and Qwen document Muon for selected matrix groups and retain AdamW for other parameters. GLM's optimizer recipe is not established by the sources used here. That is a documentation limit, not evidence that it uses a particular alternative.

The right evaluation is the quality reached for training time and resources, including optimizer overhead and stability. Muon is not executed in the serving forward pass. Therefore, it cannot directly explain our tokens-per-second ranking. We need to keep training savings separate from inference savings.
<!-- END SCRIPT -->

## Slide 6 — Day-0 speculation: ship a draft, still pay for verification

**1:10 · cumulative 6:15**

<!-- SCRIPT 6 -->
I use zero-day here to mean that native drafting and runtime support are available around a model's release. It does not mean zero computation, and it is not a separate guarantee of speed.

[Point across the three stages.] A native MTP module proposes future tokens. The target verifies the candidates. The runtime then commits valid progress and restores state for any rejected suffix. Proper rejection sampling is needed for the usual target-distribution guarantee; merely having a draft head is insufficient.

Native MTP can avoid waiting for a separately trained draft model. Runtime engineering can reduce repeated indexing and synchronization, but those optimizations are version-specific.

The break-even condition is simple: time per speculative round divided by expected committed tokens must beat ordinary time per token. Acceptance alone cannot answer that question, because drafting, verification and state handling all have costs.

This predicts a load-dependent payoff. We will test that prediction with the controlled GLM comparison in part two.
<!-- END SCRIPT -->

## Slide 7 — An architectural saving must survive the whole request

**1:05 · cumulative 7:20**

<!-- SCRIPT 7 -->
Before showing results, I want to make the causal reasoning explicit.

Suppose attention accounts for thirty percent of a request's runtime and a new kernel makes attention ten times faster. If everything else stays unchanged, the whole request improves by only 1.37 times. This is an illustrative calculation, not our measured time breakdown.

[Point to the right.] Each design therefore needs a discriminating prediction. Sparse or compressed state should affect context scaling. MoE should change the interaction between selected compute and communication. MTP should change committed progress per round. Memory-pool sizing should affect capacity and possibly scheduling.

Part two maps these regimes using local measurements. Where we lack an operator trace or matched intervention, I will explain the plausible mechanism and identify what would falsify it instead of assigning the whole ranking to architecture.
<!-- END SCRIPT -->

## Slide 8 — A fixed GPU budget; three workload axes

**1:05 · cumulative 8:25**

<!-- SCRIPT 8 -->
Each deployment uses eight H100 80-gigabyte GPUs, TP8 and expert parallelism. Main comparisons have MTP off, synthetic text prompts and 256 target output tokens.

The batch axis changes client concurrency at 16K input. The context axis changes input length at concurrency eight. The prefix axis changes sharing among sixty-four requests.

Concurrency is a client cap, not the instantaneous server batch. Output throughput includes input work, and first-token latency includes queueing and prefill.

[Point to the arm boundary.] Corrected Qwen batch results use base-util082. Its context and prefix data retain the earlier pool. Builds, precision and backends also differ across deployments. Most points have one retained run, with no repeat-based confidence intervals.

These are named deployment comparisons, with the limits stated before we interpret the curves.
<!-- END SCRIPT -->

## Slide 9 — Higher concurrency buys throughput at a latency cost

**1:30 · cumulative 9:55**

<!-- SCRIPT 9 -->
[Point to the first four rows.] To answer Kan's question about throughput at different batch sizes, we vary the client concurrency cap while keeping input at sixteen thousand tokens and output at 256. The server continuously batches requests, so this cap is not a fixed GPU batch size.

Qwen leads output throughput at every tested concurrency. At sixty-four, DeepSeek, GLM and Qwen produce about 389, 447 and 518 output tokens per second. These are whole-deployment rates across eight H100s, with input processing included in elapsed time.

[Point to the last two rows.] The latency winner depends on the metric. GLM gives the earliest median first token at concurrency sixty-four: 2.51 seconds versus 3.64 for DeepSeek and 2.94 for Qwen. Qwen has the lowest median time per output token.

Across all three models, increasing concurrency from one to sixty-four yields only 4.6 to 4.9 times more output, while median token latency grows about sixteen to nineteen times. The deployment serves more total work each second, but individual requests receive tokens more slowly.

These results compare deployed configurations. They do not isolate prefill computation, identify the dominant bottleneck, or establish equal answer quality. The next slide shows the full token-latency curves.
<!-- END SCRIPT -->

## Slide 10 — All three buy throughput with higher token latency

**1:05 · cumulative 11:00**

<!-- SCRIPT 10 -->
This chart uses exactly the same three main batch arms as the throughput slide. It shows median time per output token, so lower is better.

[Read the c64 endpoints in model order.] DeepSeek is about 149 milliseconds, GLM about 130, and Qwen about 110. Qwen has the lowest observed token latency as well as the highest output throughput at that point.

[Point to the growth summary.] For every model, moving from concurrency one to sixty-four buys less than five times the output rate while increasing token latency by about sixteen to nineteen times.

Batching may improve utilization while requests compete for resources, but we still need phase measurements to identify the mechanism. The practical choice must satisfy a latency objective. A high aggregate rate does not guarantee a good interactive experience.
<!-- END SCRIPT -->

## Slide 11 — Long input changes which metric wins

**1:05 · cumulative 12:05**

<!-- SCRIPT 11 -->
[Read both charts in the same model order.] At 131K input and concurrency eight, output throughput is about 34.7 for DeepSeek, 47.2 for GLM, and 56.5 for Qwen. First-token latency is 26.66, 12.89, and 19.84 seconds respectively. Qwen leads output rate, while GLM gives the earliest median first token.

The architecture lesson is that retained state, indexing and input processing influence different parts of a request. A smaller history representation does not automatically produce the earliest first token. Output throughput also includes input work, so this is not a decode-only curve.

Qwen's context arm retains the smaller-pool confound. DeepSeek also has an unresolved discrepancy between its 16K context anchor and a later concurrency-eight batch point.

These results identify what to investigate next. Phase-specific timing and matched startup would distinguish an input-work effect from scheduling or memory pressure.
<!-- END SCRIPT -->

## Slide 12 — The best sharing pattern differs across models

**1:05 · cumulative 13:10**

<!-- SCRIPT 12 -->
Prefix caching is enabled in every condition here. We vary one, four or sixteen shared prefixes across sixty-four requests.

[Compare all three curves.] DeepSeek and GLM peak at four prefixes, while Qwen peaks at one. There is no single best sharing pattern across the retained arms. Qwen still uses the earlier smaller pool. Fill order, eviction and scheduling are candidate explanations, but none is isolated by a trace.

This is not a cache-on versus cache-off speedup.

[Point to the state diagram.] The architectural challenge is restoring a consistent boundary. KV or compressed history must agree with recurrent checkpoints. A recurrent final state cannot be sliced backward to an arbitrary earlier prefix, but suitable checkpoints can be reused.

Branching and speculative rejection then require safe copies or rollback. Finer checkpoints trade memory and copying for less recomputation. That makes prefix reuse a state-placement and scheduling problem as well as a lookup problem.
<!-- END SCRIPT -->

## Slide 13 — MTP gains fade as concurrency rises

**1:10 · cumulative 14:20**

<!-- SCRIPT 13 -->
[Point to the first row.] One-token MTP improves recorded output throughput at concurrency one by about twenty-five percent for DeepSeek, twenty-four percent for GLM and seventeen percent for Qwen. At concurrency sixty-four, none records a gain. DeepSeek and GLM are close to flat; Qwen loses about seven percent.

[Point to the acceptance row.] GLM still accepts about seventy-two percent of draft tokens at concurrency sixty-four, yet throughput falls slightly. Acceptance alone cannot predict speedup: committed progress must pay for drafting, verification and state handling. We have not measured which overhead dominates.

Each column uses its own paired baseline. GLM provides the strongest controls. DeepSeek uses historical runs without newer cold telemetry, and Qwen retains startup and pool differences. Small changes have no established statistical significance.

The result motivates testing a load-aware draft budget. It does not establish that such a policy already works, or that MTP should always be enabled.
<!-- END SCRIPT -->

## Slide 14 — Lower token cost is purchased with higher latency

**1:00 · cumulative 15:20**

<!-- SCRIPT 14 -->
I assume two dollars fifty per GPU-hour, or twenty dollars for the eight-GPU node. Dividing that allocation cost by measured output throughput gives dollars per million output tokens, including all input work.

At concurrency sixty-four, the costs in model order are about fourteen dollars twenty-seven for DeepSeek, twelve dollars forty-three for GLM, and ten dollars seventy-three for Qwen.

[Point to GLM.] Raising concurrency reduces calculated token cost from about fifty-eight to twelve dollars per million, while median token latency rises from roughly seven to one hundred and thirty milliseconds.

That is an allocation-efficiency result, not production cost per successful task. Quality, arrivals, idle time and the best GPU count remain unmeasured. My research objective would be useful completions within explicit latency targets, with task quality held comparable.
<!-- END SCRIPT -->

## Slide 15 — Startup state can impersonate an architecture effect

**1:10 · cumulative 16:30**

<!-- SCRIPT 15 -->
The Qwen startup case is a concrete example of revising a hypothesis when the evidence changes.

[Point across the cards.] Recorded peak activation changed from 17.07 to 0.99 gigabytes. The reported cache pool increased from 2.05 million to 3.20 million tokens even though utilization was lowered. Concurrency-four throughput changed from 162.2 to 256.8 tokens per second.

An architecture-only explanation for the earlier low throughput is therefore insufficient: the checkpoint stayed the same, while runtime conditions changed.

This is not a clean utilization-only experiment, because startup changed too. Specific preemption causality is unmeasured. I use the corrected batch arm and retain the context, prefix and MTP caveats rather than transferring a correction factor across axes.

For me, this is the methodological contribution: preserve the failed explanation, locate the confound, correct the comparison, and design the next experiment to separate causes.
<!-- END SCRIPT -->

## Slide 16 — Research direction: budget state and speculation together

**1:30 · cumulative 18:00**

<!-- SCRIPT 16 -->
The findings motivate a concrete research direction: can a serving policy budget speculation and hybrid state together to improve useful throughput under latency constraints?

The hypothesis follows from two observations. GLM's MTP payoff changes with load, and Qwen's startup state changes the resources available to the scheduler. A fixed draft budget may therefore be a poor choice across changing request and memory conditions.

I would first repair the pending measurement gates, match startup conditions, and repeat the qualified comparisons. Then I would collect accepted progress, draft and verification time, state-copy work, preemption and per-request latency.

The intervention would compare MTP off, fixed draft budgets and a simple adaptive budget under the same workload, quality checks and latency targets. I would report useful completions, tails, policy overhead and failures, not just aggregate tokens.

A clear negative result would be no useful gain over fixed policies after those controls. That would tell us to investigate another source of cost rather than defend the policy.

This is the kind of work I want to pursue at SyFI: connect architectural mechanisms to measurements, correct weak explanations, and turn the remaining uncertainty into a falsifiable systems experiment. Thank you.
<!-- END SCRIPT -->

## Backup B1 / slide 17 — What is controlled, and what remains different?

**Use only for a relevant audience question; outside the timed talk.**

<!-- SCRIPT 17 -->
The comparison holds the GPU class and count, parallelism arrangement, and workload targets in common. MTP is off for the main cross-model comparison, and each point is tied to a named arm.

However, precisions, tokenizers, backends, builds and some startup pools differ. We have not established identical physical topology or equal answer quality. So I describe this as a deployment comparison under a fixed GPU budget.

The DeepSeek build bridge observed less than one percent throughput difference on its batch grid. That narrows one concern for that model and grid; it does not make every model and workload engine-matched. Repeated matched-startup comparisons are still needed.
<!-- END SCRIPT -->

## Backup B2 / slide 18 — A workload curve does not identify a dominant kernel

**Use only for a relevant audience question; outside the timed talk.**

<!-- SCRIPT 18 -->
The current request metrics identify regimes, not a dominant operator. Small GEMMs, expert dispatch, collectives, memory traffic and scheduling are all candidates, depending on the phase and load.

To discriminate between them, I would first separate prefill and decode, collect operator durations and shapes, and inspect per-rank load. Hardware memory counters and a controlled TP or EP intervention would then test specific explanations.

The engine estimates cannot substitute for this. GLM's recorded estimates omit attention, and the models have different estimator coverage. A low average modeled bandwidth over mixed request work cannot rule out individual bandwidth-bound kernels. Unknown telemetry is also not a zero measurement.
<!-- END SCRIPT -->

## Backup B3 / slide 19 — A reusable prefix needs a complete state checkpoint

**Use only for a relevant audience question; outside the timed talk.**

<!-- SCRIPT 19 -->
Yes, recurrent state can support prefix reuse through checkpoints. The limitation is that a state at the end of a long sequence does not reconstruct arbitrary earlier states.

The cache key needs the same token prefix and relevant execution identity. At a hit, every layer must resume at a compatible boundary: retained history and recurrent checkpoints must agree on position.

Immutable history can be shared, but requests that branch need protection for mutable state. Speculative rejection also requires a valid rollback point. The design tradeoff is checkpoint granularity: more checkpoints cost memory and copying, while fewer checkpoints can require more recomputation after a partial match. Our sharing sweep does not isolate those costs.
<!-- END SCRIPT -->

## Backup B4 / slide 20 — Resident parameters are not per-token arithmetic

**Use only for a relevant audience question; outside the timed talk.**

<!-- SCRIPT 20 -->
Resident weights and selected per-token arithmetic are different quantities. With MoE, different tokens can choose different experts, so a batch may touch many more experts than one token does.

Qwen also has a large n-gram lookup table. Its prior analysis counts about 51 billion parameters in that table, but a lookup does not perform a dense multiplication over the entire table each token. Communication, indexing and state management add further costs.

The counts shown are prior tensor analyses, not a new recount or a measured timing breakdown.

The three columns also show a precision difference: DeepSeek uses MXFP4 experts with FP8 attention and dense weights, while GLM and Qwen use FP8 weights. DeepSeek's main KV is FP8; the other two use BF16 KV. Therefore even the same GPU count does not create a pure architecture comparison. Recorded disk size is not runtime HBM.
<!-- END SCRIPT -->

## Backup B5 / slide 21 — More capacity and more drafts are not automatic wins

**Use only for a relevant audience question; outside the timed talk.**

<!-- SCRIPT 21 -->
I would begin with the workload objective rather than enable every feature. GLM with one draft token is a promising light-load candidate: throughput improves and median token latency falls. At concurrency sixty-four, five drafts reduce throughput and recorded cache occupancy reaches 99.2 percent. Occupancy alone does not prove the mechanism.

For FP8 KV, the alternate stack provides more reported capacity but loses about twenty-five percent throughput against its paired BF16 control at concurrency sixty-four. Its reported capacity comparison spans specific startup records, which must not be mixed with a later main pool.

The alternate stack also bypasses a version check. Successful synthetic execution does not establish numerical parity or production suitability. Those checks remain necessary before making a deployment choice.
<!-- END SCRIPT -->

## Backup B6 / slide 22 — Answer directly, then name the missing experiment

**Use only for a relevant audience question; outside the timed talk.**

<!-- SCRIPT 22 -->
[If asked about error bars.] Most cells have one retained run. Within-run latency percentiles do not measure uncertainty across repeated runs. Randomized repeats are pending.

[If asked about production cost.] We measured allocation cost for synthetic workloads. Quality, realistic arrivals, idle time and the best feasible GPU count remain unmeasured.

[If asked about cache speedup.] We compared sharing patterns with caching enabled. We need an identical cache-off control to measure cache-on versus cache-off speedup.

[If asked what might change the ranking.] Matched startup and pools, repeated runs, different workload lengths and quality requirements could all change the decision. I would test those explicitly rather than extend the ranking beyond the measured grid.
<!-- END SCRIPT -->

## Backup B7 / slide 23 — Read the earliest cause, not the loudest CUDA symptom

**Use only for a relevant audience question; outside the timed talk.**

<!-- SCRIPT 23 -->
The visible symptom was a CUDA invalid-argument error during startup. Disabling graph-related paths and trying eager execution did not resolve it. That made a graph-only explanation insufficient.

Reading earlier errors showed a compiler-cache write failure involving the TensorRT and DeepGEMM path. Eager and graph execution still shared that preparation dependency. Redirecting the additional backend-specific caches and adding a write preflight allowed graph-enabled startup.

The lesson is to find the earliest relevant exception and test a hypothesis that distinguishes causes. Free space on a volume also does not establish available user quota. The logs support this failure chain, but they do not reveal every internal transition between the failed preparation and the CUDA error.
<!-- END SCRIPT -->

## Backup B8 / slide 24 — Separate the backend change from the dtype change

**Use only for a relevant audience question; outside the timed talk.**

<!-- SCRIPT 24 -->
The original FP8 route failed because its layout required sixty-four positional dimensions while GLM's NoPE configuration had zero. That establishes an incompatible route, not a universal inability to use FP8 KV.

The alternate stack ran, so the next task is to separate its effects. Original BF16 gives about 447 tokens per second. Alternate BF16 gives 349, a twenty-two percent loss associated with the software and backend change. Alternate FP8 gives 262, another twenty-five percent loss relative to that paired BF16 control.

The ratios multiply: 0.780 times 0.751 is about 0.586. The combined reduction is about forty-one percent, not forty-seven percent, and it should not all be attributed to dtype.

The successful route remains experimental. Geometry, layout, kernels, versions and numerical validation all matter. The result establishes a measured capacity–speed tradeoff, not a production recommendation.
<!-- END SCRIPT -->
