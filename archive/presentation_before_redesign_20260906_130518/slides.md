---
marp: true
theme: default
paginate: true
size: 16:9
---

# Serving frontier MoE models on 8 H100s

### What changes with concurrency, context, caching, and speculation?

Tan Ngo · Professor Kan Zhu and UW SyFI
5 September 2026

**Three measured models. Four systems questions.**

<!--
SLIDE 1 — 0:45; cumulative 0:45.
Opening: “I benchmarked GLM, DeepSeek, and Qwen on eight H100s. The main result is that throughput, latency, and memory capacity lead to different serving choices.”
Pacing: 12 main slides = 18 minutes, followed by 2 minutes for questions. Backup slides are not part of the timed talk. Speaker notes are hidden in rendered slides; rehearse from this Markdown. If questions are outside the 20-minute slot, spend the extra two minutes on slides 5 and 9.
Detailed evidence and exact paths: report.md. All runtime tables are measured; ratios/costs are calculated; proposed mechanisms are hypotheses.
-->

---

## 2. The assignment: four questions

| Question | What I can answer |
|---|---|
| Where are the bottlenecks? | Workload regimes; exact kernel attribution remains open |
| What makes prefix caching difficult? | Sharing results and mixed-state requirements |
| Does speculative decoding help? | Controlled GLM A/B plus qualified observations |
| Which deployment costs less? | Runtime-based token cost, with an explicit price assumption |

**Kimi: architecture coverage only; no local benchmark.**

<!--
SLIDE 2 — 1:15; cumulative 2:00.
Explain scope before claiming results. These are systems measurements, not quality benchmarks. Kimi is included because task.md asks for that family; do not invent a performance ranking for it.
Transition: “First, what differs inside these models?”
Source: task.md; report.md sections 1–2.
-->

---

## 3. Similar MoE idea, different attention state

| Model | Attention stack | Experts / selected |
|---|---|---:|
| GLM-5.3-Flash | 34 recurrent KDA + 11 sparse attention | 288 / 8 |
| DeepSeek-V4-Flash | 43 sparse attention; compressed history | 256 / 6 |
| Qwen3.8-Flash-Next-FP8 | 36 recurrent GDN + 12 QSA | 512 / 10 |
| Kimi K3 — unmeasured | 69 recurrent KDA + 24 gated MLA | 896 / 16 |

**Recurrent state stays fixed with context; retained history grows.**

<!--
SLIDE 3 — 2:00; cumulative 4:00.
Define MoE simply: each token selects a few expert networks, but the deployment still holds the other weights. KDA and GDN are recurrent attention mechanisms; MLA means multi-head latent attention. Do not spend time expanding every acronym on screen.
Sparse lookup and compressed storage are distinct: reading fewer positions does not necessarily remove stored history. Mixed layers create different memory requirements within one model.
All architecture entries are configuration facts, not measured speed explanations. Qwen also has a large n-gram lookup table, discussed in backup B4.
Sources: report.md section 2 and official config links there; Kimi https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json.
Transition: “I held the GPU budget fixed and varied the workload.”
-->

---

## 4. What the benchmark measures

- **8 × H100 80GB**, tensor parallelism + expert parallelism.
- Batch: **16K input / 256 output**, concurrency 1–64.
- Context: **16K–260K input**, concurrency 8.
- Prefix: **64K shared + 2K unique**, 64 requests.

**Output tok/s includes input work. TTFT includes queueing.**

Different builds/backends; synthetic prompts; mostly one retained run per point.

<!--
SLIDE 4 — 1:30; cumulative 5:30.
Define TTFT: time to first token. TPOT: per-request average time between output tokens. Define concurrency as the client request cap, not the instantaneous server token batch.
Main comparisons use MTP off. GLM/DeepSeek main use dev20051; Qwen dev20073. The DeepSeek bridge measured less than 1% difference on its batch grid, but does not prove all model/backend combinations equivalent.
We have the same GPU class/count, not a documented same-physical-node comparison. Batch sample counts are 8/8/32/128; context has 16 requests. No repeat-based error bars. Keep these limitations brief here and use backup B1 for questions.
Source: report.md section 1; bench.sh and arm manifests.
-->

---

## 5. Qwen leads the corrected 16K throughput grid

| Concurrency | Qwen output tok/s | GLM output tok/s | DeepSeek output tok/s |
|---:|---:|---:|---:|
| 1 | 106.2 | 96.3 | 85.0 |
| 4 | 256.8 | 225.3 | 219.8 |
| 16 | 420.2 | 359.2 | 330.9 |
| 64 | **517.7** | **447.1** | **389.4** |

At concurrency 64: Qwen is **1.16× GLM**, **1.33× DeepSeek**.

This is a deployment comparison; quality was not measured.

<!--
SLIDE 5 — 2:00; cumulative 7:30.
Do not read every cell. Point to the consistent ordering and the diminishing gains. Increasing concurrency 64 times produces less than five times output throughput.
Use the corrected Qwen base-util082 arm. Its earlier c4 number, 162.2, was superseded after the startup/cache investigation. Do not use that original number for the headline comparison.
Different quantization, backends, tokenizers, and active-model sizes prevent claiming that one architecture alone caused the ranking.
Source: report.md section 3; Qwen results/base-util082, GLM results/bf16kv, DeepSeek results/mtp-off-image; batch_isl16k_c*.json.
Transition: “But the higher throughput has a latency price.”
-->

---

## 6. More concurrency is not always a better operating point

| Concurrency 8 → 64 | Throughput | Median TPOT |
|---|---|---|
| GLM | 296.9 → 447.1 tok/s; **1.51×** | 18.1 → 129.5 ms; **7.15×** |
| DeepSeek* | 281.1 → 384.9 tok/s; **1.37×** | 16.2 → 152.1 ms; **9.41×** |

**Choose concurrency against a latency objective.**

The finer sweeps suggest diminishing returns around **8–16**.

*DeepSeek uses a separate dev20073 / utilization-0.85 arm.*

<!--
SLIDE 6 — 1:30; cumulative 9:00.
Explain the operator decision: after c8, the remaining throughput gain can cost much more token latency. This is not a universal optimal concurrency; Qwen lacks the finer sweep and we did not evaluate a production latency SLO.
Do not call the shared curve shape proof of communication overhead. An operator trace and a TP/EP sweep are needed to distinguish computation, communication, and scheduling.
Source: report.md section 3; GLM bf16kv and DeepSeek util085-dev20073 eight-point grids.
-->

---

## 7. Long input changes which metric wins

### 131K input, 256 output, concurrency 8

| Model | Output tok/s | Median TTFT |
|---|---:|---:|
| Qwen* | **56.5** | 19.84 s |
| GLM | 47.2 | **12.89 s** |
| DeepSeek | 34.7 | 26.66 s |

**Highest throughput ≠ fastest first token.**

*Qwen context arm retains an earlier, smaller cache pool; rerun needed.*

<!--
SLIDE 7 — 1:30; cumulative 10:30.
Long input adds prefill work, attention/indexing work, and scheduling interactions. Output throughput is over the entire request workload, so falling tok/s does not directly measure slower decode kernels.
At 260K input, recorded Qwen/GLM cache occupancy is 97.3%/89.5%, but occupancy alone proves neither a preemption nor the cause of slowdown. Qwen's pool confound makes architectural claims especially weak.
Also disclose if asked: DeepSeek's 16K context anchor differs substantially from a later c8 batch point. It needs a same-session rerun and is not used here to explain a bottleneck.
Source: report.md section 3; ctx_isl131072_c8.json in the named context arms.
-->

---

## 8. Prefix reuse is a state-management problem

| Number of distinct prefixes | GLM tok/s | DeepSeek tok/s |
|---:|---:|---:|
| 1 | 265.3 | 198.2 |
| 4 | **365.9** | **391.6** |
| 16 | 199.4 | 114.6 |

- Maximum sharing did not maximize throughput in these runs.
- Reuse must restore **both KV and recurrent state** at one boundary.
- These compare sharing patterns, **not cache on versus off**.

<!--
SLIDE 8 — 1:30; cumulative 12:00.
Same setup: 64 requests, c8, 64K shared prefix plus 2K unique suffix, 256 output. Explain the hypothesis: fill ordering and scheduling may make four prefixes faster than one, but we did not trace the cause.
Prefix JSONs lack the cold-run/cache-hit fields used by batch/context tests. Never say all results have zero cache hits; prefix hits are intentional.
Recurrent state can be checkpointed; it cannot simply be sliced backward to an arbitrary earlier prefix. Branching, eviction, copying, and rollback require consistent state across all layer types.
Source: report.md section 4; prefix_p64k_n*.json. Further Qwen data and ratios are in the report.
-->

---

## 9. MTP helps some workloads, hurts others

### GLM: throughput relative to no speculation

| Concurrency | One draft token | Five draft tokens |
|---:|---:|---:|
| 1 | **1.24×** | **1.25×** |
| 4 | 1.06× | 1.02× |
| 16 | 0.96× | 0.95× |
| 64 | 0.99× | **0.91×** |

At **131K / concurrency 8**, one draft token: **TTFT −17%, throughput −4%**.

**Evaluate accepted progress, overhead, latency, and memory together.**

<!--
SLIDE 9 — 1:30; cumulative 13:30.
Explain speculation: draft a candidate, verify with the target, retain accepted progress. More candidates do not guarantee more useful output. At GLM c1, n1 TPOT improves from 7.06 to 5.05 ms. At c64 n5, cache occupancy reaches 99.2% and TTFT worsens substantially.
At 131K, TTFT is 12.89→10.73 seconds while throughput is 47.2→45.3. The scheduling explanation is a hypothesis, not proof that drafting accelerates prefill itself.
Why GLM only on screen? It supplies the strongest controls. Qwen has pool/startup confounds and DeepSeek's historical MTP pair has older instrumentation. Their results remain in report.md section 5, not hidden.
-->

---

## 10. Cost depends on workload and the latency target

Assumption: **$2.50/GPU-hour × 8 GPUs = $20/node-hour**.

`$/million output tokens = 20 × 1,000,000 / (3,600 × output tok/s)`

### 16K input / 256 output, concurrency 64

| Qwen | GLM | DeepSeek |
|---:|---:|---:|
| **$10.73** | **$12.43** | **$14.27** |

Includes input-processing time. No quality adjustment or production idle time.

<!--
SLIDE 10 — 1:30; cumulative 15:00.
The price is an illustrative assumption, not a current quote. Eight GPUs cost $20/hour. Divide that time cost by measured output production and normalize to one million outputs. GLM's example request costs about $0.00318 with 16K input and 256 outputs.
This is not an API price or decode-only cost. Lower concurrency costs more per token but gives lower latency. Different tokenizers and response quality mean cost per token is not cost per solved task.
For a serving decision, request goodput under TTFT/TPOT objectives. No fewer-GPU sweep establishes the optimal deployment.
Source: report.md section 6, calculated from unrounded raw throughput.
-->

---

## 11. Debugging connects architecture to the result

| Architecture / runtime feature | Observed failure | Lesson |
|---|---|---|
| Recurrent layers | Sequence-state capacity blocks startup | Budget state per active sequence |
| Prefix caching | Warmup creates 16,000 reused tokens | Warm kernels; use fresh test prefixes |
| Automatic cache sizing | Qwen c4 changes 162.2 → 256.8 tok/s | Validate startup before explaining speed |

**Follow the evidence: symptom → test → fix → bounded conclusion.**

<!--
SLIDE 11 — 1:30; cumulative 16:30.
Connect three incidents instead of listing fixes. Bug 1: recurrent layers require per-sequence state; pinning a feasible sequence cap gets past that constraint. Bug 6: overlapping warmup creates 16,000 cache-hit tokens; remove that overlap while separately warming kernels. Bug 12: Qwen's startup records 17.07 versus 0.99 GiB activation, a different pool, and 162.2→256.8 c4 tok/s. That overturns the earlier architectural story but does not prove preemptions caused the difference.
Learning guide: fix_bug.md, especially the architecture map and Bugs 1, 6, and 12. Alternative FP8 paths and remaining telemetry gaps are in backups B7–B8. Do not claim that every old guard is fully enforced.
-->

---

## 12. What I would take forward at SyFI

1. **Validate the regimes:** matched startup, repeated runs, context reruns.
2. **Identify the limiting work:** prefill/decode traces and hardware counters.
3. **Test better policies:** hybrid-state caching and scheduling against goodput.

### Main finding

**Serving choices depend on workload, state, and runtime—not parameter count alone.**

**Questions — 2 minutes**

<!--
SLIDE 12 — 1:30; cumulative 18:00, then 2:00 questions.
Closing script: “Qwen leads the corrected short-input throughput grid, but GLM can have lower first-token latency at longer input. Speculation and lower-precision cache each involve tradeoffs. The next research step is to validate and profile these regimes, then use the evidence to improve goodput.”
Do not claim a new scheduler or universal failure of existing systems. The current contribution is the measured tradeoff map and the audit discipline.
Invite feedback on which workload and latency target would matter most for the lab's next experiment.
Stop advancing here. Use the following backup slides only for questions.
-->

---

## Backup B1 — Is this a fair comparison?

| Controlled | Still different / missing |
|---|---|
| GPU class and count | Physical allocation/topology not established |
| Workload target lengths | Tokenizers and equivalent task quality |
| Main MTP-off setting | KV/weight precision and backends |
| Named arms and raw files | vLLM builds; some startup/cache pools |

**DeepSeek bridge: <1% batch-grid build effect; not a universal correction.**

<!--
Answer: “It is a deployment comparison under a fixed GPU budget, not an isolated architecture experiment.”
No repeat-based confidence intervals. Tail percentiles from 8 requests should not be treated as stable population tails.
Exact paths: report.md section 1. Do not call all models engine-matched.
-->

---

## Backup B2 — Which part is actually the bottleneck?

**Not yet identified by operator-level evidence.**

| Candidate | Test |
|---|---|
| GEMMs / expert routing | Operator time and per-rank load |
| Communication / launch overhead | Timeline and feasible TP/EP sweep |
| HBM traffic | Hardware counters within decode kernels |
| Prefill scheduling / cache pressure | Chunk-size A/B, preemption counters |

<!--
Answer: “I observed diminishing throughput returns and increasing latency. Those identify regimes, not the dominant kernel.”
Engine estimates are partial and average over mixed prefill/decode work. Low average modeled bytes/s cannot disprove a bandwidth-bound operator. The earlier active-bytes × output-throughput / concurrency calculation is invalid for this purpose.
-->

---

## Backup B3 — Can recurrent state support prefix reuse?

**Yes: save and restore state at compatible prefix boundaries.**

- KV history and recurrent checkpoints must agree on position.
- A final state cannot recover every earlier state.
- Branches and speculative rollback require safe copies/restoration.
- Cache-on/off and prewarmed controls are still needed.

<!--
Answer: “The issue is checkpoint placement, consistent state, and memory overhead—not a mathematical impossibility of caching.”
The sharing sweep did not measure actual cache-on/off speedup. The largest best/worst pattern ratio is DeepSeek 391.65/114.61 = 3.42×.
Source: report.md section 4; https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/.
-->

---

## Backup B4 — Why are parameter counts insufficient?

- MoE: **resident weights ≠ weights selected by one token**.
- Batched tokens can select different experts.
- Qwen's **51.23B lookup table** is not fully multiplied each token.
- Recurrent state, retained KV, indexing, and communication add costs.

Kimi K3: even ideal four-bit weights require **about 1,304 GiB**.

<!--
Counts are prior tensor analyses/configuration facts, not new runtime measurements. Qwen served total 176.94B, active GEMM path 7.27B; its 51.23B lookup table is roughly 29% of the served count.
Kimi: official 2.8T parameters × 0.5 bytes / 2^30 ≈ 1304 GiB before overhead. Hence architecture-only coverage, with no throughput extrapolation. Weight-only lower bounds also do not prove valid three- or five-GPU deployments for smaller models.
Sources: report.md section 2; https://huggingface.co/moonshotai/Kimi-K3.
-->

---

## Backup B5 — Should we enable MTP or FP8 KV?

| Option | Evidence | Decision still needed |
|---|---|---|
| GLM MTP n1 | +24% throughput at c1; approximately flat at c64 | Target latency and real-text acceptance |
| GLM MTP n5 | −9% throughput, 99.2% pool occupancy at c64 | No throughput case here for a larger draft budget |
| GLM FP8 KV | More reported capacity; −25% versus paired BF16 at c64 | Does capacity enable a useful workload? |

**None is a universal speed switch.**

<!--
The reported capacity ratio 1.805× spans the documented startup comparison, not necessarily the main baseline's later pool. Do not mix capacity sessions. MTP acceptance on synthetic prompts does not establish acceptance on real coding/reasoning text.
For GLM 131K context, n1 gives TTFT −16.7% and output rate −4.2%; at 260K both degrade. Cross-model MTP limitations are in report.md section 5.
-->

---

## Backup B6 — Answers to the hardest follow-ups

| Question | Answer |
|---|---|
| Where are the error bars? | Most points have one retained run; randomized repeats are next. |
| Why not use fewer GPUs? | No deployment sweep establishes the best feasible layout. |
| Is this production cost? | No: illustrative full-allocation token cost, without idle time or quality. |
| Did you replay real traffic? | No: saved workload summaries informed a synthetic grid. |
| What result would change your conclusion? | Matched reruns reversing rankings or traces identifying another dominant cost. |

Evidence and expanded answers: **report.md**.

<!--
Answer directly, state the evidence boundary, and name the experiment that resolves it. Avoid defending an unsupported causal story. Rehearsal priority: fair comparison, exact bottleneck, recurrent-state reuse, MTP controls, and cost formula.
-->

---

## Backup B7 — How to debug a misleading CUDA error

**CUDA error → earlier compiler error → cache quota → redirect JIT caches**

- Disabling CUDA graphs did not resolve the failure.
- Eager execution still needed the same compiler/cache path.
- Redirecting the overlooked caches allowed graph-enabled startup.

**A failed hypothesis test narrows the diagnosis.**

<!--
Source: fix_bug.md Bugs 2–3. Read the earliest relevant exception, not the most repeated worker error. The GPU failure followed an unsuccessful compiler-cache write. The logs do not establish every internal transition or a general CUDA behavior.
Follow-up: quotas differ from volume free space; test a small real write before loading weights. Several backends resolve their own caches, including in C++.
-->

---

## Backup B8 — A failed route is not a failed idea

| GLM FP8 investigation | What it establishes |
|---|---|
| Layout requires 64 positional dimensions; GLM has 0 | That selected route is incompatible |
| Alternate NoPE-compatible path runs | FP8 KV is not universally impossible |
| Paired BF16 348.8 → FP8 262.0 tok/s at c64 | A 25% throughput tradeoff on that stack |

**Experimental stack; numerical parity remains unverified.**

<!--
Source: fix_bug.md Bug 8; report.md section 7. The overlay bypasses a package version check and routes MoE through DeepGEMM. Do not present it as a production recipe. Compare original BF16 to alternate BF16 for the combined backend/software change; alternate BF16 to alternate FP8 for the dtype-associated effect.
If asked about validity gates: fix_bug.md section 6 documents pending harness gaps. Missing scrapes may default to zero, cache contamination prints FAIL without rejecting that point in the branch, and completed>0 does not ensure full completion. These were audited, not repaired during the document revision.
-->
