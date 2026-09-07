# Focused measurement plan for Kan's questions

Date: 7 September 2026. Status: execution guide; no new measurements are claimed here.

## Goal and scope

Produce a concise, defensible explanation of **architecture → expected cost → measured behavior → serving implication**, organized around Kan's four questions:

1. Across workloads and batch sizes, which part of the model is the bottleneck?
2. What makes prefix caching difficult?
3. When does speculative decoding help or hurt?
4. How do serving costs compare?

Prioritize a few explained, reproducible results. Reuse valid existing evidence. Spend additional GPU time on missing controls and causal evidence, not another exhaustive sweep.

Keep the current measured scope: DeepSeek-V4-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next-FP8. Kan's original email includes Kimi; disclose its absence. The current project direction excludes Kimi, so do not add its deployment or measurements without a new scope instruction.

## Start efficiently

Read README.md, WORKFLOW.md, AGENTS.md, this guide, REPRODUCE.md, experiments.md, and the relevant sections of report.md and fix_bug.md. Read only the exact launch scripts, manifests, raw points, and logs needed next.

Existing launch and benchmark guides are already set up. Reuse bench.sh, the model launch scripts, caches, and existing result conventions. Verify their current values; historical paths and script defaults may differ from the intended baseline. Do not rebuild the environment, redownload weights, or invent a second harness when a small targeted change suffices.

This file specifies work for an execution session with available, authorized compute. Adding this guide does not itself start GPU jobs. Follow the session's existing authorization and resource limits; do not repeatedly request permission already given. Never allocate extra compute or interrupt unrelated jobs based only on historical scripts.

Before running, make a short evidence checklist: question, existing evidence, specific missing control, planned run, expected decision. Every new run must resolve a named uncertainty. Group work by loaded model to minimize reloads, but alternate paired conditions where feasible and record restart/pool differences.

## Priority 0 — Make selected measurements trustworthy

Repair only validation gaps needed for the planned runs, using fix_bug.md §6:
- Require expected completions, zero failures, positive work/duration, and consistent throughput arithmetic.
- Verify endpoint/model identity and record immutable model/runtime/configuration information.
- Missing cache telemetry is unknown, never zero. Reject a claimed cold-prefix measurement if successful telemetry does not establish its protocol.
- Warm compiled kernels using disjoint prompts; a healthy endpoint is insufficient.
- Preserve raw JSON, logs, and quarantined results. Use a new directory for every session/arm/repeat. Never overwrite baseline data or let a rerun rewrite its manifest.

Record GPU type/count/topology, model revision, runtime build, TP/EP, weight/KV precision, startup state, cache pool size, scheduler limits, token lengths, request count, seed, and cache protocol. Match paired conditions except the intended intervention; record unavoidable differences.

Repeat only headline points and needed controls: start with three independent repetitions, retain slow valid runs, and show their spread. Add repetitions only if variability could change the conclusion. Use enough requests/time to avoid basing loaded behavior on a brief ramp-up; preserve comparability when changing the workload duration. Within-run percentiles are not repeat-based confidence intervals.

## Question 1 — Workload, batch size, and bottlenecks

### Reuse
Use the primary 16K-input/256-output concurrency grid: c1, c4, c16, c64.
- DeepSeek: deepseek_v4_flash/results/mtp-off-image
- GLM: GLM-5.3-Flash/results/bf16kv
- Qwen: Qwen3.8-Flash-Next-FP8/results/base-util082

Report output throughput and TPOT together. Label concurrency as a client cap, not engine batch size. Output throughput includes prefill and scheduling; TTFT includes queueing.

### Targeted reruns
- Resolve DeepSeek's 16K/c8 context-versus-batch discrepancy in one matched session/configuration before interpreting that context curve.
- Rerun only headline Qwen context points with controlled warm startup and the intended pool/configuration. The corrected batch arm does not correct old context/prefix data.
- Prefer 16K and 131,072-token inputs for the main context comparison. Keep 260K and finer concurrency sweeps as backup unless they resolve a specific question.
- The DeepSeek build bridge supports only its recorded batch grid; do not transfer it to other models or axes.

### New evidence: highest priority
Capture short representative prefill/decode profiles at 16K/c1, 16K/c64, and 131,072/c8 for each model where feasible. Start with one model to validate trace collection, then reuse that procedure. Reuse throughput runs for compatible telemetry; profile separately when profiling perturbs timing.

Record actual per-step prefill tokens, decoding sequences, and verification tokens. Attribute time to expert GEMMs, routing/dispatch/combine and collectives, attention/indexing/gather, recurrent state, and remaining overhead. Avoid double-counting overlapping kernels/communication. Use unprofiled runs for headline latency/throughput.

If naming a hardware limit, inspect counters for the dominant kernels. Operator time alone does not prove bandwidth saturation or compute saturation. Do not infer communication dominance from low partial modeled bandwidth. If attribution remains unavailable, report a hypothesis and the missing evidence.

**Deliverable:** throughput/TPOT versus concurrency, TTFT versus input length, and one compact phase/operator breakdown explaining the major change. Introduce a short-input/long-output case only if decode isolation remains unresolved; do not automatically expand the full grid.

## Question 2 — Prefix-cache benefit and implementation challenges

Use the existing 65,536-token shared prefix + 2,048-token unique suffix, 256 output tokens, c8 fixture. Compare identical request content/order under:
1. Cache disabled.
2. Cache enabled, initially empty.
3. Cache enabled, prefix explicitly prewarmed.

Compiled kernels must be warm for every condition. Confirm token-level prefix identity, unique suffixes, and the runtime's actual cache-reset/disable behavior. State whether prefix-fill work is included; report prewarm cost separately when excluded.

Measure TTFT, output throughput, successfully scraped reused-token/hit deltas, and cache/state footprint. Add eviction/preemption counters if explaining pressure. For each model, perform a small deterministic correctness check that cached resumption agrees with uncached execution under the same settings; investigate numerical differences rather than treating throughput completion as correctness.

Explain which state must be retained: attention/history/index state, recurrent checkpoints, boundary alignment, and branch copy or copy-on-write. A final recurrent state does not reconstruct arbitrary earlier boundaries. Connect each challenge to the relevant architecture and runtime source.

The existing 1/4/16-prefix sweep compares sharing patterns, not cache-on/off speedup. Keep it in backup. Investigate its nonmonotonic behavior only if it is necessary to the main conclusion.

**Deliverable:** one controlled cache comparison plus a compact per-model state/challenge table.

## Question 3 — Speculative decoding

Use GLM as the main measured case because its existing base/n1/n5 comparisons have the strongest controls. Start with c1 and c64 at 16K/256, MTP off versus n1. Reuse existing n5 evidence; rerun n5 only to explain a specific loaded-case loss.

Report throughput ratio and TPOT alongside accepted draft tokens, committed progress per round, draft length, and draft/verification/state-management time where available. Acceptance alone does not establish speedup. Preserve metric definitions and raw counters.

Explain whether progress per speculative round compensates for its extra work, state pressure, and scheduling effects. Check correctness under the supported speculative algorithm. Synthetic-prompt results do not establish representative acceptance: add one small fixed real-text workload if making a practical speculation claim, rather than replaying the whole matrix.

DeepSeek's historical off/on pair must stay paired with its historical baseline. Qwen's old base/MTP pair has startup/pool confounds. Do not substitute a newer baseline. If claiming a cross-model MTP comparison, obtain matched off/on controls for the other models at the selected low/high-load points; otherwise label those results provisional and keep them in backup.

**Deliverable:** one low-load/high-load MTP comparison and a round-cost explanation. Do not launch a draft-length search or build an adaptive policy for this report.

## Question 4 — Serving cost

Derive cost from the same accepted workload runs; no separate cost sweep is needed.

GPU-seconds/output token = GPU count / measured output tokens per second.
Dollars/million output tokens = GPU count × GPU-hour price × 1,000,000 / (3,600 × output tokens per second).

Use an explicit illustrative price, or a dated sourced price. Charge all measured work consistently: current output throughput includes input processing. Report workload, GPU count, precision, TTFT, and TPOT next to cost. Do not extrapolate eight-GPU measurements into optimal GPU counts.

If asserting cost under a service objective, specify TTFT/TPOT limits before evaluation and measure completed requests satisfying both per unit time. Aggregate medians do not give goodput. Otherwise present descriptive measured cost/latency tradeoffs. Equal token cost does not establish equal quality or cost per successful task.

**Deliverable:** one small cost table using selected light/loaded operating points. Avoid multiple cost plots that repeat the throughput ranking.

## Reporting and stopping rules

Target 8–10 main slides or an equally concise report:
- Scope/setup and one aligned architecture table: attention organization, active expert work, retained state, precision, native drafting.
- Bottleneck evidence and workload/latency tradeoffs.
- Prefix-cache comparison and state challenges.
- Speculation benefit/loss and explanation.
- Cost with latency, conclusions, and remaining gaps.

For each result state: **what changed; what was measured; why it likely happened; what evidence supports that explanation; what decision it supports.** Distinguish measured, calculated, and hypothesized claims.

Move optimizer details, full parameter accounting, fine sweeps, backend/FP8 experiments, debugging history, and untested research proposals to backup. Preserve source data and link every headline to exact run IDs.

Stop extending the experiment matrix once each question has defensible evidence or a clearly bounded limitation. Prioritize missing operator evidence and cache controls over another throughput point. Do not rerun all saved experiments.

After execution, update report.md, experiments.md/arm registry as needed, and WORKFLOW.md with actual runs, checks, findings, and unresolved gaps. Keep the existing presentation build workflow if slides are changed. The current audit hardcodes 16 main/24 total slides and 18 minutes; if a shorter deck is implemented, update its structural expectations deliberately while preserving arithmetic/link/provenance checks. Run python tools/audit_workspace.py after document/arm changes and the documented render checks after deck changes. Never report planned runs, unavailable traces, or harness repairs as completed.

