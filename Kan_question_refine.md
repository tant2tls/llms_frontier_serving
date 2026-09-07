# Claude measurement handoff for Kan's questions

Updated 8 September 2026. **Plan only: this revision performs no experiments or harness repairs.**

## Assignment and division of work

**Claude: rerun the main experiments, validate measurements, and return a compact evidence package. Codex: perform final analysis after Tan supplies Claude's new measurements.** Save tokens by focusing on execution correctness and controls; keep progress messages and the handoff brief.

Scope/order: **DeepSeek-V4-Flash → GLM-5.3-Flash → Qwen3.8-Flash-Next-FP8**, eight H100 GPUs per deployment if the authorized environment supports the recorded setup. Kimi remains outside scope.

| Kan's question | Claude measures | Codex analyzes later |
| --- | --- | --- |
| Workload/concurrency bottlenecks | Repeated throughput/latency grid, two context lengths, short phase/operator traces | Scaling, architecture implications, supported bottleneck claims |
| Prefix caching | Matched disabled/prewarmed comparison, reuse counters, correctness probe | Benefit, retained-state requirements and implementation challenges |
| Speculative decoding | Matched MTP off/n1 at low/high concurrency | Speedup, latency, acceptance and extra work |
| Serving cost | GPU count, elapsed time, actual input/output totals from those same runs | Cost/latency comparison with explicit illustrative pricing |

Leave architecture research, causal synthesis, model rankings, cost tables, report/slides/script edits and presentation redesign to Codex. Do not build an adaptive policy or expand into training/optimizer studies.

## Read only what execution needs

Read [README.md](README.md) and [WORKFLOW.md](WORKFLOW.md) first, then [AGENTS.md](AGENTS.md), this file, [task.md](task.md), [REPRODUCE.md](REPRODUCE.md), [experiments.md](experiments.md), and [fix_bug.md §6](fix_bug.md#6-remaining-gaps-in-the-saved-harness--not-fixes-performed-here). Consult report.md §§1, 3–5 only to resolve a specific control. Open exact launch scripts/manifests/logs as needed; do not load the archive or all research/logs.

Reuse the existing harness, launches, cached weights and environment. Inspect commands: bare `bench.sh` runs multiple axes and its defaults are historical. Add only selected-point execution and necessary validation; preserve patches and checks. Verify current paths, ports, executables and GPU allocation.

This handoff is for a session with authorized compute; editing it does not launch workloads. Follow authorization already given in that session. Do not allocate extra compute, change server packages or interrupt unrelated jobs from historical instructions. If resources/access are unavailable, return the concrete blocker and completed local work.

## Bounded measurement matrix

Use fixed output length 256; MTP off for B/C/P. Concurrency means **client request cap**, not observed engine batch size. Record and match arrival policy. Synthetic fixtures support serving measurements, not realistic traffic or quality claims.

| ID / priority | Models | Input / output tokens | Concurrency | Conditions | Repeats per condition |
| --- | --- | --- | --- | --- | --- |
| B — main scaling | All three | 16,384 / 256 | 1, 4, 16, 64 | Base, verified no prefix reuse | 3 |
| S — main speculation | All three | 16,384 / 256 | 1, 64 | MTP n1, paired with matching B off runs | 3 |
| P — main cache control | All three | 65,536 shared prefix + 2,048 unique suffix / 256 | 8 | Cache disabled; cache enabled and explicitly prewarmed | 3 each |
| C — context contrast | All three | 16,384 and 131,072 / 256 | 8 | Same base/configuration, verified no prefix reuse | 3 |

Full target: **30 conditions × 3 repeats = 90 unprofiled runs**: B 36, S 18 additional on-runs, P 18, C 18. B off runs serve as S controls only when configuration, workload, startup and repeat pairing match; otherwise collect additional off controls. Warmup, prewarm, correctness probes and profiles are separate.

Prioritize valid B/S/P results before extending C. Group work by loaded model to minimize reloads while preserving paired comparisons. If time is limited, finish GLM's matched S pair first and mark missing other-model pairs; do not substitute historical results. Estimate execution time from the pilot and fit work to the authorized compute budget. This target is not permission for unlimited retries.

Start B/C with existing request counts `max(8, 2 × concurrency)` at these lengths: B = 8/8/32/128; C = 16. P uses 64 measured requests. Check pilot duration and active-concurrency timelines for ramp-up/drain effects. If insufficient for a loaded comparison, increase counts consistently across that comparison before the three final repeats; retain pilots separately. Otherwise label results as finite request bursts, not steady-state capacity.

Use three separately executed repetitions, not slices of one run. Predeclare seeds and order; alternate/randomize paired conditions where feasible. Use identical request content/order within each off/on or cache pair, reset cache state between conditions, and use distinct repeat fixtures. Record shared versus restarted server processes; shared-process repeats do not measure restart variability. Retain slow valid runs; do not repeat until a preferred ranking appears.

## Controls that must hold

Verify live model/tokenizer revisions, runtime executable/build, GPU type/count/topology, TP/EP, weight/KV precision, scheduler limits and launch flags. Warm representative kernels with disjoint prompts for every configuration. A health check is insufficient. Record startup memory and usable cache/state pool capacity; equal utilization flags alone do not establish equal pools.

Historical references, never output destinations:

| Model | Configuration reference | Boundary |
| --- | --- | --- |
| DeepSeek | `deepseek_v4_flash/results/mtp-off-image`; model's `run_nomtp_image.sh` | Obtain new matched off/on controls; old build bridge covers its batch grid only. |
| GLM | `GLM-5.3-Flash/results/bf16kv`; model's `run.sh`, `_common.sh`, `run_mtp.sh` | Match base/n1 settings; avoid alternate FP8/backend experiments. |
| Qwen | `Qwen3.8-Flash-Next-FP8/results/base-util082`; model's `run.sh` | Verify warm startup/intended pool; old base context/prefix and MTP retain confounds. |

Across models, record deployed stacks; do not force package changes to claim engine matching. Within interventions, hold controllable settings fixed except cache mode or n1. Record MTP-induced capacity changes as deployment behavior; uncontrolled restart/pool changes require repair or an explicitly qualified pair. Never mix old/new MTP baselines.

**B/C:** prevent warmup/repeated seeds from introducing prefix reuse. Verify the declared cold-prefix protocol with valid before/after counters and supported cache reset. If impossible, a separately labelled cache-disabled arm can provide a control; use it consistently within comparisons and record the change.

**C:** keep both lengths in one matched configuration. Inspect historical DeepSeek 16K/c8 batch/context invocation differences; if request-generation paths differ, add a diagnostic using both paths with identical settings. An unrelated new point does not resolve the discrepancy. Qwen C must record its own controlled pool; batch corrections do not repair old context evidence.

**P:** use one shared prefix, token-level identical within a model, and unique suffixes. Preserve fixtures/token IDs and tokenizer revision. Warm kernels for both conditions. For the prewarmed arm, reset cache, fill the prefix, then measure the same 64-request fixture as the disabled arm. Save prewarm elapsed time/token work separately and declare it excluded from the measurement window. Verify actual disable/reset behavior and reuse deltas. Cache-enabled initially-empty is an optional diagnostic; it can gain hits during the run. Historical 1/4/16-prefix sweeps are sharing-pattern comparisons, not disabled/enabled controls.

**S:** verify off/n1 from live configuration, not cumulative counters alone. Save timestamped drafted/accepted token and speculative-round deltas with metric definitions/labels. Capture committed progress per round only when its definition is supported. Perform a small deterministic cached/uncached and MTP off/on correctness probe using matching prompts/settings. Preserve outputs and discrepancies; completion alone is not correctness. A fixed real-text check is optional; synthetic acceptance does not establish practical workload acceptance. No broad quality benchmark is required.

## Validation and bottleneck evidence

Repair relevant pending gates in fix_bug.md §6 before accepting runs. Use shared validation for B/C/P/S, parameterized by cache protocol:

- Require exact expected completions, zero failures, positive work/duration, intended token lengths, and throughput consistent with actual totals/duration. Preserve and reject fixed-length violations.
- Require verified identity and immutable per-run configuration; never silently adopt another served model.
- Keep timestamped raw telemetry, explicit labels/ranks/aggregation, and reject reset-derived deltas. Missing metrics are `null`/unknown with a reason, never zero. Missing required cache evidence invalidates that cache claim; missing optional operator/MTP metrics remain explicit limitations.
- Preserve raw results plus separate acceptance records and slow/tail flags. Printed warnings/exit codes are insufficient. Test partial completion, missing/multiseries/reset counters, cold-run contamination and intentional prefix reuse locally before GPU execution.

Collect available lightweight scheduler/cache telemetry: active decoding sequences, scheduled prefill/verification tokens, occupancy, preemption/evictions, GPU memory and timestamps. Avoid making instrumentation development a separate project.

Attempt short profiles at **B c1 and c64 for each model**: six captures, separate from headline timing. Validate collection on one model first. Save raw traces, commands/version, rank coverage, config/run IDs and capture windows with identifiable prefill/decode phases. Include exported kernel/operator timing and overlap semantics if readily available. Codex will interpret expert work, attention/indexing, recurrent state and communication. After one targeted fix/retry for failed collection, record the blocker and continue other measurements. A long-context profile is optional if it resolves an observed discrepancy.

No existing trace establishes a dominant bottleneck. Operator time does not prove compute/bandwidth saturation. Engine FLOP/byte counters are partial estimates; low modeled bandwidth does not prove communication dominance. Preserve hardware counters if available without launching an extensive counter campaign.

## Evidence package for Codex

Use a new session root such as `measurements/kan_<UTC timestamp>/` with unique model/arm/repeat directories. Never overwrite evidence or existing manifests; retain failed/quarantined attempts.

| File/content | Required information |
| --- | --- |
| `handoff.md` (aim for ≤500 words) | Completed/missing matrix cells, environment, patches/checks, confounds/blockers, evidence paths; factual observations only |
| `runs.csv` or `runs.jsonl`, one row per attempt | Run/model/arm/repeat/pair/session IDs; UTC start/end; accepted/rejected/qualified status/reason; workload/seed/request count; seconds; completed/failed; actual input/output totals and rates; TTFT/TPOT/E2E median and p95/p99 where collected with units; raw/config/log/telemetry paths |
| Immutable configs/workloads | Model/tokenizer revisions; hardware/runtime/launch fingerprints; TP/EP/precision/pools/scheduler; cache/MTP modes; fixture/hash/regeneration command; condition order; server-start ID; warmup/prewarm records |
| Raw evidence | Untouched benchmark JSON; per-request timing/token records where supported; server/client logs; timestamped metric snapshots; correctness outputs; profiles and capture commands |
| Validation/reproduction | Separate timing/cache/MTP validity flags, missing-field reasons, harness changes/patch, test output, exact executed commands, evidence checksums |

Preserve unrounded values and individual repeats; do not replace them with averages or pooled request percentiles. Keep missing telemetry explicit. Copy/sync the complete session folder back to this workspace for Tan; a prose table alone is insufficient.

Update WORKFLOW.md with actual execution, checks and remaining gaps. Leave published arm selection and reports/deck unchanged until Codex reviews the package. If placing new directories under model `results/` instead, register their inventory without promoting them to headline arms. Run `python tools/audit_workspace.py` after current-document/registry changes and save its output. Its scope does not certify new telemetry or repetitions. No deck build is needed without deck-source changes.

## Stop and hand back

Stop when the selected matrix and bounded profiling attempts finish, or authorized budget/access prevents further work. Record missing evidence instead of expanding into 260K, finer concurrency, n5/draft-length searches, TP/EP tuning, backend/precision sweeps, dense controls, trace replay or Kimi. Add only diagnostics needed to validate selected comparisons.

After Tan sends the package, **Codex will** verify provenance/acceptance, compare repeats and uncertainty, calculate cost from measured GPU time including input work, distinguish observation from mechanism, explain cache/MTP implications, and update the requested report/tables/presentation. Price will be explicit and illustrative unless separately sourced. This matrix does not establish equal quality, optimal GPU count, production goodput or cost per successful task.
