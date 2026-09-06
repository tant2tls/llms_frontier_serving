# Handoff for the next session

Updated 6 September 2026. Start here after README.md. This session refined documents and the local deck; it did not run new H100 experiments or fix the GPU harness.

## Completed deliverables

- report.md answers task.md with named arms, calculations, and uncertainty; now connects architecture to five debugging cases.
- fix_bug.md replaces the old narrative with 13 evidence-based incidents, study exercises, and a code-audited list of pending harness gaps.
- slides.md retains 12 timed main slides, adds debugging backup material, and supplies 20 speaker-note entries.
- SyFI_ML_Serving_refined.pptx is the refined deck; tools/ builds and checks it. The earlier PPTX is retained.
- README/AGENTS/experiments/REPRODUCE provide a small active entry set. Obsolete narratives are archived, not erased.

## Evidence boundaries to remember

1. Qwen corrected batch = base-util082. Its base context/prefix and MTP comparisons retain startup/pool confounds.
2. DeepSeek main = mtp-off-image. The finer grid uses util085-dev20073; do not splice them. The context-16K discrepancy needs a matched rerun.
3. GLM provides the strongest MTP controls. Alternate FP8 KV works, but uses an experimental version-check bypass; numerical parity is unverified.
4. Low modeled bandwidth does not prove a latency/communication bottleneck. Recurrent state can be checkpointed for reuse.
5. Prefix sweeps compare sharing patterns, not cache-on/off. No blanket “all points cold” claim is valid.
6. bench.sh has remaining validation gaps: missing cache scrapes can become zero, positive hit warnings need not reject a point, only zero completions are rejected, and prefix validation is separate.

## Next work, if requested

| User intent | Smallest useful next step |
|---|---|
| Practice presentation | Read slides.md notes; quiz from fix_bug.md §7 and report.md §9 |
| Refine slides | Edit slides.md and tools/build_powerpoint.py together; rebuild/render |
| Repair benchmark reliability | Implement the pending gates in fix_bug.md §6 with meaningful fixtures before any GPU rerun |
| Obtain stronger results | Match startup/pools, repeat/randomize runs, investigate DeepSeek context discrepancy |
| Explain the bottleneck | Capture per-operator prefill/decode traces and hardware counters; then test TP/EP/chunk-size changes |

Do not resume GPU jobs from archived to-do lists. Remote node names, paths, package versions, quotas, and process IDs describe historical sessions. Verify current environment before any authorized rerun.

## Local checks

Run `python tools/audit_workspace.py` for current-document links, arm coverage, saved completion/throughput checks, and notes/timing consistency. It does not prove telemetry validity, causality, or statistical significance. Use `python tools/check_powerpoint.py` for actual PowerPoint rendering and text bounds.
