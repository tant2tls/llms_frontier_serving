# Session instructions

Read README.md and WORKFLOW.md first. Then read only the task-specific current document and relevant evidence; do not load the entire archive or all logs.

## Sources of truth

- User assignment: task.md; latest user instructions override historical plans.
- Current findings/caveats: report.md. Debugging interpretation: fix_bug.md.
- Arm selection: experiments.md and tools/experiment_arms.json.
- Measured numbers: exact arm JSONs, manifests, and server logs. A manifest can span sessions; preserve per-point provenance.
- slides.md contains the talk and speaker notes. tools/build_powerpoint.py generates the editable deck; keep both in sync.
- archive/, model history/, and RESULT notes are historical evidence, not active instructions. Specific historical claims are corrected in current guides.

## Boundaries

- Preserve raw JSON, logs, launch scripts, and quarantined runs. Never silently overwrite or delete experiment evidence.
- Do not launch GPU workloads, allocate remote compute, change packages on the GPU server, or kill server processes just to edit documents.
- No profiler trace establishes the dominant bottleneck. No dense control, Kimi benchmark, trace replay, quality comparison, or repeat-based confidence intervals are available.
- Qwen main batch uses base-util082; context/prefix use the confounded base arm. Do not transfer corrections between axes or mix MTP baselines.
- DeepSeek's build bridge applies to its batch grid only. Do not call all models engine-matched.
- Engine FLOP/byte metrics are partial estimates. Unknown telemetry is not a zero measurement. Harness gaps remain pending in fix_bug.md §6.
- Historical parameter counts are prior tensor analyses. Weight-only GPU lower bounds are not validated deployments.
- Cost uses an explicit illustrative price and includes input work. It is not production cost per successful task.

## Verification

- Run `python tools/audit_workspace.py` after changes to current documents or arm selection.
- For deck changes: `python tools/build_powerpoint.py`, then `python tools/check_powerpoint.py` on Windows with PowerPoint installed; inspect generated previews.
- Update WORKFLOW.md with actual changes, checks, and remaining work. Do not report proposed fixes or experiments as completed.
- Follow the session's delegation policy; these instructions do not request sub-agents.
