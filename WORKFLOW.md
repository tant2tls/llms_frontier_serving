# Handoff for the next session

Updated 6 September 2026. Start here after README.md. The latest revision narrows the active report and presentation to DeepSeek, GLM and Qwen and makes their architecture and measurements directly comparable. No new H100 experiments or GPU harness repairs were performed.

## Completed deliverables

- **User's latest direction:** focus only on DeepSeek-V4-Flash, GLM-5.3-Flash and Qwen3.8-Flash-Next-FP8. Kimi must be added only if explicitly requested later. This overrides the original broader assignment and is now recorded in task.md. Preserve the two-part architecture/measurement structure and emphasize easy comparison using DeepSeek → GLM → Qwen order and orange/purple/teal identities.
- **report.md:** opens with a three-model measurement snapshot and adds a matrix of identical architecture dimensions. Comparison columns/rows are reordered consistently without changing numeric cells. Removed Kimi coverage, cost extrapolation and its Q&A. Existing mechanisms, analytical framework, proposal and evidence boundaries remain.
- **fix_bug.md:** retains all 13 incidents, adds functional/measurement/causal distinctions, a triage route and proposed discriminating validation fixtures. The saved `metric()` last-sample/default-zero behavior is documented. All harness repairs remain pending.
- **script.md:** all 24 spoken scripts remain embedded verbatim; changed architecture, latency, prefix, MTP, cost and backup scripts to follow the new visuals. Main script: 2,131 spoken words; 18-minute timing targets plus 2 minutes Q&A.
- **slides.md / SyFI_ML_Serving_refined.pptx:** 16 visible main slides and 8 hidden backups, now with 11 native editable charts. Three-column architecture and precision matrices; all three models in main-arm throughput/TPOT and prefix plots; aligned MTP panels with each pair's control status; identical-scale c1/c64 cost panels. Part I is slides 2–7; Part II is slides 8–16. No Kimi content remains in visible slides or embedded notes.
- **Build ownership:** `tools/build_powerpoint.py` is the entry point; `tools/presentation_design.py` owns visible content and reads exact arm JSONs. Edit spoken text in script.md, then rebuild; the builder regenerates slides.md and embeds scripts/references in notes.
- **Provenance:** `artifacts/presentation_evidence.json` records chart input paths and saved point dates, including the historical DeepSeek `batch_c*.json` MTP pair. This revision's prior task/documents/design/deck are preserved under `archive/before_three_model_focus_20260906_140304/`; the earlier redesign archive remains. Raw experiment JSONs, logs and launch scripts were not edited.
- **Research sources:** primary optimizer, architecture and speculation sources were checked on 6 September 2026 and cited in report.md and relevant notes. “Zero-day” is explicitly defined as native drafting/day-0 runtime support; no specific separately named zero-day algorithm was supplied.

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
| Practice presentation | Rehearse script.md against the deck; use fix_bug.md §7 and report.md §9 for questions |
| Refine slides | Edit tools/presentation_design.py and the corresponding script.md text/timing; rebuild/render |
| Repair benchmark reliability | Implement the pending gates in fix_bug.md §6 with meaningful fixtures before any GPU rerun |
| Obtain stronger results | Match startup/pools, repeat/randomize runs, investigate DeepSeek context discrepancy |
| Explain the bottleneck | Capture per-operator prefill/decode traces and hardware counters; then test TP/EP/chunk-size changes |

Do not resume GPU jobs from archived to-do lists. Remote node names, paths, package versions, quotas, and process IDs describe historical sessions. Verify current environment before any authorized rerun.

## Local checks

Completed: `python tools/build_powerpoint.py`; `python tools/audit_workspace.py` (95 selected points across 14 selected arms; 25 registered arms; 24 notes/scripts; 18-minute main timing; no issues); and `python tools/check_powerpoint.py` in installed Windows PowerPoint (24 rendered slides, backups 17–24 hidden, no text overflow).

Latest revision checks also verified that all 24 actual PPTX notes contain the corresponding script verbatim, no Kimi content remains in slides/notes, and the main throughput/TPOT chart series follow DeepSeek → GLM → Qwen with values matching exact source JSONs. Inspected full-size architecture, three-model token-latency, qualified MTP and common-scale cost previews.

Inspected the contact sheet and full-size architecture, Muon, speculation and research-proposal previews. Rendered text overflow found during the first pass was corrected and the checker rerun successfully. The checker also now handles Unicode diagnostic output without a Windows console encoding failure and saves `artifacts/powerpoint_preview/render_check.json`.

Rerun the workspace audit after current-document edits. Rebuild and render after deck/script changes. The audit proves saved arithmetic/completeness, links, inventory and synchronization only; it does not prove telemetry validity, causality or statistical significance. No training A/B, quality evaluation, dense control, new trace or GPU rerun was performed. The proposed adaptive policy has not been implemented or tested.
