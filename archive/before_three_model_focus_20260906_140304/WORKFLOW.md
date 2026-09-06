# Handoff for the next session

Updated 6 September 2026. Start here after README.md. This session redesigned the report, debugging guide and local PowerPoint, and added the full speaking script. It did not run new H100 experiments or repair the GPU harness.

## Completed deliverables

- **User's final direction:** two parts, first frontier architecture (including sparse attention, Muon/AdamW and zero-day/native speculation), then measured results, mechanisms, insights and a credible research direction for SyFI. Emphasize research reasoning, not unsupported causal claims or self-promotion.
- **report.md:** adds a decision map, cost/overhead mechanism map, sourced sparse-attention and optimizer explanations, native MTP break-even reasoning, an explicitly illustrative Amdahl calculation, and a falsifiable state/speculation research proposal. Existing numeric tables and arm boundaries remain intact.
- **fix_bug.md:** retains all 13 incidents, adds functional/measurement/causal distinctions, a triage route and proposed discriminating validation fixtures. The saved `metric()` last-sample/default-zero behavior is documented. All harness repairs remain pending.
- **script.md:** full spoken text for 16 main slides and 8 backups. The main script has 2,132 spoken words, with 18-minute timing targets (roughly 105–130 words/minute by slide, including brief pauses), then 2 minutes Q&A.
- **slides.md / SyFI_ML_Serving_refined.pptx:** synchronized 24-slide storyboard/deck, with 16 visible main slides, 8 hidden backups, 9 native editable charts, architecture/state diagrams, consistent design and embedded scripts. Part I is slides 2–7; Part II is slides 8–16.
- **Build ownership:** `tools/build_powerpoint.py` is the entry point; `tools/presentation_design.py` owns visible content and reads exact arm JSONs. Edit spoken text in script.md, then rebuild; the builder regenerates slides.md and embeds scripts/references in notes.
- **Provenance:** `artifacts/presentation_evidence.json` records chart input paths and saved point dates. Pre-redesign documents, builder/checker and deck are preserved under `archive/presentation_before_redesign_20260906_130518/`. Raw experiment JSONs, logs and launch scripts were not edited.
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

Inspected the contact sheet and full-size architecture, Muon, speculation and research-proposal previews. Rendered text overflow found during the first pass was corrected and the checker rerun successfully. The checker also now handles Unicode diagnostic output without a Windows console encoding failure and saves `artifacts/powerpoint_preview/render_check.json`.

Rerun the workspace audit after current-document edits. Rebuild and render after deck/script changes. The audit proves saved arithmetic/completeness, links, inventory and synchronization only; it does not prove telemetry validity, causality or statistical significance. No training A/B, quality evaluation, dense control, new trace or GPU rerun was performed. The proposed adaptive policy has not been implemented or tested.
