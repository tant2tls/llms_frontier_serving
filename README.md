# ML serving experiments for UW SyFI

Tan Ngo · DeepSeek-V4-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next-FP8 · Eight H100 GPUs per deployment

## Current deliverables

| Need | Start here |
|---|---|
| Present the work | [Refined PowerPoint](SyFI_ML_Serving_refined.pptx): 16 main slides + 8 hidden backups; frontier architecture, then measured serving tradeoffs; 18-minute talk + 2-minute Q&A |
| Read the results and limitations | [report.md](report.md) |
| Rehearse timing and explanations | [slides.md](slides.md), also embedded as PowerPoint speaker notes |
| Know what to say on each slide | [script.md](script.md): full spoken text for all 24 slides, embedded in the deck notes |
| Learn architecture through debugging | [fix_bug.md](fix_bug.md): 13 incidents, evidence, corrections, exercises, remaining gaps |
| Continue in a later session | [WORKFLOW.md](WORKFLOW.md) and [AGENTS.md](AGENTS.md) |
| Inspect a measured arm | [Experiment index](experiments.md), then its raw JSON, manifest, and logs |
| Rebuild or plan a rerun | [REPRODUCE.md](REPRODUCE.md) |
| Read broader family research | [Architecture reference index](final_presentation/README.md) |

The current assignment is [task.md](task.md). Compare **DeepSeek → GLM → Qwen** using the same architecture dimensions and workload metrics. Kimi is outside the active report and talk and will be added only if requested. Its earlier research remains preserved. Exact operator bottlenecks, quality, realistic traffic replay, and repeat-based uncertainty remain unmeasured.

## Workspace map

```text
report.md / slides.md / script.md        analysis, storyboard, full speaking script
fix_bug.md                              incident diagnosis and pending harness gaps
SyFI_ML_Serving_refined.pptx             current editable deck
experiments.md                         authoritative arm-selection index
tools/                                 local build, render, and audit utilities
artifacts/                             generated previews and audit output
archive/notes/                         superseded root narratives and plans
final_presentation/                    dated architecture research, not final slides
<model>/results/ + logs/                original experimental evidence
<model>/history/                       superseded per-model narratives
<model>/run*.sh                         historical Linux launch configurations
bench.sh / sending.sh / normalize.sh    historical harness/utilities; known gaps documented
```

No experiment files or logs were deleted. The previous `SyFI_ML_Serving.pptx` is retained as an earlier deliverable; use the refined filename above. PowerPoint's `~$...` file is an application lock, not a cleanup target.

The version before the redesign is preserved in [the dated presentation archive](archive/presentation_before_redesign_20260906_130518/); the version before narrowing to three models is in [the scope-change archive](archive/before_three_model_focus_20260906_140304/). The current deck uses 11 editable charts driven by exact arm JSONs, aligned three-model architecture/precision matrices, and state/optimizer/speculation diagrams. [Chart evidence ledger](artifacts/presentation_evidence.json) records source paths and saved point dates; it does not certify telemetry or causality.

This workspace currently has no `.git` directory. Archiving is file preservation, not a Git commit or remote backup. Reproducible local checks and dependencies are documented in [REPRODUCE.md](REPRODUCE.md).
