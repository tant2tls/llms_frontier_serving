# ML serving experiments for UW SyFI

Tan Ngo · GLM-5.3-Flash, DeepSeek-V4-Flash, Qwen3.8-Flash-Next-FP8 · Eight H100 GPUs per deployment

## Current deliverables

| Need | Start here |
|---|---|
| Present the work | [Refined PowerPoint](SyFI_ML_Serving_refined.pptx): 12 main slides + 8 hidden backups; 18-minute talk + 2-minute Q&A |
| Read the results and limitations | [report.md](report.md) |
| Rehearse timing and explanations | [slides.md](slides.md), also embedded as PowerPoint speaker notes |
| Learn architecture through debugging | [fix_bug.md](fix_bug.md): 13 incidents, evidence, corrections, exercises, remaining gaps |
| Continue in a later session | [WORKFLOW.md](WORKFLOW.md) and [AGENTS.md](AGENTS.md) |
| Inspect a measured arm | [Experiment index](experiments.md), then its raw JSON, manifest, and logs |
| Rebuild or plan a rerun | [REPRODUCE.md](REPRODUCE.md) |
| Read broader family research | [Architecture reference index](final_presentation/README.md) |

The original assignment is [task.md](task.md). Current conclusions live in the root report; other indexes link there instead of maintaining competing numeric summaries. Kimi is architecture-only. Exact operator bottlenecks, quality, realistic traffic replay, and repeat-based uncertainty remain unmeasured.

## Workspace map

```text
report.md / slides.md / fix_bug.md       current analysis, talk, learning guide
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

This workspace currently has no `.git` directory. Archiving is file preservation, not a Git commit or remote backup. Reproducible local checks and dependencies are documented in [REPRODUCE.md](REPRODUCE.md).
