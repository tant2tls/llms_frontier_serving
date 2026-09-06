# Rebuild locally; reproduce experiments deliberately

## Current local deliverables

From the workspace root, install the dependencies for your environment if needed:

```powershell
python -m pip install -r tools/requirements.txt
python tools/audit_workspace.py
python tools/build_powerpoint.py
python tools/check_powerpoint.py
```

The builder writes `SyFI_ML_Serving_refined.pptx`: 16 main slides and 8 hidden backups. `tools/build_powerpoint.py` calls `tools/presentation_design.py`, which owns visible text, native editable charts and diagrams. Charts read exact saved arm JSONs and export a per-point source ledger to `artifacts/presentation_evidence.json`.

Edit the full spoken script in `script.md`; the builder embeds it in the PowerPoint notes and regenerates the visible-content storyboard and notes in `slides.md`. Keep script titles/timing aligned with the design. Rebuild after changing either script or visuals. Avoid editing generated Markdown bodies independently of the design source.

The render check requires Windows and installed Microsoft PowerPoint, opens the deck without a presentation window, and saves slide previews, a contact sheet and `render_check.json` under `artifacts/powerpoint_preview/`. It checks rendered text bounds, not scientific validity. Inspect the charts and diagrams visually as well.

The local audit uses the standard library and validates current links, registered arms, saved result arithmetic, and 18-minute main-talk timing. It treats missing historical telemetry as a limitation, not fabricated evidence.

## Historical GPU environment

The GPU experiments used Linux containers. This Windows workspace is an evidence/documentation copy, not a configured eight-H100 serving environment. The original remote paths embedded in model launch scripts may not exist now.

| Model / arm | Recorded launch reference | Scope |
|---|---|---|
| GLM base | [run.sh](GLM-5.3-Flash/run.sh), [_common.sh](GLM-5.3-Flash/_common.sh) | dev20051, TP8/EP, BF16 KV, utilization 0.82 |
| GLM MTP | [run_mtp.sh](GLM-5.3-Flash/run_mtp.sh) | Match draft budget and base arm |
| GLM alternate KV stack | [FP8](GLM-5.3-Flash/run_fp8kv_fi618.sh), [BF16 control](GLM-5.3-Flash/run_bf16kv_fi618.sh) | Experimental overlay; do not assume package/ABI compatibility |
| DeepSeek main | [run_nomtp_image.sh](deepseek_v4_flash/run_nomtp_image.sh) | Match main manifest; older conda runs are distinct |
| Qwen main batch | [run.sh](Qwen3.8-Flash-Next-FP8/run.sh), [manifest](Qwen3.8-Flash-Next-FP8/results/base-util082/manifest.txt) | Explicit GPU list/TP8/utilization 0.82; script defaults differ |

Before an authorized run, check the exact executable, model revision, backend, packages, GPU topology, cache paths, and endpoint identity. Retain existing model caches. Warm representative kernels with disjoint prompts and record startup memory. See [fix_bug.md §6](fix_bug.md#6-remaining-gaps-in-the-saved-harness--not-fixes-performed-here) for the pending validation gates.

Use a **new result directory for every session/arm/repeat**; never target the published baseline directory. bench.sh skips existing files and can rewrite a manifest. Explicitly set MODEL, MODELDIR, OUTDIR, PORT, VLLM, PY, QUANT, and KV_DTYPE after verifying the current environment. A successful health request or script exit is insufficient to accept a measurement.

The longer [historical reproduction notes](archive/notes/REPRODUCE.md) preserve past commands and troubleshooting. They include obsolete claims, fixed remote paths, and unsafe-to-copy assumptions; current evidence boundaries above take precedence.
