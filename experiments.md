# Experiment index

Current interpretation: [report.md](report.md). Debugging and known validation gaps: [fix_bug.md](fix_bug.md). Machine-readable selection: [tools/experiment_arms.json](tools/experiment_arms.json).

A result folder is an experimental arm, not an interchangeable baseline. Do not merge all JSONs into one ranking. `audit_selected` identifies the 14 directories (95 saved points) reviewed for completion and arithmetic; it does not certify statistical validity, cache telemetry, or causal control.

## Arms used in the current synthesis

| Arm | Role | Boundary |
|---|---|---|
| [GLM-5.3-Flash/results/bf16kv](GLM-5.3-Flash/results/bf16kv/) | primary | GLM batch/context/prefix base; BF16 KV, utilization 0.82; points span sessions. |
| [GLM-5.3-Flash/results/bf16kv-fi618](GLM-5.3-Flash/results/bf16kv-fi618/) | experimental | Alternate BF16 stack; experimental version-check bypass. |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1](GLM-5.3-Flash/results/bf16kv-mtp-n1/) | control | GLM MTP n1 batch; compare bf16kv. |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1-context](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/) | control | GLM MTP n1 context; compare bf16kv context. |
| [GLM-5.3-Flash/results/bf16kv-mtp-n5](GLM-5.3-Flash/results/bf16kv-mtp-n5/) | control | GLM MTP n5 batch; compare bf16kv. |
| [GLM-5.3-Flash/results/fp8kv-fi618](GLM-5.3-Flash/results/fp8kv-fi618/) | experimental | Alternate FP8; paired dtype control is bf16kv-fi618; numerical parity unverified. |
| [Qwen3.8-Flash-Next-FP8/results/base](Qwen3.8-Flash-Next-FP8/results/base/) | qualified | Context/prefix usable with smaller-pool caveat; batch superseded by base-util082. |
| [Qwen3.8-Flash-Next-FP8/results/base-util082](Qwen3.8-Flash-Next-FP8/results/base-util082/) | primary | Corrected main batch only; warm compile cache, utilization 0.82. |
| [Qwen3.8-Flash-Next-FP8/results/mtp-n1](Qwen3.8-Flash-Next-FP8/results/mtp-n1/) | qualified | Compare original base only with startup/pool caveats; not corrected base-util082. |
| [deepseek_v4_flash/results/mtp-off](deepseek_v4_flash/results/mtp-off/) | historical-comparison | Older runtime; historical MTP off; missing newer cold telemetry. |
| [deepseek_v4_flash/results/mtp-off-bridge-dev20073](deepseek_v4_flash/results/mtp-off-bridge-dev20073/) | control | Build bridge versus mtp-off-image; batch-grid inference only. |
| [deepseek_v4_flash/results/mtp-off-image](deepseek_v4_flash/results/mtp-off-image/) | primary | Main batch/context/prefix base on dev20051; context discrepancy unresolved. |
| [deepseek_v4_flash/results/mtp-on-noreuse](deepseek_v4_flash/results/mtp-on-noreuse/) | historical-comparison | Older runtime MTP; compare mtp-off; missing newer cold telemetry. |
| [deepseek_v4_flash/results/util085-dev20073](deepseek_v4_flash/results/util085-dev20073/) | control | Finer grid on separate build/utilization; do not splice into main arm. |

## Additional and excluded evidence

| Arm | Status | Treatment |
|---|---|---|
| [GLM-5.3-Flash/results/_discarded-warmup-contaminated](GLM-5.3-Flash/results/_discarded-warmup-contaminated/) | quarantined | Retained measurement artifacts; exclude from headline comparisons. |
| [GLM-5.3-Flash/results/_mtpctx_queued_firstrun](GLM-5.3-Flash/results/_mtpctx_queued_firstrun/) | quarantined | Retained measurement artifacts; exclude from headline comparisons. |
| [GLM-5.3-Flash/results/_util085_queued_firstrun](GLM-5.3-Flash/results/_util085_queued_firstrun/) | quarantined | Retained measurement artifacts; exclude from headline comparisons. |
| [GLM-5.3-Flash/results/fp8kv](GLM-5.3-Flash/results/fp8kv/) | failed-route | Original incompatible fp8_ds_mla route; does not rule out alternate FP8 path. |
| [GLM-5.3-Flash/results/util085](GLM-5.3-Flash/results/util085/) | supplemental | Additional or historical arm; not included in the selected 95-file audit. Consult raw data and current caveats. |
| [Qwen3.8-Flash-Next-FP8/results/_base_c1_jitcold](Qwen3.8-Flash-Next-FP8/results/_base_c1_jitcold/) | quarantined | Retained measurement artifacts; exclude from headline comparisons. |
| [Qwen3.8-Flash-Next-FP8/results/_mtp_c1_jitcold](Qwen3.8-Flash-Next-FP8/results/_mtp_c1_jitcold/) | quarantined | Retained measurement artifacts; exclude from headline comparisons. |
| [deepseek_v4_flash/results/_v4image_firstrun_queued](deepseek_v4_flash/results/_v4image_firstrun_queued/) | quarantined | Retained measurement artifacts; exclude from headline comparisons. |
| [deepseek_v4_flash/results/mtp-off-bridge-dev20073_queued](deepseek_v4_flash/results/mtp-off-bridge-dev20073_queued/) | quarantined | Retained measurement artifacts; exclude from headline comparisons. |
| [deepseek_v4_flash/results/mtp-on](deepseek_v4_flash/results/mtp-on/) | supplemental | Additional or historical arm; not included in the selected 95-file audit. Consult raw data and current caveats. |
| [deepseek_v4_flash/results/util085-dev20073_queued](deepseek_v4_flash/results/util085-dev20073_queued/) | quarantined | Retained measurement artifacts; exclude from headline comparisons. |

Directories with `queued` anywhere in the name or a leading underscore are quarantined, not just those beginning with `_`. Some supplemental arms include failed points. Global directory counts are inventory counts, not counts of clean independent experiments.

## Per-point acceptance for a future run

Require expected completions, zero failures, positive work/duration, consistent throughput arithmetic, verified endpoint/model identity, explicit cache protocol, successful raw telemetry, stable startup, and an immutable config/run ID. Follow-up repeats must use the same retention criteria across arms. The current harness does not enforce all of these; see fix_bug.md §6.
