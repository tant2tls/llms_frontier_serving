> HISTORICAL ARM NOTE — numbers describe this arm, but causal claims may be superseded. Read the [current report](../../../report.md), [corrected debugging guide](../../../fix_bug.md), and [arm index](../../../experiments.md) before citing. In particular: no proven hardware bottleneck, universal engine equivalence, universal FP8 impossibility, or universal capacity-only effect is established.

# V4-Flash bridge arm — closing the Qwen3.8 engine confound

**Date:** 2026-09-02 · **Result: the two vLLM builds are equivalent (0.997×). The Qwen3.8 ratios
stand as published.**

## Why this arm exists

Qwen3.8-Flash-Next-FP8 cannot run in `vllm/vllm-openai:glm53-flash` (`model_type: qwen4_exp` is
unregistered there), so it was measured in `vllm/vllm-openai:qwen38-flash-next` on vLLM
**`0.1.dev20073+g8e685d198`**, while GLM-5.3-Flash and DeepSeek-V4-Flash were both measured on
**`0.1.dev20051+g487ecf187`**. That is a genuine fairness gap: a **backend swap alone cost 22%** on
GLM, so an engine delta can rival the architecture effect being measured.

`vllm/vllm-openai:qwen38-flash-next` **registers `DeepseekV4ForCausalLM`** (the full
`vllm/models/deepseek_v4/` package is present; `Glm5Next*` is **not**, so V4 is the only possible
bridge). Re-sweeping V4 here measures the build effect directly.

## Method

Flags copied **verbatim** from `../mtp-off-image/manifest.txt`, so the *only* difference from that arm
is the vLLM build. Same node (`tan-8gpus-qwen38-0-0`), same 8×H100, same CUDA 13.0 / torch
2.13.0+cu130, same TP8/EP layout, same `bench.sh` grid, same seeds, spec decode off, all points cold.

## Result [M]

| conc | dev20051 t/s | dev20073 t/s | ratio |
|--:|--:|--:|--:|
| 1 | 85.0 | 84.9 | 0.998× |
| 4 | 219.8 | 219.4 | 0.998× |
| 16 | 330.9 | 330.0 | 0.997× |
| 64 | 389.4 | 386.4 | 0.992× |

**Mean engine effect: 0.997×** (range 0.992–0.998). The newer build is 0.3% slower — inside run-to-run
noise, and **an order of magnitude smaller than the smallest architecture effect being claimed** (the
1.12× Qwen/GLM ratio at c=1).

**Consequence: every Qwen-vs-GLM and Qwen-vs-V4 ratio in the report is valid as measured.** Applying
the correction changes nothing (e.g. c=64 Qwen/GLM: 1.16× raw → 1.17× corrected).

## ⚠️ Three of the four points had to be re-run first — same trap as before

The first pass produced points that were **cold and complete yet 25–60% low**, and would have made the
build effect read **0.73×** ("the qwen38 image is 27% slower") — a completely wrong headline, and the
*second* time this exact failure mode has appeared in this project.

| conc | first pass | p99/p50 TTFT | re-run warm | understated by |
|--:|--:|--:|--:|--:|
| 4 | 89.3 | **7.96×** | **219.4** | **−59%** |
| 16 | 124.0 | **8.25×** | **330.0** | −62% |
| 64 | 284.0 | 10.4× | **386.4** | −26% |

`bench.sh`'s concurrency-aware queueing guard flagged all three (**p99 TTFT ≫ median** is the
signature). The cause is scheduler queueing while the engine is still cold — this build adds a
**DeepGEMM warmup pass** (1,261 kernels, ~5 min) that `dev20051` did not have, so `/health 200`
arrives well before the engine reaches steady state. Engine init took **538.8 s** here.

First-pass points retained in `../mtp-off-bridge-dev20073_queued/` as evidence, not deleted.

**Transferable lesson:** `/health 200` is not a readiness signal for benchmarking. Both this arm and
the Qwen c=1 points needed a warm re-run. Always check `p99/median TTFT` before citing a number.
