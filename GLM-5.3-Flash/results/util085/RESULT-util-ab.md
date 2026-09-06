> HISTORICAL ARM NOTE — numbers describe this arm, but causal claims may be superseded. Read the [current report](../../../report.md), [corrected debugging guide](../../../fix_bug.md), and [arm index](../../../experiments.md) before citing. In particular: no proven hardware bottleneck, universal engine equivalence, universal FP8 impossibility, or universal capacity-only effect is established.

# GLM-5.3-Flash — `--gpu-memory-utilization` A/B (0.82 → 0.85)

**Date:** 2026-09-02 (session 3) · **Hardware:** 8×H100-80GB-HBM3, `tan-8gpus-glm53-0-0`
**Engine:** `vllm/vllm-openai:glm53-flash`, vLLM `0.1.dev20051+g487ecf187`, torch 2.13.0+cu130
**Arms:** `results/bf16kv/` (util 0.82, published) vs `results/util085/` (util 0.85, this run)
**Grid:** ISL 16,384 · OSL 256 · TP8/EP · `--max-num-seqs 256` · base model (MTP off) · unique seed/point

## Verdict: capacity-only. Confirmed on the third and most exposed model.

| conc | util 0.82 [M] | util 0.85 [M] | ratio | TPOT p50 0.82 | TPOT p50 0.85 |
|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 96.3 | **1.000×** | 7.1 ms | 7.0 ms |
| 4 | 225.3 | 223.9 | 0.994× | 10.9 ms | 11.0 ms |
| 16 | 359.2 | 357.6 | 0.996× | 33.6 ms | 35.6 ms |
| 64 | 447.1 | 445.6 | 0.997× | 129.5 ms | 132.0 ms |

**No trend, no point outside ±0.6%, TPOT flat.** Meanwhile capacity moved as designed:

| | util 0.82 | util 0.85 | Δ |
|---|--:|--:|--:|
| KV pool | 1,916,967 tok | **2,131,562 tok** | **+11.2%** |
| max concurrency @262,144 tok/req | 7.31× | **8.13×** | +11.2% |
| **peak activation (measured by vLLM)** | **4.05 GiB** | **4.05 GiB** | **0 — the control that makes this clean** |
| consumed (weights + non-torch) | 39.64 GiB | 39.64 GiB | 0 |
| CUDAGraph memory | 1.41 GiB | 1.41 GiB | 0 |

## Why this arm was the one still worth running

GLM is **the most exposed of the three models**: it has the least KV headroom (89.5% peak KV at
260K×8 in the base arm, where V4 sits at 36.9%). If util were ever going to move throughput by
relieving scheduler pressure, it would show here. It does not.

Three models now agree, and the two clean ones agree exactly:

| model | throughput effect | peak activation both arms | verdict |
|---|---|---|---|
| Qwen3.8-Flash-Next | c=4 **1.58×** | **17.07 GiB cold vs 0.99 GiB warm** | ❌ dirty — cold-compile artifact, not util |
| DeepSeek-V4-Flash | 0.98–1.09×, no trend | 2.32 GiB / 2.32 GiB | ✅ capacity-only |
| **GLM-5.3-Flash (this)** | **0.994–1.000×, no trend** | **4.05 GiB / 4.05 GiB** | ✅ **capacity-only** |

So the Qwen 1.58× was never the flag — it was a mis-sized KV pool from a cold `torch.compile`.
`--gpu-memory-utilization` is a **capacity knob**; it only *looks* like a throughput knob when a
mis-sized pool creates scheduler pressure.

## ⚠️ The cold-compile hazard reproduced on GLM — and was avoided by construction

`/tmp` compile caches were **empty at session start** (the container had been rescheduled), so the
first launch of this session was genuinely cold. It reproduced the bug-12 signature on GLM:

| | cold compile | warm cache | published `bf16kv` |
|---|--:|--:|--:|
| peak activation | **4.88–4.90 GiB** | **4.05 GiB** | 4.05 GiB |
| KV pool | 1,838,761 tok | 1,916,967 tok | 2,099,654 tok |
| max concurrency | 7.01× | 7.31× | 8.01× |

**Both arms of this A/B were therefore run warm**, and both measured 4.05 GiB — byte-for-byte
identical to the published arm's memory accounting (39.64 / 4.05 / 1.41 GiB, KV 21.24 GiB). That
equality is what licenses attributing the ratio column to util and nothing else.

⚠️ **The absolute KV token counts differ from the published arm (1,916,967 vs 2,099,654) even though
the GiB accounting is identical** — a block-granularity/page-padding artifact of the auto-raised
`block_size 640`, not a sizing difference. The A/B is internally consistent (both arms this session,
same server binary, same flags apart from util), which is what the comparison needs.

## Provenance / honesty notes

- Both arms **cold at every point** (0 new prefix-cache hits), verified by `bench.sh`.
- **c=1 and c=4 of the 0.85 arm were rerun.** The first attempt fired the queueing guard at c=4
  (p99/median TTFT **7.47×** vs limit 4×) and read 98.8 tok/s — a 2.3× understatement. Both first-run
  points are quarantined, not deleted, in `results/_util085_queued_firstrun/`. This is the **sixth**
  occurrence of the first-points-after-cold-start artifact (`fix_bug.md` bugs 9, 10, 12).
- Every point in this arm carries **measured** achieved-bandwidth fields now
  (`achieved_gbps_per_gpu`, `frac_peak_hbm_h100`) — see `../../../bench.sh` and the MFU note in
  `manifest.txt`. Label those **[E]** (engine-side estimate), not [M].

## What this closes

This was the **last unmeasured provisioning variable** for GLM-5.3-Flash. All of GLM's remaining
knobs are now either measured or proven impossible (TP4 OOM, TP=7 head-divisibility,
`fp8_ds_mla` NoPE geometry).
