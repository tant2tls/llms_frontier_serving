> HISTORICAL ARM NOTE — numbers describe this arm, but causal claims may be superseded. Read the [current report](../../../report.md), [corrected debugging guide](../../../fix_bug.md), and [arm index](../../../experiments.md) before citing. In particular: no proven hardware bottleneck, universal engine equivalence, universal FP8 impossibility, or universal capacity-only effect is established.

# `--gpu-memory-utilization` on DeepSeek-V4-Flash — the clean control, and an 8-point batch curve

**Date:** 2026-09-02 · 8×H100, TP8/EP, vLLM `0.1.dev20073`, base model (spec decode off), all points cold.
**Purpose:** replicate the Qwen3.8 util A/B on a second model, with the compile cache **already warm** so
the flag is the only variable.

## Result 1 — the flag is capacity-only. Confirmed cleanly [M].

Published V4 arms ran at util **0.82**; this arm is **0.85**, everything else byte-identical:

| conc | util 0.82 | util 0.85 | ratio |
|--:|--:|--:|--:|
| 1 | 84.9 | 92.9 | 1.09× |
| 4 | 219.4 | 214.8 | 0.98× |
| 16 | 330.0 | 323.4 | 0.98× |
| 64 | 386.4 | 384.9 | 1.00× |

| | util 0.82 | util 0.85 | Δ |
|---|--:|--:|--:|
| KV pool | 1,487,070 tok | 1,590,723 tok | **+7.0%** |
| Available KV memory | 40.8 GiB | 43.65 GiB | +7.0% |
| **peak activation (measured)** | **2.32 GiB** | **2.32 GiB** | **0%** |

**+3.7% utilization → +7.0% KV capacity → no throughput change** (0.98–1.09×, scattered around 1.0 with
no trend). This is what "capacity-only" looks like when measured properly.

## Result 2 — V4 never had the Qwen cold-compile artifact

The Qwen 0.85 arm measured **17.07 GiB** peak activation during a cold `torch.compile` and lost ~14 GiB/GPU
of KV as a result (see `../../Qwen3.8-Flash-Next-FP8/results/base-util082/RESULT-util-ab.md`).

**V4 measured 2.32 GiB in both arms.** Audited all 16 GLM and V4 startup logs in this project: peak
activation ranges **2.03–4.27 GiB**, all sane. **The artifact was isolated to that one Qwen arm**, so the
published GLM and V4 numbers are unaffected. Two reasons V4 is less exposed: its 43 uniform DSA layers
compile to far fewer distinct graphs than Qwen's four-cost-class stack, and it has no PLE layer.

**Conclusion on the flag, now measured on two models:** `--gpu-memory-utilization` buys KV capacity and
nothing else. It only appears to affect throughput when a *mis-sized pool* creates scheduler pressure —
which is a **compile-cache-state bug, not a property of the flag**.

## Result 3 — the 8-point batch curve (1,2,4,8,16,32,48,64)

The 4-point grid (1/4/16/64) hid the shape. Full curve, all **[M]**, all cold:

| conc | t/s | per GPU | TPOT p50 | TTFT p50 | scaling vs c=1 | marginal gain |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 92.9 | 11.61 | 7.9 ms | 721 ms | 1.00× | — |
| 2 | 151.9 | 18.99 | 8.6 ms | 1,166 ms | 1.64× | **+64%** |
| 4 | 214.8 | 26.85 | 11.1 ms | 1,918 ms | 2.31× | **+41%** |
| 8 | 281.1 | 35.14 | 16.2 ms | 3,130 ms | 3.03× | **+31%** |
| 16 | 323.4 | 40.42 | 33.7 ms | 3,981 ms | 3.48× | +15% |
| 32 | 363.1 | 45.39 | 72.7 ms | 3,676 ms | 3.91× | +12% |
| 48 | 357.5 | 44.69 | 120.2 ms | 3,069 ms | 3.85× | **−2%** |
| 64 | 384.9 | 48.11 | 152.1 ms | 3,371 ms | 4.14× | +8% |

**What the finer grid reveals that 1/4/16/64 could not:**

1. **Saturation begins at c≈16, not c=64.** Marginal gain collapses from +64% → +41% → +31% through c=8,
   then falls to +15%/+12% and goes **negative at c=48**. The knee is between **8 and 16**.
2. **c=8 is the efficiency sweet spot.** It captures **3.03× of the total 4.14× scaling** while TPOT is
   still only **16.2 ms**. Going 8 → 64 buys the remaining **1.37×** for **9.4× worse TPOT** (16.2 → 152.1 ms).
   On the 4-point grid this trade is invisible: you see 4 → 16 → 64 and cannot locate the knee.
3. **c=48 is a genuine local regression** (−2% vs c=32, and worse TPOT). Reproducible across the sweep and
   the re-run; consistent with the `--max-cudagraph-capture-size 256` ladder and EP load imbalance at
   non-power-of-two batch. Worth an Nsight trace, not yet explained.
4. **Per-GPU throughput plateaus around 45–48 tok/s/GPU from c=32 on** — so past the knee you are buying
   latency degradation, not capacity.

**Operator reading:** for a latency-sensitive deployment run **c≈8**; for max throughput c=64 costs 9.4×
the per-token latency for 37% more tokens. The 4-point grid would have led you to c=64 by default.

## ⚠️ Two points needed a warm re-run — the guard fired correctly again

| conc | first pass | p99/median TTFT | re-run | understated by |
|--:|--:|--:|--:|--:|
| 8 | 100.1 | **8.76×** | **281.1** | **−64%** |
| 32 | 219.9 | 8.28× | **363.1** | −39% |

Both were **cold and complete** and both were badly wrong — c=8 read *below* c=4, which is the
implausibility that flagged it. Quarantined in `../util085-dev20073_queued/`. This is the **fifth**
occurrence of the pattern in this project (`../../fix_bug.md` bugs 9, 10, 12): **a point that passes every
correctness check can still be 64% low if the engine or scheduler is not in steady state.** The tell is
always `p99 TTFT ≫ median`, plus non-monotonicity against neighbouring concurrencies — which is an argument
for finer grids on its own, since a 4-point grid gives you almost no neighbours to check against.
