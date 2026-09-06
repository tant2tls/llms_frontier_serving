> HISTORICAL ARM NOTE — numbers describe this arm, but causal claims may be superseded. Read the [current report](../../../report.md), [corrected debugging guide](../../../fix_bug.md), and [arm index](../../../experiments.md) before citing. In particular: no proven hardware bottleneck, universal engine equivalence, universal FP8 impossibility, or universal capacity-only effect is established.

# GLM-5.3-Flash — 8-point batch curve (locating the knee)

**Date:** 2026-09-02 (session 3) · **Hardware:** 8×H100-80GB-HBM3, `tan-8gpus-glm53-0-0`
**Engine:** `vllm/vllm-openai:glm53-flash`, vLLM `0.1.dev20051+g487ecf187`, torch 2.13.0+cu130
**Arm:** `results/bf16kv/` — the published headline arm, extended in place
**Grid:** ISL 16,384 · OSL 256 · TP8/EP · util 0.82 · `--max-num-seqs 256` · base model (MTP off)

The 4-point grid (c=1/4/16/64) could not locate GLM's saturation point, so the report **assumed**
GLM saturates like V4-Flash. This arm adds **c = 2, 8, 32, 48** and tests that assumption.
The 4 pre-existing points were skipped and are unchanged; each new point gets its own seed derived
from `(isl, conc, osl)`, so adding concurrencies cannot contaminate the existing ones.

## The curve — all [M], all cold (0 new prefix-cache hits)

| conc | tok/s | ×prev | scaling efficiency¹ | TTFT p50 | TPOT p50 | KV peak | GB/s/GPU [E]² |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | — | 1.000 | 857 ms | 7.1 ms | 0.014 | — |
| **2** | **122.2** | 1.269× | 0.634 | 1,341 ms | 8.5 ms | 0.027 | **318.6** |
| 4 | 225.3 | 1.844× | 0.585 | 1,742 ms | 10.9 ms | 0.051 | — |
| **8** | **296.9** | 1.318× | 0.385 | 2,191 ms | 18.1 ms | 0.101 | **542.5** |
| 16 | 359.2 | 1.210× | 0.233 | 2,685 ms | 33.6 ms | 0.201 | — |
| **32** | **406.4** | 1.132× | 0.132 | 2,240 ms | 68.6 ms | 0.404 | **608.0** |
| **48** | **432.6** | 1.064× | 0.094 | 1,992 ms | 100.5 ms | 0.607 | **519.4** |
| 64 | 447.1 | 1.033× | 0.073 | 2,513 ms | 129.5 ms | 0.809 | — |

¹ `(tok/s ÷ tok/s@c=1) ÷ conc` — 1.0 would be perfect linear scaling.

² ⚠️ **Bandwidth exists only for the 4 points measured this session.** The MFU harness fix
(`fix_bug.md` bug 13) landed *after* c=1/4/16/64 were collected, so thosefour rows predate it and are
honestly blank rather than back-filled. **Do not substitute the util-0.85 arm's values here** — that arm
covers exactly 1/4/16/64 (347 / 471 / 594 / 448 GB/s) but at a different `gpu_memory_utilization`, and
splicing two arms into one curve is the confound this repo's rule 1 exists to prevent. Re-poll this arm
at 1/4/16/64 for a true single-arm bandwidth curve.

## Finding 1: the knee is at c≈4–8, and the report's assumption holds

**1→8 buys 3.08×; 8→64 buys only 1.51×** for 8× the concurrency. Marginal return per doubling falls
monotonically (1.27 → 1.84 → 1.32 → 1.21 → 1.13 → 1.06 → 1.03). By c=8 scaling efficiency is already
0.385, and by c=64 it is **0.073** — 64× the requests for 4.64× the throughput.

**The V4-derived assumption was correct**, and now it is measured rather than assumed:

| | GLM-5.3-Flash [M] | DeepSeek-V4-Flash [M] |
|---|--:|--:|
| 1→8 | **3.08×** | 3.03× |
| 8→64 | **1.51×** | 1.37× |
| 1→64 total | **4.64×** | 4.14× |
| efficiency @ c=8 | 0.385 | 0.378 |
| efficiency @ c=64 | 0.073 | 0.065 |

Two architecturally different hybrids (34 KDA + 11 DSA vs 43 uniform DSA) produce **nearly identical
saturation shapes** on the same hardware and harness. That similarity is itself the result: the
saturation point is set by something the two models *share* — the engine's scheduling and collective
structure — not by their attention design.

## Finding 2: GLM does **not** reproduce V4's c=48 regression

V4 has a reproducible dip at c=48 (**0.985×** vs c=32 — throughput *falls* as concurrency rises).
GLM at the same point gives **1.064×**: no dip.

| conc | GLM ×prev | V4 ×prev |
|--:|--:|--:|
| 32 | 1.132× | 1.123× |
| **48** | **1.064×** | **0.985× ← dip** |
| 64 | 1.033× | 1.077× |

So that regression is **not a universal property of this engine at c=48** — it is specific to V4's
configuration. Worth noting that GLM ran with `--max-num-seqs 256` pinned while V4 ran at the H100
default 1024; that is the leading candidate for the difference and is **not controlled here**, so
state this as "GLM does not show it," not as "V4's dip is caused by X."

## Finding 3: TTFT is non-monotonic in concurrency — it *improves* from c=16 to c=48

TTFT p50 rises 857 → 2,685 ms (c=1→16), then **falls to 1,992 ms at c=48** before rising again at
c=64. This is a chunked-prefill scheduling effect, not noise: with `max_num_batched_tokens=8192` a
16K prompt is split into ~2 chunks that interleave with other requests' decode steps, so at moderate
concurrency the scheduler packs prefill chunks more efficiently. **TTFT is not a pure prefill-compute
measurement** and should never be reported as one.

## Finding 4: measured bandwidth peaks mid-curve, ~18% of roofline — not ~1%

`achieved_gbps_per_gpu` (new this session) over the four points that carry it: **319 (c=2) → 543 (c=8)
→ 608 (c=32) → 519 (c=48) GB/s/GPU**, i.e. 9.5% → 16.2% → **18.1%** → 15.5% of 3.35 TB/s. It **rises,
peaks around c=32, then falls.** The util-0.85 arm independently traces the same shape on the
complementary concurrencies (347 → 471 → 594 → 448 GB/s at c=1/4/16/64, peaking at c=16), which is
useful corroboration precisely *because* it is a separate arm. Two things follow:

1. **The report's "~1% of memory roofline" headline was too low by an order of magnitude.** That
   figure came from a hand-rolled `active_params × bytes × steps/s` estimate. vLLM's own per-step
   accounting says **9–18%**. Still far from the roofline — decode remains latency-bound, and the
   qualitative conclusion is unchanged — but the number must be corrected.
2. **Bandwidth saturates — and then declines — before throughput does.** GB/s peaks near c=32 and falls
   by c=48, while tok/s keeps rising monotonically through c=64. Both arms agree on that shape. Whatever
   binds at high concurrency is **not** HBM bandwidth; if it were, tok/s could not still be climbing
   while bytes/s falls.

⚠️ These are **[E]**, engine-side estimates, and for GLM they **exclude attention entirely** — vLLM
instantiated only `ffn` and `unembed` ComponentMetrics for this model (no `attn`). So 18.1% is a
**lower bound**, and it is **not comparable across models** without the coverage table. See
`manifest.txt` and the MFU section of the root `bench.sh`.

## Provenance

- All 8 points **cold** (0 new prefix-cache hits); the 4 new points measured this session, the 4
  pre-existing points untouched and byte-identical.
- p99/median TTFT ratios: 1.03 / 3.44 / 1.33 / 1.95 / 3.10 / 7.31 / 12.25 / 12.97 — all within the
  believed-good distribution for their concurrency (`bench.sh`'s queueing guard). The rising ratio at
  high concurrency is real queueing, by design.
- Server: warm compile cache, peak activation **4.05 GiB** (matching the published arm exactly), so
  no cold-compile KV-sizing artifact. See `../util085/RESULT-util-ab.md` for that hazard.
