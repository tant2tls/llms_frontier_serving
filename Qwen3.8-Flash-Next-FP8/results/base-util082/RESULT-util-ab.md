> HISTORICAL ARM NOTE — numbers describe this arm, but causal claims may be superseded. Read the [current report](../../../report.md), [corrected debugging guide](../../../fix_bug.md), and [arm index](../../../experiments.md) before citing. In particular: no proven hardware bottleneck, universal engine equivalence, universal FP8 impossibility, or universal capacity-only effect is established.

# `--gpu-memory-utilization` A/B — and the confound it exposed

**Date:** 2026-09-02 · Qwen3.8-Flash-Next-FP8, 8×H100, TP8/EP, base model, all points cold.
The **only** flag changed between the two arms is `--gpu-memory-utilization` (0.85 → 0.82).

## The headline result is that the flag is NOT what moved the numbers

| conc | util 0.85 (`../base/`) | util 0.82 (this arm) | ratio | TPOT 0.85 | TPOT 0.82 |
|--:|--:|--:|--:|--:|--:|
| 1 | 107.7 | 106.2 | 0.99× | 7.0 ms | 7.0 ms |
| 4 | 162.2 | **256.8** | **1.58×** | 9.8 ms | 9.6 ms |
| 16 | 371.0 | **420.2** | **1.13×** | 32.2 ms | 26.3 ms |
| 64 | 517.8 | 517.7 | 1.00× | 110.1 ms | 110.1 ms |

**Lowering utilization produced MORE KV, not less** — the opposite of the naive reading:

| | util 0.85 | util 0.82 |
|---|--:|--:|
| KV pool | 2,048,645 tok | **3,197,331 tok** (1.56×) |
| concurrency @256K | 7.81× | **12.20×** |
| Available KV memory | 25.1 GiB | **39.18 GiB** |

## Real cause: memory profiling ran during cold torch.compile

vLLM sizes the KV pool as `util × total − weights − peak_activation − cudagraph`. It measures
`peak_activation` **at startup**, and in the first run that measurement happened while torch.compile
was still working:

| | util 0.85 run | util 0.82 run |
|---|--:|--:|
| weights + non-torch | 25.13 GiB | 24.75 GiB |
| **peak activation** | **17.07 GiB** | **0.99 GiB** |
| CUDAGraph | 1.75 GiB | 1.75 GiB |
| compilation time | **60.41 s (cold cache)** | **1.03 s (warm cache)** |
| engine init | 132.05 s | 46.41 s |

**A 17× difference in measured peak activation on identical weights and identical
`max_num_batched_tokens=8192`.** The transient compile-time allocation was attributed to activation, and
vLLM then reserved 17 GiB it did not need — **stealing ~14 GiB/GPU from the KV pool**. vLLM's own log
says as much: *"Replace gpu_memory_utilization config with `--kv-cache-memory=35225873408` (32.81 GiB)
to fully utilize gpu memory"* — i.e. it knew it was leaving 32.81 GiB unused.

## What this means for the flag itself

**Within this range, `--gpu-memory-utilization` affects KV capacity only, not decode cost — confirmed
[M]:** at c=1 and c=64 the two arms are identical (0.99×, 1.00×) with identical TPOT. The prediction in
the reports was right about the *mechanism*; it was the **cold-compile confound**, not the flag, that
moved c=4 and c=16.

Why the middle concurrencies moved: at c=4 and c=16 the run is TTFT-sensitive and a larger KV pool means
less scheduler pressure and fewer preemptions during chunked prefill. At c=1 there is nothing to schedule
and at c=64 the machine is saturated either way, so both ends are insensitive — which is exactly why the
effect looked like an architecture difference at c=4 and disappeared at the extremes.

## Consequences for the report

1. **The published Qwen batch numbers are superseded by this arm** for c=4 and c=16.
2. **The "Qwen loses at c=4 (0.72×)" finding was an artifact, not architecture.** Corrected, Qwen wins at
   **every** concurrency: 1.10× / 1.14× / 1.17× / 1.16× vs GLM.
3. **Always warm the compile cache before the arm that sets KV capacity.** Or pin `--kv-cache-memory`
   explicitly, which removes the profiling step from the critical path entirely.
4. This is a **fourth** instance of the project's recurring failure mode: *a cold engine producing
   numbers that pass every correctness check.* See `../../fix_bug.md` bugs 9, 10, 12.
