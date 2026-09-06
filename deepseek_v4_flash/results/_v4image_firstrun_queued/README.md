# QUARANTINED: V4 image-engine batch points corrupted by queueing (2026-09-02)

Three points from the first `mtp-off-image` batch sweep were discarded and rerun.
They passed BOTH of bench.sh's guards — cold (0 new prefix-cache hits) and
completed>0 — yet were badly wrong.

| point | FIRST run | RERUN | error | p99 TTFT (first) | p99 TTFT (rerun) |
|---|--:|--:|--:|--:|--:|
| c=4 | 98.7 | **219.0** | −55% | 13,252 ms | 2,633 ms |
| c=16 | 132.2 | **330.0** | −60% | 28,205 ms | 9,771 ms |
| c=64 | 290.3 | **389.0** | −25% | 38,973 ms | 38,846 ms |

## Why this matters more than the numbers

Using the first run, the "engine effect" (conda vLLM 0.28.0/cu129 → image dev/cu130)
computed to **0.64× — i.e. "the image engine is 36% slower"**. That would have been a
headline claim, and it was **an artifact of transient queueing on a freshly-started
server**, not an engine property.

With the reruns the engine effect is **1.01× (range 0.92–1.12×)** — essentially
neutral. The opposite conclusion.

## The diagnostic signal

**p99 TTFT ≫ median TTFT.** A 13–39 s p99 against a 2–4 s median means requests were
sitting in the scheduler queue, not being served. Compare the c=4 case: median TTFT
barely moved (1,902 → 1,864 ms) while p99 fell 5× and throughput doubled.

Root cause is most likely first-touch effects on a server that had just finished
startup (JIT/autotune warmup on the first heavy batch, page-cache cold weights).

## Lesson for the harness

`bench.sh`'s two guards — cold-run and zero-completion — **do not catch queueing.**
A point can be cold, complete, and still wrong by 60%.

**Proposed third guard:** flag any point where `p99_ttft_ms > 5 × median_ttft_ms`
as suspect and require a rerun. Every corrupted point here would have tripped it;
none of the good points do. (Not yet implemented — see WHY.md §9.)

Kept as evidence, not deleted.
