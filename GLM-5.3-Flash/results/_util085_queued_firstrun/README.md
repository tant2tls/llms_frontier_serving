# QUARANTINED — util 0.85 arm, first run of c=1 and c=4 (scheduler queueing)

**Not results. Kept as evidence.** These two points passed the cold check and the completion check
but were the **first points measured after a cold server start**, and c=4 tripped `bench.sh`'s
queueing guard.

| point | first run (here) | rerun (published) | p99/median TTFT first run | limit |
|---|--:|--:|--:|--:|
| c=1 | 91.5 tok/s | **96.3 tok/s** | 1.14× (passed) | 4× |
| c=4 | **98.8 tok/s** | **223.9 tok/s** | **7.47× (FIRED)** | 4× |

The c=4 point under-reported by **2.3×**. Used uncritically it would have made the util A/B read as
"util 0.85 is 56% slower at c=4" — a headline-grade wrong conclusion about a flag that is in fact
capacity-only. c=1 was quarantined alongside it because it was measured in the same cold window and
came in 5% low; its rerun is within 0.1% of the util-0.82 arm.

Both were re-measured on the **same server, same flags, warm** — so this is not a configuration
difference, only a warmup artifact.

**Signature to look for:** `p99 TTFT ≫ median TTFT` at low concurrency, or a point that is
non-monotonic against its neighbours. This is the **sixth** occurrence in this repo
(`fix_bug.md` bugs 9, 10, 12).

Published arm: `../util085/` · Writeup: `../util085/RESULT-util-ab.md`
