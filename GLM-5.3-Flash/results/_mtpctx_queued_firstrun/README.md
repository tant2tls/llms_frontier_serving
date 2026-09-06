# QUARANTINED — MTP context arm, first run of ISL 16,384 (scheduler queueing)

**Not a result. Kept as evidence.** This was the **first point measured after a cold server start**
and tripped `bench.sh`'s queueing guard.

| | first run (here) | rerun (published) | base arm, same point |
|---|--:|--:|--:|
| output tok/s | **143.5** | **314.8** | 295.9 |
| p99/median TTFT | **5.13×** | 2.14× | 1.61× |
| implied MTP ratio | **0.485×** | **1.064×** | — |

The point under-reported by **2.2×**. Used uncritically it would have published *"GLM's MTP halves
throughput at short context"* — the opposite of the true result, which is that **MTP helps at 16K
(1.064×)**. It would also have wrecked the arm's central finding, since the real trend is a *decline*
from 1.064× at 16K to 0.839× at 260K; the bad point inverted the trend's direction.

Re-measured on the **same server, same flags, warm**. **Seventh** occurrence of this artifact in this
repo (`fix_bug.md` bugs 9, 10, 12).

Published arm: `../bf16kv-mtp-n1-context/` · Writeup:
`../bf16kv-mtp-n1-context/RESULT-mtp-context-axis.md`
