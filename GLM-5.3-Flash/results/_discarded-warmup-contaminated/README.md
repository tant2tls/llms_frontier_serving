# DISCARDED: warmup-contaminated batch points (2026-09-02)

These four points were measured with `bench.sh`'s old `--num-warmups 1`, which draws
the warmup request from the SAME seeded prompt set as the measured run. With
`enable_prefix_caching=True` the measured run then re-reads the warmup's own blocks.

Every point reported exactly **16,000 new prefix-cache hits** = 25 blocks x
block_size 640 = one ISL-16384 prompt. Constant across concurrency, so it is
self-contamination, not cross-point leakage.

The bias is UNEVEN and hits the low-concurrency anchors hardest:

| point | hits / prompt tokens | inflated |
|---|---|---|
| c=1  | 16,000 / 131,072   | 12.2% |
| c=4  | 16,000 / 131,072   | 12.2% |
| c=16 | 16,000 / 524,288   |  3.1% |
| c=64 | 16,000 / 2,097,152 |  0.8% |

Numbers as measured (DO NOT CITE -- kept only to document the effect):

| label | out tok/s | tot tok/s | TTFT p50 | TPOT p50 | KV peak |
|---|--:|--:|--:|--:|--:|
| c=1  |  99.0 |  6,436.4 |   867 ms |   7.1 ms | 0.012 |
| c=4  |  89.5 |  5,821.6 | 2,749 ms |  23.1 ms | 0.051 |
| c=16 | 363.2 | 23,622.6 | 1,473 ms |  38.4 ms | 0.201 |
| c=64 | 448.7 | 29,186.8 | 2,674 ms | 130.1 ms | 0.807 |

`--num-warmups` was removed from bench.sh; the rerun lives in `../bf16kv/`.
