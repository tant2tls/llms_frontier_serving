> HISTORICAL ARM NOTE — numbers describe this arm, but causal claims may be superseded. Read the [current report](../../../report.md), [corrected debugging guide](../../../fix_bug.md), and [arm index](../../../experiments.md) before citing. In particular: no proven hardware bottleneck, universal engine equivalence, universal FP8 impossibility, or universal capacity-only effect is established.

# GLM-5.3-Flash — MTP on the CONTEXT axis (first measurement of this axis, any model)

**Date:** 2026-09-02 (session 3) · **Hardware:** 8×H100-80GB-HBM3, `tan-8gpus-glm53-0-0`
**Engine:** `vllm/vllm-openai:glm53-flash`, vLLM `0.1.dev20051+g487ecf187`, torch 2.13.0+cu130
**Arms:** `results/bf16kv/` (base, MTP off) vs `results/bf16kv-mtp-n1-context/` (this run)
**Grid:** concurrency 8 · OSL 256 · TP8/EP · util 0.82 · `--max-num-seqs 256` · BF16 KV
**Spec config:** `{"method":"mtp","num_speculative_tokens":1}` — matches the checkpoint's single MTP
layer and the V4/Qwen A/Bs. Only difference from the base arm is `--speculative-config`.

## Why this arm exists

MTP was measured on the **batch axis only** for all three models. GLM is the right model to test the
context axis because it is the only one with **`index_share_for_mtp_iteration: true`** — it shares the
sparse-attention indexer across MTP iterations, which DeepSeek-V4 does not and Qwen cannot (Qwen's
draft head is *full-attention* over a 36/48 linear-attention target).

**Hypothesis:** indexer sharing should make GLM's MTP degrade *less* with context than Qwen's does.

## Result — all [M], all cold (0 new prefix-cache hits)

| ISL | base tok/s | MTP n=1 tok/s | ratio | TTFT p50 base | TTFT p50 MTP | KV peak MTP |
|--:|--:|--:|--:|--:|--:|--:|
| 16,384 | 295.9 | **314.8** | **1.064×** | 2,629 ms | **2,101 ms** | 0.115 |
| 65,536 | 104.3 | 97.6 | 0.936× | 7,212 ms | **5,427 ms** | 0.406 |
| 131,072 | 47.2 | 45.3 | 0.958× | 12,891 ms | **10,734 ms** | 0.789 |
| 260,000 | 26.5 | 22.2 | **0.839×** | 38,476 ms | 41,067 ms | **0.969** |

Draft-token acceptance across the whole arm: **69.1%** (8,346 accepted / 12,081 drafted).

## Finding 1: the hypothesis is WRONG in its simple form — MTP degrades *more* with context, not less

The throughput ratio does not hold up as context grows: **1.064× at 16K → 0.839× at 260K**. Indexer
sharing does not buy context-robustness on the throughput metric. State this as a **refuted
prediction**, not a null result — it was a specific, falsifiable claim and the measurement falsified it.

## Finding 2: but TTFT tells the opposite story — MTP *improves* prefill latency by up to 25%

This is the genuinely interesting result, and it is the one that would have been missed by only
looking at throughput:

| ISL | TTFT change |
|--:|--:|
| 16,384 | **−20.1%** (2,629 → 2,101 ms) |
| 65,536 | **−24.8%** (7,212 → 5,427 ms) |
| 131,072 | **−16.7%** (12,891 → 10,734 ms) |
| 260,000 | +6.7% (38,476 → 41,067 ms) |

**MTP makes time-to-first-token substantially better at every context length up to 131K, while making
sustained throughput slightly worse.** These are not contradictory: the MTP layer changes the
prefill/decode scheduling mix. Because chunked prefill interleaves prefill chunks with decode steps,
a draft head that resolves decode steps in fewer scheduler iterations lets prefill chunks land sooner
— so first tokens arrive earlier even as aggregate output rate falls.

**For an agentic workload this inverts the recommendation.** TraceLab's measured median is ISL
132,092 / OSL 249 — an **ISL:OSL ratio of ~530:1**, so latency is dominated by prefill, not by
sustained decode. At ISL 131K, MTP costs **4.2% of throughput** and buys **16.7% of TTFT**. On a
530:1 workload that is a good trade, and it is the opposite of what the batch-axis A/B alone implies.

## Finding 3: the 260K point is KV-bound, and that is why it breaks

At ISL 260,000 the MTP arm hits **96.9% peak KV** and is the only point where both metrics degrade
together (0.839× throughput *and* +6.7% TTFT). The cause is capacity, measured directly:

| | base | MTP n=1 | Δ |
|---|--:|--:|--:|
| KV pool | 1,916,967 tok | **1,662,741 tok** | **−13.3%** |
| max concurrency @262,144 tok/req | 7.31× | **6.34×** | −13.3% |

The draft head costs **13.3% of KV capacity**. At 16K–131K there is headroom to absorb that; at 260K
there is not, and MTP's cost stops being amortizable. So **MTP's context ceiling is set by KV
capacity, not by draft accuracy** — acceptance stayed at 69.1% throughout.

⚠️ The **−13.3% KV cost here is much larger than the 2.4% reported for V4-Flash's MTP arm.** GLM's MTP
layer is 7.43 B params and its KV accounting differs; do not carry V4's 2.4% over to GLM.

## Cross-model context: GLM's MTP is the best-behaved of the three on the batch axis

| model | c=1 | c=4 | c=16 | c=64 | draft head design |
|---|--:|--:|--:|--:|---|
| **GLM-5.3-Flash** | **1.243×** | **1.064×** | 0.960× | 0.988× | MTP + **shared sparse indexer** |
| DeepSeek-V4-Flash | 1.25× | — | — | 0.99× | MTP, no indexer sharing |
| Qwen3.8-Flash-Next | 1.169× | 0.920× | 0.868× | 0.930× | **full-attention** draft over linear-attn target |

GLM is the only model that is still ≥1.0× at c=4, and its worst case (0.960×) is milder than Qwen's
(0.868×). So indexer sharing **does** appear to help on the batch axis — it just does not extend to
protecting throughput at long context, which is what this arm set out to test.

## Provenance / honesty notes

- **ISL 16,384 was rerun.** The first attempt was the first point after a cold server start and fired
  the queueing guard (p99/median TTFT **5.13×**, vs 1.61× for the same point in the base arm), reading
  143.5 tok/s — a **2.2× understatement** that would have produced a headline-grade wrong conclusion
  ("MTP halves throughput at short context"). The rerun reads 314.8 tok/s at p99/median **2.14×**. The
  first run is quarantined, not deleted, in `results/_mtpctx_queued_firstrun/`. **Seventh occurrence**
  of this artifact — see `fix_bug.md` bugs 9, 10, 12.
- Top context point is **ISL 260,000, not 262,144** — see `fix_bug.md` bug 7. Same value as the base
  arm, so the comparison is matched.
- Server ran with a **warm compile cache**, peak activation **4.05 GiB**, identical to the base arm —
  so no cold-compile KV-sizing confound.
- Achieved-bandwidth fields are **[E]** (engine-side estimate) and for GLM **exclude attention**
  (only `ffn` + `unembed` ComponentMetrics instantiated). Lower bound, not cross-model comparable.
