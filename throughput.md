# Throughput, TTFT, and TPOT for the three models

Prepared 6 September 2026 from saved benchmark JSONs. Model order: **DeepSeek-V4-Flash → GLM-5.3-Flash → Qwen3.8-Flash-Next-FP8**.

The main 16K-input comparison shows Qwen has the highest overall output throughput at every tested concurrency. At concurrency 64, DeepSeek / GLM / Qwen produce **389.4 / 447.1 / 517.7 output tokens/s**. GLM has the shortest median wait for the first token at that concurrency; Qwen has the shortest median time per output token.

**Separate prefill-only and aggregate decode-only throughput are not available in these saved results.** The tables report the available input/output rates over the whole benchmark and a clearly labeled per-request generation-rate proxy. They must not be presented as isolated GPU phase measurements.

## 1. What each metric means

Think of one request as: **submit prompt → wait for first token → receive remaining tokens**. Under concurrency, multiple requests share the deployment and their work can overlap.

| Metric | Meaning and calculation | Interpretation |
|---|---|---|
| TTFT, seconds | Time to first token; JSON `median_ttft_ms / 1000` | Lower is better. Includes queueing, input processing, and other serving overhead; does not isolate prefill compute. |
| TPOT, milliseconds/token | Per-request average time per output token after the first; table uses JSON `median_tpot_ms` | Lower is better. This is the median across requests of their average output spacing, not the median of individual token gaps. |
| Input rate, tokens/s | `total_input_tokens / duration` | Input work served per second over the **entire benchmark**, including time spent generating output. It is not prefill-only throughput. |
| Output rate, tokens/s | `total_output_tokens / duration`; agrees with JSON `output_throughput` | Aggregate generated output across the eight-GPU deployment, including the effect of input processing and scheduling on elapsed time. It is not decode-only throughput. |
| Generation proxy, tokens/s/request | `1000 / median_tpot_ms` | Reciprocal of the reported median TPOT: an intuitive per-request output speed after the first token. A derived summary, not a separately measured aggregate decoding rate or an average of individual request rates. |
| Prefill-only throughput | Would require input-token work and a defined, measured prefill interval | **Unavailable for all three models here.** Input length divided by TTFT is only a latency-based proxy and is not used as a prefill measurement. |
| Aggregate decode-only throughput | Would require generated-token work and a defined, measured decoding interval | **Unavailable for all three models here.** Do not multiply the generation proxy by the configured concurrency: the live number of decoding requests changes during the run. |

TTFT and TPOT below are **medians**, not means or p99 values. Rate calculations use unrounded JSON values before display rounding. Unknown phase measurements are not zero.

## 2. Main comparison: change concurrency, keep input/output lengths fixed

Each request targets **16,384 input tokens and 256 output tokens**. The benchmark uses synthetic random text, the chat-completions endpoint, and `--ignore-eos`. There are 8, 8, 32, and 128 requests at concurrency caps 1, 4, 16, and 64 respectively. Actual input counts include tokenizer/chat-template effects.

Each model runs on **8 × H100 80GB**, with tensor parallelism 8, expert parallelism, MTP off, and prefix caching enabled. Concurrency is the client cap, not a fixed engine batch size. All rates are for the complete deployment, except the per-request generation proxy.

| Concurrency cap | Model / exact JSON | TTFT (s) ↓ | TPOT (ms/token) ↓ | Input rate (tok/s) ↑ | Output rate (tok/s) ↑ | Generation proxy (tok/s/request) ↑ |
|---:|---|---:|---:|---:|---:|---:|
| 1 | [DeepSeek](deepseek_v4_flash/results/mtp-off-image/batch_isl16k_c1.json) | 0.717 | 7.94 | 5,469.7 | 85.0 | 125.97 |
| 1 | [GLM](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c1.json) | 0.857 | 7.06 | 6,168.3 | 96.3 | 141.73 |
| 1 | [Qwen](Qwen3.8-Flash-Next-FP8/results/base-util082/batch_isl16k_c1.json) | 0.598 | 6.97 | 6,821.1 | 106.2 | 143.46 |
| 4 | [DeepSeek](deepseek_v4_flash/results/mtp-off-image/batch_isl16k_c4.json) | 1.906 | 10.72 | 14,138.8 | 219.8 | 93.27 |
| 4 | [GLM](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c4.json) | 1.742 | 10.93 | 14,429.6 | 225.3 | 91.47 |
| 4 | [Qwen](Qwen3.8-Flash-Next-FP8/results/base-util082/batch_isl16k_c4.json) | 1.496 | 9.62 | 16,490.0 | 256.8 | 103.99 |
| 16 | [DeepSeek](deepseek_v4_flash/results/mtp-off-image/batch_isl16k_c16.json) | 3.946 | 32.40 | 21,283.6 | 330.9 | 30.86 |
| 16 | [GLM](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c16.json) | 2.685 | 33.57 | 23,003.8 | 359.2 | 29.79 |
| 16 | [Qwen](Qwen3.8-Flash-Next-FP8/results/base-util082/batch_isl16k_c16.json) | 2.917 | 26.32 | 26,976.3 | 420.2 | 37.99 |
| 64 | [DeepSeek](deepseek_v4_flash/results/mtp-off-image/batch_isl16k_c64.json) | 3.643 | 148.47 | 25,045.1 | 389.4 | 6.74 |
| 64 | [GLM](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c64.json) | 2.513 | 129.52 | 28,633.2 | 447.1 | 7.72 |
| 64 | [Qwen](Qwen3.8-Flash-Next-FP8/results/base-util082/batch_isl16k_c64.json) | 2.940 | 110.08 | 33,241.8 | 517.7 | 9.08 |

**How to read this:** at concurrency 1, Qwen's median first token arrives after about 0.60 seconds, and its median TPOT corresponds to about 143.46 tokens/s/request once output starts. Its whole-run output rate is lower, 106.2 tokens/s, because that denominator also includes waiting for input processing and other overhead.

At concurrency 64, Qwen's aggregate output rate rises to 517.7 tokens/s while its generation proxy falls to 9.08 tokens/s/request. The deployment completes more total work per second, but individual requests receive output more slowly. GLM's median TTFT is shorter than Qwen's at concurrency 16 and 64, even though Qwen's TPOT and aggregate output rate are better.

The input and output rates in this fixed-length workload mostly track each other because every completed request has approximately the same input/output token ratio. They are not independent evidence of prefill and decode efficiency. The median TTFT also need not increase monotonically with the concurrency cap: request count, scheduling, and finite-run effects differ, and these are not repeated trials.

## 3. Long-context comparison: change input length, keep concurrency at 8

Each point has **16 requests**, a concurrency cap of **8**, and **256 output tokens per request**. Input length is a target. Qwen here uses the earlier `base` arm with its smaller cache pool; the corrected main batch arm does not supply this context sweep.

| Input target (tokens) | Model / exact JSON | TTFT (s) ↓ | TPOT (ms/token) ↓ | Input rate (tok/s) ↑ | Output rate (tok/s) ↑ | Generation proxy (tok/s/request) ↑ |
|---:|---|---:|---:|---:|---:|---:|
| 16,384 | [DeepSeek](deepseek_v4_flash/results/mtp-off-image/ctx_isl16384_c8.json) | 3.119 | 23.29 | 6,722.4 | 104.5 | 42.94 |
| 16,384 | [GLM](GLM-5.3-Flash/results/bf16kv/ctx_isl16384_c8.json) | 2.629 | 16.66 | 18,951.5 | 295.9 | 60.04 |
| 16,384 | [Qwen*](Qwen3.8-Flash-Next-FP8/results/base/ctx_isl16384_c8.json) | 2.330 | 15.63 | 20,719.0 | 322.7 | 63.98 |
| 65,536 | [DeepSeek](deepseek_v4_flash/results/mtp-off-image/ctx_isl65536_c8.json) | 12.311 | 125.17 | 12,266.0 | 47.9 | 7.99 |
| 65,536 | [GLM](GLM-5.3-Flash/results/bf16kv/ctx_isl65536_c8.json) | 7.212 | 47.96 | 26,698.1 | 104.3 | 20.85 |
| 65,536 | [Qwen*](Qwen3.8-Flash-Next-FP8/results/base/ctx_isl65536_c8.json) | 9.287 | 33.62 | 29,344.2 | 114.5 | 29.75 |
| 131,072 | [DeepSeek](deepseek_v4_flash/results/mtp-off-image/ctx_isl131072_c8.json) | 26.665 | 144.17 | 17,760.1 | 34.7 | 6.94 |
| 131,072 | [GLM](GLM-5.3-Flash/results/bf16kv/ctx_isl131072_c8.json) | 12.891 | 116.69 | 24,190.7 | 47.2 | 8.57 |
| 131,072 | [Qwen*](Qwen3.8-Flash-Next-FP8/results/base/ctx_isl131072_c8.json) | 19.836 | 63.48 | 28,923.0 | 56.5 | 15.75 |
| 260,000 | [DeepSeek](deepseek_v4_flash/results/mtp-off-image/ctx_isl260000_c8.json) | 61.822 | 242.28 | 16,985.9 | 16.7 | 4.13 |
| 260,000 | [GLM](GLM-5.3-Flash/results/bf16kv/ctx_isl260000_c8.json) | 38.476 | 147.65 | 26,886.2 | 26.5 | 6.77 |
| 260,000 | [Qwen*](Qwen3.8-Flash-Next-FP8/results/base/ctx_isl260000_c8.json) | 21.338 | 231.33 | 25,495.6 | 25.1 | 4.32 |

*Qwen context measurements retain the startup/cache-pool confound described in [report.md](report.md). DeepSeek's 16K context result remains inconsistent with its separate finer concurrency grid and needs a matched rerun; that grid is not spliced into these tables.*

At 131K input, GLM returns the first token sooner than Qwen (12.89 versus 19.84 seconds), while Qwen has better TPOT and aggregate output rate. At 260K, Qwen has shorter median TTFT, while GLM has better TPOT and overall output rate. Choosing by first-token responsiveness can therefore give a different answer from choosing by output rate.

As prompts get longer, output throughput falls partly because the benchmark does more input work for the same 256 output tokens. The falling output rate alone cannot establish how much slower the decoding phase becomes. TPOT also includes scheduling effects while other requests may be processing input.

## 4. Worked calculation from one saved run

For [Qwen, main grid, concurrency 64](Qwen3.8-Flash-Next-FP8/results/base-util082/batch_isl16k_c64.json), 128 requests completed with zero recorded failures:

```text
Benchmark duration = 63.28968291543424 seconds
Total input tokens = 2,103,865
Total output tokens = 32,768 = 128 requests × 256 tokens

Input rate = 2,103,865 / 63.28968291543424 = 33,241.8 tokens/s
Output rate = 32,768 / 63.28968291543424 = 517.7 tokens/s
Median TTFT = 2939.918457530439 / 1000 = 2.940 seconds
Median TPOT = 110.08028010746428 milliseconds/token
Generation proxy = 1000 / 110.08028010746428 = 9.08 tokens/s/request
```

The JSON's `total_token_throughput` counts **input + output** tokens over the same duration. For this run it is 33,759.6 tokens/s; that number must not be labeled decoding throughput.

## 5. Exact arms, dates, and limits

| Model | Selected batch arm | Selected context arm | Runtime / qualification |
|---|---|---|---|
| DeepSeek | [mtp-off-image](deepseek_v4_flash/results/mtp-off-image/) | Same arm | [Manifest](deepseek_v4_flash/results/mtp-off-image/manifest.txt): dev20051; unresolved context discrepancy. |
| GLM | [bf16kv](GLM-5.3-Flash/results/bf16kv/) | Same arm | [Manifest](GLM-5.3-Flash/results/bf16kv/manifest.txt): dev20051, BF16 KV; directory spans sessions. |
| Qwen | [base-util082](Qwen3.8-Flash-Next-FP8/results/base-util082/) | [base](Qwen3.8-Flash-Next-FP8/results/base/) | dev20073; [batch manifest](Qwen3.8-Flash-Next-FP8/results/base-util082/manifest.txt) and [context manifest](Qwen3.8-Flash-Next-FP8/results/base/manifest.txt) describe different startup/pool conditions. |

Per-point provenance is the exact linked JSON and its `date` field, not the folder manifest timestamp. All 24 listed points have date prefix `20260902`; the saved time suffixes below are transcribed without assigning a timezone that the JSON does not encode.

| Model | Batch times at c1 / c4 / c16 / c64 | Context times at 16K / 65K / 131K / 260K |
|---|---|---|
| DeepSeek | 060808 / 064030 / 064127 / 064440 | 061418 / 061620 / 061904 / 062416 |
| GLM | 015503 / 015541 / 015636 / 015829 | 015913 / 020028 / 020235 / 025206 |
| Qwen | 091756 / 091835 / 091925 / 092112 | 074558 / 074710 / 074908 / 075258 |

These compare deployed configurations with different model/tokenizer/precision choices and some different runtime builds. DeepSeek's separate build bridge supports only its batch-grid comparison; it does not make all models engine-matched. No equal-quality comparison, real traffic replay, repeat-based confidence intervals, or operator trace establishes a dominant bottleneck. Cache validation gaps remain pending in [fix_bug.md §6](fix_bug.md); saved cold-run flags do not independently prove every point was cold.

To measure prefill-only and decode-only rates, a future authorized experiment would need phase-resolved timing and token accounting with an explicit treatment of overlap, queueing, and cached input. No such experiment was run to create this document. Current interpretations and arm selection remain in [report.md](report.md) and [experiments.md](experiments.md).
