# Speculative decoding results for DeepSeek, GLM, and Qwen

Prepared 6 September 2026 from the selected saved benchmark JSONs.

**MTP increases recorded output throughput at concurrency 1 for all three models, but none of the selected n=1 pairs improves throughput at concurrency 64.** GLM provides the strongest controlled comparison. DeepSeek and Qwen results carry additional limitations, so the speedup percentages are observations from their own pairs, not a controlled ranking of speculative-decoding implementations.

| Model / MTP setting | Throughput change at c1 | Throughput change at c64 | Evidence status |
|---|---:|---:|---|
| DeepSeek n=1 | +24.9% | -0.9% | Historical runtime; missing newer cold-run telemetry |
| GLM n=1 | +24.3% | -1.2% | Strongest controls in this workspace |
| Qwen n=1 | +16.9% | -7.0% | Startup/cache-pool confounded |
| GLM n=5, additional test | +25.4% | -8.7% | Longer-draft comparison against GLM base |

## 1. Workload and metric definitions

The batch comparisons target **16,384 synthetic random input tokens and 256 output tokens per request**. Client concurrency caps are 1, 4, 16, and 64, with 8, 8, 32, and 128 requests respectively. Each deployment uses eight H100 80GB GPUs, TP8, and expert parallelism. These are finite synthetic workloads, not real conversation replay. Actual input-token counts vary with tokenization and chat templates.

MTP means multi-token prediction: the model drafts candidate tokens and verifies them before committing output. Here, n=1 or n=5 is the configured number of speculative draft tokens per round. It does not mean every draft is accepted.

| Metric | Definition |
|---|---|
| Output throughput, tok/s | JSON `output_throughput` = total output tokens / whole benchmark duration, across the eight-GPU deployment. Includes the effect of input work and scheduling; not isolated decode-only throughput. |
| Throughput change | `100 * (MTP throughput / off throughput - 1)`. Positive means more aggregate output per second. |
| TTFT, seconds | JSON `median_ttft_ms / 1000`: median wait for the first token, including queueing and input processing. Lower is better. |
| TPOT, ms/token | JSON `median_tpot_ms`: median across requests of their average time per output token after the first. Lower is better. |
| Draft acceptance, % | Saved `spec_decode_acceptance_rate`; accepted draft tokens / draft tokens, expressed as a percentage. A benchmark telemetry result, not an answer-quality score. |

TTFT and TPOT cells show **off / MTP**, in that order. Throughput cells link to the exact raw JSONs. Changes are calculated before rounding. Concurrency is a client cap, not a fixed GPU batch size. Separate prefill-only and aggregate decode-only rates remain unavailable; see [throughput.md](throughput.md).

## 2. Correct baseline for each model

| Model | MTP off | MTP on | Required qualification |
|---|---|---|---|
| DeepSeek | [mtp-off](deepseek_v4_flash/results/mtp-off/) | [mtp-on-noreuse](deepseek_v4_flash/results/mtp-on-noreuse/) | Historical pair on the older runtime. Do not substitute `mtp-off-image` from the main throughput comparison. The folder name does not independently prove zero cache reuse. |
| GLM | [bf16kv](GLM-5.3-Flash/results/bf16kv/) | [bf16kv-mtp-n1](GLM-5.3-Flash/results/bf16kv-mtp-n1/), [bf16kv-mtp-n5](GLM-5.3-Flash/results/bf16kv-mtp-n5/) | Same dev20051 runtime family, BF16 KV, utilization 0.82; strongest controls, but no randomized repeats. |
| Qwen | [base](Qwen3.8-Flash-Next-FP8/results/base/) | [mtp-n1](Qwen3.8-Flash-Next-FP8/results/mtp-n1/) | Both use utilization 0.85, but startup and cache pools differ. Do not substitute corrected `base-util082`, which also changes utilization. |

The off numbers for DeepSeek and Qwen therefore differ from the main grid in [throughput.md](throughput.md). That is intentional: the speculative comparison must retain its own control arm.

## 3. DeepSeek: historical MTP n=1 pair

| Concurrency cap | Off (tok/s) | MTP (tok/s) | Throughput change | TTFT off / MTP (s) | TPOT off / MTP (ms/token) | MTP draft acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | [92.3](deepseek_v4_flash/results/mtp-off/batch_c1.json) | [115.3](deepseek_v4_flash/results/mtp-on-noreuse/batch_c1.json) | +24.9% | 0.714 / 0.738 | 7.99 / 5.11 | 44.9% |
| 4 | [222.0](deepseek_v4_flash/results/mtp-off/batch_c4.json) | [251.8](deepseek_v4_flash/results/mtp-on-noreuse/batch_c4.json) | +13.4% | 1.914 / 1.630 | 10.51 / 9.10 | 70.3% |
| 16 | [293.7](deepseek_v4_flash/results/mtp-off/batch_c16.json) | [339.9](deepseek_v4_flash/results/mtp-on-noreuse/batch_c16.json) | +15.8% | 3.994 / 3.543 | 43.13 / 32.58 | 73.7% |
| 64 | [383.4](deepseek_v4_flash/results/mtp-off/batch_c64.json) | [380.1](deepseek_v4_flash/results/mtp-on-noreuse/batch_c64.json) | -0.9% | 3.687 / 5.375 | 151.42 / 137.35 | 62.1% |

At c1, output throughput rises 24.9% and TPOT falls from 7.99 to 5.11 ms/token, while TTFT increases slightly. At c4 and c16, both aggregate output rate and median latencies improve in this historical pair. At c64, throughput falls 0.9% and TTFT rises from 3.69 to 5.38 seconds even though TPOT improves.

These runs lack the newer cold-run/cache-hit fields. The [off manifest](deepseek_v4_flash/results/mtp-off/manifest.txt) records vLLM 0.28.0, and the historical [MTP launch script](deepseek_v4_flash/run.sh) specifies one draft token. This is weaker evidence than GLM's pair. The exact c64 MTP JSON rounds to **380.1 tok/s**; the existing report's 380.2 display differs slightly, so this document uses the raw JSON.

## 4. GLM: MTP n=1

| Concurrency cap | Off (tok/s) | MTP (tok/s) | Throughput change | TTFT off / MTP (s) | TPOT off / MTP (ms/token) | MTP draft acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | [96.3](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c1.json) | [119.7](GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c1.json) | +24.3% | 0.857 / 0.721 | 7.06 / 5.05 | 72.0% |
| 4 | [225.3](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c4.json) | [239.8](GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c4.json) | +6.4% | 1.742 / 1.274 | 10.93 / 11.82 | 73.7% |
| 16 | [359.2](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c16.json) | [344.7](GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c16.json) | -4.0% | 2.685 / 2.829 | 33.57 / 32.97 | 71.7% |
| 64 | [447.1](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c64.json) | [441.9](GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c64.json) | -1.2% | 2.513 / 3.022 | 129.52 / 123.80 | 71.7% |

At c1, MTP improves output throughput 24.3% and reduces both TTFT and TPOT. At c4, throughput improves 6.4% and TTFT falls, but TPOT increases from 10.93 to 11.82 ms/token. At c16 and c64, output throughput is lower with MTP.

Acceptance stays near 72% even at c64, where there is no throughput gain. Acceptance alone therefore does not predict speedup. Drafting, verification, state handling, and scheduling have costs, but the saved measurements do not isolate which cost dominates.

## 5. Qwen: MTP n=1, qualified comparison

| Concurrency cap | Off (tok/s) | MTP (tok/s) | Throughput change | TTFT off / MTP (s) | TPOT off / MTP (ms/token) | MTP draft acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | [107.7](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c1.json) | [125.9](Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c1.json) | +16.9% | 0.602 / 0.658 | 6.96 / 5.56 | 64.5% |
| 4 | [162.2](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c4.json) | [149.3](Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c4.json) | -8.0% | 2.250 / 2.306 | 9.84 / 16.52 | 63.8% |
| 16 | [371.0](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c16.json) | [322.2](Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c16.json) | -13.2% | 3.168 / 1.954 | 32.16 / 37.01 | 58.0% |
| 64 | [517.8](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c64.json) | [481.5](Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c64.json) | -7.0% | 2.942 / 3.888 | 110.06 / 109.77 | 56.8% |

At c1, output throughput increases 16.9% and TPOT improves, while TTFT increases slightly. At c4, c16, and c64, throughput is lower with MTP. At c16, TTFT improves markedly even though output throughput falls 13.2% and TPOT worsens.

These are observations from the original base/MTP pair. Startup/cache-pool differences prevent attributing the full change to MTP. They do not establish how MTP would perform against the corrected main batch baseline.

## 6. GLM: longer drafts, MTP n=5

| Concurrency cap | Off (tok/s) | MTP (tok/s) | Throughput change | TTFT off / MTP (s) | TPOT off / MTP (ms/token) | MTP draft acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | [96.3](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c1.json) | [120.8](GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c1.json) | +25.4% | 0.857 / 0.734 | 7.06 / 4.86 | 30.1% |
| 4 | [225.3](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c4.json) | [230.7](GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c4.json) | +2.4% | 1.742 / 1.676 | 10.93 / 11.66 | 33.9% |
| 16 | [359.2](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c16.json) | [342.1](GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c16.json) | -4.7% | 2.685 / 2.119 | 33.57 / 33.80 | 31.8% |
| 64 | [447.1](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c64.json) | [408.3](GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c64.json) | -8.7% | 2.513 / 10.724 | 129.52 / 101.15 | 30.1% |

At c1, n=5 produces 120.8 tok/s versus 119.7 tok/s for n=1: only about 0.9% more in these single runs. The extra draft length has no demonstrated repeat-based advantage.

At c64, n=5 loses 8.7% output throughput versus off, and TTFT rises from 2.51 to 10.72 seconds. Its sampled cache-pool occupancy reaches **99.2%**, versus 92.8% for n=1. This is occupancy of the engine pool, not total GPU-memory utilization, and it does not by itself prove the cause of the slowdown. TPOT improves despite worse TTFT and overall output rate, illustrating why one latency metric is insufficient.

The n=5 acceptance percentage uses all five candidate positions in its denominator; it is not directly interchangeable with n=1 acceptance as a measure of progress per verification round.

## 7. GLM: long-context MTP n=1

Input length varies below; concurrency remains 8.

| Input target (tokens) | Off (tok/s) | MTP (tok/s) | Throughput change | TTFT off / MTP (s) | TPOT off / MTP (ms/token) | MTP draft acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | [295.9](GLM-5.3-Flash/results/bf16kv/ctx_isl16384_c8.json) | [314.8](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl16384_c8.json) | +6.4% | 2.629 / 2.101 | 16.66 / 17.22 | 72.2% |
| 65,536 | [104.3](GLM-5.3-Flash/results/bf16kv/ctx_isl65536_c8.json) | [97.6](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl65536_c8.json) | -6.4% | 7.212 / 5.427 | 47.96 / 57.17 | 66.9% |
| 131,072 | [47.2](GLM-5.3-Flash/results/bf16kv/ctx_isl131072_c8.json) | [45.3](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl131072_c8.json) | -4.2% | 12.891 / 10.734 | 116.69 / 122.95 | 63.9% |
| 260,000 | [26.5](GLM-5.3-Flash/results/bf16kv/ctx_isl260000_c8.json) | [22.2](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl260000_c8.json) | -16.1% | 38.476 / 41.067 | 147.65 / 151.34 | 72.8% |

This sweep uses **16 requests, concurrency cap 8, and 256 output tokens** at every input length. Its paired MTP arm is [bf16kv-mtp-n1-context](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/).

At 131K input, median TTFT improves from 12.89 to 10.73 seconds (16.7% shorter), while throughput falls 4.2% and TPOT worsens. At 260K, throughput falls 16.1%, both median latencies worsen, and sampled cache-pool occupancy reaches **96.9%**.

An earlier first token does not establish faster prefill computation: TTFT also reflects queueing and scheduling. No timeline isolates the mechanism. The selected synthesis has no comparable controlled MTP context pair for DeepSeek or Qwen; historical files are not substituted to fill those gaps.

## 8. What these results support

For the synthetic 16K workload, the recorded benefit of one-token MTP is largest at low concurrency. GLM's controlled results support considering n=1 for light load; its loaded results give no throughput reason to assume MTP should always be enabled. DeepSeek and Qwen observations require the qualifications above.

MTP can improve TPOT while worsening TTFT or overall throughput. Read all three metrics together. Under speculative decoding, one response event can contain multiple tokens, so individual inter-token-gap statistics are also not interchangeable with per-request TPOT.

No real-text acceptance evaluation, answer-quality comparison, repeat-based confidence intervals, or phase trace was produced here. The measured acceptance percentages do not establish quality parity. Small throughput differences, especially around 1%, have no established statistical significance. Missing telemetry is unknown, not zero; remaining harness validation work is documented in [fix_bug.md](fix_bug.md).

A future stronger comparison would match startup/cache pools, repair the documented measurement gates, repeat and randomize matched workloads, and record draft/verification/state costs with representative text and quality checks. These are pending experiments, not completed work.

## 9. Per-point provenance

Each throughput cell links to its exact JSON. The ledger below preserves each JSON's saved `date` value; no timezone is inferred from that field. Folder manifests can span sessions and must not replace per-point dates.

| Exact result JSON | Saved date |
|---|---|

| [deepseek_v4_flash/results/mtp-off/batch_c1.json](deepseek_v4_flash/results/mtp-off/batch_c1.json) | 20260901-163828 |
| [deepseek_v4_flash/results/mtp-on-noreuse/batch_c1.json](deepseek_v4_flash/results/mtp-on-noreuse/batch_c1.json) | 20260901-161819 |
| [deepseek_v4_flash/results/mtp-off/batch_c4.json](deepseek_v4_flash/results/mtp-off/batch_c4.json) | 20260901-163911 |
| [deepseek_v4_flash/results/mtp-on-noreuse/batch_c4.json](deepseek_v4_flash/results/mtp-on-noreuse/batch_c4.json) | 20260901-161851 |
| [deepseek_v4_flash/results/mtp-off/batch_c16.json](deepseek_v4_flash/results/mtp-off/batch_c16.json) | 20260901-164016 |
| [deepseek_v4_flash/results/mtp-on-noreuse/batch_c16.json](deepseek_v4_flash/results/mtp-on-noreuse/batch_c16.json) | 20260901-161949 |
| [deepseek_v4_flash/results/mtp-off/batch_c64.json](deepseek_v4_flash/results/mtp-off/batch_c64.json) | 20260901-164237 |
| [deepseek_v4_flash/results/mtp-on-noreuse/batch_c64.json](deepseek_v4_flash/results/mtp-on-noreuse/batch_c64.json) | 20260901-162156 |
| [GLM-5.3-Flash/results/bf16kv/batch_isl16k_c1.json](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c1.json) | 20260902-015503 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c1.json](GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c1.json) | 20260902-051951 |
| [GLM-5.3-Flash/results/bf16kv/batch_isl16k_c4.json](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c4.json) | 20260902-015541 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c4.json](GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c4.json) | 20260902-052557 |
| [GLM-5.3-Flash/results/bf16kv/batch_isl16k_c16.json](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c16.json) | 20260902-015636 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c16.json](GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c16.json) | 20260902-052137 |
| [GLM-5.3-Flash/results/bf16kv/batch_isl16k_c64.json](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c64.json) | 20260902-015829 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c64.json](GLM-5.3-Flash/results/bf16kv-mtp-n1/batch_isl16k_c64.json) | 20260902-052331 |
| [Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c1.json](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c1.json) | 20260902-075950 |
| [Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c1.json](Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c1.json) | 20260902-081646 |
| [Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c4.json](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c4.json) | 20260902-074236 |
| [Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c4.json](Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c4.json) | 20260902-081038 |
| [Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c16.json](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c16.json) | 20260902-074330 |
| [Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c16.json](Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c16.json) | 20260902-081134 |
| [Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c64.json](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c64.json) | 20260902-074516 |
| [Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c64.json](Qwen3.8-Flash-Next-FP8/results/mtp-n1/batch_isl16k_c64.json) | 20260902-081326 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c1.json](GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c1.json) | 20260902-053552 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c4.json](GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c4.json) | 20260902-053630 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c16.json](GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c16.json) | 20260902-053726 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c64.json](GLM-5.3-Flash/results/bf16kv-mtp-n5/batch_isl16k_c64.json) | 20260902-053926 |
| [GLM-5.3-Flash/results/bf16kv/ctx_isl16384_c8.json](GLM-5.3-Flash/results/bf16kv/ctx_isl16384_c8.json) | 20260902-015913 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl16384_c8.json](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl16384_c8.json) | 20260902-113155 |
| [GLM-5.3-Flash/results/bf16kv/ctx_isl65536_c8.json](GLM-5.3-Flash/results/bf16kv/ctx_isl65536_c8.json) | 20260902-020028 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl65536_c8.json](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl65536_c8.json) | 20260902-112139 |
| [GLM-5.3-Flash/results/bf16kv/ctx_isl131072_c8.json](GLM-5.3-Flash/results/bf16kv/ctx_isl131072_c8.json) | 20260902-020235 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl131072_c8.json](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl131072_c8.json) | 20260902-112351 |
| [GLM-5.3-Flash/results/bf16kv/ctx_isl260000_c8.json](GLM-5.3-Flash/results/bf16kv/ctx_isl260000_c8.json) | 20260902-025206 |
| [GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl260000_c8.json](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/ctx_isl260000_c8.json) | 20260902-112751 |

Selection and interpretation follow [experiments.md](experiments.md), [tools/experiment_arms.json](tools/experiment_arms.json), and [report.md](report.md). No raw JSON, logs, launch scripts, or GPU workloads were changed to prepare this document.
