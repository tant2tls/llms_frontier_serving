# Serving three frontier MoE models on 8 H100 GPUs

Tan Ngo · For Professor Kan Zhu and UW SyFI · Final synthesis: 6 September 2026

Assignment: [task.md](task.md). Presentation: [slides.md](slides.md). Architecture-to-debugging study guide: [fix_bug.md](fix_bug.md).

## Executive findings

This study measures **GLM-5.3-Flash, DeepSeek-V4-Flash, and Qwen3.8-Flash-Next-FP8** on an eight-GPU H100 configuration. Kimi is covered through architecture analysis, without a local performance measurement. The central systems lesson is that serving behavior depends on workload, cache state, backend, and scheduling as well as architecture.

1. **Qwen leads the corrected 16K-input throughput grid.** At concurrency 64: Qwen 517.7 output tokens/s, GLM 447.1, DeepSeek 389.4. These are deployed configurations, not isolated architectural effects or equal-quality answers.
2. **Concurrency trades latency for throughput.** From concurrency 1 to 64, throughput grows about 4.6–4.9× while median time per output token grows about 16–19×. Finer GLM/DeepSeek sweeps show diminishing returns around concurrency 8–16.
3. **Long inputs change the tradeoff.** At 131K input, GLM has lower median first-token latency than Qwen despite lower throughput. Qwen's long-context arm has a known cache-sizing confound, so its capacity findings are provisional.
4. **Speculative decoding is workload-dependent.** GLM MTP with one draft token improves throughput 24% at concurrency 1, but provides no observed throughput gain at 64. At 131K input it reduces median TTFT by 17% while reducing throughput by 4%.
5. **Memory savings need not increase speed.** GLM's alternate FP8-KV stack increases reported capacity; switching BF16 to FP8 KV on that stack loses 25% throughput at concurrency 64.
6. **Exact kernel bottlenecks remain unproven.** Request metrics establish tradeoffs. Incomplete engine estimates cannot prove hardware bandwidth saturation or communication dominance.

## 1. Scope, evidence, and method

### Evidence convention

- **[M] Measured:** raw request metrics or recorded server observations.
- **[A] Analytical:** calculations, configuration facts, or prior tensor accounting.
- **[E] Engine estimate:** modeled FLOPs/bytes combined with observed scheduling activity.
- **[H] Hypothesis:** an explanation requiring another experiment.

Result tables are [M]; derived ratios and costs are [A] from [M] inputs. Historical reports are supporting notes, not independent experiments. [report_previous.md](archive/notes/report_previous.md) preserves the superseded synthesis; the present report corrects several of its causal claims and comparisons.

### Setup and metrics

Each deployment uses eight NVIDIA H100 80GB HBM3 GPUs, TP8, and expert parallelism. Main image runs record driver 575.57.08 and PyTorch 2.13.0+cu130. GLM and the main DeepSeek arm use vLLM `0.1.dev20051+g487ecf187`; Qwen uses `0.1.dev20073+g8e685d198`. Host names differ between allocations: this establishes the same hardware class/count, not necessarily the same physical node. No measured interconnect topology, power, or energy results are supplied.

[bench.sh](bench.sh) uses the chat-completions endpoint. Main comparisons are text-only with MTP off and prefix caching enabled. Batch/context points use synthetic random prompts, unique seeds by workload shape, and `--ignore-eos` with 256 output tokens.

| Axis | Input / output tokens | Client concurrency cap | Requests per point |
|---|---|---|---|
| Batch | 16,384 / 256 | 1, 4, 16, 64; extra points for GLM/DeepSeek | 8, 8, 32, 128 on main grid |
| Context | 16,384; 65,536; 131,072; 260,000 / 256 | 8 | 16 |
| Prefix | 65,536 shared prefix + 2,048 suffix / 256 | 8 | 64; 1, 4, or 16 distinct prefixes |

Lengths are targets: chat templates and tokenization affect actual counts. Use JSON `total_input_tokens` for exact accounting. The 260,000 target leaves headroom below the configured 262,144-token limit.

**Concurrency is not instantaneous engine batch size.** Continuous batching mixes prefill tokens, decoding requests, and speculative verification. An infinite request-rate setting with a concurrency cap produces a finite workload, not a production arrival trace or steady-state capacity test.

- Output throughput = `total_output_tokens / duration`: includes input processing and scheduling, not just decoding.
- TTFT = time to first token, including queueing and chunked prefill; not isolated prefill kernel time.
- TPOT = per-request average time between output tokens, summarized across requests; distinct from individual inter-token-latency percentiles.
- Cache peak = sampled engine pool occupancy. JSON `peak_kv_cache_usage_perc` stores fractions: `0.725` means 72.5%. It is not total HBM utilization.

Most cells have one retained run, small request counts, and no repeat-based confidence intervals. A within-run p99 does not quantify uncertainty across runs. Small differences are observations, not statistically established improvements.

### Arm provenance

| Use | Result directory | Qualification |
|---|---|---|
| GLM main | [bf16kv](GLM-5.3-Flash/results/bf16kv/) | BF16 KV; utilization 0.82 |
| DeepSeek main | [mtp-off-image](deepseek_v4_flash/results/mtp-off-image/) | FP8 KV; utilization 0.82; dev20051 |
| Qwen corrected batch | [base-util082](Qwen3.8-Flash-Next-FP8/results/base-util082/) | BF16 KV; utilization 0.82; warm compile cache |
| Qwen context/prefix | [base](Qwen3.8-Flash-Next-FP8/results/base/) | Utilization 0.85; smaller pool from cold startup profiling |
| GLM MTP batch | [n1](GLM-5.3-Flash/results/bf16kv-mtp-n1/) / [n5](GLM-5.3-Flash/results/bf16kv-mtp-n5/) | Compare with GLM bf16kv |
| GLM MTP context | [n1-context](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/) | Same context grid as GLM base |
| DeepSeek finer grid | [util085-dev20073](deepseek_v4_flash/results/util085-dev20073/) | Separate utilization/build arm |
| Engine bridge | [mtp-off-bridge-dev20073](deepseek_v4_flash/results/mtp-off-bridge-dev20073/) | DeepSeek batch only |

The DeepSeek bridge reports dev20073/dev20051 throughput ratios **0.998, 0.998, 0.997, 0.992** across the main batch grid. That limits the observed build effect for DeepSeek on this grid. It does **not** make all models engine-matched or isolate architecture, and it does not validate context/prefix comparisons across builds.

## 2. Architecture and performance implications

All three measured checkpoints are **mixture-of-experts (MoE)** models. Routing activates a subset of experts per token, reducing arithmetic compared with activating all weights while retaining a large resident weight footprint and adding dispatch/combine communication.

| Model | Attention organization [A] | Routed experts / selected [A] | Serving implication |
|---|---|---|---|
| GLM-5.3-Flash | 34 Kimi Delta Attention (KDA) + 11 sparse-attention layers | 288 / 8 | Recurrent state plus sequence-growing attention cache |
| DeepSeek-V4-Flash | 43 sparse-attention layers; compressed history | 256 / 6 | Smaller history representation, plus compression/indexing work |
| Qwen3.8-Flash-Next-FP8 | 36 Gated DeltaNet (GDN) + 12 Qwen sparse-attention (QSA) layers | 512 / 10 | Recurrent state, retained attention KV, and an n-gram lookup table |
| Kimi K3 — not measured | 69 KDA + 24 gated MLA layers | 896 / 16 | Hybrid state at a much larger weight footprint |

Sources: local [GLM analysis](GLM-5.3-Flash/report.md), [DeepSeek analysis](deepseek_v4_flash/report.md), [Qwen analysis](Qwen3.8-Flash-Next-FP8/report.md), and [Kimi review](final_presentation/Kimi_series.md). Official configuration cross-checks: [GLM](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json), [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json), [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json), [Kimi](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json). Online sources checked 5 September 2026; rolling sources do not replace recorded manifests.

**The similarity:** all use sparse expert computation. GLM, Qwen, and Kimi also mix recurrent and history-retaining attention inside one stack. **The difference:** recurrent layers store fixed-size state per sequence; attention retains sequence-growing history. Sparse attention can limit the history consulted without eliminating its storage. Compression reduces history size but adds work. These costs need separate terms.

### Connect architecture to debugging and results

| Feature | Runtime consequence | Evidence to study | Limit of the inference |
|---|---|---|---|
| Recurrent layers | State capacity also scales with active sequences | [Bug 1](fix_bug.md#bug-1--mamba-state-capacity-blocks-a-hybrid-model): sequence cap blocked startup | More available state does not guarantee higher throughput |
| Prefix caching | Warmup can change the measured workload | [Bug 6](fix_bug.md#bug-6--benchmark-warmup-creates-the-cache-hits-being-measured): 16,000 reused tokens | Reuse fraction is not throughput speedup |
| Cache geometry and precision | A dtype requires a compatible layout/kernel | [Bug 8](fix_bug.md#bug-8--one-incompatible-fp8-layout-is-not-universal-fp8-failure): failed route, successful alternate stack | Backend and dtype effects must be separated |
| Automatic pool sizing | Startup state changes capacity available for serving | [Bug 12](fix_bug.md#bug-12--a-plausible-architecture-story-explains-a-startup-confound): Qwen c4 changed 162.2→256.8 tok/s | The causal contribution of preemptions is not measured |
| Hybrid execution | Estimators need explicit operation coverage | [Bug 13](fix_bug.md#bug-13--enabled-counters-absent-collection-incomplete-accounting): missing components | Partial estimates cannot identify the hardware bottleneck |

Read these as a chain: **architecture → runtime requirement → observed failure/result → discriminating test → bounded conclusion**. That chain links the architecture survey to the experiments without assuming every performance difference is architectural.

### Weight and state accounting [A]

| Model | Prior tensor-accounted total / active GEMM parameters | Recorded disk footprint | Main experiment precision |
|---|---|---|---|
| GLM | 321.34B / 17.38B | 305.8 GiB | FP8 weights; BF16 KV |
| DeepSeek | 290.91B / 14.08B | 148.6 GiB | MXFP4 experts + FP8 attention/dense; FP8 KV |
| Qwen | 176.94B served / 7.27B | 172.8 GiB | FP8 weights; BF16 KV |

Counts come from the existing per-model tensor analyses, not a new recount of complete checkpoints. Active-parameter definitions and treatment of embeddings/MTP differ from vendor labels. Disk size is not runtime HBM. A weight-only GPU-count lower bound establishes neither a valid TP layout nor room for runtime state.

Qwen's prior analysis identifies a **51.23B n-gram embedding table**, about 29% of its served total. Lookup touches selected entries, not the entire table each token. Treating all table parameters as active arithmetic overestimates work. Table offloading was not measured.

```text
HBM needed = weights + workspace + graph buffers
           + sequence-growing history + recurrent state + padding/metadata

Conventional KV bytes/token = sum_over_layers(2 × KV_heads × head_dim × bytes)
```

For Qwen's 12 retained-KV layers: `12 × 2 × 2 × 256 × 2 = 24,576 bytes/token = 24 KiB/token`, before indexer/state overhead. Prior analysis estimates **24.75 KiB/token including indexer keys**. Eight sequences of 131,072 tokens imply about **24.75 GiB across the model** for this component alone; rank placement, grouping, and padding require separate accounting.

Kimi K3's official card reports 2.8T parameters. Even an ideal four-bit payload is `2.8e12 × 0.5 = 1.4e12 bytes`, approximately **1,304 GiB**, exceeding this node before scales and runtime state. This explains architecture-only coverage; no throughput or cost is extrapolated. [Official Kimi card](https://huggingface.co/moonshotai/Kimi-K3)

## 3. Q1 — Workload and concurrency: what limits performance?

### Main batch grid: 16K input, 256 output

| Concurrency | Qwen tok/s | GLM tok/s | DeepSeek tok/s | Qwen TPOT ms | GLM TPOT ms | DeepSeek TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 106.2 | 96.3 | 85.0 | 7.0 | 7.1 | 7.9 |
| 4 | 256.8 | 225.3 | 219.8 | 9.6 | 10.9 | 10.7 |
| 16 | 420.2 | 359.2 | 330.9 | 26.3 | 33.6 | 32.4 |
| 64 | 517.7 | 447.1 | 389.4 | 110.1 | 129.5 | 148.5 |

Source: `batch_isl16k_c{1,4,16,64}.json` in the three main batch directories in §1. Qwen at concurrency 64 is **1.16× GLM and 1.33× DeepSeek**. It leads this observed throughput grid; no accuracy or equal-text comparison was performed.

### Finer concurrency grid

| Concurrency | GLM tok/s | GLM TPOT ms | DeepSeek tok/s | DeepSeek TPOT ms |
|---:|---:|---:|---:|---:|
| 1 | 96.3 | 7.1 | 92.9 | 7.9 |
| 2 | 122.2 | 8.5 | 151.9 | 8.6 |
| 4 | 225.3 | 10.9 | 214.8 | 11.1 |
| 8 | 296.9 | 18.1 | 281.1 | 16.2 |
| 16 | 359.2 | 33.6 | 323.4 | 33.7 |
| 32 | 406.4 | 68.6 | 363.1 | 72.7 |
| 48 | 432.6 | 100.5 | 357.5 | 120.2 |
| 64 | 447.1 | 129.5 | 384.9 | 152.1 |

Sources: GLM `bf16kv` and DeepSeek `util085-dev20073`. DeepSeek's curve is a separate arm from the main comparison. GLM points span sessions, adding temporal variability.

Concurrency 8→64 buys **1.51× throughput for 7.15× TPOT on GLM**, and **1.37× for 9.41× on DeepSeek**. This suggests a latency-sensitive operating region around 8–16, not a universal optimum. Qwen lacks the finer grid. DeepSeek's small dip at 48 has no established mechanism or repeat-based significance.

### Context sweep at concurrency 8

| Input tokens | Qwen tok/s* | GLM tok/s | DeepSeek tok/s | Qwen cache peak* | GLM cache peak | DeepSeek cache peak |
|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 322.7 | 295.9 | 104.5 | 7.2% | 10.2% | 18.1% |
| 65,536 | 114.5 | 104.3 | 47.9 | 28.2% | 36.6% | 21.9% |
| 131,072 | 56.5 | 47.2 | 34.7 | 56.1% | 72.5% | 27.0% |
| 260,000 | 25.1 | 26.5 | 16.7 | 97.3% | 89.5% | 36.9% |

Source: `ctx_isl{length}_c8.json` in §1's context arms. *Qwen uses the earlier smaller `base` pool, not corrected `base-util082`; rerun before treating its cache limit as architectural.*

At 131K, median TTFT is **19.84 s Qwen, 12.89 s GLM, 26.66 s DeepSeek**. The throughput leader is not the first-token-latency leader. High occupancy indicates headroom risk, not proof of preemption or a capacity-caused slowdown. Pool sizes and cache semantics differ across models.

DeepSeek's 16K/context point is much slower than its 16K/concurrency-8 point in the separate finer-grid arm (**104.5 versus 281.1 tok/s**). Build, utilization, and session differ; the batch-only bridge does not explain this discrepancy. Flag it for a same-session rerun, and avoid using it to identify an architectural bottleneck. Context throughput also includes extra input work; it is not a decode-only scaling curve.

### Bottleneck attribution: observation versus explanation

| Regime | Observation | Candidate mechanism [H] | Required check |
|---|---|---|---|
| Low concurrency | Low aggregate throughput; lower TPOT | Small GEMMs, weight/state traffic, launches, collectives | Decode operator trace and hardware counters |
| High concurrency | Diminishing gains; rising TPOT | Scheduling, compute/communication contention | Actual per-step batches; TP/EP sweep |
| Long input | Higher TTFT; lower output rate | Input work, attention/indexing, queueing | Separate prefill/decode timing; chunk-size sweep |
| High cache occupancy | Little pool headroom | Allocation pressure or preemption | Preemption counters; matched pool-size A/B |

GLM's available engine estimates report roughly **319–608 GB/s/GPU**, or 9.5–18.1% of an assumed 3,350 GB/s peak. They include FFN and unembedding but **omit attention**, and span different utilization arms. They are [E], not hardware-counter measurements. A partial whole-request average cannot rule out bandwidth-bound individual kernels.

The older estimate `active_weight_bytes × output_tok/s ÷ concurrency` is unsuitable: tokens can select different experts, live batch differs from client concurrency, and output throughput includes prefill. Under simplified independent uniform routing, distinct experts scale as `E × [1 − (1 − k/E)^B]` for B tokens. Real routing and reuse need measurement. Total/active parameters alone do not predict the bottleneck.

## 4. Q2 — Prefix caching: observed effects and implementation challenges

| Distinct 64K prefixes among 64 requests | Qwen tok/s* | GLM tok/s | DeepSeek tok/s |
|---:|---:|---:|---:|
| 1 | 453.9 | 265.3 | 198.2 |
| 4 | 398.1 | 365.9 | 391.6 |
| 16 | 262.2 | 199.4 | 114.6 |

Source: `prefix_p64k_n{1,4,16}.json` in the main context/prefix arms. *Qwen pool caveat applies.* All complete 64 requests without failure. Prefix JSONs do **not** contain cache-hit deltas or cold-run assertions; reuse is intentional.

Best/worst observed patterns differ by **1.73× Qwen, 1.83× GLM, 3.42× DeepSeek**. These are sharing-pattern ratios, **not cache-on/cache-off speedups**: no identical uncached control exists. GLM and DeepSeek peak at four prefixes. Fill ordering, locality, eviction, and scheduling could explain this [H]; no trace isolates them.

Implementation requirements:

1. **Correct identity:** match token prefix, model revision, adapter, positions, and relevant execution state. Similar text is insufficient.
2. **Consistent state boundary:** retained KV can be shared; recurrent layers need saved state at the reused boundary. A final recurrent state cannot reconstruct arbitrary earlier states. Checkpointing enables reuse with memory/copy costs; recurrent state is not inherently uncacheable.
3. **Different layouts:** full KV, compressed/indexer history, and recurrent state require compatible alignment, resume positions, and group-aware eviction.
4. **Safe branching:** immutable prefix data can be shared; divergent requests need state copies or copy-on-write. Speculative rejection must roll state back correctly.
5. **Granularity tradeoff:** finer checkpoints allow more reuse but cost metadata/storage; coarser checkpoints require more recomputation after partial matches.

GLM's manifest resolves attention block size 640 and Mamba block size 128; corrected Qwen resolves 4 and 16. These represent different organizations, not a measured 160× efficiency difference. DeepSeek requests launch block size 256 while its exported cache metric says 4; logical blocks and backend storage blocks must not be equated without implementation inspection.

vLLM documents heterogeneous state sizing, padding, and coordination across cache groups. This supports the design challenge; current documentation does not establish behavior in the recorded development builds. [Hybrid cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)

**Benchmark lesson:** warm compiled kernels with unrelated prompts, then use fresh prompts for cold-prefix tests. Reusing seeds across points can accidentally benchmark cached prompts. `/health` does not establish completion of kernel warmup. The recorded incidents are in [fix_bug.md](fix_bug.md).

## 5. Q3 — Speculative decoding / multi-token prediction

MTP drafts candidates and the target verifies them. Benefit depends on accepted progress relative to drafting, verification, state, and scheduling costs. Acceptance rate alone does not determine speedup.

### Strongest controlled evidence: GLM batch A/B

| Concurrency | Base tok/s | MTP n=1 tok/s | n=1 / base | MTP n=5 tok/s | n=5 / base |
|---:|---:|---:|---:|---:|---:|
| 1 | 96.3 | 119.7 | 1.24× | 120.8 | 1.25× |
| 4 | 225.3 | 239.8 | 1.06× | 230.7 | 1.02× |
| 16 | 359.2 | 344.7 | 0.96× | 342.1 | 0.95× |
| 64 | 447.1 | 441.9 | 0.99× | 408.3 | 0.91× |

Sources: GLM `bf16kv`, `bf16kv-mtp-n1`, `bf16kv-mtp-n5`. At concurrency 1, n=1 reduces median TPOT **7.06→5.05 ms**. At 64, n=5 has **99.2% cache occupancy** and TTFT **10.72 s versus 2.51 s** for base. More draft tokens are not justified by throughput in these loaded cases.

### GLM context A/B: latency and throughput diverge

| Input tokens | Base → MTP tok/s | Throughput ratio | Median TTFT base → MTP | TTFT change |
|---:|---:|---:|---:|---:|
| 16,384 | 295.9 → 314.8 | 1.064× | 2.63 → 2.10 s | −20.1% |
| 65,536 | 104.3 → 97.6 | 0.936× | 7.21 → 5.43 s | −24.8% |
| 131,072 | 47.2 → 45.3 | 0.958× | 12.89 → 10.73 s | −16.7% |
| 260,000 | 26.5 → 22.2 | 0.839× | 38.48 → 41.07 s | +6.7% |

Source: [GLM MTP context](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/). The 131K point offers a possible latency/cost tradeoff. At 260K both metrics worsen and cache occupancy reaches 96.9%. Scheduling changes may explain earlier first tokens [H]; MTP does not directly eliminate input work, and no timeline proves the mechanism.

### Other models: weaker controls

| Concurrency | Qwen original base → MTP tok/s | Ratio | DeepSeek historical off → no-reuse MTP tok/s | Ratio |
|---:|---:|---:|---:|---:|
| 1 | 107.7 → 125.9 | 1.17× | 92.3 → 115.3 | 1.25× |
| 4 | 162.2 → 149.3 | 0.92× | 222.0 → 251.8 | 1.13× |
| 16 | 371.0 → 322.2 | 0.87× | 293.7 → 339.9 | 1.16× |
| 64 | 517.8 → 481.5 | 0.93× | 383.4 → 380.2 | 0.99× |

Sources: Qwen `base`/`mtp-n1`; DeepSeek `mtp-off`/`mtp-on-noreuse`. Qwen's equal-utilization pair retains startup/pool confounds; substituting corrected `base-util082` would also change utilization. DeepSeek's historical pair uses the older runtime and lacks newer cold-run fields. Do not substitute newer baselines or call this a fully controlled three-model A/B. Real-text acceptance and answer quality remain untested.

## 6. Q4 — Serving cost with explicit assumptions

For GPU-hour price p, GPU count G=8, and output throughput T:

```text
GPU-seconds/output token = G / T
$/million output tokens = G × p × 1,000,000 / (3,600 × T)
```

Assume **$2.50/GPU-hour**, or $20/node-hour, for illustration. This is not a current quote. The formula charges the entire measured workload, including input processing, to output tokens. It is neither a decode-only cost nor an API tariff.

| Workload | Qwen $/million output | GLM $/million output | DeepSeek $/million output |
|---|---:|---:|---:|
| 16K input, concurrency 1 | 52.29 | 57.68 | 65.33 |
| 16K input, concurrency 64 | 10.73 | 12.43 | 14.27 |
| 131K input, concurrency 8 | 98.38* | 117.59 | 160.26 |

Costs [A] use unrounded JSON throughput. *Qwen long-context caveat applies.* Scale by `actual_GPU_hour_price / 2.50` for another price assumption. GLM at concurrency 64 costs about **$0.00318 per request** with 256 output tokens, including its 16K input work under this allocation model.

At concurrency 64, throughput/GPU is **64.72 Qwen, 55.88 GLM, 48.67 DeepSeek tok/s/GPU**. This normalizes eight-GPU runs; it does not predict one-GPU performance. There is no measured optimal GPU count, energy efficiency, total ownership cost, or cost per successful task. Tokenizers and quality differ.

A deployment decision should compare **goodput under explicit TTFT/TPOT objectives**. The cheapest token in the table may violate an interactive latency target. Production cost also needs realistic arrival rates, repeats, equivalent tasks, and idle-time accounting.

## 7. Additional systems findings

### FP8 KV: backend and dtype must be separated

| GLM arm, concurrency 64 | Output tok/s | Interpretation |
|---|---:|---|
| Original BF16 (`bf16kv`) | 447.1 | Main baseline |
| Alternate BF16 (`bf16kv-fi618`) | 348.8 | −22.0%: software/backend-path change |
| Alternate FP8 (`fp8kv-fi618`) | 262.0 | −24.9% versus alternate BF16 |

The full drop is 41.4%; attributing it all to dtype is incorrect. The prior [GLM capacity accounting](GLM-5.3-Flash/report.md) reports **2,099,654→3,790,580 tokens, or 1.805×**, across its reported capacity comparison. Those startup pools differ from the main baseline's later 1,916,967-token pool; retain session provenance. The successful alternate arm supersedes the old blanket statement that GLM FP8 KV is impossible on H100.

The saved alternate-stack scripts bypass a FlashInfer version check and combine newer Python with older compiled artifacts, routing MoE through DeepGEMM. This is an experimental configuration, not a validated production recipe. Synthetic completion does not establish numerical parity. The paired dtype effect includes dtype-dependent kernel/dequantization behavior. See [Bug 8](fix_bug.md#bug-8--one-incompatible-fp8-layout-is-not-universal-fp8-failure).

### Startup state changes measured capacity

GLM's utilization 0.82→0.85 control grows reported capacity 11.2%, while its four batch throughputs change by less than 0.7%. This supports little throughput sensitivity **on that grid**, not a universal claim that utilization can never affect speed. [GLM control](GLM-5.3-Flash/results/util085/RESULT-util-ab.md)

Qwen's earlier startup records peak activation **17.07 GiB versus 0.99 GiB** in the warm arm. Despite lowering utilization, reported pool capacity grows **2,048,645→3,197,331 tokens**, and concurrency-4 throughput rises **162.2→256.8 tok/s**. Startup and utilization changed together: this exposes a confound rather than a utilization-only speedup. Only corrected batch results become headline evidence. [Qwen audit](Qwen3.8-Flash-Next-FP8/results/base-util082/RESULT-util-ab.md)

## 8. Contribution to SyFI and next experiments

The contribution is an empirical **workload/configuration tradeoff map**, backed by raw results and documented measurement failures. It motivates testing layer-aware memory accounting and workload-aware serving policies. It does not establish a new scheduler, prove that existing engines assume uniform layers, or identify a dominant kernel.

| Priority | Experiment | Question resolved |
|---|---|---|
| 1 | Matched-startup Qwen base/MTP/context/prefix; same-session DeepSeek context rerun | Which rankings survive controlled pools and startup? |
| 2 | Randomized repeated runs with longer windows | Are small differences reproducible? |
| 3 | Profile prefill, decode, GEMMs, dispatch, attention, hardware memory traffic | What actually limits each regime? |
| 4 | Sweep chunked-prefill budget and feasible TP/EP layouts | Can goodput improve at fixed latency objectives? |
| 5 | Cache-off, cold-fill, and prewarmed-prefix controls with state/eviction/preemption counters | What savings come from reuse, and what does hybrid state cost? |
| 6 | Representative task/session replay with real text and quality evaluation | What is cost per successful task and realistic MTP acceptance? |

Saved TraceLab summaries motivated the input-length grid; no real trace replay was performed. Do not interpret reusable-prefix estimates as guaranteed achievable cache hits. No dense baseline, Kimi run, multimodal test, disaggregated deployment, or validated layer-reduction extrapolation is included.

The document audit also found remaining harness gaps: unavailable cache telemetry can default to zero, positive cache-hit warnings do not reject a point in that branch, and completion validation rejects zero rather than all partial failures. The retained JSON audit checks saved fields; it cannot independently prove successful original telemetry. These are documented as **pending**, not fixed, in [fix_bug.md §6](fix_bug.md#6-remaining-gaps-in-the-saved-harness--not-fixes-performed-here).

## 9. Professor questions: answers to rehearse

| Question | Defensible answer |
|---|---|
| What did you establish? | Named deployment tradeoffs, GLM MTP/backend A/Bs, and measurement confounds. |
| Which model is best? | Qwen leads corrected 16K throughput; GLM has lower observed 131K median TTFT. Quality and optimal deployments are unmeasured. |
| Which part is the bottleneck? | I have hypotheses, not operator attribution. Separate prefill/decode traces and counters are next. |
| Why not claim bandwidth is low? | Partial modeled counters over mixed request work cannot rule out bandwidth-bound kernels. |
| Is concurrency batch size? | No; it caps client requests. Engine token batches vary each step. |
| Is the comparison fair? | GPU class/count and workload targets match; precision, tokenizers, backends, builds, and some pools differ. This compares deployments. |
| Does the bridge remove engine confounds? | Only a small DeepSeek batch-grid build effect was observed; other models and axes remain uncontrolled. |
| Why no Kimi? | Architecture is covered; even ideal four-bit K3 weights exceed node memory. No performance is extrapolated. |
| Can recurrent state be cached? | Yes, via compatible checkpoints at reused boundaries, together with other layer state. |
| How much does prefix caching accelerate inference? | We measured sharing patterns, not on/off. Best/worst differs up to 3.42× for DeepSeek. |
| Why does MTP improve TTFT? | That is observed for some GLM contexts. Scheduling is a hypothesis, not a traced cause. |
| Should MTP always be enabled? | No. GLM benefits at low concurrency; loaded throughput is flat/worse. Evaluate latency, memory, and quality too. |
| Is FP8 KV impossible on Hopper? | No blanket claim is valid: GLM's alternate stack ran it, with a speed/capacity tradeoff. |
| Why not fewer GPUs? | It may reduce cost, but no successful deployment sweep establishes the optimum. Eight GPUs were the experimental budget. |
| Where are error bars? | Most cells have one retained run. Within-run percentiles are not repeat-based confidence intervals. |
| Is $10.73 a real quote? | No: $2.50/GPU-hour times observed runtime, normalized to outputs and including input work. |
| What would you build for SyFI? | Validate regimes first, then test state-aware cache/scheduling policies against measured goodput. |

Presentation discipline: use **“observed,” “for this arm,” and “hypothesis.”** Avoid “proved communication-bound,” “all engines matched,” “cache speedup,” and “cheapest model” without the missing controls.
