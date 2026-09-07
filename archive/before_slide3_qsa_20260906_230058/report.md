# DeepSeek, GLM and Qwen: architecture and serving comparison

Tan Ngo · For Professor Kan Zhu and UW SyFI · Final synthesis: 6 September 2026

Assignment: [task.md](task.md). Presentation: [editable PowerPoint](SyFI_ML_Serving_refined.pptx), [storyboard](slides.md), and [complete speaking script](script.md). Architecture-to-debugging study guide: [fix_bug.md](fix_bug.md).

**Research question:** when do architectural reductions in computation or state translate into useful serving performance? The report has two connected parts: frontier design and its testable implications (§§1–2), then local measurements, diagnosis and a proposed research direction (§§3–9). Published training results and mechanisms are distinguished from this study's inference measurements.

## Executive findings

This study compares **DeepSeek-V4-Flash, GLM-5.3-Flash, and Qwen3.8-Flash-Next-FP8** on an eight-GPU H100 configuration. These three checkpoints are the complete current scope. Tables and presentation comparisons follow **DeepSeek → GLM → Qwen**; model colors are orange, purple and teal respectively. The central systems lesson is that serving behavior depends on workload, cache state, backend, and scheduling as well as architecture.

### Three-model result snapshot

Read across a row to compare one metric under one workload. All rates include input processing; latency entries are request medians. The columns keep model identity fixed instead of reordering by the winner.

| Workload / metric | DeepSeek | GLM | Qwen |
| --- | ---: | ---: | ---: |
| 16K input, c64: output tok/s ↑ | 389.4 | 447.1 | 517.7 |
| Same 16K/c64: median TPOT, ms ↓ | 148.5 | 129.5 | 110.1 |
| Same 16K/c64: illustrative $/million output ↓ | 14.27 | 12.43 | 10.73 |
| 131K input, c8: output tok/s ↑ | 34.7 | 47.2 | 56.5* |
| Same 131K/c8: median TTFT, s ↓ | 26.66 | 12.89 | 19.84* |
| Prefix sharing: best retained pattern | 4 prefixes | 4 prefixes | 1 prefix* |
| MTP comparison strength | Historical runtime pair | Strongest paired controls | Startup/pool confounded |

Sources: exact main batch, `ctx_isl131072_c8.json`, and `prefix_p64k_n*.json` arms in §1; full grids in §§3–5. Costs use the §6 illustrative $2.50/GPU-hour assumption. *Qwen context/prefix use the earlier `base` pool, while batch uses corrected `base-util082`.* MTP comparisons are qualified separately; these rows do not imply a fully matched architecture experiment or equal answer quality.

1. **Qwen leads the corrected 16K-input throughput grid.** At concurrency 64: Qwen 517.7 output tokens/s, GLM 447.1, DeepSeek 389.4. These are deployed configurations, not isolated architectural effects or equal-quality answers.
2. **Concurrency trades latency for throughput.** From concurrency 1 to 64, throughput grows about 4.6–4.9× while median time per output token grows about 16–19×. Finer GLM/DeepSeek sweeps show diminishing returns around concurrency 8–16.
3. **Long inputs change the tradeoff.** At 131K input, GLM has lower median first-token latency than Qwen despite lower throughput. Qwen's long-context arm has a known cache-sizing confound, so its capacity findings are provisional.
4. **Speculative decoding is workload-dependent.** GLM MTP with one draft token improves throughput 24% at concurrency 1, but provides no observed throughput gain at 64. At 131K input it reduces median TTFT by 17% while reducing throughput by 4%.
5. **Memory savings need not increase speed.** GLM's alternate FP8-KV stack increases reported capacity; switching BF16 to FP8 KV on that stack loses 25% throughput at concurrency 64.
6. **Exact kernel bottlenecks remain unproven.** Request metrics establish tradeoffs. Incomplete engine estimates cannot prove hardware bandwidth saturation or communication dominance.

### What decision does each finding support?

| Objective | Evidence to use | Decision supported now | What could change it |
| --- | --- | --- | --- |
| Maximize output rate on this 16K grid | Corrected main batch arms, §3 | Qwen is the observed throughput leader | Different quality target, workload, runtime or repeated results |
| Reduce waiting at long input | 131K/c8 median TTFT, §3 | GLM is the observed first-token-latency leader | Matched Qwen startup; phase timing; larger samples |
| Tune interactive concurrency | Finer GLM/DeepSeek curves, §3 | Evaluate the latency cost of moving beyond c8–16 | Actual per-request latency targets and arrival traffic |
| Decide whether to speculate | GLM paired MTP arms, §5 | Test n1 at light load; avoid assuming longer drafts help | Real-text acceptance, phase costs, quality and state pressure |
| Increase history capacity | GLM alternate BF16/FP8 pair, §7 | Price additional capacity against its speed loss | Validated kernels, numerical parity and matched startup |

The result is an **operating-point map**, not a universal model ranking. Most observations have one retained run; the table describes evidence-supported choices for further evaluation, not production recommendations.

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
| --- | --- | --- | --- |
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
| --- | --- | --- |
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

### Three-model architecture matrix

| Comparison dimension | DeepSeek-V4-Flash | GLM-5.3-Flash | Qwen3.8-Flash-Next-FP8 |
| --- | --- | --- | --- |
| Attention organization | 43 sparse-attention layers with compressed history | 34 recurrent KDA + 11 sparse-attention layers | 36 recurrent GDN + 12 QSA layers |
| State that must be retained | Compressed history, indices and local/tail state | Recurrent checkpoints plus retained sparse-attention history | Recurrent checkpoints plus retained QSA history/index state |
| Routed experts / selected per token | 256 / 6 | 288 / 8 | 512 / 10 |
| Prior total / active GEMM parameters | 290.91B / 14.08B | 321.34B / 17.38B | 176.94B served / 7.27B |
| Main measured weights / cache | MXFP4 experts + FP8 attention/dense / FP8 KV | FP8 weights / BF16 KV | FP8 weights / BF16 KV |
| Published optimizer information used here | Mixed Muon + AdamW | Recipe not established by the reviewed sources | Mixed Muon + AdamW |
| Local MTP evidence | Older off/on-noreuse pair | Base versus n1/n5; paired context n1 | Original base versus n1, pool/startup confounded |
| Serving question to test [H] | Do compression savings exceed indexing and tail-management work? | When do recurrent-state and speculative costs offset reduced history work? | How much does the smaller GEMM path contribute after matching startup and kernels? |

Configuration and prior tensor sources are linked under “Checkpoint organization and retained evidence” below; optimizer and MTP qualifications are explained in their dedicated sections. Qwen's 51.23B n-gram table is counted in served capacity but is not a full per-token matrix multiplication. “Not established” is a documentation limit, not a claim that GLM uses a particular alternative optimizer.

**The main architectural contrast:** DeepSeek compresses the history representation throughout its attention stack; GLM and Qwen combine recurrent state with history retrieval. **The main experimental contrast:** all three have main batch measurements, but the strongest local MTP controls belong to GLM. Keep those two comparisons separate.

### A design map: remove work, account for the replacement

| Frontier mechanism | Work or resource it targets | New obligation / possible loss | Relevant test |
| --- | --- | --- | --- |
| MoE routing | Selected feed-forward arithmetic | Resident experts, dispatch/combine, load imbalance | Expert GEMM and collective timing versus live tokens per expert |
| Sparse attention | Attention over irrelevant history positions | Index scoring, top-k, irregular gather; retrieval quality | Context-length crossover with a matched attention control |
| Compressed history | Stored/read history representation | Compression, indexing and incomplete tails | Bytes by cache group, phase time, exact prefix restoration |
| Recurrent/attention hybrid | Sequence-growing state in recurrent layers | Per-sequence state and checkpoint/rollback management | Active-sequence capacity separately from token capacity |
| Native MTP | Serial target decode iterations per committed token | Draft/verify work, rejected state and scheduler overhead | Committed tokens per round and phase time versus load |
| Muon / mixed optimizer recipe | Training convergence and update geometry | Orthogonalization, partitioning and stability engineering | Quality reached per training GPU-hour; no local training A/B |

This is a mechanism map [A], not an operator attribution for the local runs. The key question is **which cost shrinks, which cost replaces it, and at what workload does the saving exceed the replacement?**

### Sparse attention: selected computation is only one term

For sequence length S and K selected history entries per query, dense prefill attention-score work scales as `O(S²)`; the selected attention component scales roughly as `O(SK)`. That is not the total sparse path:

```text
Sparse-path time = index construction/scoring + selection + gather
                 + attention on selected entries + state management
```

DeepSeek V4 combines compressed sparse attention with heavily compressed attention and a local window; Qwen QSA uses a compressed lightweight indexer. These designs reduce different parts of history work. In particular, compressing an indexer's candidate sequence by a fixed factor does not make all index scoring constant-time. [DeepSeek V4 report, §2.3](https://arxiv.org/html/2606.19348v1), [Qwen architecture report, §2.1.2](https://arxiv.org/html/2608.30320v1)

**Prediction [H]:** long contexts offer more opportunity to amortize indexing and selection, while short contexts may favor highly fused dense kernels. **Local evidence [M]:** §3 measures context-sensitive request behavior. **Missing discriminating test:** a matched dense/sparse or indexer intervention with phase timing and retrieval-quality checks. Neither a lower cache occupancy nor a higher output rate alone measures sparse-attention speedup.

### Muon versus AdamW: training efficiency has a different causal path

AdamW uses elementwise first/second-moment adaptation with decoupled weight decay. Muon applies approximate orthogonalization to a matrix momentum update, typically using Newton–Schulz iterations. The distinction is update geometry, not a smaller inference matrix. [AdamW paper](https://arxiv.org/abs/1711.05101), [Muon author explanation](https://kellerjordan.github.io/posts/muon/)

**Optimizer-state memory [A]:** standard Muon keeps one momentum tensor, while AdamW keeps first- and second-moment tensors. At equal state dtype and comparable sharding, this is **50% less persistent optimizer-state tensor memory per parameter assigned to Muon**: 4 versus 8 bytes with FP32 states. It is not a 50% reduction in total training memory; weights, gradients, activations, optional master weights, temporary workspace, and auxiliary AdamW groups require separate accounting. This is buffer-count arithmetic, not a local memory measurement. [Reference implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py), [PyTorch / DeepSpeed explanation](https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/)

**Matrix geometry [A]:** Muon's matrix products couple entries and approximately orthogonalize the update, reshaping its singular-value spectrum. AdamW's adaptive rescaling uses each coordinate's own moment estimates without explicit normalization of matrix singular directions. This does not mean AdamW's gradients are independent of the rest of the network, or that Muon estimates the full Hessian of the loss. [Muon author explanation](https://kellerjordan.github.io/posts/muon/)

| Question | Precise answer |
| --- | --- |
| Does Muon replace AdamW everywhere? | No. The reference implementation explicitly separates suitable matrix weights from auxiliary AdamW parameters. |
| What do the three models' sources establish? | DeepSeek and Qwen use mixed parameter-group assignments. GLM-5.3-Flash's optimizer recipe is not established by the sources used here; do not infer it from the other models. |
| What is the systems challenge? | Matrix orthogonalization changes optimizer computation and how matrices should be partitioned/batched. |
| Can Muon explain the local inference ranking? | Not directly: optimizer updates are absent from inference. A training recipe can affect learned quality and feasible model design, but this study has no training or equal-quality control. |

Sources: [Muon reference implementation](https://github.com/KellerJordan/Muon), [DeepSeek V4 optimizer and framework sections](https://arxiv.org/html/2606.19348v1), [Qwen official release](https://qwen.ai/blog?id=qwen3.8-flash-next). These primary references were consulted on 6 September 2026. The appropriate evaluation is quality reached per training resource budget, including optimizer overhead, rather than treating “trained with Muon” as a measured serving acceleration.

### Native MTP and “zero-day” speculative decoding

Here **zero-day** is descriptive: native draft capability and serving support available around release. It is not a separate mathematical algorithm or a promise of zero overhead. A model-provided MTP module can remove the need to wait for a separately trained external draft; support and optimized execution still depend on the engine version. DeepSeek documents MTP as both a training component and a potential inference drafter. SGLang's Qwen launch implementation illustrates additional runtime work, including sharing selection information across draft steps. These external implementation disclosures do not describe the exact saved local vLLM runs. [DeepSeek V3 repository](https://github.com/deepseek-ai/DeepSeek-V3), [SGLang Qwen day-0 support](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)

The protocol is **draft → target verification → commit valid progress / restore rejected state**. Preserving the target sampling distribution requires the appropriate acceptance and residual-resampling algorithm, together with correct runtime state; simply verifying a candidate list is insufficient. This theoretical property does not establish numerical parity of the saved experimental backends. [Speculative decoding paper](https://arxiv.org/abs/2211.17192)

```text
Speculative time / useful token ≈ E[round time] / E[committed tokens per round]
Round time includes drafting, verification, state operations and scheduling.
Benefit requires this ratio to beat the ordinary time / token at the same load.
```

The expression is a steady-work analytical lens [A], not a fit to these finite benchmarks. It predicts why acceptance alone cannot choose draft length, and why a policy that helps c1 may hurt at c64. §5 provides the local test; a phase trace is still needed to explain the mechanism.

### From an efficient component to an efficient request

An Amdahl calculation prevents kernel improvements from becoming unsupported end-to-end claims. If fraction f of original elapsed time is accelerated by r, while other work stays constant:

```text
Request speedup = 1 / [(1 − f) + f/r]
Illustration: f = 0.30 and r = 10 → 1 / (0.70 + 0.03) = 1.37×
```

The 30% fraction and 10× improvement are **illustrative assumptions**, not measured values. Real overlap, changing batch composition and queueing can invalidate the fixed-fraction model. Its purpose is to show why component efficiency must be connected to phase cost and then to the user objective.

**Why might these deployments be efficient?** MoE reduces selected arithmetic, recurrence/compression reduce portions of history work, and MTP can amortize serial decoding. **Why is Qwen fastest on the main grid?** Its selected GEMM path and hybrid organization are plausible contributors, but no ablation separates them from precision, kernels, startup and scheduling. The local data supports the ranking; it does not allocate the speed advantage among architectural causes.

### Checkpoint organization and retained evidence

All three measured checkpoints are **mixture-of-experts (MoE)** models. Routing activates a subset of experts per token, reducing arithmetic compared with activating all weights while retaining a large resident weight footprint and adding dispatch/combine communication.

| Model | Attention organization [A] | Routed experts / selected [A] | Serving implication |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash | 43 sparse-attention layers; compressed history | 256 / 6 | Smaller history representation, plus compression/indexing work |
| GLM-5.3-Flash | 34 recurrent KDA + 11 sparse-attention layers | 288 / 8 | Recurrent state plus sequence-growing attention cache |
| Qwen3.8-Flash-Next-FP8 | 36 Gated DeltaNet (GDN) + 12 Qwen sparse-attention (QSA) layers | 512 / 10 | Recurrent state, retained attention KV, and an n-gram lookup table |

Sources: local [DeepSeek analysis](deepseek_v4_flash/report.md), [GLM analysis](GLM-5.3-Flash/report.md), and [Qwen analysis](Qwen3.8-Flash-Next-FP8/report.md). Official configuration cross-checks: [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json), [GLM](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json), [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json). Online sources checked 5 September 2026; rolling sources do not replace recorded manifests.

**The similarity:** all use sparse expert computation. GLM and Qwen also mix recurrent and history-retaining attention inside one stack. **The difference:** recurrent layers store fixed-size state per sequence; attention retains sequence-growing history. Sparse attention can limit the history consulted without eliminating its storage. Compression reduces history size but adds work. These costs need separate terms.

### Connect architecture to debugging and results

| Feature | Runtime consequence | Evidence to study | Limit of the inference |
| --- | --- | --- | --- |
| Recurrent layers | State capacity also scales with active sequences | [Bug 1](fix_bug.md#bug-1--mamba-state-capacity-blocks-a-hybrid-model): sequence cap blocked startup | More available state does not guarantee higher throughput |
| Prefix caching | Warmup can change the measured workload | [Bug 6](fix_bug.md#bug-6--benchmark-warmup-creates-the-cache-hits-being-measured): 16,000 reused tokens | Reuse fraction is not throughput speedup |
| Cache geometry and precision | A dtype requires a compatible layout/kernel | [Bug 8](fix_bug.md#bug-8--one-incompatible-fp8-layout-is-not-universal-fp8-failure): failed route, successful alternate stack | Backend and dtype effects must be separated |
| Automatic pool sizing | Startup state changes capacity available for serving | [Bug 12](fix_bug.md#bug-12--a-plausible-architecture-story-explains-a-startup-confound): Qwen c4 changed 162.2→256.8 tok/s | The causal contribution of preemptions is not measured |
| Hybrid execution | Estimators need explicit operation coverage | [Bug 13](fix_bug.md#bug-13--enabled-counters-absent-collection-incomplete-accounting): missing components | Partial estimates cannot identify the hardware bottleneck |

Read these as a chain: **architecture → runtime requirement → observed failure/result → discriminating test → bounded conclusion**. That chain links the architecture survey to the experiments without assuming every performance difference is architectural.

### Weight and state accounting [A]

| Model | Prior tensor-accounted total / active GEMM parameters | Recorded disk footprint | Main experiment precision |
| --- | --- | --- | --- |
| DeepSeek | 290.91B / 14.08B | 148.6 GiB | MXFP4 experts + FP8 attention/dense; FP8 KV |
| GLM | 321.34B / 17.38B | 305.8 GiB | FP8 weights; BF16 KV |
| Qwen | 176.94B served / 7.27B | 172.8 GiB | FP8 weights; BF16 KV |

Counts come from the existing per-model tensor analyses, not a new recount of complete checkpoints. Active-parameter definitions and treatment of embeddings/MTP differ from vendor labels. Disk size is not runtime HBM. A weight-only GPU-count lower bound establishes neither a valid TP layout nor room for runtime state.

Qwen's prior analysis identifies a **51.23B n-gram embedding table**, about 29% of its served total. Lookup touches selected entries, not the entire table each token. Treating all table parameters as active arithmetic overestimates work. Table offloading was not measured.

```text
HBM needed = weights + workspace + graph buffers
           + sequence-growing history + recurrent state + padding/metadata

Conventional KV bytes/token = sum_over_layers(2 × KV_heads × head_dim × bytes)
```

For Qwen's 12 retained-KV layers: `12 × 2 × 2 × 256 × 2 = 24,576 bytes/token = 24 KiB/token`, before indexer/state overhead. Prior analysis estimates **24.75 KiB/token including indexer keys**. Eight sequences of 131,072 tokens imply about **24.75 GiB across the model** for this component alone; rank placement, grouping, and padding require separate accounting.

## 3. Q1 — Workload and concurrency: what limits performance?

### Main batch grid: 16K input, 256 output

| Concurrency | DeepSeek tok/s | GLM tok/s | Qwen tok/s | DeepSeek TPOT ms | GLM TPOT ms | Qwen TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 85.0 | 96.3 | 106.2 | 7.9 | 7.1 | 7.0 |
| 4 | 219.8 | 225.3 | 256.8 | 10.7 | 10.9 | 9.6 |
| 16 | 330.9 | 359.2 | 420.2 | 32.4 | 33.6 | 26.3 |
| 64 | 389.4 | 447.1 | 517.7 | 148.5 | 129.5 | 110.1 |

Source: `batch_isl16k_c{1,4,16,64}.json` in the three main batch directories in §1. Qwen at concurrency 64 is **1.16× GLM and 1.33× DeepSeek**. It leads this observed throughput grid; no accuracy or equal-text comparison was performed.

### Finer concurrency grid

| Concurrency | DeepSeek tok/s | DeepSeek TPOT ms | GLM tok/s | GLM TPOT ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 92.9 | 7.9 | 96.3 | 7.1 |
| 2 | 151.9 | 8.6 | 122.2 | 8.5 |
| 4 | 214.8 | 11.1 | 225.3 | 10.9 |
| 8 | 281.1 | 16.2 | 296.9 | 18.1 |
| 16 | 323.4 | 33.7 | 359.2 | 33.6 |
| 32 | 363.1 | 72.7 | 406.4 | 68.6 |
| 48 | 357.5 | 120.2 | 432.6 | 100.5 |
| 64 | 384.9 | 152.1 | 447.1 | 129.5 |

Sources: GLM `bf16kv` and DeepSeek `util085-dev20073`. DeepSeek's curve is a separate arm from the main comparison. GLM points span sessions, adding temporal variability.

Concurrency 8→64 buys **1.51× throughput for 7.15× TPOT on GLM**, and **1.37× for 9.41× on DeepSeek**. This suggests a latency-sensitive operating region around 8–16, not a universal optimum. Qwen lacks the finer grid. DeepSeek's small dip at 48 has no established mechanism or repeat-based significance.

### Context sweep at concurrency 8

| Input tokens | DeepSeek tok/s | GLM tok/s | Qwen tok/s* | DeepSeek cache peak | GLM cache peak | Qwen cache peak* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 104.5 | 295.9 | 322.7 | 18.1% | 10.2% | 7.2% |
| 65,536 | 47.9 | 104.3 | 114.5 | 21.9% | 36.6% | 28.2% |
| 131,072 | 34.7 | 47.2 | 56.5 | 27.0% | 72.5% | 56.1% |
| 260,000 | 16.7 | 26.5 | 25.1 | 36.9% | 89.5% | 97.3% |

Source: `ctx_isl{length}_c8.json` in §1's context arms. *Qwen uses the earlier smaller `base` pool, not corrected `base-util082`; rerun before treating its cache limit as architectural.*

At 131K, median TTFT is **19.84 s Qwen, 12.89 s GLM, 26.66 s DeepSeek**. The throughput leader is not the first-token-latency leader. High occupancy indicates headroom risk, not proof of preemption or a capacity-caused slowdown. Pool sizes and cache semantics differ across models.

DeepSeek's 16K/context point is much slower than its 16K/concurrency-8 point in the separate finer-grid arm (**104.5 versus 281.1 tok/s**). Build, utilization, and session differ; the batch-only bridge does not explain this discrepancy. Flag it for a same-session rerun, and avoid using it to identify an architectural bottleneck. Context throughput also includes extra input work; it is not a decode-only scaling curve.

### Bottleneck attribution: observation versus explanation

| Regime | Observation | Candidate mechanism [H] | Required check |
| --- | --- | --- | --- |
| Low concurrency | Low aggregate throughput; lower TPOT | Small GEMMs, weight/state traffic, launches, collectives | Decode operator trace and hardware counters |
| High concurrency | Diminishing gains; rising TPOT | Scheduling, compute/communication contention | Actual per-step batches; TP/EP sweep |
| Long input | Higher TTFT; lower output rate | Input work, attention/indexing, queueing | Separate prefill/decode timing; chunk-size sweep |
| High cache occupancy | Little pool headroom | Allocation pressure or preemption | Preemption counters; matched pool-size A/B |

GLM's available engine estimates report roughly **319–608 GB/s/GPU**, or 9.5–18.1% of an assumed 3,350 GB/s peak. They include FFN and unembedding but **omit attention**, and span different utilization arms. They are [E], not hardware-counter measurements. A partial whole-request average cannot rule out bandwidth-bound individual kernels.

The older estimate `active_weight_bytes × output_tok/s ÷ concurrency` is unsuitable: tokens can select different experts, live batch differs from client concurrency, and output throughput includes prefill. Under simplified independent uniform routing, distinct experts scale as `E × [1 − (1 − k/E)^B]` for B tokens. Real routing and reuse need measurement. Total/active parameters alone do not predict the bottleneck.

## 4. Q2 — Prefix caching: observed effects and implementation challenges

| Distinct 64K prefixes among 64 requests | DeepSeek tok/s | GLM tok/s | Qwen tok/s* |
| ---: | ---: | ---: | ---: |
| 1 | 198.2 | 265.3 | 453.9 |
| 4 | 391.6 | 365.9 | 398.1 |
| 16 | 114.6 | 199.4 | 262.2 |

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
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96.3 | 119.7 | 1.24× | 120.8 | 1.25× |
| 4 | 225.3 | 239.8 | 1.06× | 230.7 | 1.02× |
| 16 | 359.2 | 344.7 | 0.96× | 342.1 | 0.95× |
| 64 | 447.1 | 441.9 | 0.99× | 408.3 | 0.91× |

Sources: GLM `bf16kv`, `bf16kv-mtp-n1`, `bf16kv-mtp-n5`. At concurrency 1, n=1 reduces median TPOT **7.06→5.05 ms**. At 64, n=5 has **99.2% cache occupancy** and TTFT **10.72 s versus 2.51 s** for base. More draft tokens are not justified by throughput in these loaded cases.

### GLM context A/B: latency and throughput diverge

| Input tokens | Base → MTP tok/s | Throughput ratio | Median TTFT base → MTP | TTFT change |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 295.9 → 314.8 | 1.064× | 2.63 → 2.10 s | −20.1% |
| 65,536 | 104.3 → 97.6 | 0.936× | 7.21 → 5.43 s | −24.8% |
| 131,072 | 47.2 → 45.3 | 0.958× | 12.89 → 10.73 s | −16.7% |
| 260,000 | 26.5 → 22.2 | 0.839× | 38.48 → 41.07 s | +6.7% |

Source: [GLM MTP context](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/). The 131K point offers a possible latency/cost tradeoff. At 260K both metrics worsen and cache occupancy reaches 96.9%. Scheduling changes may explain earlier first tokens [H]; MTP does not directly eliminate input work, and no timeline proves the mechanism.

### Other models: weaker controls

| Concurrency | DeepSeek historical off → no-reuse MTP tok/s | Ratio | Qwen original base → MTP tok/s | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 92.3 → 115.3 | 1.25× | 107.7 → 125.9 | 1.17× |
| 4 | 222.0 → 251.8 | 1.13× | 162.2 → 149.3 | 0.92× |
| 16 | 293.7 → 339.9 | 1.16× | 371.0 → 322.2 | 0.87× |
| 64 | 383.4 → 380.2 | 0.99× | 517.8 → 481.5 | 0.93× |

Sources: Qwen `base`/`mtp-n1`; DeepSeek `mtp-off`/`mtp-on-noreuse`. Qwen's equal-utilization pair retains startup/pool confounds; substituting corrected `base-util082` would also change utilization. DeepSeek's historical pair uses the older runtime and lacks newer cold-run fields. Do not substitute newer baselines or call this a fully controlled three-model A/B. Real-text acceptance and answer quality remain untested.

## 6. Q4 — Serving cost with explicit assumptions

For GPU-hour price p, GPU count G=8, and output throughput T:

```text
GPU-seconds/output token = G / T
$/million output tokens = G × p × 1,000,000 / (3,600 × T)
```

Assume **$2.50/GPU-hour**, or $20/node-hour, for illustration. This is not a current quote. The formula charges the entire measured workload, including input processing, to output tokens. It is neither a decode-only cost nor an API tariff.

| Workload | DeepSeek $/million output | GLM $/million output | Qwen $/million output |
| --- | ---: | ---: | ---: |
| 16K input, concurrency 1 | 65.33 | 57.68 | 52.29 |
| 16K input, concurrency 64 | 14.27 | 12.43 | 10.73 |
| 131K input, concurrency 8 | 160.26 | 117.59 | 98.38* |

Costs [A] use unrounded JSON throughput. *Qwen long-context caveat applies.* Scale by `actual_GPU_hour_price / 2.50` for another price assumption. GLM at concurrency 64 costs about **$0.00318 per request** with 256 output tokens, including its 16K input work under this allocation model.

At concurrency 64, throughput/GPU is **64.72 Qwen, 55.88 GLM, 48.67 DeepSeek tok/s/GPU**. This normalizes eight-GPU runs; it does not predict one-GPU performance. There is no measured optimal GPU count, energy efficiency, total ownership cost, or cost per successful task. Tokenizers and quality differ.

A deployment decision should compare **goodput under explicit TTFT/TPOT objectives**. The cheapest token in the table may violate an interactive latency target. Production cost also needs realistic arrival rates, repeats, equivalent tasks, and idle-time accounting.

## 7. Additional systems findings

### FP8 KV: backend and dtype must be separated

| GLM arm, concurrency 64 | Output tok/s | Interpretation |
| --- | ---: | --- |
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

### A falsifiable research direction: budget speculation and hybrid state together

**Hypothesis [H].** A simple draft-budget policy conditioned on live load and state pressure can improve useful throughput over fixed speculation settings for hybrid models. The motivation is the GLM light-load/loaded MTP reversal and the sensitivity of available resources to Qwen startup state. Neither observation establishes that such a policy will win; related adaptive approaches must be reviewed before claiming novelty.

| Research step | Concrete design | Acceptance / falsification |
| --- | --- | --- |
| Establish valid baselines | Repair §6 harness gaps in fix_bug.md; immutable matched startup/pools; randomized repeats | Keep the same retention criteria for all policies, including slow runs |
| Explain the round | Record accepted/committed tokens, draft/verify/state-copy time, live batches, preemptions and per-request timelines | Determine whether overhead or acceptance explains each loss; absent counters remain unknown |
| Intervene | Compare MTP off, fixed n1/n5 and an explicitly specified adaptive draft budget | Same model, precision, traffic, cache protocol, quality checks and latency objectives |
| Evaluate | Completed requests meeting **both** TTFT and TPOT targets per unit time; tails, failures, policy overhead and memory | A median below a target is not the fraction of requests meeting it |
| Falsify | Compare against the strongest fixed policy within each matched workload | No useful gain after controls, or gains erased by policy/state overhead, reject this hypothesis for that regime |

Targets must be specified before comparing policies; they are not inferred from the best-looking curve. This is proposed work, not an implemented controller, a replay result or a demonstrated new contribution. It turns the present uncertainty into a reviewable systems experiment.

| Priority | Experiment | Question resolved |
| --- | --- | --- |
| 1 | Matched-startup Qwen base/MTP/context/prefix; same-session DeepSeek context rerun | Which rankings survive controlled pools and startup? |
| 2 | Randomized repeated runs with longer windows | Are small differences reproducible? |
| 3 | Profile prefill, decode, GEMMs, dispatch, attention, hardware memory traffic | What actually limits each regime? |
| 4 | Sweep chunked-prefill budget and feasible TP/EP layouts | Can goodput improve at fixed latency objectives? |
| 5 | Cache-off, cold-fill, and prewarmed-prefix controls with state/eviction/preemption counters | What savings come from reuse, and what does hybrid state cost? |
| 6 | Representative task/session replay with real text and quality evaluation | What is cost per successful task and realistic MTP acceptance? |

Saved TraceLab summaries motivated the input-length grid; no real trace replay was performed. Do not interpret reusable-prefix estimates as guaranteed achievable cache hits. No dense baseline, multimodal test, disaggregated deployment, or validated layer-reduction extrapolation is included.

The document audit also found remaining harness gaps: unavailable cache telemetry can default to zero, positive cache-hit warnings do not reject a point in that branch, and completion validation rejects zero rather than all partial failures. The retained JSON audit checks saved fields; it cannot independently prove successful original telemetry. These are documented as **pending**, not fixed, in [fix_bug.md §6](fix_bug.md#6-remaining-gaps-in-the-saved-harness--not-fixes-performed-here).

## 9. Professor questions: answers to rehearse

| Question | Defensible answer |
| --- | --- |
| What did you establish? | Named deployment tradeoffs, GLM MTP/backend A/Bs, and measurement confounds. |
| Which model is best? | Qwen leads corrected 16K throughput; GLM has lower observed 131K median TTFT. Quality and optimal deployments are unmeasured. |
| Which part is the bottleneck? | I have hypotheses, not operator attribution. Separate prefill/decode traces and counters are next. |
| Why not claim bandwidth is low? | Partial modeled counters over mixed request work cannot rule out bandwidth-bound kernels. |
| Is concurrency batch size? | No; it caps client requests. Engine token batches vary each step. |
| Is the comparison fair? | GPU class/count and workload targets match; precision, tokenizers, backends, builds, and some pools differ. This compares deployments. |
| Does the bridge remove engine confounds? | Only a small DeepSeek batch-grid build effect was observed; other models and axes remain uncontrolled. |
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
