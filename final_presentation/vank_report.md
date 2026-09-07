# DeepSeek, GLM and Qwen: architecture and serving comparison

Tan Ngo · For Professor Kan Zhu and UW SyFI · Final synthesis: 6 September 2026 · Revision 2 (refined)

Assignment: [task.md](task.md). Presentation: [editable PowerPoint](SyFI_ML_Serving_refined.pptx), [storyboard](slides.md), and [complete speaking script](script.md). Architecture-to-debugging study guide: [fix_bug.md](fix_bug.md). Companion architecture survey (broader scope, includes Kimi K2→K3): [reference_report_v1_standalone.md](reference_report_v1_standalone.md).

**Research question:** when do architectural reductions in computation or state translate into useful serving performance? The report has two connected parts: frontier design and its testable implications (§§1–2), then local measurements, diagnosis and a proposed research direction (§§3–9). Published training results and mechanisms are distinguished from this study's inference measurements.

### What changed in this revision

No measured value, arm, or provenance entry was altered. The revision adds: (1) a first-party architecture matrix with layer, head, indexer, draft-module and precision facts for all three checkpoints (§2.1–2.3, §2.10); (2) an analytical bytes-per-step and prefill-rate model that is reconciled against the local grids, including the observation that the 16K-input grid and the context sweep are input-dominated (§3.5–3.7); (3) a mechanism-level account of what prefix caching must store for recurrent, compressed and indexed attention, with a documented engine failure mode to rule out for GLM (§4.3–4.6); (4) draft-module cost accounting and published speculative-decoding results that frame the local MTP A/Bs (§5.4–5.6); (5) API-price context and a per-request cost comparison that includes input work (§6.3–6.4); (6) a related-work review that the §8 research direction previously called for (§8.2). New material is tagged [A] when derived from configuration or local JSON, [P] when it comes from published external measurements or vendor documents, and [H] when it is a candidate explanation that still needs a discriminating test.

## Executive findings

This study compares **DeepSeek-V4-Flash, GLM-5.3-Flash, and Qwen3.8-Flash-Next-FP8** on an eight-GPU H100 configuration. These three checkpoints are the complete measured scope. Tables and presentation comparisons follow **DeepSeek → GLM → Qwen**; model colors are orange, purple and teal respectively. The central systems lesson is that serving behavior depends on workload, cache state, backend, and scheduling as well as architecture.

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
7. **The measured grids are input-dominated, and the architectures were built for exactly that regime.** On the 16K/256 grid, input tokens outnumber output tokens 64:1; in the context sweep the ratio reaches 1,000:1. Derived marginal input-token rates (§3.6) stay roughly flat from 64K to 260K for GLM (22–30K tok/s) and fall for DeepSeek (32K → 16K tok/s), which is the direction the vendors' prefill-scaling claims predict but not yet a phase-timed confirmation [A/H]. At concurrency 1 all three decode 10–30× slower than an HBM-bandwidth floor computed from their active weights, so batch-1 decode on this build is launch/latency-bound, not bandwidth-bound [A].

### What decision does each finding support?

| Objective | Evidence to use | Decision supported now | What could change it |
| --- | --- | --- | --- |
| Maximize output rate on this 16K grid | Corrected main batch arms, §3 | Qwen is the observed throughput leader | Different quality target, workload, runtime or repeated results |
| Reduce waiting at long input | 131K/c8 median TTFT, §3 | GLM is the observed first-token-latency leader | Matched Qwen startup; phase timing; larger samples |
| Tune interactive concurrency | Finer GLM/DeepSeek curves, §3 | Evaluate the latency cost of moving beyond c8–16 | Actual per-request latency targets and arrival traffic |
| Decide whether to speculate | GLM paired MTP arms, §5 | Test n1 at light load; avoid assuming longer drafts help | Real-text acceptance, phase costs, quality and state pressure |
| Increase history capacity | GLM alternate BF16/FP8 pair, §7 | Price additional capacity against its speed loss on Hopper; expect the opposite sign on Blackwell [P] | Validated kernels, numerical parity and matched startup |
| Choose between self-hosting and an API | §6.4 per-request comparison | On this prefill-heavy grid an 8×H100 node at $2.50/GPU-hour is near API list-price parity only when input tokens are priced; cache-hit pricing tips agentic traffic toward the APIs | Real cache-hit rates, utilisation, Blackwell hardware, promo expiry |

The result is an **operating-point map**, not a universal model ranking. Most observations have one retained run; the table describes evidence-supported choices for further evaluation, not production recommendations.

## 1. Scope, evidence, and method

### Evidence convention

- **[M] Measured:** raw request metrics or recorded server observations.
- **[A] Analytical:** calculations, configuration facts, or prior tensor accounting.
- **[E] Engine estimate:** modeled FLOPs/bytes combined with observed scheduling activity.
- **[H] Hypothesis:** an explanation requiring another experiment.
- **[P] Published:** vendor documents, serving-engine cookbooks, or independent benchmarks cited inline. Hardware, software version, workload and metric definitions differ from the local runs; [P] values contextualise local results and never substitute for them.

Result tables are [M]; derived ratios and costs are [A] from [M] inputs. Historical reports are supporting notes, not independent experiments. [report_previous.md](archive/notes/report_previous.md) preserves the superseded synthesis; the present report corrects several of its causal claims and comparisons.

One metric caution applies to every [P] throughput figure: engine cookbooks and benchmark sites do not agree on what "tokens per second per GPU" counts. The SGLang cookbook's `tokens_per_sec_per_gpu` counts prompt plus output tokens (its GLM-5.3-Flash notes give the aggregate output figure separately); SemiAnalysis InferenceX's AgentX traces count prefix-cached prompt tokens; Artificial Analysis reports output tokens per second per user on hosted APIs. Where a [P] figure is quoted below, its definition is stated, and it is not placed in the same column as a local [M] value.

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

**Input-to-output ratio [A].** Every point on the main grid moves 16,384 input tokens for 256 output tokens, so a run's wall time is `total_input / prefill_rate + decode_time`. Because output throughput divides only the 256-token outputs by that wall time, the metric is dominated by input processing whenever prefill time exceeds decode time, which §3.5 shows is the case at concurrency 16 and above and throughout the context sweep. This does not invalidate the comparisons; it identifies what they compare.

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

### 2.1 Three-model architecture matrix

The first four rows repeat the local tensor accounting; the remaining rows are first-party configuration facts [A/P] added in this revision, with the source noted per row. Vendor "active" labels differ from the local GEMM-parameter definition (see §2.10).

| Comparison dimension | DeepSeek-V4-Flash | GLM-5.3-Flash | Qwen3.8-Flash-Next-FP8 |
| --- | --- | --- | --- |
| Attention organization | 43 sparse-attention layers with compressed history | 34 recurrent KDA + 11 sparse-attention layers | 36 recurrent GDN + 12 QSA layers |
| State that must be retained | Compressed history, indices and local/tail state | Recurrent checkpoints plus retained sparse-attention history | Recurrent checkpoints plus retained QSA history/index state |
| Routed experts / selected per token | 256 / 6 | 288 / 8 | 512 / 10 |
| Prior total / active GEMM parameters | 290.91B / 14.08B | 321.34B / 17.38B | 176.94B served / 7.27B |
| Main measured weights / cache | MXFP4 experts + FP8 attention/dense / FP8 KV | FP8 weights / BF16 KV | FP8 weights / BF16 KV |
| Layer layout (first-party) | 43 layers, hidden 4,096; layers 0–1 sliding-window only, then Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) interleaved; all layers MoE, first 3 MoE layers hash-routed ([tech report §4.2.1](https://arxiv.org/pdf/2606.19348); [NeMo config notes](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/deepseek-v4-flash)) | 45 layers, hidden 4,096; strict 3:1 rhythm (DSA at layers 3, 7, … 43), first 3 layers dense FFN, 42 MoE layers ([tensor teardown](https://kgptalkie.com/tutorials/llm-benchmarking/glm-5-3-vs-glm-5-3-flash-architecture-teardown); [NeMo](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thudm/glm-5-3-flash)) | 48 layers, hidden 2,560; 12 × (3 GDN + 1 QSA) with an MoE after every block ([architecture report](https://arxiv.org/html/2608.30320); [spec sheet](https://intuitionlabs.ai/articles/qwen3-8-flash-next-architecture-memory)) |
| Global-attention path | Single KV head broadcast to 64 query heads (head dim 512, query compression 1,024, grouped output projection); CSA compresses 4 tokens per KV entry with an overlapping compressor, a 64-head × 128-dim lightning indexer picks top-512 compressed entries (≈2,048 raw tokens); HCA compresses 128 tokens per entry and attends densely; every layer keeps a 128-token sliding window (tech report §4.2.1; [HF blog](https://huggingface.co/blog/deepseekv4)) | MLA with a 512-wide latent, NoPE (`qk_rope_head_dim: 0`); DSA indexer with 32 heads × 128 dims selects top-2,048 tokens; IndexPool/KPool compresses 4 indexer keys into 1 and always keeps the tail (teardown; [Z.ai](https://docs.z.ai/guides/vlm/glm-5.3-flash)) | GQA with 24 query / 2 KV heads at head dim 256, RoPE; QSA scores 4-token micro-blocks with a 4-query-head MQA indexer (partial RoPE on 64 of 128 dims) and selects 512 blocks = 2,048 tokens plus the tail block (architecture report §2.1.2) |
| Recurrent path | none | KDA: 64 heads × 128, conv-4, channel-wise decay; per-layer state 64 × 128 × 128 (teardown) | Gated DeltaNet: 48 value heads / 16 QK heads × 128; sigmoid output gate (architecture report §2.1.1) |
| Residual stream | 4-stream manifold-constrained hyper-connections (mHC), 20 Sinkhorn iterations (tech report) | 4-stream mHC, 20 Sinkhorn iterations, 0.011% of parameters; norms retained (teardown) | 4-branch Gated Residual; branch-mixing operator dropped at decode; residual state FP8-storable (architecture report §2.2) |
| Draft module | Preview checkpoint: MTP depth 1 (sliding-window-only block); 0731 checkpoint: fused DSpark draft, no MTP head ([vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)) | One "NextN" layer: a full block with its own 288-expert MoE and its own indexer, 7.43B parameters, reusing the last index pick (`index_share_for_mtp_iteration`), single residual stream (teardown) | 4B single-layer MTP trained multi-step; QSA inside the draft; top-k indices reused across draft steps (architecture report §2.1.2) |
| Positional encoding | Dual RoPE bases: θ=10,000 on SWA layers, θ=160,000 with YaRN on compressed layers (NeMo) | NoPE in attention; order carried by KDA convolutions; indexer keys still RoPE'd (teardown) | RoPE retained; NoPE rejected after post-training showed endless generation (architecture report §2.1.1) |
| Precision kept high | attention/router/norms FP8; KV FP8 with BF16 RoPE dims; indexer FP4 on Blackwell (HF blog; vLLM recipe) | entire KDA stack BF16, `A_log`/`dt_bias` FP32, `kv_b_proj` BF16 (teardown) | n-gram table not below 4-bit in community quants ([Unsloth](https://unsloth.ai/docs/models/qwen3.8-next)) |
| Published optimizer information used here | Mixed Muon + AdamW | Recipe not established by the reviewed sources | Mixed Muon + AdamW |
| Local MTP evidence | Older off/on-noreuse pair | Base versus n1/n5; paired context n1 | Original base versus n1, pool/startup confounded |
| Serving question to test [H] | Do compression savings exceed indexing and tail-management work? | When do recurrent-state and speculative costs offset reduced history work? | How much does the smaller GEMM path contribute after matching startup and kernels? |

Configuration and prior tensor sources are linked under "Checkpoint organization and retained evidence" below; optimizer and MTP qualifications are explained in their dedicated sections. Qwen's 51.23B n-gram table is counted in served capacity but is not a full per-token matrix multiplication. "Not established" is a documentation limit, not a claim that GLM uses a particular alternative optimizer.

**The main architectural contrast:** DeepSeek compresses the history representation throughout its attention stack; GLM and Qwen combine recurrent state with history retrieval. **The main experimental contrast:** all three have main batch measurements, but the strongest local MTP controls belong to GLM. Keep those two comparisons separate.

**What the three share [A].** All three are sparse MoEs with 5–7% of weights active per token, all support 1M-token contexts (Qwen natively 262K, 1M via YaRN), all widen the residual stream to four branches, all ship a native draft module, and all restrict full-context attention to a small subset of the stack: 11 of 45 layers (GLM), 12 of 48 (Qwen), and — for DeepSeek — every layer, but over a history compressed 4× or 128× and, in CSA layers, pruned to a fixed 2,048-token budget. The differences are in *what remains per token or per request* once history work is reduced, and that is where serving behaviour diverges (§2.3, §4).

### 2.2 A design map: remove work, account for the replacement

| Frontier mechanism | Work or resource it targets | New obligation / possible loss | Relevant test |
| --- | --- | --- | --- |
| MoE routing | Selected feed-forward arithmetic | Resident experts, dispatch/combine, load imbalance | Expert GEMM and collective timing versus live tokens per expert |
| Sparse attention | Attention over irrelevant history positions | Index scoring, top-k, irregular gather; retrieval quality | Context-length crossover with a matched attention control |
| Compressed history | Stored/read history representation | Compression, indexing and incomplete tails | Bytes by cache group, phase time, exact prefix restoration |
| Recurrent/attention hybrid | Sequence-growing state in recurrent layers | Per-sequence state and checkpoint/rollback management | Active-sequence capacity separately from token capacity |
| Index pooling / index sharing (GLM KPool; GLM-5.3 shares one index across 4 layers; Qwen 4-token micro-blocks) | The indexer's full-context scan, the last O(S²) term | Pooled keys must be recomputed at unaligned boundaries; shared picks assume cross-layer similarity (Qwen reports this limits IndexShare in hybrids [P]) | Indexer-latency share of prefill versus context length |
| Widened residual (mHC / Gated Residual) | Signal propagation and training scaling, not inference speed | 4× residual activation traffic; drafters must read a multi-stream hidden state | No inference speed test is meaningful; check draft acceptance per position |
| Off-accelerator n-gram table (Qwen) | Capacity without per-token FLOPs | Host-memory lookup latency and prefetch; 51 GB of host or device memory | Prefill/decode with table on GPU versus host versus SSD |
| Native MTP | Serial target decode iterations per committed token | Draft/verify work, rejected state and scheduler overhead | Committed tokens per round and phase time versus load |
| Muon / mixed optimizer recipe | Training convergence and update geometry | Orthogonalization, partitioning and stability engineering | Quality reached per training GPU-hour; no local training A/B |

This is a mechanism map [A], not an operator attribution for the local runs. The key question is **which cost shrinks, which cost replaces it, and at what workload does the saving exceed the replacement?**

### 2.3 What each attention path stores and reads [A]

The three designs delete different terms from the per-step cost. Per-token and per-request quantities below are derived from the configuration facts in §2.1; the arithmetic is shown so it can be checked against the config files.

| Quantity | DeepSeek-V4-Flash | GLM-5.3-Flash | Qwen3.8-Flash-Next-FP8 |
| --- | --- | --- | --- |
| Sequence-growing history per token | ≈ 2.5–3.5 KB (vendor: ~7% of V3.2's ≈ 35 KB/token FP8 at 1M) | 11 layers × 512 latent + 11 × 128/4 pooled index = 5,984 elements → 6.0 KB all-FP8, 11.6 KB with BF16 latent (teardown) | 12 layers × 2 KV heads × 256 × (K,V) = 12,288 elements → 24.0 KiB BF16 + ~0.75 KiB indexer keys (local prior analysis: 24.75 KiB) |
| Fixed per-request state | 128-token sliding window per layer (negligible) | 34 × 64 × 128 × 128 = 35.65M elements → 71 MB BF16, ~142 MB if the engine keeps the temporal state in FP32 | 36 × 48 × 128 × 128 = 28.3M elements → ~57 MB BF16, ~113 MB FP32 |
| History bytes *read* per decode token at 128K | Index scan over 128K/4 compressed entries (FP4 keys on Blackwell) + top-512 entries + HCA over 128K/128 entries ≈ tens of MB | 128K × 352 B pooled-index bytes + 2,048 × 5.6 KB selected ≈ 57 MB | 128K × ~768 B index + 2,048 × 12 KB selected ≈ 120 MB |
| Full-context term in prefill | CSA indexer O(S²/4) in ~half the 43 layers (64 heads); HCA dense O(S²/128) in the other half; compressors | DSA indexer O(S²/4) in 11 layers (32 heads); KDA chunked scan O(S) in 34 layers | QSA indexer O(S²/4) in 12 layers (4 heads); GDN chunked scan O(S) in 36 layers |
| Draft-module active cost per drafted token | Preview MTP: one sliding-window block (small); 0731 DSpark: ~7 GB fused draft module, block draft of 7 | ≈ 0.4B active parameters (8 of 288 experts + shared + attention of a 4,096-wide block) although the layer stores 7.43B; no extra index scan because the pick is shared | ≈ 0.1–0.2B active (10 of 512 experts at dim 640 + attention); indices reused across steps |

Reading the table. (1) For all three, the *bandwidth* cost of history at decode time is tens of megabytes per request even at 128K, because sparse selection caps the entries read; only the indexer scan grows with context, at roughly 0.3–0.8 KB per context token. Long context therefore costs **capacity** (0.4 GB, 1.5 GB and 3.2 GB per 128K request for DeepSeek, GLM and Qwen at their measured KV precisions) rather than per-step bandwidth. (2) The two recurrent designs add a per-request cost that does not depend on context: one GLM checkpoint costs as much memory as ~6K tokens of its BF16-latent KV, one Qwen checkpoint as much as ~2.3K tokens. (3) In prefill, the remaining super-linear term is the indexer, and the three differ by an order of magnitude in how many indexer-heads × layers they run: roughly 20 CSA layers × 64 heads for DeepSeek, 11 × 32 for GLM, 12 × 4 for Qwen, all over a 4×-compressed key sequence, plus DeepSeek's HCA dense term. This ordering is a candidate explanation for the DeepSeek context-sweep behaviour in §3.6, not a demonstrated one [H].

### 2.4 Sparse attention: selected computation is only one term

For sequence length S and K selected history entries per query, dense prefill attention-score work scales as `O(S²)`; the selected attention component scales roughly as `O(SK)`. That is not the total sparse path:

```text
Sparse-path time = index construction/scoring + selection + gather
                 + attention on selected entries + state management
```

DeepSeek V4 combines compressed sparse attention with heavily compressed attention and a local window; Qwen QSA uses a compressed lightweight indexer. These designs reduce different parts of history work. In particular, compressing an indexer's candidate sequence by a fixed factor does not make all index scoring constant-time: Qwen's own ablation reports that QSA matches full attention at a relative indexer latency of 0.25, and that its kernel-level attention speed-up over dense GQA at 1M tokens is 7.6× in prefill and 4.9× in decode with three MTP steps — with gains beginning only above 64K context [P]. GLM's launch material reports 3.0× less attention compute and 4.4× less KV cache than the 753B GLM-5.3; a tensor-level derivation gives 8.0× on KV per token, and the two have not been reconciled [P]. [DeepSeek V4 report, §2.3](https://arxiv.org/html/2606.19348v1), [Qwen architecture report, §2.1.2](https://arxiv.org/html/2608.30320v1), [Z.ai overview](https://docs.z.ai/guides/vlm/glm-5.3-flash)

**Prediction [H]:** long contexts offer more opportunity to amortize indexing and selection, while short contexts may favor highly fused dense kernels. **Local evidence [M]:** §3 measures context-sensitive request behavior, and §3.6 derives marginal input-token rates from it. **Missing discriminating test:** a matched dense/sparse or indexer intervention with phase timing and retrieval-quality checks. Neither a lower cache occupancy nor a higher output rate alone measures sparse-attention speedup.

### 2.5 Muon versus AdamW: training efficiency has a different causal path

AdamW uses elementwise first/second-moment adaptation with decoupled weight decay. Muon applies approximate orthogonalization to a matrix momentum update, typically using Newton–Schulz iterations. The distinction is update geometry, not a smaller inference matrix. [AdamW paper](https://arxiv.org/abs/1711.05101), [Muon author explanation](https://kellerjordan.github.io/posts/muon/)

| Question | Precise answer |
| --- | --- |
| Does Muon replace AdamW everywhere? | No. The reference implementation explicitly separates suitable matrix weights from auxiliary AdamW parameters. Qwen's report states the split: Muon for two-dimensional weights that act as linear maps; AdamW for embeddings, the n-gram table, the output head, the MoE router and the Gated-Residual low-rank projections, with Newton–Schulz fixed at 8 steps for stability [P]. |
| What do the three models' sources establish? | DeepSeek and Qwen use mixed parameter-group assignments. GLM-5.3-Flash's optimizer recipe is not established by the sources used here (the GLM-5 technical report, arXiv:2602.15763, is the place to check); do not infer it from the other models. |
| What is the systems challenge? | Matrix orthogonalization changes optimizer computation and how matrices should be partitioned/batched. Qwen repartitions the data-parallel gradient buffer by estimated orthogonalization cost and captures the fragmented optimizer step in a CUDA graph [P]. |
| Can Muon explain the local inference ranking? | Not directly: optimizer updates are absent from inference. A training recipe can affect learned quality and feasible model design, but this study has no training or equal-quality control. |

Sources: [Muon reference implementation](https://github.com/KellerJordan/Muon), [DeepSeek V4 optimizer and framework sections](https://arxiv.org/html/2606.19348v1), [Qwen official release](https://qwen.ai/blog?id=qwen3.8-flash-next), [Qwen architecture report §3.1](https://arxiv.org/html/2608.30320). These primary references were consulted on 6 September 2026. The appropriate evaluation is quality reached per training resource budget, including optimizer overhead, rather than treating "trained with Muon" as a measured serving acceleration.

### 2.6 Native MTP and "zero-day" speculative decoding

Here **zero-day** is descriptive: native draft capability and serving support available around release. It is not a separate mathematical algorithm or a promise of zero overhead. A model-provided MTP module can remove the need to wait for a separately trained external draft; support and optimized execution still depend on the engine version. DeepSeek documents MTP as both a training component and a potential inference drafter. SGLang's Qwen launch implementation illustrates additional runtime work, including sharing selection information across draft steps. These external implementation disclosures do not describe the exact saved local vLLM runs. [DeepSeek V3 repository](https://github.com/deepseek-ai/DeepSeek-V3), [SGLang Qwen day-0 support](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)

What the three checkpoints actually ship, from first-party sources [P]:

| | Draft mechanism | What it costs per drafted token [A] | Published acceptance / gain |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash | Preview: MTP depth 1, a sliding-window-only block. 0731: the MTP head is removed and a fused **DSpark** module (DFlash-style parallel drafter + low-rank Markov head + trained confidence head) drafts 7 tokens per step with confidence-scheduled verification ([DSpark paper](https://arxiv.org/html/2607.05147v1), [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)) | Preview MTP: small; DSpark: one parallel draft pass for the whole block | DeepSeek production: 60–85% faster per-user generation than MTP-1 at matched throughput; a community single-stream run went 26.3 → 39.9 (MTP-1) → 60.3 tok/s (DSpark) ([VentureBeat](https://venturebeat.com/orchestration/deepseek-open-sources-dspark-a-new-framework-to-speed-up-llm-inference-by-up-to-85)) |
| GLM-5.3-Flash | One NextN layer: a full block with its own 288-expert MoE and indexer; shares the last index pick for the draft iteration | ≈ 0.4B active parameters per drafted token, i.e. roughly 2–3% of a target step's active weights, plus the fixed launch cost of one extra layer forward, LM head and verification | SGLang's adaptive MTP recipe (5 steps / top-1 / 6 draft tokens) on 4× GB300: 1.57× aggregate output throughput at concurrency 16 versus spec-off, with accept length pinned at 3.0 by simulation; DCP arms measured 3.9 real accept length ([SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash)) |
| Qwen3.8-Flash-Next | 4B single-layer MTP trained with a multi-step objective; QSA inside the draft; indices reused across steps | ≈ 0.1–0.2B active per drafted token | Mean accepted length 4.06 under four-step speculation (MT-Bench 3.44, GSM8K 4.19, MATH 4.29, HumanEval 4.24, MBPP 4.12), unchanged by index reuse ([architecture report Table 4](https://arxiv.org/html/2608.30320)) |

The protocol is **draft → target verification → commit valid progress / restore rejected state**. Preserving the target sampling distribution requires the appropriate acceptance and residual-resampling algorithm, together with correct runtime state; simply verifying a candidate list is insufficient. This theoretical property does not establish numerical parity of the saved experimental backends. [Speculative decoding paper](https://arxiv.org/abs/2211.17192)

```text
Speculative time / useful token ≈ E[round time] / E[committed tokens per round]
Round time includes drafting, verification, state operations and scheduling.
Benefit requires this ratio to beat the ordinary time / token at the same load.
```

The expression is a steady-work analytical lens [A], not a fit to these finite benchmarks. It predicts why acceptance alone cannot choose draft length, and why a policy that helps c1 may hurt at c64. Two model-specific terms belong inside "round time" for these checkpoints: for the recurrent models, the KDA/GDN state must be made rewindable (snapshot per draft step, or replay of stored raw inputs — SGLang's ReplaySSM cuts that draft-window memory ~32× for Kimi K3's KDA layers [P]); and for an MoE target, every verified position routes to its own experts, so verification increases the number of distinct experts streamed per step (§3.5). §5 provides the local test; a phase trace is still needed to explain the mechanism.

### 2.7 From an efficient component to an efficient request

An Amdahl calculation prevents kernel improvements from becoming unsupported end-to-end claims. If fraction f of original elapsed time is accelerated by r, while other work stays constant:

```text
Request speedup = 1 / [(1 − f) + f/r]
Illustration: f = 0.30 and r = 10 → 1 / (0.70 + 0.03) = 1.37×
```

The 30% fraction and 10× improvement are **illustrative assumptions**, not measured values. Real overlap, changing batch composition and queueing can invalidate the fixed-fraction model. Its purpose is to show why component efficiency must be connected to phase cost and then to the user objective.

**Why might these deployments be efficient?** MoE reduces selected arithmetic, recurrence/compression reduce portions of history work, and MTP can amortize serial decoding. **Why is Qwen fastest on the main grid?** Its selected GEMM path and hybrid organization are plausible contributors, but no ablation separates them from precision, kernels, startup and scheduling. The local data supports the ranking; it does not allocate the speed advantage among architectural causes. Two structural facts narrow the candidates without settling them [A]: Qwen streams the fewest weight bytes per token (≈ 6.4 GB versus ≈ 18 GB for GLM in FP8; §3.5) and runs the cheapest indexer (4 heads × 12 layers; §2.3), so it should lead in both a bandwidth-bound decode regime and an indexer-bound prefill regime. DeepSeek's small active count does not translate on H100 for two hardware-specific reasons: Hopper has no FP4 tensor cores, so its MXFP4 experts execute through weight-only dequantisation kernels rather than the native FP4 path used in published Blackwell results, and its FP4 indexer cache is documented for Blackwell only, so the indexer runs at higher precision here [A/P].

### 2.8 Checkpoint organization and retained evidence

All three measured checkpoints are **mixture-of-experts (MoE)** models. Routing activates a subset of experts per token, reducing arithmetic compared with activating all weights while retaining a large resident weight footprint and adding dispatch/combine communication.

| Model | Attention organization [A] | Routed experts / selected [A] | Serving implication |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash | 43 sparse-attention layers; compressed history | 256 / 6 | Smaller history representation, plus compression/indexing work |
| GLM-5.3-Flash | 34 recurrent KDA + 11 sparse-attention layers | 288 / 8 | Recurrent state plus sequence-growing attention cache |
| Qwen3.8-Flash-Next-FP8 | 36 Gated DeltaNet (GDN) + 12 Qwen sparse-attention (QSA) layers | 512 / 10 | Recurrent state, retained attention KV, and an n-gram lookup table |

Sources: local [DeepSeek analysis](deepseek_v4_flash/report.md), [GLM analysis](GLM-5.3-Flash/report.md), and [Qwen analysis](Qwen3.8-Flash-Next-FP8/report.md). Official configuration cross-checks: [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json), [GLM](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json), [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json). Online sources checked 5 September 2026; rolling sources do not replace recorded manifests.

**The similarity:** all use sparse expert computation. GLM and Qwen also mix recurrent and history-retaining attention inside one stack. **The difference:** recurrent layers store fixed-size state per sequence; attention retains sequence-growing history. Sparse attention can limit the history consulted without eliminating its storage. Compression reduces history size but adds work. These costs need separate terms.

Expert geometry differs more than the "routed / selected" row shows [A]: a DeepSeek or GLM expert is 3 × 4,096 × 2,048 ≈ 25M parameters, a Qwen expert 3 × 2,560 × 640 ≈ 4.9M. Under simplified independent uniform routing the fraction of experts touched per decode step, `1 − (1 − k/E)^B`, is 31% / 36% / 27% at 16 concurrent tokens and 78% / 84% / 72% at 64 for DeepSeek / GLM / Qwen; real routing is skewed, so these are pessimistic bounds (§3.5). By 64 concurrent decode tokens a node is streaming most of the expert set every step regardless of how few experts each token selects, which is why per-token cost keeps falling with batch until the whole expert set is read per step.

### 2.9 Connect architecture to debugging and results

| Feature | Runtime consequence | Evidence to study | Limit of the inference |
| --- | --- | --- | --- |
| Recurrent layers | State capacity also scales with active sequences | [Bug 1](fix_bug.md#bug-1--mamba-state-capacity-blocks-a-hybrid-model): sequence cap blocked startup | More available state does not guarantee higher throughput |
| Prefix caching | Warmup can change the measured workload | [Bug 6](fix_bug.md#bug-6--benchmark-warmup-creates-the-cache-hits-being-measured): 16,000 reused tokens | Reuse fraction is not throughput speedup |
| Cache geometry and precision | A dtype requires a compatible layout/kernel | [Bug 8](fix_bug.md#bug-8--one-incompatible-fp8-layout-is-not-universal-fp8-failure): failed route, successful alternate stack | Backend and dtype effects must be separated |
| Automatic pool sizing | Startup state changes capacity available for serving | [Bug 12](fix_bug.md#bug-12--a-plausible-architecture-story-explains-a-startup-confound): Qwen c4 changed 162.2→256.8 tok/s | The causal contribution of preemptions is not measured |
| Hybrid execution | Estimators need explicit operation coverage | [Bug 13](fix_bug.md#bug-13--enabled-counters-absent-collection-incomplete-accounting): missing components | Partial estimates cannot identify the hardware bottleneck |
| Recurrent-state checkpoint placement | Prefix reuse for KDA/GDN requires a checkpoint at the shared boundary; engines keep checkpoints only at aligned block boundaries | vLLM issue #45238 documents hit rates dropping to 0% when the aligned checkpoint lands in the request-unique suffix [P]; §4.4 shows the local prefix geometry is near that condition for GLM | Whether the local build was affected is unverified; per-group hit counters would settle it |
| Single-KV-head attention under TP | A latent/MQA KV cannot be split by head, so each TP rank may hold a full copy | GLM's recorded BF16 pool (1,916,967 tokens) is consistent with per-rank replication at ~11.6 KB/token [A]; §2.10 | Replication versus DCP is an engine layout choice, not an architectural constant |

Read these as a chain: **architecture → runtime requirement → observed failure/result → discriminating test → bounded conclusion**. That chain links the architecture survey to the experiments without assuming every performance difference is architectural.

### 2.10 Weight and state accounting [A]

| Model | Prior tensor-accounted total / active GEMM parameters | Recorded disk footprint | Main experiment precision |
| --- | --- | --- | --- |
| DeepSeek | 290.91B / 14.08B | 148.6 GiB | MXFP4 experts + FP8 attention/dense; FP8 KV |
| GLM | 321.34B / 17.38B | 305.8 GiB | FP8 weights; BF16 KV |
| Qwen | 176.94B served / 7.27B | 172.8 GiB | FP8 weights; BF16 KV |

Counts come from the existing per-model tensor analyses, not a new recount of complete checkpoints. Active-parameter definitions and treatment of embeddings/MTP differ from vendor labels: DeepSeek's card says 284B / 13B; the local 290.91B matches Hugging Face's 291B tensor count for the *preview* checkpoint, which includes the depth-1 MTP block, and the 148.6 GiB (≈ 160 GB) footprint matches the preview rather than the ~167 GB 0731 build that carries a fused DSpark module and no MTP head [P] — so the local DeepSeek arms appear to use the April preview weights, which is consistent with the existence of a local DeepSeek MTP arm (verify via `num_nextn_predict_layers` and the absence of a `dspark_*` block in the recorded config) [A/H]; GLM's card says 320B / 18B, and an independent tensor census gives 321.32B with 16.9B active before embeddings and the draft path [P]; Qwen's card says 125B + 51B n-gram + 4B MTP with 6B active [P]. Disk size is not runtime HBM. A weight-only GPU-count lower bound establishes neither a valid TP layout nor room for runtime state.

Qwen's prior analysis identifies a **51.23B n-gram embedding table**, about 29% of its served total. Lookup touches selected entries, not the entire table each token. Treating all table parameters as active arithmetic overestimates work. Table offloading was not measured; the vendor designed the table for host-memory placement with asynchronous prefetch, and the vLLM recipe requires that offload on 80 GB-class GPUs [P].

```text
HBM needed = weights + workspace + graph buffers
           + sequence-growing history + recurrent state + padding/metadata

Conventional KV bytes/token = sum_over_layers(2 × KV_heads × head_dim × bytes)
```

For Qwen's 12 retained-KV layers: `12 × 2 × 2 × 256 × 2 = 24,576 bytes/token = 24 KiB/token`, before indexer/state overhead. Prior analysis estimates **24.75 KiB/token including indexer keys**. Eight sequences of 131,072 tokens imply about **24.75 GiB across the model** for this component alone; rank placement, grouping, and padding require separate accounting.

The same accounting for the other two [A]: GLM's 11 MLA layers cache a 512-wide latent plus a 4:1-pooled 128-wide indexer key, 5,984 elements per token, i.e. **11.6 KB/token with a BF16 latent and FP8 index (the measured configuration) or 6.0 KB all-FP8**; its 34 KDA layers add a **fixed ~71 MB per active sequence** (BF16) that does not grow with context. DeepSeek's compressed paths give roughly **2.5–3.5 KB/token** (the vendor states ~7% of V3.2's cache at 1M tokens; V3.2 cached 576 elements × 61 layers in FP8), with no recurrent state.

**Consistency check against the recorded pools [A/H].** GLM's main baseline reports a 1,916,967-token BF16 pool. At 11.6 KB/token that is ≈ 22 GB; with ≈ 38 GB of FP8 weights per GPU after TP8 and a 0.82 utilisation cap on 80 GB, ≈ 27 GB per GPU remains for KV, state and workspace. The pool size is therefore consistent with **each TP rank holding a full copy of the latent KV** (a single latent head cannot be split across ranks), not with an eighth of it. The alternate-stack FP8 pool (3,790,580 tokens, 1.805× the 2,099,654-token BF16 pool of that session) is consistent with the latent halving while the FP8 index stays fixed (5,632 of 5,984 elements halve → 1.89× predicted). Qwen's corrected pool of 3,197,331 tokens at 24.75 KiB would be 79 GB if replicated, so its two KV heads must be sharded (each head on four ranks: ≈ 39 GB per rank), which fits alongside ≈ 22 GB of weights per rank. Both readings are inferences from pool sizes, not inspected layouts; a layout dump or a decode-context-parallel A/B (vLLM's `--decode-context-parallel-size`, which gave Kimi-K2-Thinking 8× KV capacity on 8×H200 [P]) would test them and, for GLM, could lift the 89.5% occupancy seen at 260K in §3.

### 2.11 Out-of-scope reference: where Kimi K2 → K3 sits [P]

The assignment names Kimi K2–K3, but neither fits the 8×H100 budget: K2.x is a 1T / 32B all-MLA MoE (INT4 QAT builds need 8×H200), and K3 is a 2.8T / 104B hybrid with 69 KDA + 24 MLA layers, natively MXFP4, needing 8×B300 or 16×B200. They are relevant here as the two ends of the design axis the three measured models occupy: K2 is the pre-hybrid baseline — ≈ 35 KB/token of KV in FP8, KV-bandwidth-bound at 128K+ context, no native draft head — and K3 is the largest deployment of the recurrent-plus-attention idea, whose serving stack had to add a second mutable-state allocator, branch-aware state checkpoints, ReplaySSM for speculation, and context parallelism for the MLA layers. The companion document [reference_report_v1_standalone.md](reference_report_v1_standalone.md) covers both; nothing in this section is locally measured.

## 3. Q1 — Workload and concurrency: what limits performance?

### 3.1 Main batch grid: 16K input, 256 output

| Concurrency | DeepSeek tok/s | GLM tok/s | Qwen tok/s | DeepSeek TPOT ms | GLM TPOT ms | Qwen TPOT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 85.0 | 96.3 | 106.2 | 7.9 | 7.1 | 7.0 |
| 4 | 219.8 | 225.3 | 256.8 | 10.7 | 10.9 | 9.6 |
| 16 | 330.9 | 359.2 | 420.2 | 32.4 | 33.6 | 26.3 |
| 64 | 389.4 | 447.1 | 517.7 | 148.5 | 129.5 | 110.1 |

Source: `batch_isl16k_c{1,4,16,64}.json` in the three main batch directories in §1. Qwen at concurrency 64 is **1.16× GLM and 1.33× DeepSeek**. It leads this observed throughput grid; no accuracy or equal-text comparison was performed.

### 3.2 Finer concurrency grid

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

### 3.3 Context sweep at concurrency 8

| Input tokens | DeepSeek tok/s | GLM tok/s | Qwen tok/s* | DeepSeek cache peak | GLM cache peak | Qwen cache peak* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | 104.5 | 295.9 | 322.7 | 18.1% | 10.2% | 7.2% |
| 65,536 | 47.9 | 104.3 | 114.5 | 21.9% | 36.6% | 28.2% |
| 131,072 | 34.7 | 47.2 | 56.5 | 27.0% | 72.5% | 56.1% |
| 260,000 | 16.7 | 26.5 | 25.1 | 36.9% | 89.5% | 97.3% |

Source: `ctx_isl{length}_c8.json` in §1's context arms. *Qwen uses the earlier smaller `base` pool, not corrected `base-util082`; rerun before treating its cache limit as architectural.*

At 131K, median TTFT is **19.84 s Qwen, 12.89 s GLM, 26.66 s DeepSeek**. The throughput leader is not the first-token-latency leader. High occupancy indicates headroom risk, not proof of preemption or a capacity-caused slowdown. Pool sizes and cache semantics differ across models.

DeepSeek's 16K/context point is much slower than its 16K/concurrency-8 point in the separate finer-grid arm (**104.5 versus 281.1 tok/s**). Build, utilization, and session differ; the batch-only bridge does not explain this discrepancy. Flag it for a same-session rerun, and avoid using it to identify an architectural bottleneck. Context throughput also includes extra input work; it is not a decode-only scaling curve.

### 3.4 Bottleneck attribution: observation versus explanation

| Regime | Observation | Candidate mechanism [H] | Required check |
| --- | --- | --- | --- |
| Low concurrency | Low aggregate throughput; lower TPOT | Small GEMMs, weight/state traffic, launches, collectives | Decode operator trace and hardware counters |
| High concurrency | Diminishing gains; rising TPOT | Scheduling, compute/communication contention; on this grid, chunked-prefill tokens sharing steps with decode (§3.5) | Actual per-step batches; TP/EP sweep; per-step prefill-token counts |
| Long input | Higher TTFT; lower output rate | Input work, attention/indexing, queueing | Separate prefill/decode timing; chunk-size sweep |
| High cache occupancy | Little pool headroom | Allocation pressure or preemption | Preemption counters; matched pool-size A/B |

GLM's available engine estimates report roughly **319–608 GB/s/GPU**, or 9.5–18.1% of an assumed 3,350 GB/s peak. They include FFN and unembedding but **omit attention**, and span different utilization arms. They are [E], not hardware-counter measurements. A partial whole-request average cannot rule out bandwidth-bound individual kernels.

The older estimate `active_weight_bytes × output_tok/s ÷ concurrency` is unsuitable: tokens can select different experts, live batch differs from client concurrency, and output throughput includes prefill. Under simplified independent uniform routing, distinct experts scale as `E × [1 − (1 − k/E)^B]` for B tokens. Real routing and reuse need measurement. Total/active parameters alone do not predict the bottleneck.

### 3.5 An explicit bytes-per-step floor, reconciled with the grid [A]

The following model replaces the unsuitable estimate above with one that separates the terms. For a decode step carrying B tokens at context S on an 8-GPU node:

```text
bytes/step ≈ W_nonexpert + f(B)·W_experts          f(B) = 1 − (1 − k/E)^B  (uniform routing)
           + B · A(S)                               A(S) = index scan + selected entries (§2.3)
           + B · 2 · S_rec                          recurrent state read + write (GLM, Qwen only)
floor(ms)  = bytes/step ÷ (8 × 3.35 TB/s)          H100 HBM3, no overlap losses
```

| Point | DeepSeek | GLM | Qwen |
| --- | --- | --- | --- |
| Weight bytes streamed per token at B=1 | ≈ 10 GB (≈ 7 GB FP8 non-expert + 6 × 43 × 12.6 MB FP4 experts) | ≈ 18 GB (≈ 9.5 GB non-expert + 8 × 42 × 25 MB) | ≈ 6.4 GB (≈ 4 GB non-expert + 10 × 48 × 4.9 MB) |
| Floor at B=1 | 0.38 ms | 0.67 ms | 0.24 ms |
| Measured TPOT at c1 (16K input) [M] | 7.9 ms | 7.1 ms | 7.0 ms |
| Ratio measured / floor | ≈ 21× | ≈ 11× | ≈ 29× |
| Bytes per step at B=64, S=16K | ≈ 116 GB (78% of experts) | ≈ 280 GB (84% of experts, + 9 GB state r/w) | ≈ 102 GB (72% of experts, + 7 GB state r/w) |
| Floor at B=64 | 4.3 ms | 10.5 ms | 3.8 ms |
| Measured TPOT at c64 [M] | 148.5 ms | 129.5 ms | 110.1 ms |
| Ratio measured / floor | ≈ 34× | ≈ 12× | ≈ 29× |

Three conclusions follow, each bounded.

1. **Batch-1 decode is not bandwidth-bound on this build.** All three sit 10–30× above their HBM floor at concurrency 1, and their TPOTs are nearly identical (7.0–7.9 ms) despite a 3× spread in bytes per token. That pattern is the signature of a fixed per-step cost — kernel launches, TP all-reduces, sampling, scheduler overhead — rather than weight streaming. It matches what the SGLang team reported for a much larger hybrid (Kimi K3), where the batch-1 campaign gained almost everything from launch elimination and communication fusion, not faster GEMMs [P]. A decode operator trace remains the required check; the model says where to look.
2. **At concurrency 64 the decode step is still far from the weight-streaming floor, and the excess is consistent with prefill sharing the step.** The main-grid wall times, derived from output tokens ÷ throughput, are 84.1 s (DeepSeek), 73.3 s (GLM) and 63.3 s (Qwen) for 128 requests. Decoding 32,768 output tokens at the B=64 floors would take 2–6 s; the remaining ≥ 90% of wall time is input processing of 2,097,152 tokens plus overhead. The implied lower bound on aggregate prefill rate is **24.9K / 28.6K / 33.1K input tok/s** for DeepSeek / GLM / Qwen. Under chunked prefill every decode step that also carries a prefill chunk is stretched by that chunk's time, so the measured TPOT at c64 is best read as **decode latency under prefill interference**, and the "diminishing returns beyond c8–16" observation as a statement about prefill/decode contention on an input-heavy grid [H]. Per-step prefill-token counts from the engine would confirm or refute this.
3. **The c1 phase split also ranks the models.** From the c1 rows, per-request wall time is 3.01 / 2.66 / 2.41 s (DeepSeek / GLM / Qwen); subtracting 255 × TPOT of decode leaves ≈ 1.0 / 0.85 / 0.62 s for a 16K prefill, i.e. ≈ 16K / 19K / 26K input tok/s at batch 1. The throughput ranking on the main grid therefore holds in both phases, which is consistent with Qwen streaming the fewest weight bytes and running the lightest indexer (§2.3) but does not separate those from kernel maturity.

The uniform-routing coverage term is pessimistic: real routing is skewed and engines can exploit it. On Blackwell, SGLang measured GLM-5.3-Flash at 2,660 aggregate output tok/s (665 per GPU) at concurrency 64 on 4× GB300 with 1K inputs — above what uniform coverage would allow on that hardware — so the model bounds rather than predicts [P].

### 3.6 Marginal input-token rates from the context sweep [A]

Because each context point processes 16 requests of length L with 256 outputs, the difference in wall time between adjacent lengths is almost entirely input work (decode adds ≈ 2 waves × 255 × TPOT ≈ 9 s at c8, small against 25–127 s differences). Wall time is `4,096 output tokens ÷ throughput`; the marginal rate is `16 × ΔL ÷ Δwall`.

| Interval | Δ input tokens | DeepSeek marginal tok/s | GLM marginal tok/s | Qwen marginal tok/s* |
| --- | ---: | ---: | ---: | ---: |
| 16K → 64K | 786,432 | 17.0K (16K point anomalous) | 30.9K | 34.1K |
| 64K → 131K | 1,048,576 | 32.3K | 22.1K | 28.5K |
| 131K → 260K | 2,062,848 | 16.2K | 30.4K | 22.7K |

Derived from the §3.3 throughputs. *Qwen's 260K point ran at 97.3% pool occupancy on the smaller `base` pool, so its final-interval decline is confounded by capacity.* DeepSeek's 16K point is the flagged anomaly, so its first interval is unreliable.

What can be said: between 64K and 260K, **GLM's marginal input cost is flat** (22–30K tok/s, no sign of super-linear growth to 260K), which is what a stack of 34 linear-scan layers plus 11 pooled-indexer layers should produce; **DeepSeek's marginal rate halves** from the middle to the last interval (32.3K → 16.2K), which is the direction its heavier indexer/compressor load predicts (§2.3) but could equally come from scheduling or pool pressure at 36.9% occupancy — the required check is phase timing at 131K and 260K. **Qwen's decline** coincides with a 97% pool, so it says nothing architectural until the corrected pool is rerun. The 131K median TTFTs (GLM 12.89 s < Qwen 19.84 s < DeepSeek 26.66 s) are consistent with the same ordering of per-request input cost, with the Qwen/GLM order affected by the pool confound.

### 3.7 Published reference points for the same models [P]

These are quoted to bound what engine maturity and hardware could change, not to be compared cell-for-cell with §3.1–3.3. Definitions differ (§1).

| Model | Published configuration | Published result | Nearest local point |
| --- | --- | --- | --- |
| GLM-5.3-Flash | SGLang, 8× H100 TP8/EP8, 8K in / 1K out, adaptive MTP 5/1/6 with accept length pinned at 3.0 by simulation | 212.6 aggregate output tok/s at concurrency 1 (single user), TTFT 0.35 s; 1,176 at concurrency 16, TPOT 10.2 ms ([SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash)) | Local vLLM 16K/256, spec off: 96.3 at c1 (119.7 with MTP n1), 359 at c16, TPOT 33.6 ms |
| GLM-5.3-Flash | SGLang, 4× GB300 TP4/EP4 FP8, 1K in / 256 out, spec off | 1,161 / 2,660 / 4,828 aggregate output tok/s at concurrency 16 / 64 / 256; FP8 KV + TRT-LLM DSA is 2.9–5.7% faster than BF16 KV on Blackwell with 1.8× KV capacity | Local BF16 → FP8 KV on the Hopper alternate stack loses 24.9% (§7) |
| DeepSeek-V4-Flash | Vendor API (Blackwell serving, DSpark), Artificial Analysis | ≈ 108–133 output tok/s per user; recommended vLLM layout is DP-attention + EP (`--data-parallel-size 4`) rather than TP8 ([vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)) | Local 85.0 tok/s at c1 with TP8 on H100, MTP off |
| Qwen3.8-Flash-Next | vLLM recipe, 4× H100 with the n-gram table offloaded to host RAM | ≈ 1,430 output tok/s at concurrency 64 (whole node); GDN startup failure mitigated by capping sequences at 256 ([recipe via IntuitionLabs](https://intuitionlabs.ai/articles/qwen3-8-flash-next-architecture-memory)) | Local 8× H100: 517.7 at c64 with 16K inputs (input-dominated; §3.5); local Bug 1 is the same startup failure |
| Qwen3.8-Flash-Next | TRT-LLM on GB300 NVL72 | >16K tok/s per GPU at the throughput end, >200 tok/s per user at the interactive end ([NVIDIA](https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-on-nvidia-gb300-nvl72-for-agentic-coding/)) | — |

The local c1 rate for GLM is roughly half of SGLang's published single-user rate on the same GPU class, with a shorter prompt and speculation on in the published run; the local c16 aggregate is a third of the published one. Both gaps are inside what speculation, prompt length and kernel maturity can explain, so the local ranking should be re-tested on a released image before it is attributed to architecture.

### 3.8 Hopper-specific factors that shape the local ranking [A/P]

- **DeepSeek's FP4 experts have no native path on H100.** Hopper lacks FP4 tensor cores, so MXFP4 expert weights run through weight-only dequantisation kernels: the HBM-traffic saving survives, the tensor-core throughput of published Blackwell runs does not. The FP4 lightning-indexer cache is likewise documented for Blackwell (SM100) only, so the indexer scan here runs at higher precision. DeepSeek's smallest-active-parameter advantage is therefore least visible on exactly this hardware. Published FP8 + DP/EP and FP8 + pipeline-parallel failures on Hopper-class parts were open issues during 2026 ([Morph summary](https://www.morphllm.com/deepseek-v4-flash)).
- **GLM must keep BF16 KV on Hopper in vLLM.** The vLLM recipe states Hopper does not support FP8 KV for this model; the FP8-KV + TRT-LLM DSA path that is faster on Blackwell is Blackwell-only ([vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)). The local alternate-stack FP8 result (§7) is consistent with that.
- **Qwen's 51 GB n-gram table competes with KV for 80 GB of HBM** unless offloaded, and its FP8 block-128 quantisation is incompatible with plain TP8 on H200 (TEP8 required) [P]. Which placement the local Qwen arms used is recorded in their manifests and should be stated next to any capacity claim.
- **All three replicate or shard KV differently under TP8** (§2.10), so "cache peak" percentages are not comparable pool sizes.

## 4. Q2 — Prefix caching: observed effects and implementation challenges

### 4.1 Observed sharing patterns

| Distinct 64K prefixes among 64 requests | DeepSeek tok/s | GLM tok/s | Qwen tok/s* |
| ---: | ---: | ---: | ---: |
| 1 | 198.2 | 265.3 | 453.9 |
| 4 | 391.6 | 365.9 | 398.1 |
| 16 | 114.6 | 199.4 | 262.2 |

Source: `prefix_p64k_n{1,4,16}.json` in the main context/prefix arms. *Qwen pool caveat applies.* All complete 64 requests without failure. Prefix JSONs do **not** contain cache-hit deltas or cold-run assertions; reuse is intentional.

Best/worst observed patterns differ by **1.73× Qwen, 1.83× GLM, 3.42× DeepSeek**. These are sharing-pattern ratios, **not cache-on/cache-off speedups**: no identical uncached control exists. GLM and DeepSeek peak at four prefixes. Fill ordering, locality, eviction, and scheduling could explain this [H]; no trace isolates them.

A bound on what the numbers can mean [A]: with no reuse at all, each point would process 64 × ≈ 67.6K ≈ 4.3M input tokens; at the §3.5 prefill rates (25–33K tok/s) that is 130–175 s, i.e. ≈ 95–125 output tok/s. All nine cells except DeepSeek's 16-prefix point (114.6) exceed that band, and the best cells exceed it 2–4×, so substantial reuse occurred in every arm even though its fraction is unrecorded. Reading the table as "cache hit rate" is still not possible: the 1- and 4-prefix points should require the same number of cold prefills under a concurrency cap of 8, yet DeepSeek's differ by 2×.

### 4.2 Implementation requirements

1. **Correct identity:** match token prefix, model revision, adapter, positions, and relevant execution state. Similar text is insufficient.
2. **Consistent state boundary:** retained KV can be shared; recurrent layers need saved state at the reused boundary. A final recurrent state cannot reconstruct arbitrary earlier states. Checkpointing enables reuse with memory/copy costs; recurrent state is not inherently uncacheable.
3. **Different layouts:** full KV, compressed/indexer history, and recurrent state require compatible alignment, resume positions, and group-aware eviction.
4. **Safe branching:** immutable prefix data can be shared; divergent requests need state copies or copy-on-write. Speculative rejection must roll state back correctly.
5. **Granularity tradeoff:** finer checkpoints allow more reuse but cost metadata/storage; coarser checkpoints require more recomputation after partial matches.

GLM's manifest resolves attention block size 640 and Mamba block size 128; corrected Qwen resolves 4 and 16. These represent different organizations, not a measured 160× efficiency difference. DeepSeek requests launch block size 256 while its exported cache metric says 4; logical blocks and backend storage blocks must not be equated without implementation inspection.

vLLM documents heterogeneous state sizing, padding, and coordination across cache groups. This supports the design challenge; current documentation does not establish behavior in the recorded development builds. [Hybrid cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)

### 4.3 What a cached prefix must contain, per model [A]

| Cache component | DeepSeek-V4-Flash | GLM-5.3-Flash | Qwen3.8-Flash-Next-FP8 |
| --- | --- | --- | --- |
| Append-only history | CSA entries (one per 4 tokens, overlapping compressor), HCA entries (one per 128 tokens), 128-token windows, FP4/FP8 indexer keys per compressed entry | 512-wide latent per token per MLA layer (BF16 here) + one pooled 128-wide indexer key per 4 tokens per layer | K and V per token per QSA layer (2 heads × 256) + one indexer key per 4-token micro-block |
| Mutable per-request state | none beyond the window | KDA state: 71 MB BF16 / ~142 MB FP32 per sequence, overwritten every token | GDN state: ~57 MB BF16 / ~113 MB FP32, overwritten every token |
| Reuse granularity forced by the design | 4 tokens (CSA) and 128 tokens (HCA); the overlapping compressor makes the last compressed entries depend on tokens past the boundary, so they must be recomputed on extension | 4 tokens (pooled index; tail always selected) plus the engine's checkpoint interval for the recurrent state | 4 tokens (micro-block; only fully observed blocks are scored) plus the checkpoint interval |
| Cost of one recurrent checkpoint in KV-token equivalents | — | 71 MB ÷ 11.6 KB ≈ 6,100 tokens of latent KV | 57 MB ÷ 24.75 KiB ≈ 2,300 tokens of KV |
| If a checkpoint were kept every attention block | — | at 640-token blocks: 111 KB/token, ≈ 9.6× the KV itself | at 16-token blocks: 3.6 MB/token — infeasible, so engines keep sparse checkpoints |

The last two rows are the quantitative form of requirement 5. A recurrent checkpoint is so much larger than a token of KV that engines cannot keep one per block; they keep a sparse set (vLLM's `align` mode retains one aligned checkpoint per request; SGLang keeps a per-path capped LRU set and prioritises branching points [P]). Every reuse therefore has three possible outcomes — exact checkpoint hit, replay from an earlier checkpoint, or full recomputation — and the local JSONs do not record which occurred.

### 4.4 A documented failure mode to rule out for GLM [H]

vLLM issue #45238 (June 2026) reports that for GDN/KDA hybrids in `align` mode the only retained recurrent checkpoint is the one at the request's last aligned block boundary, and that prefix-cache hits silently fall to 0% whenever `floor((prompt_len − 1) / block) · block > shared_prefix_len`, i.e. whenever that boundary lands inside the request-unique suffix [P]. The local GLM prefix geometry is close to that condition: with a 640-token attention block, a 65,536-token shared prefix (102.4 blocks) and a ≈ 67,600-token request, the last aligned boundary is at 67,200 tokens, 1,664 tokens past the shared prefix. Qwen's 4/16-token blocks make the same condition harmless, and DeepSeek has no recurrent state. Whether the recorded GLM build was affected is unverified — §4.1 shows reuse of *something* occurred — but three cheap checks would settle it and are recommended before any GLM prefix number is reused: read the per-cache-group hit counters (`vllm:prefix_cache_hits_total` / `queries_total`) for each point; rerun one GLM point with the shared prefix padded to a multiple of 640 tokens; and record `mamba_cache_mode` / `mamba_block_size` from the manifest. If the aligned run is materially faster, the local GLM cells measured attention-only reuse with recurrent replay, which would also help explain why GLM shows the smallest 1-versus-16-prefix ratio (1.33×) of the three.

For the DeepSeek and GLM 1-versus-4-prefix inversions, two engine-level hypotheses are worth testing alongside the fill-order and eviction ones: run order (a cold-compile or first-run effect on the first arm executed; the JSON timestamps identify it), and in-flight duplicate prefill (engines do not deduplicate identical prefixes that are being computed concurrently, so up to 8 cold 64K prefills can run at once under the concurrency cap; this does not by itself separate 1 from 4 prefixes, but it changes with arrival timing).

### 4.5 What production engines do about these problems [P]

| Problem | Mechanism in current engines | Relevance to the local arms |
| --- | --- | --- |
| Mutable recurrent state shared across requests | SGLang (Kimi K3 day-0): copy-on-write restore into a private slot before mutation, snapshot into a ping-pong buffer, donate the slot index to the radix tree; ordered on the forward stream, no device-wide sync ([LMSYS](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)) | vLLM's development builds used a different, `align`-mode design; race-freedom under overlap scheduling is a property to verify, not assume |
| Which checkpoints to keep | SGLang: sparse set, per-path cap, LRU, priority to branching points (Marconi); vLLM: Marconi-style admission policy and a retention interval for mamba groups ([tracking issue](https://github.com/vllm-project/vllm/issues/26201)) | The 4-prefix optimum could reflect an admission/eviction policy interacting with checkpoint size |
| Checkpoint memory pressure | dasc: drop short-horizon KDA/GDN channels from checkpoints (2.63× smaller on Kimi-Linear; TTFT −42.6%, input throughput +68.4% under a fixed checkpoint budget) ([arXiv:2608.30386](https://arxiv.org/html/2608.30386)); INT8 checkpoints compose with it | Directly relevant to the §8 "state pressure" variable |
| Two allocators with different unit sizes | SGLang unified pool: recurrent states fill from one end, KV blocks from the other (K3: 54 MB state blocks vs 27 KB KV blocks) | GLM's KDA pool capping concurrency (Bug 1; `--mamba-full-memory-ratio`) is the same failure in a static-pool design |
| Cross-engine / tiered caching | LMCache treats recurrent state as an opaque page; cached-vs-fresh output is score-equivalent, not bit-exact; no sharing across engines with different kernel block sizes ([LMCache docs](https://docs.lmcache.ai/mp/hybrid_models.html)); SGLang HiCache L1+L2 validated on GLM-5.3-Flash within 4–5% of no-HiCache on a no-reuse dataset ([SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash)) | A host tier would change the 260K occupancy picture in §3.3 |
| Compressed / pooled entries | DeepSeek recipes require the hybrid KV manager (`--no-disable-hybrid-kv-cache-manager`) and 256-token blocks so CSA (4) and HCA (128) groups align; FP8 KV with BF16 RoPE dims must follow the checkpoint's own layout — a naive q8 KV in llama.cpp produced malformed output ([vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash), [Morph](https://www.morphllm.com/deepseek-v4-flash)) | Consistent with the local "block size 256 vs exported 4" observation: two block systems coexist |
| Prefix identity across turns | DeepSeek V4 keeps reasoning content in context across user turns when tools are involved ([HF blog](https://huggingface.co/blog/deepseekv4)); GLM exposes `clear_thinking` (default false); reasoning-effort levels inject different system prompts | A harness that strips `reasoning_content` or changes effort level changes the prefix bytes and forfeits hits — a benchmark-design hazard for real-text replays (§8) |

### 4.6 Why the hit rate is worth more than it looks [P]

On the vendors' own APIs a cache hit is priced at $0.007 vs $0.22 per million input tokens for DeepSeek-V4-Flash (31× discount, off-peak), $0.03 vs $0.15 for GLM-5.3-Flash (5×) and ≈ $0.016 vs $0.15 for Qwen3.8-Flash (≈ 9×); observed hit rates on agentic traffic run 80–90% (OpenRouter reported 91.1% for Qwen3.8-Flash; Moonshot's Mooncake reports ~90% on coding) ([Morph](https://www.morphllm.com/deepseek-v4-flash), [eesel](https://www.eesel.ai/blog/qwen38-flash-next-review)). For a self-hosted deployment the same hit saves the prefill work that §3.5 shows dominates these grids. That is the strongest argument for treating the prefix arms as the most decision-relevant experiment in this study — and for repeating them with the counters and controls above before drawing on them.

**Benchmark lesson:** warm compiled kernels with unrelated prompts, then use fresh prompts for cold-prefix tests. Reusing seeds across points can accidentally benchmark cached prompts. `/health` does not establish completion of kernel warmup. The recorded incidents are in [fix_bug.md](fix_bug.md).

## 5. Q3 — Speculative decoding / multi-token prediction

MTP drafts candidates and the target verifies them. Benefit depends on accepted progress relative to drafting, verification, state, and scheduling costs. Acceptance rate alone does not determine speedup.

### 5.1 Strongest controlled evidence: GLM batch A/B

| Concurrency | Base tok/s | MTP n=1 tok/s | n=1 / base | MTP n=5 tok/s | n=5 / base |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96.3 | 119.7 | 1.24× | 120.8 | 1.25× |
| 4 | 225.3 | 239.8 | 1.06× | 230.7 | 1.02× |
| 16 | 359.2 | 344.7 | 0.96× | 342.1 | 0.95× |
| 64 | 447.1 | 441.9 | 0.99× | 408.3 | 0.91× |

Sources: GLM `bf16kv`, `bf16kv-mtp-n1`, `bf16kv-mtp-n5`. At concurrency 1, n=1 reduces median TPOT **7.06→5.05 ms**. At 64, n=5 has **99.2% cache occupancy** and TTFT **10.72 s versus 2.51 s** for base. More draft tokens are not justified by throughput in these loaded cases.

### 5.2 GLM context A/B: latency and throughput diverge

| Input tokens | Base → MTP tok/s | Throughput ratio | Median TTFT base → MTP | TTFT change |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 295.9 → 314.8 | 1.064× | 2.63 → 2.10 s | −20.1% |
| 65,536 | 104.3 → 97.6 | 0.936× | 7.21 → 5.43 s | −24.8% |
| 131,072 | 47.2 → 45.3 | 0.958× | 12.89 → 10.73 s | −16.7% |
| 260,000 | 26.5 → 22.2 | 0.839× | 38.48 → 41.07 s | +6.7% |

Source: [GLM MTP context](GLM-5.3-Flash/results/bf16kv-mtp-n1-context/). The 131K point offers a possible latency/cost tradeoff. At 260K both metrics worsen and cache occupancy reaches 96.9%. Scheduling changes may explain earlier first tokens [H]; MTP does not directly eliminate input work, and no timeline proves the mechanism.

### 5.3 Other models: weaker controls

| Concurrency | DeepSeek historical off → no-reuse MTP tok/s | Ratio | Qwen original base → MTP tok/s | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 92.3 → 115.3 | 1.25× | 107.7 → 125.9 | 1.17× |
| 4 | 222.0 → 251.8 | 1.13× | 162.2 → 149.3 | 0.92× |
| 16 | 293.7 → 339.9 | 1.16× | 371.0 → 322.2 | 0.87× |
| 64 | 383.4 → 380.2 | 0.99× | 517.8 → 481.5 | 0.93× |

Sources: Qwen `base`/`mtp-n1`; DeepSeek `mtp-off`/`mtp-on-noreuse`. Qwen's equal-utilization pair retains startup/pool confounds; substituting corrected `base-util082` would also change utilization. DeepSeek's historical pair uses the older runtime and lacks newer cold-run fields. Do not substitute newer baselines or call this a fully controlled three-model A/B. Real-text acceptance and answer quality remain untested.

### 5.4 Why the light-load gain is only ~1.25×: a draft-round cost model [A/H]

With random-token prompts and `--ignore-eos`, acceptance is unmeasured locally, but the published accept lengths for these draft modules are 3–4 tokens (Qwen: mean 4.06 at four steps; GLM: 3.9 real accept length in SGLang's DCP arms, 3.0 pinned in its headline rows [P]). If acceptance were the limit, an n=1 draft should approach 1.8× at batch 1, and n=5 should approach 3–4×. The local c1 gains (1.24× and 1.25×) are far below both, and — decisively — **n=5 gains nothing over n=1**. §3.5 shows the batch-1 step is dominated by fixed per-step cost rather than weight streaming (7.1 ms measured against a 0.67 ms bandwidth floor). In that regime a draft round adds its own fixed cost:

```text
gain ≈ (1 + committed extra tokens) × T_target / (T_target + n × T_draft_step + T_verify_overhead)
```

GLM's draft is one full block (its own MoE and indexer, ≈ 0.4B active parameters per drafted token, §2.6), plus an LM head and sampling; if each draft step costs ≈ 0.4–0.5 of a launch-bound target step, then n=1 with ≈ 0.8 accepted tokens gives ≈ 1.8 / 1.45 ≈ 1.24×, and n=5 with ≈ 3 accepted tokens gives ≈ 4 / (1 + 2.3 + overhead) ≈ 1.2× — both matching the observed values. This is a fit to two numbers, not a measurement; the check is per-step draft/verify timing. The implication, if it holds, is that on this build the speed-up ceiling is set by draft-step overhead, not by acceptance — which is exactly what fused draft paths change: SGLang's published GLM-5.3-Flash recipe reaches 3.94 ms TPOT at concurrency 1 with a five-step draft on GB300, and 1.57× aggregate output throughput at concurrency 16 versus spec-off [P].

### 5.5 Why the loaded cases flatten or reverse [A/H]

Two mechanisms are specific to these architectures and predict the sign of the c16–c64 results:

1. **Verification is not free for an MoE target.** Every verified position routes to its own experts, so a batch of 64 requests with n=5 drafts routes 384 positions per step; under the §2.8 coverage bound that is ≈ 100% of GLM's experts versus 84% for 64 positions, i.e. more weight bytes per step for tokens most of which are rejected. Dense targets do not pay this. The published analogue: SGLang turns speculation off in its high-throughput recipe, and its Kimi K3 work found verify-all and confidence-trimmed verification break even through batch 8, after which trimming wins by +68% (chat, accept 2.7) and +24% (math, accept 5.0) at batch 256 [P].
2. **Draft tokens consume state and token budget.** For a recurrent model the KDA state must be rewindable across the draft window; a snapshot-per-step implementation holds up to (n + 1) × 71 MB per request — ≈ 27 GB across 64 requests at n=5 — which is a candidate cause of the n=5 arm's 99.2% occupancy and 10.7 s TTFT [H]; SGLang's ReplaySSM replaced snapshots with ~1 KB of raw inputs per step for Kimi K3, a ~32× reduction in draft-window memory [P]. Verified positions also count against the engine's per-step token budget, so a large draft window leaves fewer tokens per step for chunked prefill and lengthens TTFT under load — the direction observed at c64/n5. The check for both is the split of pool occupancy between state and KV, and the per-step prefill-token counts, across the three GLM batch arms.

The TTFT *reductions* with n=1 at 16K–131K (§5.2) run in the opposite direction and are not explained by either mechanism. Candidates [H]: enabling speculation changes scheduler defaults in some engines (SGLang, for example, resets its running-request cap to 48 when speculation is on unless overridden [P]); or shorter decode residency frees batch slots sooner. A diff of the effective engine configuration between the base and MTP arms is the first check.

### 5.6 What the published draft mechanisms would change [P]

| Model | Draft actually available | Published effect | Local status |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash | Preview: MTP-1 (sliding-window block). 0731: DSpark — 7-token semi-autoregressive block drafts with a trained confidence head that trims verification per request against a profiled engine cost curve | 60–85% faster per-user generation than MTP-1 at matched throughput in DeepSeek's serving; single-stream community run 26.3 → 39.9 (MTP-1) → 60.3 tok/s | Local arm is the historical MTP pair on the preview weights (§2.10); DSpark untested |
| GLM-5.3-Flash | NextN layer, index shared across the draft iteration | 1.57× at c16, 3.94 ms TPOT at c1 (SGLang, GB300, accept pinned 3.0) | Local: 1.24× at c1, flat at c64 |
| Qwen3.8-Flash-Next | 4B MTP with QSA index reuse | mean accept 4.06 at four steps; 1–2 GB extra memory locally per Unsloth | Local: confounded pair; 1.17× at c1, ≤1× beyond |
| (cautionary) Kimi K2.5 with a mismatched community MTP head | — | 39% acceptance and *lower* throughput than no speculation (947 → 869 tok/s on 8×B200); matched drafts expected 80–90% ([community card](https://huggingface.co/yrrhall/Kimi-K2.5-MTP)) | Illustrates why a native, checkpoint-matched draft is the precondition, not the guarantee, of a gain |

The published mechanisms also settle what "state rollback" costs for the recurrent models: either snapshot per draft step (memory) or replay of stored inputs (compute), and every drafted position must obtain sparse-attention indices — GLM and Qwen avoid a per-draft indexer scan by sharing the last pick, DeepSeek by making its draft block sliding-window only [P]. None of this replaces the missing local phase trace; it tells the trace what to record.

## 6. Q4 — Serving cost with explicit assumptions

### 6.1 Illustrative self-hosting cost

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

### 6.2 Goodput, not tokens

A deployment decision should compare **goodput under explicit TTFT/TPOT objectives**. The cheapest token in the table may violate an interactive latency target. Production cost also needs realistic arrival rates, repeats, equivalent tasks, and idle-time accounting.

### 6.3 API list prices for the same three models (September 2026) [P]

| Model | Input (miss) | Input (cache hit) | Output | Notes |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | $0.22 off-peak / $0.44 peak | $0.007 / $0.014 | $0.66 / $1.32 | Peak 01:00–04:00 and 06:00–10:00 UTC after the 16 Aug 2026 repricing; third-party hosts from $0.065 / $0.18 ([Morph](https://www.morphllm.com/deepseek-v4-flash)) |
| GLM-5.3-Flash | $0.15 (promo $0.075) | $0.03 (promo $0.015) | $0.50 (promo $0.25) | Promotion ends 9 Sep 2026; no context tiering ([codersera](https://codersera.com/blog/glm-5-3-flash-complete-guide-2026/)) |
| Qwen3.8-Flash | $0.15–0.16 | ≈ $0.016 | $0.47 | Flat across 1M context; the hosted model is a production build of the open Flash-Next weights ([Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-flash-next)) |

The §6.1 figures ($10.7–14.3 per million output tokens at c64) are 20× the API output prices because they charge 16K input tokens to every 256 output tokens; the comparison has to be made per request with input priced.

### 6.4 Per-request comparison on the local 16K/256 workload [A]

Self-host cost per request at concurrency 64 = `$20/h ÷ (T × 3600 / 256)`; API cost = `16,384 × input price + 256 × output price`, list prices, no cache hits.

| Model | Self-host $/request (8×H100, $2.50/GPU-h) | API list $/request | Self-host ÷ API | Same with 90% cache hits on input (API side) |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek | 0.00365 | 0.00377 off-peak / 0.00755 peak | 0.97× / 0.48× | API ≈ $0.00063 → self-host 5.8× |
| GLM | 0.00318 | 0.00259 list / 0.00129 promo | 1.23× / 2.46× | API ≈ $0.00082 → self-host 3.9× |
| Qwen | 0.00275 | 0.00258 | 1.07× | API ≈ $0.00060 → self-host 4.6× |

Read with the caveats of §6.1 (single runs, illustrative price, 100% utilisation implied). Two conclusions are robust to those caveats: on a prefill-heavy synthetic workload with no reuse, a rented 8×H100 node is near API list-price parity for all three — which says the vendor APIs are priced close to Hopper cost, not that self-hosting is free; and once realistic prefix reuse (80–90% on agentic traffic, §4.6) is priced at cache-hit rates, the APIs are several times cheaper unless the self-hosted stack achieves the same reuse. A self-hosted node does capture reuse too, so the fair comparison is measured goodput with the §4 counters on both sides.

### 6.5 Cost per task, not per token [P]

Reasoning verbosity dominates real bills: Artificial Analysis measured DeepSeek-V4-Flash-0731 emitting 210M output tokens to run its Intelligence Index against a 100M median for its size, Qwen3.8-Flash-Next ≈ 200M against a 110M median, and Z.ai quotes GLM-5.3-Flash at $0.045 per Index task (discounted) ([Morph](https://www.morphllm.com/deepseek-v4-flash), [eesel](https://www.eesel.ai/blog/qwen38-flash-next-review), [Z.ai](https://docs.z.ai/guides/vlm/glm-5.3-flash)). All three default to a high or maximum reasoning effort. The local runs, with `--ignore-eos` and fixed 256-token outputs, cannot see this term at all; a real-text replay with recorded output lengths (§8, priority 6) is the only way to price it.

### 6.6 Published self-hosting reference points [P]

SGLang's GLM-5.3-Flash cookbook reaches 1,207 output tok/s per GPU at concurrency 256 on 4× GB300 (FP8, 1K/256, spec off), i.e. ≈ $0.58 per million output tokens at the same illustrative $2.50/GPU-hour before pricing input work, and 1,480–1,538 with NVFP4 experts; SemiAnalysis InferenceX estimates $0.03–0.14 per million tokens for DeepSeek-V4-Pro on B200 across 59–145 tok/s per user on cached agentic traces, and $0.10–0.30 for Kimi K3 on B300 ([InferenceX](https://inferencex.semianalysis.com/compare/deepseek-v4-b200-vs-b300)). These use different metrics and hardware (§1) and are cited only to show that the gap between the local §6.1 numbers and API prices is mostly hardware generation, batch depth and prefix reuse — the three levers §8 proposes to measure.

## 7. Additional systems findings

### 7.1 FP8 KV: backend and dtype must be separated

| GLM arm, concurrency 64 | Output tok/s | Interpretation |
| --- | ---: | --- |
| Original BF16 (`bf16kv`) | 447.1 | Main baseline |
| Alternate BF16 (`bf16kv-fi618`) | 348.8 | −22.0%: software/backend-path change |
| Alternate FP8 (`fp8kv-fi618`) | 262.0 | −24.9% versus alternate BF16 |

The full drop is 41.4%; attributing it all to dtype is incorrect. The prior [GLM capacity accounting](GLM-5.3-Flash/report.md) reports **2,099,654→3,790,580 tokens, or 1.805×**, across its reported capacity comparison. Those startup pools differ from the main baseline's later 1,916,967-token pool; retain session provenance. The successful alternate arm supersedes the old blanket statement that GLM FP8 KV is impossible on H100.

The saved alternate-stack scripts bypass a FlashInfer version check and combine newer Python with older compiled artifacts, routing MoE through DeepGEMM. This is an experimental configuration, not a validated production recipe. Synthetic completion does not establish numerical parity. The paired dtype effect includes dtype-dependent kernel/dequantization behavior. See [Bug 8](fix_bug.md#bug-8--one-incompatible-fp8-layout-is-not-universal-fp8-failure).

Two published facts sharpen the interpretation without changing it [P]. The 1.805× capacity gain is what the cache geometry predicts (§2.10: the 512-wide latent halves, the FP8 index does not, 1.89× expected). And the sign of the speed effect is hardware-specific: the vLLM recipe states Hopper must run BF16 KV for this model, while on Blackwell SGLang measured FP8 KV with a TRT-LLM sparse-attention backend 2.9–5.7% *faster* than BF16 KV across concurrency 16–256 with 1.8× KV capacity ([vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash), [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash)). The local result is therefore best stated as "FP8 KV costs 25% on this Hopper kernel path", not as a property of the model.

### 7.2 Startup state changes measured capacity

GLM's utilization 0.82→0.85 control grows reported capacity 11.2%, while its four batch throughputs change by less than 0.7%. This supports little throughput sensitivity **on that grid**, not a universal claim that utilization can never affect speed. [GLM control](GLM-5.3-Flash/results/util085/RESULT-util-ab.md)

Qwen's earlier startup records peak activation **17.07 GiB versus 0.99 GiB** in the warm arm. Despite lowering utilization, reported pool capacity grows **2,048,645→3,197,331 tokens**, and concurrency-4 throughput rises **162.2→256.8 tok/s**. Startup and utilization changed together: this exposes a confound rather than a utilization-only speedup. Only corrected batch results become headline evidence. [Qwen audit](Qwen3.8-Flash-Next-FP8/results/base-util082/RESULT-util-ab.md)

The general form of this problem — a pool sized at startup for one kind of state while a second kind (recurrent checkpoints) competes for the same bytes — is what SGLang's unified memory pool for Kimi K3 removes by letting both kinds allocate from one region [P]; for the vLLM builds used here, `--mamba-full-memory-ratio` / `--max-mamba-cache-size` are the equivalent manual knobs and should be recorded in every manifest alongside utilization.

### 7.3 KV replication under TP8 [A/H]

§2.10 infers from the recorded pool sizes that GLM's latent KV is replicated on every TP rank while Qwen's two KV heads are sharded. If confirmed by a layout dump, this means GLM's effective KV capacity on this node is one-eighth of what the same bytes would hold under position-sharded decode, and that its 89.5% occupancy at 260K (§3.3) is a layout limit before it is an architectural one. vLLM's decode context parallelism gave Kimi-K2-Thinking 8× KV capacity and +43% output throughput on 8×H200 [P]; whether it supports GLM's sparse-MLA layers in the recorded build is unknown and is the first thing to check before the 260K rerun.

## 8. Contribution to SyFI and next experiments

The contribution is an empirical **workload/configuration tradeoff map**, backed by raw results and documented measurement failures. It motivates testing layer-aware memory accounting and workload-aware serving policies. It does not establish a new scheduler, prove that existing engines assume uniform layers, or identify a dominant kernel.

### 8.1 A falsifiable research direction: budget speculation and hybrid state together

**Hypothesis [H].** A simple draft-budget policy conditioned on live load and state pressure can improve useful throughput over fixed speculation settings for hybrid models. The motivation is the GLM light-load/loaded MTP reversal and the sensitivity of available resources to Qwen startup state. Neither observation establishes that such a policy will win; related adaptive approaches must be reviewed before claiming novelty — that review is now in §8.2, and it narrows the claim.

| Research step | Concrete design | Acceptance / falsification |
| --- | --- | --- |
| Establish valid baselines | Repair §6 harness gaps in fix_bug.md; immutable matched startup/pools; randomized repeats; add a released engine image as a reference arm | Keep the same retention criteria for all policies, including slow runs |
| Explain the round | Record accepted/committed tokens, draft/verify/state-copy time, live batches, per-step prefill tokens, state-pool versus KV-pool occupancy, preemptions and per-request timelines | Determine whether overhead, acceptance, or state pressure explains each loss; absent counters remain unknown |
| Intervene | Compare MTP off, fixed n1/n5, the strongest published load-conditioned policy available in the engine (DSpark-style confidence trimming or an adaptive token budget), and an explicitly specified budget that also reads state-pool pressure | Same model, precision, traffic, cache protocol, quality checks and latency objectives |
| Evaluate | Completed requests meeting **both** TTFT and TPOT targets per unit time; tails, failures, policy overhead and memory | A median below a target is not the fraction of requests meeting it |
| Falsify | Compare against the strongest fixed and load-conditioned policies within each matched workload | No useful gain after controls, or gains erased by policy/state overhead, reject this hypothesis for that regime |

Targets must be specified before comparing policies; they are not inferred from the best-looking curve. This is proposed work, not an implemented controller, a replay result or a demonstrated new contribution. It turns the present uncertainty into a reviewable systems experiment.

### 8.2 Related work that already conditions speculation or state on load [P]

| Prior mechanism | What it conditions on | What it does not do | Consequence for the hypothesis |
| --- | --- | --- | --- |
| DSpark confidence-scheduled verification (DeepSeek, June 2026; open-sourced as DeepSpec; deployed for V4) | A trained per-position confidence head plus a one-time profile of the engine's marginal verify-token cost at each load level; trims each request's verify budget per step | Does not read recurrent-state pressure; drafter is DFlash-style, not native MTP ([paper](https://arxiv.org/html/2607.05147v1)) | The "live load" half of the hypothesis is an existing, deployed technique; it must be the baseline, not the contribution |
| SGLang trim planner for Kimi K3 | Same confidence × cost-staircase planning; measured break-even to batch 8 and +68%/+24% at batch 256 | Small-batch early exit noted as follow-up ([LMSYS](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)) | Provides the measured shape of the verify-cost curve the local trace should reproduce |
| vLLM 0.28 adaptive speculative token budget (K3 DSpark, ~60% better TTFT) and SGLang adaptive MTP for GLM-5.3-Flash (`--speculative-adaptive`) | Engine-side budget adaptation | Not documented in the local dev builds ([vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.28.0), [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash)) | An off-the-shelf adaptive arm exists for at least GLM; compare against it |
| ReplaySSM (Tri Dao, 2026) | Replaces per-draft-step state snapshots with stored raw inputs and a fold kernel; ~32× less draft-window memory on K3 | Addresses state *volume* of speculation, not the decision to speculate | Removes most of the state-pressure term the hypothesis leans on; the experiment must run with and without such a path |
| dasc (Meituan/ECNU, Aug 2026) and Marconi (2025) | Compress recurrent checkpoints by retention horizon; admit/evict hybrid prefix entries by reuse utility | Prefix caching, not speculation ([dasc](https://arxiv.org/html/2608.30386)) | Defines "state pressure" measurably: checkpoint pool occupancy and eviction rate |
| HyperDFlash (ByteDance, June 2026) | Per-position acceptance of native MTP under hyper-connected residuals; block drafting aligned to mHC | Not a scheduling policy ([paper](https://arxiv.org/pdf/2606.26744)) | Explains why native MTP acceptance decays past the first positions — relevant to the n=5 result |

The defensible narrowed claim: *jointly* conditioning the draft budget on load **and** on recurrent-state pool pressure — with the state term measured as checkpoint occupancy and eviction rate rather than inferred — has not been reported for hybrid models; whether it beats DSpark-style trimming plus a ReplaySSM-style state path is the open question, and a null result is a legitimate outcome.

### 8.3 Experiment priorities

| Priority | Experiment | Question resolved |
| --- | --- | --- |
| 1 | Matched-startup Qwen base/MTP/context/prefix; same-session DeepSeek context rerun | Which rankings survive controlled pools and startup? |
| 2 | Randomized repeated runs with longer windows | Are small differences reproducible? |
| 3 | Profile prefill, decode, GEMMs, dispatch, attention, hardware memory traffic; log per-step prefill-token counts and draft/verify times | What actually limits each regime, and how much of c64 TPOT is prefill interference (§3.5)? |
| 4 | Sweep chunked-prefill budget and feasible TP/EP layouts; test DP-attention + EP for DeepSeek and DCP for GLM (§7.3) | Can goodput improve at fixed latency objectives? |
| 5 | Cache-off, cold-fill, and prewarmed-prefix controls with per-group hit counters, aligned-prefix rerun for GLM (§4.4), state/eviction/preemption counters | What savings come from reuse, and what does hybrid state cost? |
| 6 | Representative task/session replay with real text, recorded output lengths, reasoning content retained in history, and quality evaluation | What is cost per successful task and realistic MTP acceptance? |
| 7 | Repeat the c1/c16 GLM points on a released engine image (SGLang or vLLM stable) on the same node | How much of the gap to published H100 numbers (§3.7) is engine maturity? |
| 8 | On Blackwell if available: FP8 KV + TRT-LLM DSA for GLM; native FP4 experts and FP4 indexer cache for DeepSeek | Does the Hopper-specific ordering in §3.8 reverse? |

Saved TraceLab summaries motivated the input-length grid; no real trace replay was performed. Do not interpret reusable-prefix estimates as guaranteed achievable cache hits. No dense baseline, multimodal test, disaggregated deployment, or validated layer-reduction extrapolation is included.

The document audit also found remaining harness gaps: unavailable cache telemetry can default to zero, positive cache-hit warnings do not reject a point in that branch, and completion validation rejects zero rather than all partial failures. The retained JSON audit checks saved fields; it cannot independently prove successful original telemetry. These are documented as **pending**, not fixed, in [fix_bug.md §6](fix_bug.md#6-remaining-gaps-in-the-saved-harness--not-fixes-performed-here).

## 9. Professor questions: answers to rehearse

| Question | Defensible answer |
| --- | --- |
| What did you establish? | Named deployment tradeoffs, GLM MTP/backend A/Bs, and measurement confounds. |
| Which model is best? | Qwen leads corrected 16K throughput; GLM has lower observed 131K median TTFT. Quality and optimal deployments are unmeasured. |
| Which part is the bottleneck? | I have hypotheses, not operator attribution. At c1 all three sit 10–30× above their HBM floor, so the step is launch/latency-bound; at c64 the excess over the floor is consistent with prefill chunks sharing steps. Separate prefill/decode traces and counters are next. |
| Why not claim bandwidth is low? | Partial modeled counters over mixed request work cannot rule out bandwidth-bound kernels. |
| Is concurrency batch size? | No; it caps client requests. Engine token batches vary each step. |
| What does your grid actually measure? | Input work: 64 input tokens per output token on the main grid, up to 1,000 in the context sweep. The rankings are prefill-plus-scheduling rankings; the derived marginal input rates (§3.6) are the closest thing to a scaling curve. |
| Why is DeepSeek slowest with the fewest active parameters? | On H100 its FP4 experts run through dequantisation kernels and its FP4 indexer cache is Blackwell-only; it also runs an indexer in roughly twice as many layers with twice the heads. Those are candidate causes, not attributions. |
| Is the comparison fair? | GPU class/count and workload targets match; precision, tokenizers, backends, builds, and some pools differ. This compares deployments. |
| Does the bridge remove engine confounds? | Only a small DeepSeek batch-grid build effect was observed; other models and axes remain uncontrolled. |
| Why do published H100 numbers for GLM exceed yours? | Different prompt length, speculation on, a released image with fused sparse-attention kernels, and a different metric. The gap is inside what those explain; priority 7 tests it. |
| Can recurrent state be cached? | Yes, via compatible checkpoints at reused boundaries, together with other layer state. One GLM checkpoint costs as much memory as ~6K tokens of KV, so engines keep them sparsely and replay in between. |
| How much does prefix caching accelerate inference? | We measured sharing patterns, not on/off. Best/worst differs up to 3.42× for DeepSeek, and every cell exceeds the no-reuse bound, but the reuse fraction per point is unrecorded. |
| Could GLM's prefix result be an engine artifact? | Possibly: a documented vLLM align-mode failure puts the retained recurrent checkpoint inside the unique suffix for exactly this geometry. Per-group hit counters and an aligned rerun would tell. |
| Why does MTP improve TTFT? | That is observed for some GLM contexts. Scheduling is a hypothesis, not a traced cause; a configuration diff between arms is the first check. |
| Should MTP always be enabled? | No. GLM benefits at low concurrency; loaded throughput is flat/worse. On an MoE target every verified position streams its own experts, so verification is not free. Evaluate latency, memory, and quality too. |
| Why is the light-load gain only 1.24×? | Because the step is launch-bound, a draft round adds fixed cost; the fit in §5.4 reproduces both n1 and n5 with ≈ 0.45 of a target step per draft step. Fused draft kernels change that, per SGLang's published GB300 results. |
| Isn't an adaptive draft budget already DSpark? | The load-conditioned part is; DSpark and SGLang's planner are the baselines. The open part is conditioning jointly on recurrent-state pressure, and it may fail. |
| Is FP8 KV impossible on Hopper? | No blanket claim is valid: GLM's alternate stack ran it, with a speed/capacity tradeoff. On Blackwell the published sign is reversed. |
| Why not fewer GPUs? | It may reduce cost, but no successful deployment sweep establishes the optimum. Eight GPUs were the experimental budget. |
| Where are error bars? | Most cells have one retained run. Within-run percentiles are not repeat-based confidence intervals. |
| Is $10.73 a real quote? | No: $2.50/GPU-hour times observed runtime, normalized to outputs and including input work. Per request, the same run is within ±25% of the vendors' list prices before cache hits. |
| What would you build for SyFI? | Validate regimes first, then test state-aware cache/scheduling policies against measured goodput. |

Presentation discipline: use **"observed," "for this arm," and "hypothesis."** Avoid "proved communication-bound," "all engines matched," "cache speedup," and "cheapest model" without the missing controls.

## Appendix A. External references consulted for this revision

All accessed 6 September 2026. Local files are linked inline in the sections above.

- DeepSeek, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* (arXiv:2606.19348): https://arxiv.org/pdf/2606.19348
- Hugging Face, *DeepSeek-V4: a million-token context that agents can actually use*: https://huggingface.co/blog/deepseekv4
- vLLM recipes, DeepSeek-V4-Flash: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash · NVIDIA NeMo AutoModel notes: https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/deepseek-v4-flash
- DeepSeek, *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation* (arXiv:2607.05147): https://arxiv.org/html/2607.05147v1 · VentureBeat coverage: https://venturebeat.com/orchestration/deepseek-open-sources-dspark-a-new-framework-to-speed-up-llm-inference-by-up-to-85
- ByteDance, *HyperDFlash* (arXiv:2606.26744): https://arxiv.org/pdf/2606.26744
- Z.ai, GLM-5.3-Flash overview: https://docs.z.ai/guides/vlm/glm-5.3-flash · Hugging Face model card: https://huggingface.co/zai-org/GLM-5.3-Flash
- KGP Talkie, *GLM 5.3 vs GLM 5.3 Flash architecture teardown* (tensor-level census): https://kgptalkie.com/tutorials/llm-benchmarking/glm-5-3-vs-glm-5-3-flash-architecture-teardown
- SGLang cookbook, GLM-5.3-Flash (H100/H200/B200/B300/GB300 benchmarks): https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash · vLLM recipe: https://recipes.vllm.ai/zai-org/GLM-5.3-Flash · NVIDIA NeMo AutoModel: https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thudm/glm-5-3-flash
- Qwen Team, *On the Design of Qwen3.8-Next Architecture* (arXiv:2608.30320): https://arxiv.org/html/2608.30320 · GitHub README: https://github.com/QwenLM/Qwen3.8-Flash-Next · NVIDIA GB300 blog: https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-on-nvidia-gb300-nvl72-for-agentic-coding/ · spec/recipe summary: https://intuitionlabs.ai/articles/qwen3-8-flash-next-architecture-memory · Unsloth notes: https://unsloth.ai/docs/models/qwen3.8-next
- LMSYS/SGLang, *Day-0 support for Kimi K3* (state moves, unified pool, DSpark trimming, ReplaySSM, DCP): https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support · vLLM v0.28.0 release notes: https://github.com/vllm-project/vllm/releases/tag/v0.28.0
- vLLM hybrid prefix caching: tracking issue https://github.com/vllm-project/vllm/issues/26201 · align-mode 0%-hit bug https://github.com/vllm-project/vllm/issues/45238 · retention interval PR https://github.com/vllm-project/vllm/pull/45845 · SGLang MambaRadixCache for KDA https://github.com/sgl-project/sglang/issues/26575
- PyTorch blog, *Hybrid Models Meet SGLang*: https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/ · LMCache hybrid-model docs: https://docs.lmcache.ai/mp/hybrid_models.html
- *dasc: Decay-Aware State Compression for Hybrid Linear-Attention Serving* (arXiv:2608.30386): https://arxiv.org/html/2608.30386
- Kimi K2 Thinking vLLM recipe (DCP capacity/throughput): https://recipes.vllm.ai/moonshotai/Kimi-K2-Thinking · community Kimi-K2.5 MTP card (mismatched-draft acceptance): https://huggingface.co/yrrhall/Kimi-K2.5-MTP
- Pricing and hosted-API measurements: Morph (DeepSeek V4 Flash) https://www.morphllm.com/deepseek-v4-flash · codersera (GLM-5.3-Flash) https://codersera.com/blog/glm-5-3-flash-complete-guide-2026/ · Artificial Analysis (Qwen3.8-Flash-Next) https://artificialanalysis.ai/models/qwen3-8-flash-next · eesel (Qwen review, cache-hit rate) https://www.eesel.ai/blog/qwen38-flash-next-review · Yotta Labs (GLM vs DeepSeek hosting) https://www.yottalabs.ai/post/glm-5-3-flash-vs-deepseek-v4-flash-2026
- SemiAnalysis InferenceX (DeepSeek V4 Pro on B200/B300; blog index): https://inferencex.semianalysis.com/compare/deepseek-v4-b200-vs-b300 · https://inferencex.semianalysis.com/blog