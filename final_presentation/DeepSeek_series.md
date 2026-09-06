# DeepSeek-V3 to DeepSeek-V4 Architecture and Serving Performance Report

## Release verification and scope

**Scope.** This report begins with DeepSeek-V3 and follows the verified lineage through R1, V3.1, V3.2, the V4 preview, and the latest V4 production snapshots. DeepSeek-V2 is mentioned only where it originated Multi-Head Latent Attention (MLA) and DeepSeekMoE.

**Verification date and research cutoff.** **4 September 2026 (UTC+7).** The latest verified text-only production snapshots are **DeepSeek-V4-Flash-0731** (released 31 July 2026, public beta) and **DeepSeek-V4-Pro-0813** (released 13 August 2026, general availability). The newest official V4-family endpoint is **DeepSeek-V4-Flash-Vision-Exp**, released 21 August 2026, but it is an experimental multimodal derivative, not a newer text backbone. The rolling API identifiers `deepseek-v4-flash` and `deepseek-v4-pro` currently resolve to 0731 and 0813, respectively. [Official changelog](https://api-docs.deepseek.com/updates/) and [current model/pricing page](https://api-docs.deepseek.com/quick_start/pricing/).

**Release classes.** The 24 April 2026 V4 release was explicitly a **preview**, with open weights for Pro and Flash plus a technical report. Flash-0731 kept the preview architecture and size and was “only re-post-trained,” according to the official changelog. Pro-0813 is the GA production snapshot; DeepSeek discloses agent/post-training improvements but no evidence of a changed backbone. Both production API models support non-thinking and thinking modes with `low`, `high`, and `max` reasoning effort. `V4-Pro-Max` and `V4-Flash-Max` are reasoning-effort configurations, not separately downloadable base checkpoints. [Preview announcement](https://api-docs.deepseek.com/news/news260424/), [GA announcement](https://api-docs.deepseek.com/news/news260813/), [V4 report](https://arxiv.org/abs/2606.19348).

**Weights and reports.** Downloadable official weights exist for V3 Base/Instruct, R1 and R1-Zero, V3.1 Base/Instruct/Terminus, V3.2-Exp, V3.2, V3.2-Speciale, and the V4 preview Pro/Flash checkpoints. V4 production 0731/0813 repositories are referenced by serving recipes and official API mappings; the report does not assume that every rolling API build is bit-identical to the April weights. V3, R1, V3.2, and V4 have technical reports. V3.1 has official release notes and model cards, but no comparably complete standalone architecture report located by this cutoff.

**Naming reconciliation.** Papers use family/capability names such as V4-Pro-Max; Hugging Face uses immutable checkpoint names; GitHub often hosts architecture code rather than every dated weight; API docs expose stable aliases that silently advance; announcements use “official,” “preview,” or “GA.” Therefore an API result must record both alias and resolved snapshot. No official evidence was found for a text checkpoint newer than Pro-0813 or Flash-0731. Catalog entries such as “V4-Flash-0901,” “V4.1,” or claims that 0731 changed the architecture are rejected unless they appear in DeepSeek’s own documentation, repositories, model cards, or reports.

**Purpose.** Connect architecture to computation, memory traffic, communication, cache behavior, latency, throughput, deployment complexity, and cost.

**Terminology.** *Officially reported* means stated by DeepSeek; *independently measured* means a reproducible external run; *analytically estimated* means calculated here; *not disclosed* means no primary-source value was found. “Active parameters” is not exact FLOPs. “Reasoning mode” is a post-trained behavior plus decoding/API policy, not necessarily a different backbone.

**Disclosure limitations.** DeepSeek has not disclosed V4 GPU type/count, GPU-hours, wall-clock duration, full production-snapshot training data, or whether the API snapshots exactly equal public files. Hardware cost examples therefore use official API prices, while self-hosting capacity is analytical and must be benchmarked.

## 1. Executive summary

1. **V3 was a systems co-design milestone.** Its 671B/37B-active MoE combined MLA, fine-grained experts, auxiliary-loss-free balancing, MTP, FP8 training, and DualPipe. Officially: 14.8T tokens, 2.664M H800 GPU-hours for pretraining, and 2.788M for full training. [V3 repository/report](https://github.com/deepseek-ai/DeepSeek-V3).
2. **R1 is primarily post-training.** It uses the V3-derived 671B/37B backbone. The major serving change is longer serialized reasoning, not new MLA or MoE kernels. [R1 report](https://arxiv.org/abs/2501.12948), [TensorRT-LLM guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v3/README.md).
3. **V3.1 is official.** It continued pretraining for 840B tokens, extended context/agent behavior, updated tokenizer/template, and offered one checkpoint with think/non-think modes. Terminus is a refinement. [V3.1 release](https://api-docs.deepseek.com/news/news250821/).
4. **V3.2 changes attention.** DSA adds a learned lightning indexer and top-k sparse MLA; the rest remains close to the V3 family. V3.2-Speciale is a high-compute reasoning post-training variant, not a new architecture. [V3.2 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2).
5. **V4 is a true backbone change.** It retains DeepSeekMoE and V3-style MTP but replaces MLA/DSA with interleaved CSA and HCA, adds mHC residual pathways, hash-routed early MoE layers, and trains with Muon. [V4 report](https://arxiv.org/html/2606.19348v1).
6. **V4 has two capacity tiers.** Flash is 284B/13B-active, 43 layers, 256 routed experts with 6 selected plus 1 shared; Pro is 1.6T/49B-active, 61 layers, 384 routed experts with 6 selected plus 1 shared. Both support 1M context. These are not merely width-scaled copies because layer, hidden, head, top-k, and expert configurations differ.
7. **Prefill bottlenecks move with length.** Short prompts pay extra compression/indexing overhead; very long prompts benefit from CSA’s approximately `O(S*K/m)` selected attention and HCA’s `O(S^2/m')` compressed dense attention rather than conventional dense `O(S^2)`.
8. **Decode is often bandwidth/latency bound at low batch.** Active expert weights, compressed history, small GEMMs, kernel launches, and all-to-all dominate. Flash should be easier at batch 1, but this is an analytical expectation, not a measured guarantee.
9. **At high concurrency, MoE gets more efficient and more dangerous.** Larger expert token groups improve GEMM utilization, while expert imbalance and dispatch/combine traffic can saturate NVLink/RDMA.
10. **MLA reduces, but does not eliminate, sequence-dependent cache.** Correct V3/R1 prefix reuse restores compressed KV latent plus decoupled RoPE state and exact position/layout metadata.
11. **V4 prefix state is heterogeneous.** It includes compressed CSA/HCA entries, lightning-indexer-related entries, SWA state, and incomplete compression tails. DeepSeek’s report explicitly describes block-aligned storage and tail recomputation.
12. **MTP is conditional acceleration.** It helps when acceptance is high and target verification amortizes multiple steps. At high concurrency, extra draft work and state rollback may be neutral or harmful.
13. **Reasoning cost is output-length cost.** Per-token backbone compute can remain similar while long hidden traces increase latency, cache residence, scheduler unfairness, and billable output.
14. **V4 used more tokens.** Flash used 32T and Pro 33T versus V3’s 14.8T, but elapsed time is unknown. Muon, larger batches, newer clusters, FP4/FP8 paths, and better overlap can reduce time per token without proving lower total GPU-hours.
15. **Latest verified status.** Flash-0731 is architecture-preserving re-post-training; Pro-0813 is the latest production text snapshot. Vision-Exp is newer but experimental and multimodal.

```text
DeepSeek-V3: MLA + DeepSeekMoE + loss-free balancing + MTP + FP8 + DualPipe
        |
DeepSeek-R1: same V3-class backbone + cold start/GRPO/reasoning behavior
        |
DeepSeek-V3.1: 840B continued-pretraining tokens + hybrid modes + agents
        |
DeepSeek-V3.2: DSA sparse indexer + scaled reasoning/agent post-training
        |
DeepSeek-V4: CSA/HCA + mHC + adjusted DeepSeekMoE + Muon + native 1M context
        |-- Flash-0731: same architecture, re-post-trained production snapshot
        `-- Pro-0813: GA production snapshot, stronger agents/effort control
```

## 2. Release timeline and disclosure matrix

All unqualified values in this table are **officially reported**; “ND” means not disclosed.

| Model | Date | Identifier/status | Weights | Type/modes | Params total/active | Experts | Attention/context | Training | Hardware/GPU-hours/FLOPs | License/report |
|---|---:|---|---|---|---:|---|---|---|---|---|
| V3 | 2024-12-26 | `DeepSeek-V3`; production | Open Base + post-trained | instruct | 671B/37B | 256 routed, top-8; 1 shared | MLA; 128K | 14.8T | H800; 2.664M pretrain, 2.788M full; reported FLOPs ND | model license initially, later MIT snapshots; [report](https://arxiv.org/abs/2412.19437) |
| R1 | 2025-01-20 | `DeepSeek-R1` | Open post-trained + distills | reasoning | 671B/37B | same V3 class | MLA; 128K | base tokens inherited; RL data ND | post-train compute ND | MIT; [report](https://arxiv.org/abs/2501.12948) |
| V3-0324 | 2025-03-25 | dated production | Open | instruct | 671B/37B | V3 | MLA; 128K | ND | ND | MIT; V3 report |
| V3.1 | 2025-08-21 | `DeepSeek-V3.1` | Open Base + post-trained | think/non-think | 671B/37B | V3 class | MLA; 128K | +840B continued pretraining | ND | MIT; release notes |
| V3.1-Terminus | 2025-09-22 | `DeepSeek-V3.1-Terminus` | Open | hybrid | 671B/37B | V3 class | MLA; 128K | ND | ND | MIT; release notes |
| V3.2-Exp | 2025-09-29 | experimental | Open | hybrid | 671B/37B | V3 class | DSA; 128K | aligned to Terminus; exact ND | ND | MIT; report |
| V3.2 | 2025-12-01 | production | Open | think/non-think | 671B/37B | V3 class | DSA; 128K | ND | ND | MIT; technical report |
| V3.2-Speciale | 2025-12-01 | temporary API + open weights | Open | high-compute reasoning, no tools | 671B/37B | same | DSA; 128K | post-training ND | ND | MIT; V3.2 report |
| V4-Flash Preview | 2026-04-24 | preview | Open | base-derived post-trained; think/non-think/max | 284B/13B | 256 routed, top-6; 1 shared | CSA/HCA; native 1M | 32T | hardware/GPU-hours/FLOPs ND | MIT; V4 report |
| V4-Pro Preview | 2026-04-24 | preview | Open | base-derived post-trained; think/non-think/max | 1.6T/49B | 384 routed, top-6; 1 shared | CSA/HCA; native 1M | 33T | ND | MIT; V4 report |
| V4-Flash-0731 | 2026-07-31 | API production beta | snapshot availability documented by engines | re-post-trained; low/high/max | same as preview | same | same architecture; 1M | additional post-training ND | ND | API terms / checkpoint MIT where downloadable |
| V4-Pro-0813 | 2026-08-13 | API GA | snapshot availability documented by engines | post-trained; low/high/max | same disclosed family size | same disclosed design | 1M | additional post-training ND | ND | API terms / checkpoint MIT where downloadable |
| V4-Flash-Vision-Exp | 2026-08-21 | API experimental | API-only in official release | multimodal + text modes | ND in official API page | ND | V4 text stack + vision; 1M | ND | ND | no dedicated report found |

## 3. DeepSeek-V3: the starting point

V3 became an industry moment because it demonstrated coherent algorithm-system co-design at frontier scale, not merely a low headline cost. A sparse 671B model still needs the entire checkpoint distributed across memory, but activates about 37B parameters per token. MLA reduces cache traffic; fine-grained MoE reduces arithmetic; router control avoids quality-damaging auxiliary pressure; FP8 improves matrix/communication efficiency; DualPipe overlaps pipeline and expert traffic; MTP improves representation learning and can draft future tokens. The reported 2.788M H800 GPU-hours excludes failed research, data generation, salaries, and earlier experiments. [V3 official repository](https://github.com/deepseek-ai/DeepSeek-V3).

### 3.1 DeepSeekMoE

A basic top-k MoE routes each token to a few large experts. DeepSeekMoE segments FFNs into many finer experts, adds always-active shared experts for common knowledge, and selects routed experts for specialization. In V3, the official configuration uses 256 routed experts, 8 selected per token, and one shared expert. Routing computes affinities, adds training-time balancing control, chooses experts subject to group/node constraints, permutes tokens into expert-contiguous buffers, performs expert GEMMs, then all-to-all combines outputs.

The distinction between **total** and **active** parameters reduces arithmetic but not weight residency. FP8 weights for 671B parameters are roughly 671 GB before metadata; BF16 is about 1.342 TB. Checkpoint loading, host RAM, storage, and fault recovery follow total size. At batch 1, each selected expert may receive one token, making GEMMs tiny and dispatch latency prominent. Small batches improve little unless requests route similarly. Medium/high batches produce larger expert matrices and higher arithmetic intensity, but all-to-all bytes and hot experts grow. A single node benefits from NVLink/NVSwitch; multiple nodes expose RDMA bandwidth and tail latency.

There is no universal “tokens per expert” threshold. Efficiency depends on expert dimensions, dtype, GPU generation, fusion, and grouped-GEMM kernel. Measure occupancy versus tokens/expert. As a rule, dozens to hundreds of tokens per expert per microbatch are more favorable than one to eight, but this is an analytical heuristic, not a DeepSeek number. Offline balanced training does not guarantee balanced production traffic: code, one language, or repeated agent templates can create expert hot spots.

### 3.2 Multi-Head Latent Attention

For conventional MHA/GQA, assuming `L` layers, sequence length `S`, `Hkv` KV heads, head dimension `D`, and `b` bytes/element:

```text
Cache_MHA = L * S * Hkv * D * 2 * b bytes
```

The factor 2 stores keys and values. Ignoring allocator padding and scales, decode reads approximately the same history bytes per generated token:

```text
Bytes_read_MHA/token ~= L * S * Hkv * D * 2 * b
```

MLA down-projects hidden state into a compressed KV latent `c_t^KV` of dimension `Dc`, retains a decoupled positional/RoPE key state of dimension `Dr`, and reconstructs or algebraically absorbs up-projections into adjacent weights. A simplified cache is:

```text
Cache_MLA ~= L * S * (Dc * bc + Dr * br) bytes
Reduction ~= (2 * Hkv * D * b) / (Dc * bc + Dr * br)
Bytes_read_MLA/token ~= L * S * (Dc * bc + Dr * br)
```

Assumptions: every layer caches one latent and positional component per token; scales/padding are omitted; absorbed projection avoids materializing full K/V but does not make computation free. Low-rank query projection reduces query-side work; low-rank KV projection produces the compressed latent; decoupled RoPE separates position from content. Weight absorption can transform the query/output path so attention consumes latent entries directly.

Serving gains are lower per-token cache storage, lower HBM history reads during decode, greater concurrent sequence capacity, and lower long-context cost. Challenges are custom kernels, fused projections, absorbed-weight layouts, precision-sensitive latent quantization, checkpoint conversion, and engine-specific prefix formats. MLA absolutely does **not** remove sequence growth: it stores less state for every preceding token.

### 3.3 Auxiliary-loss-free expert balancing

Conventional MoE adds a load-balancing loss to discourage router collapse, but its gradient competes with language modeling and may force semantically poor routing. V3 keeps a per-expert routing bias used for selection; after observing loads, it raises bias for underloaded experts and lowers it for overloaded experts, without placing the main balancing pressure in the LM loss. A small sequence-wise term may remain in later designs. This improves training quality/stability and reduces serving hot spots, but serving distributions can differ from training.

### 3.4 Multi-Token Prediction

MTP adds sequential prediction modules trained to predict tokens beyond the immediate next token. It is first a **training objective**. At inference, the attached MTP module can propose several tokens; the main model verifies them in one target pass, commits the accepted prefix, rejects at the first mismatch, and rolls back unaccepted cache state. This differs from a separate small draft model, prompt lookup, and n-gram matching. Native MTP shares representations and checkpoint state but consumes memory for draft modules. V3’s repository notes 14B MTP-module parameters in addition to the 671B main model.

### 3.5 FP8 training

V3 applies FP8 to major GEMMs with higher-precision accumulation and uses higher precision for sensitive operations, master weights, optimizer states, and selected communications. Scaling is fine-grained, with block/tile strategies and special handling for outliers and accumulation. FP8 reduces operand memory, HBM traffic, and communication, allowing more effective compute per GPU-hour without changing token count. It does not imply that released inference files use the identical training format or scales.

### 3.6 DualPipe and distributed training

Pipeline parallelism splits layers across stages; ordinary 1F1B schedules leave fill/drain bubbles and can collide all-to-all with compute. DualPipe schedules forward/backward work bidirectionally and overlaps expert dispatch/combine with matrix multiplication. It was motivated partly by H800 cross-node bandwidth constraints. DualPipe is a training schedule, not an inference requirement, although its overlap principles inspire serving. GPU-hours equal GPU count times elapsed hours, but do not disclose count, wall time, failures, data generation, or R&D cost.

## 4. DeepSeek-R1: architecture versus post-training

R1-Zero applies large-scale GRPO-style RL directly to a V3-Base-class model, producing reflection, verification, and long reasoning but also repetition, language mixing, and poor readability. R1 adds cold-start SFT, reasoning-oriented RL, rejection-sampled SFT, and a final alignment RL stage. GRPO scores groups of sampled outputs relative to their group statistics, avoiding a separate critic. Verifiable math/code rewards provide cleaner signals than subjective tasks. Six smaller dense Qwen/Llama derivatives distill R1 behavior. [Official R1 repository](https://github.com/deepseek-ai/DeepSeek-R1).

Direct answers:

1. **Different core kernels?** No for the main 671B model. TensorRT-LLM explicitly shares the V3/R1 code path.
2. **MLA changed?** No evidence.
3. **DeepSeekMoE changed?** No evidence.
4. **Active parameters changed?** Reported 37B, the same V3 class.
5. **Primary serving change?** More generated reasoning tokens.
6. **Why cost rises?** Similar per-token backbone work multiplied by far more serialized tokens.
7. **System effects.** Long traces grow MLA cache, keep requests resident, delay later users, reduce scheduler fairness, and amplify p99.
8. **Counting hidden reasoning.** Count all generated tokens that consume model compute and provider billing, even if not shown to the user.
9. **What is reasoning mode?** Primarily post-trained behavior activated by template/decoding/API policy. It need not be a separate backbone, though a provider may route to a distinct snapshot.

## 5. DeepSeek-V3.1 and DeepSeek-V3.2

V3.1 is official and deserves a release subsection, not a separate kernel architecture: 840B continued-pretraining tokens, updated tokenizer/template, 128K context, hybrid think/non-think, and stronger agent post-training. Terminus refined language consistency and agents. V3.2-Exp then introduced DSA on Terminus; production V3.2 scaled RL and synthesized tool-use trajectories. [V3.2-Exp repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp), [V3.2 release/model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2).

### 5.1 DeepSeek Sparse Attention

DSA adds a learned lightning indexer. For each query it scores historical compressed/index keys, selects top-k entries, gathers their MLA KV state, and computes sparse attention. Dense prefill is `O(S^2)` attention-score work; selected-context attention is approximately `O(S*K)` when `K << S`. Actual cost includes index projection, score construction, top-k/sorting, metadata, irregular gathers, and launches. NVIDIA documents a typical V3.2 top-k of 2048. [TensorRT-LLM V3.2 analysis](https://nvidia.github.io/TensorRT-LLM/1.3.0rc25/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.html).

Sparse attention can lose relevant context if the indexer misses it. Reads are less regular than dense FlashAttention, so short contexts may be slower. The crossover depends on GPU, sequence, batch, and kernel. Prefix caches need persistent MLA KV/index keys; query-specific top-k selections should generally be recomputed because a new query changes scores.

### 5.2 Agentic and reasoning post-training

V3.2’s gains come from multiple causes: DSA lowers long-context cost; long-context training enables larger windows; synthetic tool interactions teach tool formatting and recovery; scaled RL improves reasoning; Speciale spends greater test-time compute; agent data improves coding/search planning. Benchmark gains cannot be attributed to DSA alone without matched post-training and reasoning budgets.

## 6. DeepSeek-V4 architecture

V4 retains Transformer, DeepSeekMoE, and one-depth MTP, but changes the backbone through CSA/HCA and mHC. It adjusts routing affinity from sigmoid to `sqrt(softplus)`, removes node-count routing constraints, adds a small sequence balance loss, and uses hash routing in the first three MoE layers. It also uses FP4 QAT for routed experts and the indexer path in post-training. [V4 technical report](https://arxiv.org/html/2606.19348v1).

### 6.1 Pro versus Flash

| Feature | V4-Flash | V4-Pro |
|---|---:|---:|
| Total / active | 284B / 13B | 1.6T / 49B |
| Layers / hidden | 43 / 4096 | 61 / 7168 |
| Experts | 256 routed, 6 selected, 1 shared | 384 routed, 6 selected, 1 shared |
| Expert intermediate | 2048 | 3072 |
| Query heads / head dim | 64 / 512 | 128 / 512 |
| CSA compression / top-k | 4 / 512 | 4 / 1024 |
| HCA compression | 128 | 128 |
| Query compression | 1024 | 1536 |
| Initial attention | 2 pure SWA layers | 2 HCA layers |
| Context | 1,048,576 | 1,048,576 |
| Pretraining tokens | 32T | 33T |

Flash is not simply a smaller Pro: it has different depth, dimensions, experts, top-k, and initial-layer attention. Flash likely wins batch-1 latency and throughput/GPU because fewer active and resident weights are involved. Pro requires more expert storage and likely more communication, but can deliver greater capability per request. Cost per successful task remains empirical: Flash may need more reasoning tokens/retries; Pro may solve in fewer attempts.

### 6.2 Compressed Sparse Attention

CSA generates weighted compressed KV summaries, one entry per `m=4` original tokens, using overlapping `a/b` branches. A low-rank multi-head lightning indexer scores compressed index keys, selects top-k compressed entries, then shared-KV MQA performs exact attention on selected summaries plus a 128-token uncompressed sliding window. Pro selects 1024; Flash 512.

At short prefill, compression, indexer, and top-k overhead may exceed savings. At long prefill, reducing sequence by four and selecting fixed K changes the dominant cost. At 1M, candidate scoring itself is substantial and may be low-arithmetic-intensity. Batch-1 decode reads summary/index caches, runs top-k, sparse gather, SWA, and MoE. High concurrency can batch these operations but introduces selection/load imbalance. Shared-prefix workloads benefit only if compressed blocks and state tails restore exactly.

### 6.3 Heavily Compressed Attention

HCA is not MLA. It combines every `m'=128` tokens into one learned compressed KV entry and performs dense MQA across those summaries, plus SWA. Relative to MHA/GQA, it compresses sequence state; relative to MLA, it compresses across tokens rather than only channel rank; relative to CSA, it uses much stronger compression but no sparse top-k over summaries.

V4 stores mixed-precision entries, BF16 for RoPE dimensions and FP8 elsewhere, plus sliding-window and incomplete-block state. At 1M, DeepSeek analytically reports Pro at 10% and Flash at 7% of V3.2 KV cache; versus BF16 GQA8 with head dimension 128, V4 is about 2%. These are report-level analytical comparisons at 1M, not independent measured serving RSS. Prefix portability is hard because block ratios, precision/scales, SWA tail, and engine layout must match.

### 6.4 Manifold-Constrained Hyper-Connections

A conventional residual stream is one vector `x`; HC expands it to `n_hc` pathways and computes `X_{l+1}=B_l X_l + C_l F_l(A_l X_l)`. Unconstrained dynamic mixing can amplify singular values across deep stacks. mHC constrains `B_l` to the doubly stochastic Birkhoff polytope using 20 Sinkhorn-Knopp iterations and bounds input/output maps with sigmoid. V4 uses expansion factor 4.

This improves signal/gradient stability and feature reuse, but expands activation state and pipeline communication. DeepSeek uses recomputation and fusion to mitigate memory. At inference, mHC adds within-layer dynamic mixing and small matrix work; it changes transient/residual activations but not history cache semantics. It therefore should not independently alter prefix-cache format, although checkpoint and kernels must support it.

### 6.5 Muon optimizer

Most matrix parameters use Muon; embeddings, prediction head, RMSNorm weights, and mHC static biases/gates remain AdamW. Muon momentum is 0.95; orthogonalization uses 10 hybrid Newton-Schulz iterations, first 8 aggressive and last 2 stabilizing, with BF16 matrix multiplies and FP32 local reduction where needed. Flash peaks at batch 75.5M tokens and learning rate `2.7e-4`; Pro at 94.4M and `2.0e-4`, both after 2,000 warmup steps.

Muon does not reduce deployed inference FLOPs or alter inference architecture. Faster convergence and stable larger batches can reduce tokens or elapsed time, but V4 actually used far more tokens than V3. Orthogonalization and full-matrix optimizer state add overhead, partially offset by hybrid ZeRO and batched updates.

## 7. Training tokens, compute, and elapsed time

| Model | Total/active | Tokens | Precision/hardware | Reported GPU-hours/FLOPs/wall time |
|---|---:|---:|---|---|
| V3 | 671B/37B | 14.8T | FP8 mixed; H800 | 2.664M pretrain, 2.788M full; FLOPs/time ND |
| R1 | 671B/37B | inherited base; post-train ND | V3 checkpoint; RL hardware ND | ND |
| V3.2 | 671B/37B | continued/post-train ND | FP8 weights; hardware ND | ND |
| V4-Flash | 284B/13B | 32T | mixed precision; hardware ND | ND |
| V4-Pro | 1.6T/49B | 33T | mixed precision; hardware ND | ND |
| Pro-0813 | same disclosed family size | extra post-training ND | ND | ND |

Two rough formulas:

```text
Dense upper-style estimate: C ~= 6 * N_total * T
MoE active estimate:       C ~= 6 * N_active * T
```

**Analytical estimates:** V3 active-style: `6*37B*14.8T = 3.286e24 FLOPs`; Flash: `2.496e24`; Pro: `9.702e24`. Dense total-style estimates are V3 `5.958e25`, Flash `5.453e25`, Pro `3.168e26`. These are not DeepSeek-reported compute. Active-style omits attention, embeddings, shared/dense layers, routing, MTP, sparse indexes, recomputation, optimizer, communication, and imbalance. Total-style grossly overcounts inactive experts.

V4 Flash used 2.16x and Pro 2.23x V3’s token count. Active parameters decreased for Flash and increased 32% for Pro. Flash’s active-style estimate is lower than V3 despite more tokens. V4 could have lower wall time through larger clusters, better kernels, overlap, and optimizer convergence, but no wall-time/GPU-hour disclosure proves it. Lower FLOPs may still fail to reduce GPU-hours because MoE communication, low utilization, optimizer work, long-context stages, and failures consume time.

## 8. Bottleneck analysis by workload and batch size

```text
request latency = queueing + preprocessing + prefill + decode + tool/runtime overhead
phase time ~= max(FLOPs/compute, bytes/bandwidth, communication/link bandwidth)
```

| Workload | V3/R1 | V3.2 | V4-Flash | V4-Pro |
|---|---|---|---|---|
| 1K chat, C=1 | weights/small expert GEMMs | same + index overhead | weights, launches, all-to-all | heavier weights/EP latency |
| 8K coding, C=4 | MoE + MLA reads | DSA crossover uncertain | compressed attention + MoE | MoE/communication |
| 32K RAG, C=16 | dense MLA attention | index/top-k/gather | CSA candidate/top-k | CSA + larger MoE |
| 128K document | prefill compute/cache | sparse gather/index | compression/index | top-k 1024, context parallelism |
| 256K agent | attention + long decode | DSA + serialized tokens | CSA/HCA + tool stalls | same + expert traffic |
| 1M analysis | impractical/unsupported native | unsupported advertised 128K | compression/index/cache capacity | same, larger cluster |
| shared prompt, 90% hit | MLA prefix restore | restore MLA/index keys | heterogeneous cache restore | cache I/O + larger state |
| offline batch, C=64+ | all-to-all/compute | sparse kernels + all-to-all | expert GEMM/links | link saturation/imbalance |

Across input lengths 1K, 8K, 32K, 128K, 256K, and 1M, dense attention becomes progressively worse; sparse/compressed designs introduce a crossover that must be measured. Across concurrency 1, 4, 16, 64, and saturation, expert GEMMs improve until network/cache/scheduler limits dominate.

### 8.1 Batch-1 decode

Every token may require reading selected expert weights, compressed history, and routing metadata. Small expert GEMMs underfill GPUs; collectives and launches do not amortize. Flash’s 13B active footprint is favorable analytically, but only optimized CSA/HCA/mHC/FP4 kernels can realize it. MTP helps if several proposals are accepted.

### 8.2 High-concurrency decode

Continuous batching supplies more tokens/expert, raising GEMM intensity. Eventually expert hot spots, all-to-all, KV capacity, fragmentation, and fairness dominate. MoE communication dominates when dispatch+combine time exceeds overlappable expert compute, which depends on topology and token distribution rather than a universal concurrency number.

### 8.3 Long-context prefill

V3 MLA still performs dense attention; V3.2 DSA changes selected attention toward `O(S*K)`; V4 CSA compresses and selects, while HCA attends densely over `S/128` summaries. Chunked prefill controls activation peaks. Context parallelism and prefill/decode disaggregation can isolate long prompts. Index construction and sparse gather can erase gains at short lengths.

### 8.4 Reasoning workloads

Long CoT is serialized, grows cache, keeps requests alive, lowers scheduler turnover, and worsens p99. Retry behavior multiplies cost. Compare **cost per correct solution**, not merely cost per generated token.

## 9. Prefix-cache design

### 9.1 V3 and R1

A correct key includes exact token IDs, model/tokenizer revision, chat template, LoRA identity, quantization/context settings, and parallel layout. State includes compressed MLA KV latent, decoupled RoPE key state, position metadata, scales, and block layout. Raw text is insufficient because templates and tokenization differ. Weight-absorbed formats may be engine-specific; partial-block hits require exact offsets.

### 9.2 V3.2

Persist prefix-dependent MLA latent and indexer keys/layout. Candidate selections are query-dependent and should be recomputed on extension. Any cached block metadata must match the DSA RoPE layout; DeepSeek corrected an indexer RoPE-layout discrepancy in its demo, illustrating why logit tests matter.

### 9.3 V4

The official report is unusually explicit: cache compressed CSA/HCA entries, indexer dimensions, SWA state, and uncompressed tail state until a block is compressible. Classical blocks cover `lcm(m,m')` original tokens; prefix reuse can load complete compressed blocks, while incomplete tails are recomputed. SWA may be fully stored, periodically checkpointed, or reconstructed. This is not a conventional uniform PagedAttention cache.

### 9.4 General challenges and correctness plan

Include tokenizer/template/reasoning/tool-schema revisions, adapters, quantization scales, context scaling, partial blocks, eviction, prefix-aware routing, replica affinity, distributed transfer, privacy, tenant isolation, and poisoning controls.

Test deterministic cached versus uncached runs using identical token IDs; compare restored latent/state tensors, attention outputs, logits, and continuation tokens. Cover exact hits, partial blocks, one-token mismatches, long prefixes, mode/revision changes, eviction/reload, and parallel-layout changes. A fast hit is invalid if logits differ beyond the defined precision tolerance.

## 10. Speculative decoding

Native MTP drafts 1-N tokens, target verifies, accepts the longest valid prefix, commits only accepted MLA/DSA/CSA/HCA/MoE state, and discards rejected speculative state. Proposal lengths 1-4 are safer starting points than assuming longer is better. Code and repetitive text often accept more than creative text; difficult math/tool boundaries and high temperature often accept less. These are hypotheses to measure.

MTP reduces batch-1 latency when target verification is efficient and acceptance amortizes serial steps. It improves throughput when spare compute exists and verification batches well. At saturation, draft work, larger temporary cache, and scheduler complexity can hurt. The same high-level protocol spans V3/V3.2/V4, but cache commit/rollback kernels differ. Prefix caching must restore a committed boundary before speculation.

Experiment: proposal `{1,2,3,4}`, concurrency `{1,4,16,64}`, prompt `{1K,8K,32K,128K}`, output `{128,512,2K}`, greedy and production sampling, reasoning/non-reasoning, prefix hit `{0,50,90}%`. Measure acceptance, accepted tokens/verification, TTFT, inter-token latency, accepted tok/s, energy/accepted token, cache memory, and quality. Compare native MTP, separate draft, prompt lookup, and n-gram speculation. vLLM and SGLang expose V4 DSpark/MTP paths but version and backend must be recorded. [vLLM V4 docs](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v4/), [SGLang V4 docs](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4).

## 11. Serving-cost comparison

### 11.1 Official API prices observed 2026-09-04

DeepSeek, USD per 1M tokens, all regions on the public price page; peak is weekdays 01:00-04:00 and 06:00-10:00 UTC. Cache creation is priced as cache miss; cache read as hit. No separate batch or long-context tier is listed.

| Model alias (snapshot) | Cache hit off/peak | Cache miss off/peak | Output off/peak | Context/output |
|---|---:|---:|---:|---:|
| `deepseek-v4-flash` (0731) | $0.007/$0.014 | $0.22/$0.44 | $0.66/$1.32 | 1M / max 384K |
| `deepseek-v4-pro` (0813) | $0.022/$0.044 | $0.66/$1.32 | $1.98/$3.96 | 1M / max 384K |
| Vision-Exp | Flash prices | Flash prices | Flash prices | 1M / 384K |

Reasoning has no separate token rate; it increases output-token volume. Low/high/max are effort controls. [Official pricing](https://api-docs.deepseek.com/quick_start/pricing/).

**Worked API examples, Flash off-peak, analytically computed:**

1. 8K input + 1K output, no hit: `0.008*0.22 + 0.001*0.66 = $0.00242`.
2. 100K + 5K: `$0.022 + $0.00330 = $0.02530`.
3. 1M + 10K: `$0.22000 + $0.00660 = $0.22660`.
4. Repository agent, 100K input with 90K cached + 5K output: `0.09*0.007 + 0.01*0.22 + 0.005*0.66 = $0.00613`.
5. 8K input + 20K hidden reasoning + 1K visible answer: `0.008*0.22 + 0.021*0.66 = $0.01562`.

Pro off-peak is exactly 3x these listed rates: `$0.00726`, `$0.07590`, `$0.67980`, `$0.01839`, and `$0.04686`. Arithmetic was recomputed from the official per-million rates.

### 11.2 Self-hosting cost

Approximate weight-only storage: V3 BF16 1.342 TB, FP8 671 GB, INT4/FP4 335.5 GB; Flash BF16 568 GB, FP8 284 GB, 4-bit 142 GB; Pro BF16 3.2 TB, FP8 1.6 TB, 4-bit 800 GB. Actual public V4 packages include mixed precision, MTP, scales, embeddings, and padding, so file size differs. vLLM reports practical variants around 200 GB for Flash-0731 FP8 and 960 GB for Pro-0813 in its recipes, which must not be confused with pure parameter-count arithmetic. [vLLM Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash), [Pro recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro).

V3/R1 practical FP8 typically needs at least 8 H200/B200-class GPUs plus cache/activations; Flash can fit one high-memory 8-GPU node in optimized low precision; Pro generally needs a larger or multi-node high-memory deployment unless native FP4 and aggressive sharding are available. Host RAM should exceed checkpoint plus staging buffers; NVMe loading of hundreds of GB can take minutes; expert parallelism needs NVLink/NVSwitch within node and high-bandwidth RDMA across nodes.

```text
cost per 1M accepted tokens = infrastructure dollars/hour / accepted tokens/hour * 1,000,000
```

Include GPU, CPU, RAM, storage, network, redundancy, idle reserve, SLO headroom, cache hit rate, speculative acceptance, failures, retries, and reasoning tokens. No defensible dollar figure is supplied without a selected cloud SKU and measured accepted throughput. Provider price is not physical cost; total parameters are not per-token FLOPs; active parameters are not exact FLOPs; and cheap tokens can be expensive if task success is low.

## 12. Hooks for cross-family comparison

### DeepSeek versus Qwen

Compare MLA/HCA compressed per-token and per-block state with Gated DeltaNet plus QSA recurrent/sparse state; DSA/CSA/HCA with Qwen Sparse Attention; fine-grained DeepSeekMoE with Qwen ultra-sparse MoE; mHC with gated residuals; native MTP acceptance; Muon scope; host-memory capacity; total/active parameters; and all-to-all bytes. Control tokenizer, context, reasoning budget, and kernel maturity.

### DeepSeek versus Kimi

Compare attention cache representation, sparse selector quality, expert granularity, 1M prefill, long-agent decode, MTP/draft support, and prefix-cache state. Ask whether Kimi’s design avoids V4’s heterogeneous compression tails or shifts cost elsewhere.

### DeepSeek versus GLM

Compare experts selected/shared, attention compression, residual pathways, reasoning RL, long-context retrieval quality, and cost per successful agent task.

No family should be declared superior without controlling checkpoint quality, hardware, precision, engine/version, context, batch/concurrency, reasoning budget, cache hit, speculation, and task-success target.

## 13. Benchmark and profiling plan

Matrix: prompts `{1K,8K,32K,128K,256K,1M where supported}`; outputs `{128,512,2K,8K,budget-controlled reasoning}`; concurrency `{1,4,16,64,saturation}`; prefix hits `{0,50,90,99}%`; decode `{AR,MTP,separate draft,prompt lookup}`.

Report TTFT and inter-token latency p50/p95/p99; input tok/s/GPU; accepted output tok/s/GPU; requests/s under SLO; GPU utilization; HBM bandwidth; cache/request; expert load distribution; all-to-all duration/bytes; sparse index/gather time; MTP acceptance; energy/accepted token; dollars/completed task; success and retries.

Instrument MLA projection/cache reads, router, token permutation, grouped expert GEMMs, dispatch/combine, indexer, top-k, sparse gather, compression, mHC, MTP proposal/verification, cache lookup/restore, and scheduler queueing. Record vLLM/SGLang/TensorRT-LLM commit, CUDA/driver, FlashMLA/DeepGEMM/DeepEP versions, topology, precision, power limit, and fallback kernels. SGLang’s published V4 page reports detailed commands and measurements, but reuse only rows with complete hardware/version/workload fields; otherwise label “Non-reproducible from published information.” [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4), [TensorRT-LLM V3 guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v3/README.md).

## 14. Conclusions and open questions

### High-confidence conclusions

V3 is 671B/37B-active with MLA, DeepSeekMoE, loss-free balancing, MTP, FP8 training, and DualPipe. R1 keeps the V3 architecture and changes post-training/behavior. V3.2 adds DSA. V4 introduces CSA/HCA and mHC, retains MTP and adjusted DeepSeekMoE, and trains Flash/Pro on 32T/33T tokens. Flash-0731 is architecture-preserving re-post-training; Pro-0813 is the latest GA text snapshot.

### Analytical implications

Flash should be easier for low-latency serving; Pro should need more EP and storage. V4’s compressed attention should dominate at very long context but may lose at short context. MoE accuracy/compute efficiency does not guarantee batch-1 latency. Reasoning tokens can dominate total cost.

### Independent observations

vLLM, SGLang, and TensorRT-LLM all provide architecture-specific implementations. SGLang describes three coherent cache pools and speculative-state handling; vLLM provides V4 sparse backends and production-snapshot recipes. Framework benchmark claims remain version- and topology-specific.

### Undisclosed information

V4 GPU cluster, GPU-hours, wall time, exact production-snapshot extra tokens/data, detailed API routing, production quantization, full training FLOPs, and API cache implementation are not disclosed.

### Questions requiring experiments

Where is the sparse-attention crossover? What tokens/expert saturate each GPU? What MTP acceptance holds by domain and effort? Are cached and uncached logits equivalent across layouts? What is cost per correct agent task? Does 0731/0813 alter only weights/template or hidden serving policy?

### Engineering risks

Unoptimized MLA/CSA/HCA kernels; MoE all-to-all; sparse top-k/index memory; incorrect prefix tails; reasoning-token explosion; low MTP acceptance; cache fragmentation; mixed-precision drift; fallback kernels; and API alias drift.

## 15. Required references

### Official technical reports

- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DeepSeek-V3.2 paper](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348)

### Official repositories

- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

### Official model cards

- [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
- [DeepSeek-V3.2-Speciale](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale)
- [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DeepSeek V4 collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4)

### Official API and pricing

- [Changelog](https://api-docs.deepseek.com/updates/)
- [V4 preview](https://api-docs.deepseek.com/news/news260424/)
- [V4-Pro GA](https://api-docs.deepseek.com/news/news260813/)
- [Models and pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [V3.1 release](https://api-docs.deepseek.com/news/news250821/)

### Serving-framework documentation

- [vLLM V4 implementation](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v4/)
- [vLLM V4 systems article](https://vllm.ai/blog/2026-04-24-deepseek-v4)
- [SGLang V4 deployment](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [TensorRT-LLM V3/R1/V3.2](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v3/README.md)
- [TensorRT-LLM V3.2 optimization](https://nvidia.github.io/TensorRT-LLM/1.3.0rc25/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.html)

### Reproducible vendor measurements

- [SGLang V4 Day-0 systems report](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) (vendor/framework team; inspect exact configuration before comparison)
- [vLLM Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) (framework recipe, not model-vendor measurement)

### Community implementation notes

- [StreamIndex CSA indexer study](https://arxiv.org/abs/2605.02568) (independent, synthetic layer-level shapes; not end-to-end checkpoint behavior)
- [FlashMemory-DeepSeek-V4](https://arxiv.org/abs/2606.09079) (independent derivative, not an official DeepSeek checkpoint)

## 16. Notes for the next session

1. Reverify the latest DeepSeek releases and immutable aliases.
2. Reverify official pricing and peak windows.
3. Preserve architecture versus post-training distinctions.
4. Preserve official versus estimated compute labels.
5. Add profiler traces when available.
6. Record kernel and serving-engine versions and commits.
7. Test prefix-cache logit equivalence.
8. Record MTP acceptance, not only speedup.
9. Compare cost per successful task.
10. Flag fallback or missing kernels.
11. Extend the same workload matrix to Qwen, GLM, and Kimi.
12. Avoid cross-family conclusions until serving conditions are controlled.
