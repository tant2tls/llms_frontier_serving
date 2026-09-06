# GLM-4.7 Forward Architecture and Serving Performance Report

## Release verification and scope

**Scope.** This report begins at GLM-4.7 and follows only releases corroborated by Z.ai/Zhipu AI or an official `zai-org` repository/model card. It is an architecture and serving report, not a benchmark chronology.

**Verification date.** Research cutoff: **4 September 2026, 18:16 GMT+7** (11:16 UTC; 19:16 UTC+8). The latest official release found is **GLM-5.3-Flash**, released **26 August 2026**. The latest flagship is **GLM-5.3**, released **18 August 2026**; the latest open-weight text flagship checkpoint is also GLM-5.3, whose weights were published after launch. The latest fast tier, newest redesigned backbone, latest open-weight multimodal checkpoint, and latest multimodal model are all GLM-5.3-Flash. Official release notes give GLM-5 on 12 February, GLM-5.1 on 7 April, GLM-5.2 on 16 June, GLM-5.3 on 18 August, and GLM-5.3-Flash on 26 August 2026 [[official release notes](https://docs.z.ai/release-notes/new-released)].

**Verified lineage.**

```text
GLM-4.7 (same disclosed 4.5/4.6-class backbone; new post-training)
  +-- GLM-4.7-Flash (separate smaller MLA/MoE backbone, not a shrunk checkpoint)
  +-- GLM-4.7-FlashX (official API latency tier; weights/architecture not disclosed)
  +-- official FP8 checkpoint(s) (same graph, stored arithmetic differs)
GLM-5 (new 744B/40B MoE + MLA/DSA base)
  -> GLM-5.1 (same family/backbone; post-training refresh)
  -> GLM-5.2 (continued/mid-training plus IndexShare, 1M context, revised MTP)
  -> GLM-5.3 (same base as 5.2; post-training only)
GLM-5.3-Flash (parallel new 320B/18B native-multimodal base;
               hybrid linear+sparse attention + mHC + IndexPool)
```

**Product-state classification.** GLM-4.7, 5, 5.1, 5.2, 5.3, 4.7-Flash and 5.3-Flash have downloadable official weights. GLM-4.7-FlashX is an official production API tier but no official downloadable checkpoint was found. `glm-5.3`, `glm-5.3-flash`, and similar API names are product identifiers, not immutable weight hashes; an API provider can update implementation, quantization, kernels, or routing without changing the name. For reproducibility, pin a Hugging Face/ModelScope commit and record file hashes. The official GLM-5 GitHub repository has no GitHub Releases, so its `main` branch is rolling [[repository](https://github.com/zai-org/GLM-5)] [[releases page](https://github.com/zai-org/GLM-5/releases)].

**Modes.** “Standard” means ordinary autoregressive generation. “Thinking” emits or internally accounts for reasoning. “Forced thinking” means the API refuses disabled thinking, as GLM-5.3 and 5.3-Flash do. “Interleaved thinking” permits reasoning between tool calls. “Preserved thinking” returns prior reasoning blocks into later turns. “Turn-level thinking” toggles the template per turn. These are request/template and state semantics, not necessarily different weight sets. GLM-5.3 supports `low`, `high`, and `max` reasoning effort and cannot disable reasoning [[5.3 API guide](https://docs.z.ai/guides/llm/glm-5.3)]; 5.3-Flash likewise only supports enabled thinking and recommends `clear_thinking:false` [[5.3-Flash guide](https://docs.z.ai/guides/vlm/glm-5.3-flash)].

**Technical reports.** GLM-4.7 cards point backward to the GLM-4.5 report, creating a disclosure gap. GLM-5 has an official technical report linked by the repository/model cards. GLM-5.2 has the official IndexShare paper and detailed launch note. No separate full architecture report was found for 5.1 or 5.3; 5.3 is explicitly post-training-only. GLM-5.3-Flash has an official technical launch article and released configuration, but not a comparably complete training report as of cutoff.

**Excluded as unverified.** `GLM-5.4`, `GLM-5.3-Pro`, `GLM-5-Turbo`, `GLM-5.2-Fast`, provider-specific `ox-alpha` after de-anonymization, and any GLM-5.3.1/5.4 benchmark-catalog alias are **Unverified and excluded** unless an official Z.ai/Zhipu/zai-org source appears. `ox-alpha` was an official anonymous pre-release experiment, not a stable checkpoint name [[5.3-Flash announcement](https://z.ai/blog/glm-5.3-flash)].

### Purpose, terminology, disclosure limits, and evidence labels

The purpose is to connect architecture to computation, traffic, communication, persistent state, latency, throughput, deployment complexity, and cost. **Total parameters** means all stored model parameters; **active parameters** is a coarse per-token routing count, not exact FLOPs. **Native context** means configuration/training support; **advertised context** means provider acceptance. **Prefix cache** means exact reusable model state for identical tokenized prefixes, not merely provider billing cache.

Labels are used as follows: **Officially reported** for primary-source statements/configurations; **Independently measured** for reproducible third-party experiments; **Analytically estimated** for equations or arithmetic here; **Not disclosed** when the primary record is silent; **Unverified and excluded** for unsupported names. Package sizes include metadata/sharding and may differ from parameter-count arithmetic. No independent end-to-end profiler trace was found for the final two releases; performance claims without released commands, engine commits, and traces are marked **Non-reproducible from published information**.

## 1. Executive summary

1. **Officially reported:** GLM-5.3-Flash is the latest verified release and a genuinely new native-multimodal base, not a smaller GLM-5.3. It has 320B total/18B active parameters, 45 layers, hybrid linear+sparse attention, mHC, IndexPool, and 30T multimodal pretraining tokens [[announcement](https://z.ai/blog/glm-5.3-flash)].
2. **Officially reported:** GLM-5.3 is the same base model as 5.2; all gains are attributed to post-training. Therefore its attention graph, active parameter count, and KV layout should match the pinned 5.2 configuration [[5.3 announcement](https://z.ai/blog/glm-5.3)] [[model card](https://huggingface.co/zai-org/GLM-5.3)].
3. **Officially reported:** GLM-5 is the major flagship backbone break: 744B total, 40B active, 28.5T tokens, 78 layers, 256 routed experts, top-8, one shared expert, MLA plus DSA, and one MTP layer [[GLM-5 announcement](https://z.ai/blog/glm-5)] [[configuration](https://huggingface.co/zai-org/GLM-5/raw/main/config.json)].
4. **Officially reported:** GLM-5.2 changes serving-relevant architecture via IndexShare, sharing one DSA indexer’s top-k positions across four layers, and extends configured context to 1,048,576. It also revises MTP and reports up to 20% greater acceptance length [[5.2 technical note](https://z.ai/blog/glm-5.2)].
5. **Officially reported / disclosure-limited:** GLM-5.1 appears to reuse GLM-5’s backbone while scaling long-horizon multi-turn SFT/RL and process-quality training. No primary source found a new base architecture; treat observed gains as post-training unless a pinned config proves otherwise [[5.1 announcement](https://z.ai/blog/glm-5.1)] [[release notes](https://docs.z.ai/release-notes/new-released)].
6. **Officially reported:** GLM-4.7’s released config is a 92-layer, 160-routed-expert, top-8 MoE with one shared expert, 96 query and 8 KV heads. Its card points to GLM-4.5’s report, so inherited training totals are not automatically 4.7 facts [[configuration](https://huggingface.co/zai-org/GLM-4.7/raw/main/config.json)] [[card](https://huggingface.co/zai-org/GLM-4.7)].
7. **Analytically estimated:** GLM-4.7 BF16 KV is 376,832 bytes/token/sequence before allocator overhead. At 200K tokens this is about 70.2 GiB, making long-context concurrency KV-capacity and KV-bandwidth limited despite MoE sparsity.
8. Batch-1 decode for all large MoEs is usually weight-bandwidth and collective-latency sensitive; high concurrency improves GEMM utilization but can become expert all-to-all, KV capacity, scheduler, or network bound. Sparse activation does not shrink total stored weights.
9. GLM-5/5.2/5.3’s DSA reduces attention work by selecting 2,048 positions, but the indexer and irregular gathers remain history-dependent. IndexShare reduces repeated indexer work, not all KV storage [[5.2 note](https://z.ai/blog/glm-5.2)].
10. GLM-5.3-Flash reduces flagship attention compute and average KV by reported factors of 3.0 and 4.4, but the whole model does not have constant history: sparse layers retain history/index metadata, vision prefixes consume tokens/state, and linear states remain per layer/request [[Flash announcement](https://z.ai/blog/glm-5.3-flash)].
11. Interleaved/preserved reasoning increases token volume, decode occupancy, and state identity. Exact prior reasoning blocks, tool schemas, template, thinking controls, and tokenizer revision belong in a cache key.
12. GLM-5.2 has native MTP support; ordinary framework prompt lookup is not evidence of model-native speculation. Verification must commit recurrent/sparse state only for accepted tokens.
13. **Officially reported pricing observed 4 September 2026:** international per-million-token list prices are $1.40 input/$0.26 cached/$4.40 output for 5.2 and 5.3; 5.3-Flash is temporarily $0.075 input/$0.015 cached/$0.25 output, with list prices twice those values, promotion ending 9 September UTC+8 [[pricing](https://docs.z.ai/guides/overview/pricing)].
14. Self-hosting feasibility is dominated by weight residency, topology, quantization, KV/workspace, reliability replicas, and utilization. A 744B BF16 checkpoint is about 1.488 TB by parameter arithmetic, before runtime workspace.
15. Cross-family comparisons must pin checkpoint hash, tokenizer/template, reasoning budget, hidden reasoning billing, precision, optimized kernel path, topology, context, concurrency, cache hit rate, tool environment, and successful-task criterion.

## 2. Release timeline and disclosure matrix

Numbers are **Officially reported** unless marked otherwise. ND = **Not disclosed**.

| Model | Date | Identifier/state | Weights | Base relation/type | Core configuration | Context/output | Training/license/report |
|---|---:|---|---|---|---|---|---|
| GLM-4.7 | 2025-12-22 | `glm-4.7`; production rolling API | BF16 and official FP8 | post-trained 4.5/4.6-class text MoE | ~355B/32B; 92L; d=5120; 160 routed, top-8, 1 shared; GQA 96Q/8KV, hd=128 | config 202,752; API 200K/128K | tokens/hardware ND; MIT; card cites 4.5 report |
| GLM-4.7-Flash | 2026-01-19 | `glm-4.7-flash`; production | yes | separate lightweight text MoE+MLA | parameter totals ND in config; 47L; d=2048; 64 routed, top-4, 1 shared; 20 heads; MLA ranks q=768, kv=512 | 202,752; API 200K/128K | ND; MIT; no dedicated report |
| GLM-4.7-FlashX | 2026-01-19 | `glm-4.7-flashx`; production rolling API | no official weights found | API latency tier, architecture ND | ND | 200K/128K | ND |
| GLM-5 | 2026-02-12 | `glm-5`; production | BF16, FP8 | new text MoE base | 744B/40B; 78L; d=6144; 256 routed, top-8, 1 shared; MLA+DSA top-2048; 1 MTP | 202,752; API 200K/128K | 28.5T; MIT; technical report |
| GLM-5.1 | 2026-04-07 | `glm-5.1`; production | BF16, FP8 | same family; post-training refresh | same 744B/40B class; exact changed fields ND | 200K/128K | multi-turn SFT/RL disclosed qualitatively; MIT; no separate full report |
| GLM-5.2 | 2026-06-16 | `glm-5.2`; production | BF16, FP8 | continued/mid-trained 5.x with IndexShare | 78L, d=6144, 256/top-8/1; MLA+DSA, shared index every 4 layers; MTP revised | 1,048,576/128K API | 128K mid-training disclosed; amount ND; MIT; IndexShare paper/note |
| GLM-5.3 | 2026-08-18 | `glm-5.3`; production forced-thinking | weights released after launch; full package about 756 GB | exactly same base as 5.2; post-training only | same pinned graph as 5.2 | 1M/128K | added long-horizon/coding RL; custom GLM-5.3 license on card; no new architecture report |
| GLM-5.3-Flash | 2026-08-26 | `glm-5.3-flash`; production forced-thinking | yes; FP8-tagged card | new native-multimodal MoE base | 320B/18B; 45L; hybrid linear+sparse; mHC; IndexPool; detailed expert/head fields ND here | 1M/128K | 30T multimodal; MIT; technical launch article |

The complete requested field set is not publicly populated. Query/KV heads, exact per-layer pattern, linear-state dimensions, vision encoder/token count, optimizer, accelerator-hours, reported training FLOPs, and wall-clock are **Not disclosed** for 5.3-Flash in the sources available. The released config is the authoritative place to pin those values when stable; do not backfill from marketing diagrams. GLM-4.7’s exact config is available [[raw config](https://huggingface.co/zai-org/GLM-4.7/raw/main/config.json)], GLM-4.7-Flash’s is materially different [[raw config](https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/config.json)], and GLM-5’s is available [[raw config](https://huggingface.co/zai-org/GLM-5/raw/main/config.json)].

## 5. GLM-4.7 baseline

### 3.1 Backbone, variants, and quantization

GLM-4.7 is best classified as a new post-trained checkpoint on the 4.5/4.6-class architecture. The unchanged 92-layer shape and the card’s reliance on the GLM-4.5 report support inheritance, while the launch emphasizes coding, tool use, and reasoning behavior rather than new pretraining [[launch](https://z.ai/blog/glm-4.7)] [[card](https://huggingface.co/zai-org/GLM-4.7)]. It should not need fundamentally different kernels from 4.6 when configuration matches, although a new template, reasoning parser, MTP support, and official FP8 path can require newer engines.

The full model configuration confirms 92 layers, hidden width 5,120, 160 routed experts, top-8, one shared expert, three initial dense replacements, 96 query heads, eight KV heads, 128 head dimension, BF16 metadata, and one next-token-prediction layer [[config](https://huggingface.co/zai-org/GLM-4.7/raw/main/config.json)]. Total/active numbers commonly inherited from the official GLM-4.5 report are 355B/32B, but because 4.7 does not publish an independent parameter accounting, retain that inheritance label rather than pretending the 4.5 report measured the post-trained files.

Flash is not merely a smaller flagship. Its config changes model class to `Glm4MoeLiteForCausalLM`, 47 layers, width 2,048, 64 routed experts, top-4, one shared expert, and MLA-like low-rank q/kv projections; it uses 20 heads and no GQA reduction in the same representation [[Flash config](https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/config.json)]. FlashX is architecturally **Not disclosed** and should be treated as an API speed tier, not assumed identical to Flash. FP8 does not change layer topology or learned function in an exact mathematical sense; it changes stored/compute arithmetic and scales, can change logits numerically, and needs compatible GEMM/MoE kernels.

### 3.2 GLM-4.7 MoE serving

For each sparse FFN layer, a router scores experts, selects top-8, normalizes/scales gates, permutes tokens by expert, runs grouped GEMMs, weights outputs, and unpermutes/combines them. The shared expert runs for every token. “32B active” is not exact per-token FLOPs because attention, embeddings, router, shared expert, initial dense layers, projections, and MTP remain outside the simple routed count.

At batch 1, selected experts receive tiny token groups. Weight reads and launch/collective latency dominate; expert parallel all-to-all may cost more than the saved arithmetic. At small batch, token imbalance creates stragglers. At high concurrency, grouped GEMMs improve but router skew, permutation buffers, dispatch/combine, and network saturation emerge. A single node avoids inter-node latency but must hold all 355B weights, typically requiring aggressive quantization or large-HBM GPUs. Multi-node expert parallelism lowers per-device weights but adds all-to-all on every MoE layer. Sparse activation does not reduce checkpoint size, initial loading, host-RAM staging, replica cost, or the requirement that cold experts remain addressable.

### 3.3 Attention and KV cache

For conventional GQA/MQA:

```text
KV_bytes = L * S * H_kv * D_head * 2 * bytes_per_element
KV_bytes_per_token = L * H_kv * D_head * 2 * bytes_per_element
```

For GLM-4.7 BF16, `L=92`, `H_kv=8`, `D_head=128`, and bytes=2:

```text
KV/token/sequence = 92*8*128*2*2 = 376,832 bytes = 368 KiB
KV(S) = 376,832*S bytes
KV(200,000) ~= 75.37 GB = 70.20 GiB
```

These are **Analytically estimated** logical tensors, excluding page fragmentation, alignment, scales, MTP, temporary attention buffers, and engine metadata. In decode, conventional attention approximately reads one K and one V element for each retained past position, so history traffic per generated token is approximately `376,832*S` bytes across layers, plus query/output traffic. FlashAttention-like kernels reduce intermediates, not the requirement to consume history.

Correct restoration requires exact K/V tensors or an engine-equivalent paged representation, per-layer positions/RoPE state, sequence length, cache dtype/scales, and parallel ownership. GLM-4.7-Flash’s MLA cache is not interchangeable with flagship GQA state.

### 3.4 Thinking-state semantics

Interleaved thinking allows new reasoning after a tool result rather than forcing a single initial chain. Preserved thinking forwards earlier reasoning blocks into later requests. Turn-level controls modify whether the chat template requests reasoning. These affect token IDs, context length, output-token billing, scheduler occupancy, and cache identity. If an API requires reasoning blocks to be returned, preserve them byte-for-byte and in original order; normalization can retokenize. A provider may separately maintain a session cache, but model-correctness still depends on reproducing the same token/position stream. Switching thinking mode can change special tokens or generation prefix, invalidating a logical prefix even if visible user text is unchanged.

## 6. GLM-5 base generation

| Field | GLM-4.7 | GLM-5 | Serving consequence |
|---|---|---|---|
| Scale | ~355B/32B inherited | 744B/40B official | ~2.1x stored weights; modestly higher active compute |
| Layers/width | 92/5120 | 78/6144 | fewer, wider layers; different partitioning |
| Experts | 160, top-8, one shared | 256, top-8, one shared | more expert shards and routing fan-out |
| Attention | 96Q/8KV GQA | MLA + DSA top-2048 | different cache/kernels; sparse index/gather |
| Context | ~200K | ~200K initially | same product scale, different attention economics |
| Pretraining | inherited/ND for 4.7 | 28.5T tokens | new base, not only post-training |
| MTP | one next-n layer | one MTP layer | native speculation path possible |

GLM-5’s raw config reports 78 layers, width 6,144, q-LoRA rank 2,048, KV-LoRA rank 512, 64 attention heads, qk dimensions 192 non-RoPE + 64 RoPE, value dimension 256, and a lightweight 32-head/index-head-128 DSA indexer selecting top 2,048 positions [[config](https://huggingface.co/zai-org/GLM-5/raw/main/config.json)]. This is a new kernel family relative to 4.7 GQA. The announcement reports 744B/40B and 28.5T pretraining tokens, compared with 355B/32B and 23T for 4.5 [[launch](https://z.ai/blog/glm-5)]. Optimizer, exact hardware, training precision schedule, accelerator-hours, and wall time are **Not disclosed** in the consulted public summary.

MLA stores a compressed latent representation plus positional components instead of ordinary per-head K/V, while DSA adds indexer keys/metadata and selected gathers. Exact bytes must be read from engine layouts, not conventional GQA equations. DSA changes decode from scanning every token with full attention to indexing history and attending a selected set, but index scoring and top-k selection can still grow with history.

## 7. GLM-5.1 and GLM-5.2

### 7.1 GLM-5.1

Official release notes identify multi-turn SFT, RL, and a process-quality evaluation framework aimed at autonomous operation up to eight hours [[release notes](https://docs.z.ai/release-notes/new-released)]. The launch describes hundreds of rounds and thousands of tool calls, which is workload behavior, not a new residual/attention design [[announcement](https://z.ai/blog/glm-5.1)]. Because vLLM documents 5.1 as a refreshed 5 model and serves both with the same model path/kernels, this report classifies 5.1 as same-backbone post-training absent contrary config evidence [[vLLM recipe](https://docs.vllm.ai/projects/recipes/en/stable/GLM/GLM5.html)]. Runtime changes arise primarily from longer trajectories, more tool serialization, and reasoning tokens, not new KV layout.

### 7.2 GLM-5.2: one-million-token architecture

GLM-5.2 is not merely an API limit increase. It was trained with IndexShare from mid-training at 128K sequences. One lightweight DSA indexer is placed at the first of each four-layer group, and its top-k indices are reused by the next three layers. Those layers still compute their own queries, attention weights, value combinations, and projections. The official note reports 2.9x fewer per-token indexer FLOPs at 1M and explicitly warns that KV size is not proportionally reduced [[technical note](https://z.ai/blog/glm-5.2)]. Thus “lossless 1M” is a provider quality claim supported by selected long-horizon evaluations, not an architectural proof of perfect retrieval.

The 5.2 configuration raises maximum positions to 1,048,576 and codifies the repeating full/shared indexer pattern [[config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)]. Prefill remains expensive: sparse main attention is roughly `O(S*k)` for `k=2048`, while index construction/scoring and top-k have additional sequence-dependent work. Decode reads compressed attention state, indexer keys/metadata, and selected value data. CPU cache management, scheduler overhead, irregular gathers, and KV capacity become first-order at 1M.

The MTP layer shares the first proposal step’s top-k indices and KV state across later proposal steps, adds rejection sampling and TV loss, and reports acceptance length 4.56 to 5.47 in an ablation, a 20% increase [[technical note](https://z.ai/blog/glm-5.2)]. This is native model support, not generic prompt lookup.

## 8. GLM-5.3 flagship

The primary statement is unusually explicit: “same base model as GLM-5.2; every gain comes from post-training” [[announcement](https://z.ai/blog/glm-5.3)] [[official card](https://huggingface.co/zai-org/GLM-5.3)]. Accordingly, 5.3 should not require different fundamental attention/MoE kernels, change KV layout, or change active parameters versus a pinned 5.2 graph. It may require updated reasoning/tool parsing and provider policy. The API forces thinking and exposes low/high/max effort, context 1M, output 128K [[API guide](https://docs.z.ai/guides/llm/glm-5.3)].

Forced thinking expands generated tokens and serial dependency length. Even if hidden from visible output, reasoning occupies decode slots, KV/state, bandwidth, and potentially billable output accounting. At saturation, longer occupancy reduces requests/s and increases queueing. Gains should be attributed to expanded coding/long-horizon environments and post-training compute, not to new pretraining or attention. Cyber evaluations are capability/risk evidence only; this report provides no operational exploitation guidance.

## 9. GLM-5.3-Flash architecture

### 9.1 Flagship versus Flash

| Property | GLM-5.3 | GLM-5.3-Flash |
|---|---|---|
| Base lineage | GLM-5.2 base, post-trained | newly trained base |
| Parameters | 744B/40B class | 320B/18B |
| Layers | 78 | 45 |
| Attention | MLA + DSA + IndexShare | hybrid linear + sparse + IndexPool |
| Residual | conventional disclosed GLM-5 path | mHC |
| Modality | text | native image/video/text/file input, text output |
| Context/output | 1M/128K | 1M/128K |
| Pretraining | 28.5T for original 5 base; later amount ND | 30T multimodal |
| Logical cache | compressed MLA + DSA index state | recurrent linear state + sparse history/index + vision state |
| Likely batch-1 | very large weight traffic/collectives | lower active weights/layers; specialized kernels |
| High concurrency | MoE all-to-all, KV/index | MoE, sparse gathers, recurrent-state batching |
| Target | maximum coding/agent quality | low-cost multimodal agent throughput |

All Flash values in this table are **Officially reported** by the launch [[source](https://z.ai/blog/glm-5.3-flash)]. Exact expert count/top-k, hidden/FFN sizes, head counts, and layer pattern are **Not disclosed** in that article and must come from a pinned released config. Flash is not a compressed flagship: base data, modality, attention, residual path, layer count, active scale, and kernels differ.

### 9.2 Hybrid sparse and linear attention

Linear attention maintains a fixed-dimensional recurrent summary per layer/head under the chosen formulation. Per token it updates state, often conceptually `M_t = f(M_{t-1}, k_t, v_t)` and computes `o_t = g(q_t, M_t)`. If state dimensions are `d_k x d_v`, logical recurrent bytes are approximately `L_linear * H * d_k*d_v*b`, independent of S. Prefill can be `O(S*d_k*d_v)` and decode update constant in S, but exact gated formulation, dimensions, normalization, and precision are **Not disclosed** in the announcement.

Sparse attention retains sequence-dependent token/block K/V or compressed variants, indexer vectors, selected positions, and positional metadata. Main attention is approximately `O(k*d)` per query after selection; index cost depends on pooling and candidate mechanism. IndexPool compresses four indexer key vectors into one weighted pooled vector, reducing index latency/memory at 1M [[announcement](https://z.ai/blog/glm-5.3-flash)]. Selection introduces irregular gathers, low arithmetic intensity, and missed-context risk if relevant positions are not selected. It can lose to dense fused attention at short contexts due to index overhead.

Reported averages versus 5.3 are 3.0x less attention compute and 4.4x smaller KV, not constant total history [[announcement](https://z.ai/blog/glm-5.3-flash)]. At 128K/256K/1M, sparse indexing and capacity dominate more than at 8K. Context parallelism must partition history and merge top-k candidates; recurrent states need deterministic ownership/reduction. Production uses tensor parallelism for linear attention, ReplaySSM, W8A8, hybrid INT8/FP8/BF16 cache, Layer Split, and encode-prefill-decode disaggregation [[same source](https://z.ai/blog/glm-5.3-flash)]. Those are production implementation disclosures, not portable checkpoint guarantees.

### 9.3 mHC

A conventional residual stream carries one vector through layers. Hyper-connections maintain multiple pathways and dynamically mix them; manifold constraints keep mixing transformations in a stable admissible set, improving gradient flow and scaling stability. mHC can increase within-layer activation traffic, mixing operations, fusion requirements, and tensor-parallel communication. It does not inherently add sequence-length-dependent persistent decode history: once a token has passed the layer, its path activations need not be retained unless attention/state uses them. It can change the representation expected by each layer but not necessarily the external prefix-cache schema beyond model revision. Exact path count, matrices, fusion, and overhead are **Not disclosed** in the launch, so quantitative claims are inappropriate.

### 9.4 Multimodal serving

The API accepts video/image/text/file and returns text [[API guide](https://docs.z.ai/guides/vlm/glm-5.3-flash)]. The vision encoder/tokenizer, patch size, token count versus resolution, projector, and positional scheme are **Not disclosed** in the cited public page. Image/video requests add decode-independent preprocessing, media fetch/validation, frame sampling, encoder compute, visual tokens, projection, and longer prefill. Visual prefix caching is valid only for identical decoded pixels/transforms, encoder revision, resize/crop/frame policy, visual token IDs, and multimodal positions. EPD disaggregation permits encoding to scale separately from prefill and decode [[announcement](https://z.ai/blog/glm-5.3-flash)].

## 10. Training tokens, compute, and elapsed time

| Model | Total/active | Tokens | Post-training | Modalities | Hardware/precision/hours/FLOPs/wall time |
|---|---|---|---|---|---|
| 4.7 | ~355B/32B inherited | 23T belongs to 4.5 report, not proven new 4.7 training | coding/reasoning disclosed | text | ND |
| 4.7-Flash | ND | ND | disclosed qualitatively | text | BF16 config; other fields ND |
| 5 | 744B/40B | 28.5T pretraining | slime async RL | text | exact hardware/hours/FLOPs/wall ND |
| 5.1 | same class | continued amount ND | multi-turn SFT, RL, process quality | text | ND |
| 5.2 | same class | 128K mid-training disclosed; amount ND | agentic RL/OPD; OPD about two days | text | KV FP8 in rollout stack; other totals ND |
| 5.3 | same as 5.2 | no new base | scaled post-training only | text | ND |
| 5.3-Flash | 320B/18B | 30T multimodal | agentic/visual coding disclosed | native multimodal | Chinese chips for serving; training hardware/hours/FLOPs ND |

**Analytically estimated**, not Z.ai-reported:

```text
Dense upper-style estimate: C ~= 6*N_total*T
MoE active-style estimate: C ~= 6*N_active*T
GLM-5 active-style ~= 6*40e9*28.5e12 = 6.84e24 FLOPs
GLM-5 dense-style  ~= 6*744e9*28.5e12 = 1.272e26 FLOPs
Flash active-style ~= 6*18e9*30e12 = 3.24e24 FLOPs
Flash dense-style  ~= 6*320e9*30e12 = 5.76e25 FLOPs
```

These omit attention, dense/shared layers, embeddings, router, multimodal encoder, sparse index, linear updates, recomputation, optimizer, communication, imbalance, and post-training. More tokens can coexist with lower elapsed time through better utilization, parallelism, lower active compute, fewer layers, and faster hardware. Lower FLOPs can fail to reduce accelerator-hours when communication, memory, imbalance, or low arithmetic intensity dominates.

## 11. Bottleneck analysis

```text
request latency = queueing + preprocessing + prefill + decode + tool/runtime overhead
phase time ~= max(FLOPs/sustained_compute,
                  bytes_moved/sustained_bandwidth,
                  communication/link_bandwidth)
```

| Workload | 4.7 | 5/5.2 | 5.3 | 5.3-Flash |
|---|---|---|---|---|
| Batch-1 decode | weight bandwidth, launches, MoE collectives | weight/MLA state, DSA index, MoE | same as 5.2 plus longer forced reasoning | lower active weights; recurrent kernels, sparse gathers, MoE |
| High-concurrency decode | expert GEMM then all-to-all/KV | all-to-all, KV/index capacity | occupancy from forced thinking | state batching, sparse index, expert network |
| 128K-1M prefill | 4.7 only to ~200K; dense attention expensive | DSA compute/index; 5.2 IndexShare helps | same graph | multimodal encode, linear scan, sparse index |
| Long reasoning | serialized decode/KV growth | MTP may help if accepted | forced; scheduler occupancy high | forced; lower per-token cost but many tokens |
| Agents/tools | tool/network waits, preserved state | same plus huge context | long horizon, serialization | vision encode/GUI waits plus EPD |

At 1K/8K prompts, weight movement and MoE dominate decode; sparse machinery may not amortize. At 32K/128K, prefill compute and KV bandwidth rise. At 256K/1M, capacity, CPU page/index management, context parallel communication, and irregular gathers dominate. Outputs 128/512 are TTFT-sensitive; 2K/8K outputs amplify decode and reasoning occupancy. Concurrency 1/4 exposes latency and underutilization; 16/64 improves GEMMs but stresses KV, all-to-all, and scheduler; saturation adds queueing.

Repository agents with shared prompts benefit strongly from exact prefix reuse, but tool results and preserved thinking rapidly fork prefixes. RAG is often prefill-bound and should deduplicate document tokenization. One-million-token analysis should use 5.2/5.3/Flash only with measured retrieval quality and memory headroom. Offline batches maximize throughput if expert balance and sequence lengths are bucketed. Optimize **cost per correct solution** or **successful agent task**, including retries, tool fees, hidden reasoning, and wall time, rather than raw tokens/s.

## 12. Prefix-cache correctness

A portable logical key must contain:

```text
CacheEntry {
  exact token and visual-token IDs;
  model, weight hash, tokenizer, processor and chat-template revision;
  thinking mode/effort and exact preserved reasoning blocks;
  tool-schema revision and tool-result serialization;
  adapter identity;
  per-layer KV, compressed MLA, recurrent state;
  sparse indexer vectors, pooled keys, selected-block metadata;
  positional/RoPE and multimodal positional state;
  cache dtype and quantization scales;
  context configuration;
  engine, page-table and parallel-layout metadata;
}
```

For 4.7, store GQA K/V per layer plus MTP-relevant state. For 4.7-Flash, store its MLA compressed state. For 5/5.1, store MLA/DSA state and index metadata. For 5.2/5.3, also preserve shared-index ownership/pattern and any MTP draft state; selected indices may be prefix- and current-query-dependent, so distinguish reusable indexer keys from query-specific selection results. For 5.3-Flash, linear recurrent states are prefix-dependent and fixed-size per layer; sparse history/index remains growing; IndexPool state and visual encoder outputs/tokens are reusable only with identical preprocessing. Page tables and shard ownership are engine-specific and usually not portable across parallel layouts.

Raw text is insufficient because whitespace, template, tool JSON ordering, reasoning blocks, visual transforms, or tokenizer revisions alter IDs. Preserved reasoning should be returned exactly when the protocol requires it. Test cached versus uncached execution at identical IDs: compare layer states, attention outputs, logits within dtype tolerance, greedy continuations, and sampled distributions. Cover exact/partial hits, one-token mismatch, thinking toggle, edited reasoning, tool-schema change, visual reuse, model revision, eviction/reload, BF16/FP8, and changed tensor/expert parallelism. A cache is correct only if logits match within justified numerical tolerance, not if visible prose “looks similar.”

## 13. Speculative decoding

GLM-5/5.1 configurations include one MTP layer; GLM-5.2 officially improves multi-step MTP with IndexShare/KVShare, rejection sampling, and end-to-end TV loss [[5.2 note](https://z.ai/blog/glm-5.2)]. No official Medusa/EAGLE head or separate draft checkpoint was verified. Framework prompt lookup/n-gram speculation is framework-level only.

The draft proposes 1-4 tokens; the target verifies them in parallel; the longest accepted prefix is committed; rejected suffix tokens and all corresponding MLA/KV, sparse indices, recurrent updates, MoE side effects, and multimodal positions are rolled back. With linear attention, speculative state must be shadowed or checkpointed because updates are sequential. Continuous batching adds per-request acceptance divergence. MoE routes proposals and verification independently, potentially increasing expert traffic.

Experiment over proposal length 1-4; concurrency 1/4/16/64; prompts 1K/8K/32K/128K; outputs 128/512/2K; reasoning on/off where allowed; cache hits 0/50/90%. Measure acceptance, accepted tokens/verification, TTFT, inter-token latency, output throughput, energy, cache memory, and quality. At saturation, target verification and draft overhead can reduce throughput even when single-request latency improves.

## 14. Serving costs

### 14.1 Official API pricing

Observed 4 September 2026 from the international Z.ai pricing page, USD per 1M tokens [[source](https://docs.z.ai/guides/overview/pricing)]:

| Model | Input | Cached input | Output | Note |
|---|---:|---:|---:|---|
| GLM-5.3-Flash | $0.075 | $0.015 | $0.25 | temporary 50% discount; list $0.15/$0.03/$0.50; ends Sep 9 UTC+8 |
| GLM-5.3 | $1.40 | $0.26 | $4.40 | output includes provider-defined reasoning accounting; verify invoice |
| GLM-5.2 | $1.40 | $0.26 | $4.40 | rolling API name |
| GLM-5.1 | $1.40 | $0.26 | $4.40 | rolling API name |
| GLM-5 | $1.00 | $0.20 | $3.20 | rolling API name |
| GLM-4.7 | $0.60 | $0.11 | $2.20 | rolling API name |
| GLM-4.7-FlashX | $0.07 | $0.01 | $0.40 | API-only architecture ND |
| GLM-4.7-Flash | free | free | free | quota/rate limits apply |

Cached-input storage was shown as limited-time free. The page does not make an immutable snapshot guarantee. Mainland-China BigModel pricing and coding-plan subscriptions must remain separate; they were not reproduced here because a stable, same-time primary table with all requested RMB tiers was not retrieved. **Not disclosed/needs reverification:** cache creation charge semantics, reasoning-token visibility, batch discount, rate limits, and subscription point conversion. Do not currency-convert or turn a quota subscription into token pricing.

Checked examples at list pricing, excluding tool fees: 5.3 for 8K input +1K output costs `0.008*1.4 + 0.001*4.4 = $0.0156`; 100K+5K costs `$0.14+$0.022=$0.162`; 1M+10K costs `$1.40+$0.044=$1.444`. With 90% cache reuse on a 100K repository prefix plus 10K fresh input and 5K output: `0.09M*0.26 + 0.01M*1.4 + 0.005M*4.4 = $0.0594`. At 5.3-Flash promotional rates the same cases are $0.00085, $0.00875, $0.0775, and $0.00335 respectively. These are **Analytically estimated** from observed rates; long reasoning adds output cost and occupancy.

### 14.2 Self-hosting

**Analytical raw weight bytes** before metadata/scales:

| Model | BF16 | FP8 | ideal 4-bit |
|---|---:|---:|---:|
| GLM-4.7 ~355B | 710 GB | 355 GB | 177.5 GB |
| GLM-5.x 744B | 1,488 GB | 744 GB | 372 GB |
| GLM-5.3-Flash 320B | 640 GB | 320 GB | 160 GB |

Actual packages differ: the official GLM-5.3 HF tree is about 756 GB, consistent with near-8-bit storage plus metadata [[files](https://huggingface.co/zai-org/GLM-5.3/tree/main)]; GLM-4.7’s observed tree is about 717 GB, close to BF16 arithmetic [[files](https://huggingface.co/zai-org/GLM-4.7)]. Quantized deployment needs scales, sometimes higher-precision shared layers, KV, workspace, CUDA graphs, and safety margin. GLM-5.3 vLLM-Ascend guidance requires two 8x128GB A3 nodes or four A2-class nodes for a W8A8C8 path, illustrating that package fit is not sufficient [[vLLM-Ascend](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.3.html)].

Practical cost requires measured accepted tokens/hour:

```text
cost per 1M accepted tokens = infrastructure $/hour /
                              accepted tokens/hour * 1,000,000
```

For example, an **Analytically estimated** $80/hour deployment at 40,000 accepted tokens/hour costs $2,000 per 1M accepted tokens; at 400,000 tokens/hour it costs $200. These illustrative numbers are not claimed GLM throughput. Include idle replicas, host RAM, NVMe load time, failures, interconnect, engineering, vision encoding, and utilization. API prices can be far below low-utilization self-hosting because the provider multiplexes requests and uses optimized/quantized kernels.

## 15. Hooks for cross-family comparison

### GLM versus DeepSeek

Ask whether Flash hybrid attention reduces persistent bytes versus MLA+DSA/CSA/HCA under identical context; whether GLM expert routing has the same locality/load balance as DeepSeekMoE; whether both mHC implementations use identical path count/constraints; how preserved reasoning compares with R1/V4 semantics; which state is portable; whether native MTP exists; and how topologies change all-to-all.

### GLM versus Qwen

Compare Flash linear state with Gated DeltaNet, and sparse index/gather with Qwen Sparse Attention; expert granularity and shared experts; residual/hyper-connection design; native vision integration; MTP versus framework speculation; reasoning token budgets; and cache bytes at 8K, 128K, and 1M.

### GLM versus Kimi

Compare fixed recurrent versus growing attention state, expert structure, long-context retrieval/indexing, visual prefix tokenization, coding-agent post-training, speculative support, and cache restoration semantics.

Never rank without checkpoint hash, hardware, precision, engine/commit, optimized versus fallback kernels, context, batch, concurrency, reasoning budget, cache hit, tool environment, and task-success target.

## 16. Benchmark and profiling plan

Cross prompt lengths 1K/8K/32K/128K/256K/1M with outputs 128/512/2K/8K, concurrency 1/4/16/64/saturation, prefix hits 0/50/90/99%, thinking modes, text/multimodal, and ordinary/speculative decode. Unsupported cells are N/A, not failures.

Measure TTFT and inter-token latency p50/p95/p99; input and accepted output tokens/s/GPU; requests/s under SLO; SM and HBM utilization; cache bytes/request; expert load and all-to-all bytes/duration; index and gather time; recurrent-state update; vision preprocessing/encoding; speculative acceptance; joules/accepted token; dollars/completed task; task success and retries.

Instrument attention projections, cache reads, recurrence, router, permutation, expert GEMMs, collectives, sparse index/gather, visual encoder, MTP proposal/verification, cache lookup/restore, queueing, and tools. Record engine/commit, CUDA/driver, kernels, topology, quantization, power limit, model hash, tokenizer/template/tool schema, and reasoning settings. Explicitly flag fallback kernels. Publish warmup, sampling, seeds, traces, and failed requests.

## 17. Conclusions

### High-confidence official conclusions

GLM-5 is the major full-size backbone redesign after 4.7. GLM-5.2 adds trained IndexShare and revised MTP for 1M context. GLM-5.3 shares 5.2’s base and is post-training-only. GLM-5.3-Flash is a separate new multimodal backbone with hybrid attention and mHC. The latest verified release at cutoff is GLM-5.3-Flash.

### Analytical implications

Flagship batch-1 decode is weight/communication sensitive; long context is index/KV/capacity sensitive. Flash should reduce weight and attention traffic, but requires specialized recurrent, sparse, multimodal, and mHC kernels. MoE sparsity lowers active arithmetic, not residency. Reasoning and tools make cost per successful task more useful than token price.

### Independent observations

No independent, fully reproducible profiler suite covering final release, pinned engine, hardware, precision, and all context/concurrency cells was found. Provider throughput anecdotes are therefore excluded. Published benchmark claims without full environment are **Non-reproducible from published information**.

### Undisclosed information

Exact 5.3-Flash expert/head/layer pattern, linear recurrence equations/state dimensions, mHC path count, vision encoder/tokenization, training hardware, optimizer, accelerator-hours, FLOPs, and wall time remain **Not disclosed** in the primary sources reviewed.

### Questions requiring experiments and engineering risks

Test logit-equivalent prefix restoration, DSA/IndexPool misses, recurrent-state rollback, multimodal cache keys, MTP acceptance under reasoning, expert imbalance, cross-node all-to-all, 1M CPU scheduling, quantization drift, forced-thinking occupancy, and optimized versus fallback kernels. Principal risks are silent API alias advancement, template drift, cache false hits, underestimated KV/index memory, fallback paths, and misleading low-utilization self-hosting economics.

## 17. References

### 1. Official technical reports

- [GLM-5 technical report link hub](https://github.com/zai-org/GLM-5)
- [IndexShare / GLM-5.2 technical launch note](https://z.ai/blog/glm-5.2)
- [GLM-4.5 report referenced by 4.7 card](https://huggingface.co/zai-org/GLM-4.7)

### 2. Official repositories

- [zai-org/GLM-5](https://github.com/zai-org/GLM-5)

### 3. Official model cards and configurations

- [GLM-4.7](https://huggingface.co/zai-org/GLM-4.7), [config](https://huggingface.co/zai-org/GLM-4.7/raw/main/config.json)
- [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash), [config](https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/config.json)
- [GLM-5 config](https://huggingface.co/zai-org/GLM-5/raw/main/config.json)
- [GLM-5.2 config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)
- [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)

### 4. Official API, pricing, and release notes

- [Release notes](https://docs.z.ai/release-notes/new-released)
- [International model pricing](https://docs.z.ai/guides/overview/pricing)
- [GLM-5.3 API](https://docs.z.ai/guides/llm/glm-5.3)
- [GLM-5.3-Flash API](https://docs.z.ai/guides/vlm/glm-5.3-flash)
- [GLM-4.7 announcement](https://z.ai/blog/glm-4.7)
- [GLM-5 announcement](https://z.ai/blog/glm-5)
- [GLM-5.1 announcement](https://z.ai/blog/glm-5.1)
- [GLM-5.3 announcement](https://z.ai/blog/glm-5.3)
- [GLM-5.3-Flash announcement](https://z.ai/blog/glm-5.3-flash)

### 5. Serving-framework documentation

- [vLLM GLM-5/5.1 recipe](https://docs.vllm.ai/projects/recipes/en/stable/GLM/GLM5.html)
- [vLLM-Ascend GLM-5.3](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.3.html)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)

### 6. Reproducible vendor measurements

None accepted as fully reproducible for the complete matrix. Official performance plots are vendor-reported and **Non-reproducible from published information** where engine commits/traces are absent.

### 7. Community implementation notes

No community claim is used to establish lineage or exact architecture. Community notes should be used only after checking their values against pinned official configs.

## Notes for the next session

1. Reverify releases and immutable aliases.
2. Reverify international and China pricing.
3. Preserve architecture versus post-training distinctions.
4. Preserve official versus estimated compute labels.
5. Add profiler traces.
6. Record engine and kernel versions.
7. Test prefix-cache logit equivalence.
8. Test preserved-thinking exactness.
9. Record speculative acceptance.
10. Compare cost per successful task.
11. Flag fallback kernels.
12. Record tokenizer, template, tool-schema, and thinking modes.
13. Extend the controlled matrix to DeepSeek, Qwen, and Kimi.
14. Avoid cross-family conclusions until conditions are controlled.
15. Recheck whether GLM-5.3 still shares the GLM-5.2 base.
16. Recheck whether GLM-5.3-Flash remains the newest redesigned backbone.
