# Qwen Series Architecture and Serving Performance Report

**Scope:** Qwen3.6 through Qwen3.8-Flash-Next  
**As of:** 2026-09-04  
**Purpose:** Working technical report for comparison with GLM, DeepSeek, and Kimi. It emphasizes architecture-to-systems implications rather than benchmark scores alone.

---

## 1. Executive summary

From Qwen3.6 through the main Qwen3.8 release, Qwen reused the same broad backbone introduced by Qwen3-Next and scaled in Qwen3.5: a 3:1 mixture of **Gated DeltaNet (GDN)** linear/recurrent token-mixing layers and **Gated Attention** global-attention layers, combined with either dense FFNs or sparse Mixture-of-Experts (MoE) FFNs. The models also include **multi-token prediction (MTP)** for self-speculative decoding and support native 262K context, usually extensible to about 1M tokens. Qwen3.6 primarily improved agentic coding, stability, and thinking-context preservation. Qwen3.7 was mostly a hosted capability/product release. Qwen3.8 scaled the existing design to a 2.4T-total, 95B-active MoE. [Qwen3.6 model card](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B), [Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8).

Qwen3.8-Flash-Next is the meaningful architectural break and an early preview of Qwen4. It combines GDN with **Qwen Sparse Attention (QSA)**, adds a four-branch **Gated Residual**, introduces a 51B-parameter host-resident **n-gram embedding** table, and uses a refined **Muon plus AdamW** optimization recipe. The main backbone has 125B parameters with only 6B activated per token. Qwen reports about one-third the active parameters, one-third the training tokens, and roughly one-ninth the training FLOPs of its 397B-A17B predecessor while retaining similar or better pretraining benchmark quality. [Technical report](https://arxiv.org/abs/2608.30320), [official release](https://qwen.ai/blog?id=qwen3.8-flash-next), [GitHub repository](https://github.com/QwenLM/Qwen3.8-Flash-Next).

The key serving implication is that there is no single bottleneck:

- **Short prompt, batch 1 decode:** weight and expert memory bandwidth, kernel-launch overhead, and inter-GPU latency dominate. MTP can help most here.
- **Long prompt prefill:** token-mixing computation and activation memory dominate. Full attention becomes expensive with context length; GDN and QSA reduce this pressure.
- **High-concurrency decode:** KV/state capacity, memory bandwidth, scheduler efficiency, and MoE all-to-all communication dominate. MTP's benefit usually shrinks because ordinary continuous batching already exposes parallelism.
- **Very large MoE serving:** total weight storage and expert placement remain major costs even when active FLOPs are low. A 2.4T model does not become cheap merely because only 95B parameters are active.
- **Flash-Next:** the 51B n-gram table introduces a new host-memory/NVMe locality and prefetch problem, while its GDN recurrent state makes prefix caching more complex than caching a normal Transformer KV sequence.

---

## 2. Model lineage and architecture

### 2.1 Release-level summary

| Model | Availability | Core architecture | Parameters | Native context | Main change |
|---|---|---|---:|---:|---|
| Qwen3.6-27B | Open weights | Dense, 3 GDN layers per 1 Gated Attention layer, MTP | 27B active | 262K | Coding, stability, thinking preservation |
| Qwen3.6-35B-A3B | Open weights | Hybrid attention, sparse MoE, MTP | 35B total, about 3B active | 262K | Lower arithmetic per token |
| Qwen3.7-Plus | Hosted only | Same broad GDN plus Gated Attention lineage; details undisclosed | Undisclosed | 1M service context | Multimodal GUI and interactive agents |
| Qwen3.8-27B | Open weights | Dense hybrid GDN/Gated Attention, MTP | 27B active | 262K | Better coding, office, and multimodal workflows |
| Qwen3.8-2.4T-A95B | Open weights | 3:1 GDN/Gated Attention, 512-expert MoE, MTP | 2.4T total, 95B active | 262K | Max-class capacity and long-horizon agents |
| Qwen3.8-Flash-Next | Open weights | GDN plus QSA, sparse MoE, Gated Residual, n-gram embedding, MTP | 125B main, 6B active, plus 51B n-gram | 262K | Qwen4-preview efficiency redesign |

Sources: [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.8-2.4T-A95B model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B), [Qwen3.7-Plus service page](https://www.qwencloud.com/models/qwen3.7-plus), [Qwen3.8-Flash-Next report](https://arxiv.org/abs/2608.30320).

### 2.2 Shared Qwen3.6 to Qwen3.8 backbone

For Qwen3.6-27B, the published layout is:

```text
16 x [
    3 x (Gated DeltaNet -> dense FFN)
    1 x (Gated Attention -> dense FFN)
]
```

This produces 64 layers: 48 GDN layers and 16 full-attention layers. The model uses a 5,120 hidden dimension, GQA in global-attention layers, a 248,320 padded vocabulary, and multi-step MTP training. [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B).

For Qwen3.8-2.4T-A95B, the published layout is:

```text
23 x [
    3 x (Gated DeltaNet -> MoE)
    1 x (Gated Attention -> MoE)
]
```

This produces 92 layers, 69 GDN layers, and 23 global-attention layers. It has hidden size 8,192, 512 routed experts, ten routed experts selected per token, and one always-active shared expert. The model has 2.4T total parameters and about 95B activated per step. [Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B).

### 2.3 What Gated DeltaNet changes

A conventional full-attention layer retains key and value tensors for every processed token. Its prefill attention work grows roughly quadratically with context length, while its decode-time KV-cache reads grow linearly with the number of cached tokens.

GDN instead compresses prior context into a recurrent state. This changes the systems profile:

- State size does not grow linearly with sequence length in GDN layers.
- Long-context decode avoids reading a full KV history in three out of four layers.
- Prefix cache storage is potentially much smaller for GDN layers.
- Cache reuse becomes semantically more complex because the cached object is a recurrent state at an exact token boundary, not simply a collection of independently addressable KV blocks.
- The implementation needs specialized fused convolution, recurrence, gating, and state-update kernels. A fallback implementation can erase the theoretical advantage.

Qwen3.8-Flash-Next retains three GDN layers out of every four and uses QSA for precise retrieval in the remaining layer. NVIDIA describes GDN as compressing history into fixed-size recurrent state and reports that the hybrid design materially improves long-context throughput. [NVIDIA deployment analysis](https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-on-nvidia-gb300-nvl72-for-agentic-coding/), [Qwen3.8-Flash-Next report](https://arxiv.org/abs/2608.30320).

### 2.4 MoE implications

MoE separates **total capacity** from **active arithmetic**. Qwen3.8's 2.4T model activates about 95B parameters, or roughly 4% of the total, on a token. Flash-Next activates about 6B in a 125B main model. This reduces matrix-multiplication work compared with a dense model of the same total size, but it does not eliminate:

- storage for all experts;
- HBM or host-memory placement decisions;
- router execution;
- token permutation and dispatch;
- inter-GPU all-to-all communication;
- load imbalance and expert hot spots;
- small or irregular GEMMs at low batch sizes;
- synchronization between tensor, expert, and pipeline parallel groups.

At low batch size, MoE can be latency-inefficient because each selected expert receives too few tokens to form large GEMMs. At high aggregate batch size, expert batching improves arithmetic intensity, but all-to-all traffic and load balance become first-order bottlenecks. Qwen's 512-expert layout therefore benefits strongly from continuous batching and expert-aware placement. [Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B), [SGLang Qwen3.5 architecture guide](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5).

### 2.5 Qwen3.8-Flash-Next changes

Flash-Next introduces four changes with direct serving consequences:

1. **Qwen Sparse Attention:** a lightweight indexer selects relevant micro-blocks rather than attending densely to all prior tokens. This reduces long-context prefill and decode work, but introduces index construction, top-k selection, sparse gather, irregular memory access, and cache-metadata overhead.
2. **Gated Residual:** the residual stream is widened into four branches with elementwise control of reads and writes. This can improve optimization and information flow, but increases implementation complexity and may increase activation traffic.
3. **N-gram embedding:** a 51B lookup table adds capacity with little arithmetic. It can be stored in host memory and prefetched asynchronously, shifting the problem from GEMM throughput to row-lookup locality, PCIe/CXL/NVLink transfer, page faults, and prefetch overlap.
4. **Muon optimization:** relevant mainly to training. The report says it allows larger optimal learning rates and batch sizes and removes batch-size warmup in the tested recipe, improving stability and reducing wasted training work.

Sources: [technical report](https://arxiv.org/abs/2608.30320), [official blog](https://qwen.ai/blog?id=qwen3.8-flash-next), [repository](https://github.com/QwenLM/Qwen3.8-Flash-Next).

---

## 3. Bottleneck analysis by workload and batch size

### 3.1 Performance model

It is useful to split a request into:

```text
request latency = queueing + vision encoding + prefill + decode + tool/runtime overhead
```

The two language-model phases behave differently:

```text
Prefill: process many prompt tokens in parallel
Decode: generate one or a few accepted tokens per sequence per iteration
```

A simple roofline-oriented model is:

```text
phase time ~= max(compute FLOPs / sustained FLOP/s,
                  bytes moved / sustained bandwidth,
                  communication / link bandwidth)
```

For MoE, add routing imbalance and collective latency. For QSA, add indexing and sparse-gather work. For multimodal requests, add vision-encoder cost and visual-token expansion.

### 3.2 Bottleneck matrix

| Workload | Low batch / low concurrency | Medium batch | High batch / high concurrency | Qwen-specific implication |
|---|---|---|---|---|
| Short prompt, short response | Kernel launch, weight reads, communication latency | Weight bandwidth and GEMM utilization | Scheduler and HBM bandwidth | Dense 27B is operationally simple; MTP can reduce iterations |
| Short prompt, long response | Decode weight bandwidth | Decode memory bandwidth | KV/state capacity and bandwidth | MoE active weights help, but expert dispatch matters |
| Long prompt, short response | Prefill compute and attention | Prefill compute, activation memory | Chunked-prefill scheduling | GDN helps; QSA helps more at very long context |
| Long prompt, long response | Prefill then memory-bound decode | Mixed prefill/decode interference | KV/state memory and fairness | Separate prefill/decode pools may be preferable |
| Shared long system/repository prefix | Repeated prefill dominates without cache | Cache lookup and block reuse | Cache capacity, eviction, routing locality | Hybrid GDN state plus attention KV must be restored consistently |
| Multimodal document/video | Vision encoder and tokenization | Encoder batching and LM prefill | Visual-token memory pressure | Cache transformed visual tokens only if preprocessing is identical |
| Agentic coding with many turns | Repeated prefix prefill and output decode | Prefix-cache hit rate | Cache churn and long-tail output | Thinking preservation improves logical continuity but enlarges prefixes |
| Offline batch generation | Underutilized if batch is too small | GEMM-efficient | Compute or communication-bound | Disable latency-only optimizations if they reduce throughput |

### 3.3 Batch-1 decode

At batch 1, each decode step has very few tokens. Large weight tensors must be read to produce one next token, so the execution is typically memory-bandwidth-bound. Dense Qwen3.6-27B reads all dense weights each iteration. A sparse MoE reads only selected experts, but also performs routing and may communicate across devices. Qwen3.8-2.4T-A95B additionally requires a very large distributed deployment because all 2.4T weights must be resident or otherwise available even though 95B are active per token.

The likely priority order is:

1. quantize weights and possibly KV/state;
2. minimize tensor/expert-parallel latency;
3. fuse GDN and MoE kernels;
4. use MTP speculative decoding;
5. avoid over-sharding a model that could fit on fewer devices.

Dense Qwen3.6-27B may beat a nominally lower-active-parameter MoE in single-user latency when the MoE is poorly placed or communication-bound. Conversely, 35B-A3B can win when experts fit locally and the runtime has good sparse kernels.

### 3.4 High-batch decode

With continuous batching, many sequences contribute tokens to each iteration. GEMMs become larger and hardware utilization improves. The bottleneck shifts toward:

- HBM bandwidth for weights and cache/state;
- cache capacity, fragmentation, and paging;
- expert all-to-all bandwidth;
- router balance;
- scheduler fairness across different sequence lengths;
- interleaving prefill requests with decode tokens.

MTP can be less beneficial here. Verification creates wider token blocks and can disturb batching; acceptance rates vary by request; and the server may already be throughput-efficient from ordinary batching. Benchmark both output tokens per GPU-second and per-user inter-token latency rather than assuming a batch-1 speedup carries over.

### 3.5 Long-context prefill

For full attention, attention work grows roughly as the square of prompt length. Qwen's 3:1 hybrid reduces the number of layers paying this full cost. Flash-Next's QSA reduces it further by selecting micro-blocks. NVIDIA reports up to 7.6x prefill and 4.9x decode speedups for QSA over full attention in the tested long-context setup, plus 8.6x prefill throughput relative to Qwen3.7-Plus at a 1M-token context with 90% prefix-cache hit rate. These are vendor measurements on specific hardware and should not be generalized without reproduction. [NVIDIA analysis](https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-on-nvidia-gb300-nvl72-for-agentic-coding/).

Long prefill also stresses activation memory. Chunked prefill is usually required to prevent one huge request from blocking decode traffic. The correct chunk size depends on model, attention backend, GDN kernel, and service-level objective.

### 3.6 Flash-Next's n-gram bottleneck

The 51B n-gram table is a lookup structure rather than a conventional dense layer. It can be offloaded to CPU memory or mmap-backed storage. The required engineering includes:

- pin or mmap the table;
- derive row indices early;
- coalesce and deduplicate row requests;
- prefetch asynchronously;
- overlap host transfer with GPU backbone work;
- control NUMA placement;
- monitor page faults and tail latency;
- decide whether popular rows merit an HBM cache.

A community DGX Spark implementation reports that the n-gram table occupies about 44 GiB in its quantized checkpoint and can be served from NVMe via mmap so more unified memory remains for cache. Treat this as an implementation case study, not an official guarantee. [DGX Spark implementation](https://github.com/blazux/qwen3.8-Flash-DGX).

---

## 4. Prefix caching

### 4.1 What must be cached

For a pure Transformer, a prefix-cache entry usually contains per-layer K and V blocks plus metadata. For Qwen's hybrid architecture, a logically complete entry may need:

```text
CacheEntry = {
  token_ids and exact boundary,
  model/tokenizer/template/version identity,
  attention-layer KV blocks,
  GDN recurrent/convolution state,
  QSA index and selected-block metadata where applicable,
  multimodal encoder outputs or visual tokens where applicable,
  positional state and rope-scaling configuration,
  precision/quantization metadata
}
```

Restoring only attention KV while recomputing or zeroing GDN state is incorrect. A community Flash-Next deployment found a real bug where a block-size mismatch caused cache hits to restore an all-zero recurrent state, yielding wrong behavior until patched. This illustrates that a cache hit can be fast but semantically invalid if hybrid state is incomplete. [Implementation report](https://github.com/blazux/qwen3.8-Flash-DGX).

### 4.2 Key challenges

#### Exact-match semantics

The cache key must include token IDs, not raw strings. Tiny differences in system prompt, chat template, tool schema, image preprocessing, reasoning flags, or special-token placement break reuse. Include at least:

- checkpoint hash;
- tokenizer revision;
- chat-template revision;
- LoRA/adaptor identity;
- quantization mode if state formats differ;
- rope/YaRN settings;
- thinking/non-thinking or reasoning-effort configuration;
- multimodal preprocessing version.

#### Hybrid-state block boundaries

KV caches naturally support block-level paging. Recurrent state is stateful at a sequence boundary and cannot always be assembled by concatenating independently cached blocks. A robust design may checkpoint the GDN state at each reusable block boundary, or reconstruct from the nearest prior checkpoint. The time-space trade-off is:

```text
more recurrent checkpoints -> larger cache, faster restoration
fewer recurrent checkpoints -> smaller cache, more recomputation
```

#### QSA metadata

QSA needs compressed indexing and selected micro-block state. Determine whether cached index structures are reusable when a prefix is extended, whether selection is query-dependent, and which portions must be recomputed. The cache manager should version QSA metadata independently because index-kernel changes can alter layout.

#### Multimodal prefixes

Images and video create many transformed tokens. Reuse is valid only if resizing, frame sampling, normalization, positional encoding, and encoder weights are identical. Caching raw images is insufficient. Caching encoder outputs can save substantial work but consumes more memory and can create privacy concerns.

#### Cache capacity

Hybrid Qwen reduces KV growth in GDN layers, but global-attention or QSA layers still retain context-dependent state. At high concurrency, cache capacity can dominate HBM. Use paged allocation, prefix trees or hash-chained blocks, admission policy, and cost-aware eviction. A good eviction score should account for:

```text
reuse value ~= probability of reuse * recompute cost / bytes retained
```

A long repository prefix may deserve retention more than many short chat prefixes.

#### Distributed placement

A cache entry is tied to the parallel layout that created it. Moving it between replicas can involve large transfers. Use prefix-aware request routing so matching requests are sent to the worker already holding the prefix. For disaggregated prefill/decode, define a state-transfer format that includes GDN and QSA state, not only KV blocks.

#### Correctness testing

For every cache implementation, compare cached and uncached outputs using deterministic decoding across:

- exact prefix hit;
- partial-block hit;
- one-token mismatch;
- multiple chat templates;
- multimodal input;
- long context;
- thinking mode changes;
- model revision changes;
- eviction and reload;
- tensor/expert-parallel configurations.

Do not validate solely through latency or token equality on one short prompt. Check logits, recurrent state, and generated continuation over many steps.

---

## 5. Speculative decoding and MTP

### 5.1 Mechanism

Qwen3.6 and Qwen3.8 models are trained with multi-token prediction. At serving time, the MTP head can propose future tokens. The main model verifies those proposals in a wider pass and accepts a valid prefix. This is self-speculative decoding, so a separate draft model is not necessarily required.

A simplified speed model is:

```text
speedup ~= accepted tokens per verification step
           / (verification cost + drafting overhead relative to one AR step)
```

Actual speed depends on acceptance rate, proposal length, verification-kernel efficiency, batch size, sampling, and hardware.

### 5.2 When it helps

MTP is most attractive when:

- batch size is small;
- autoregressive decode is memory-bandwidth-bound;
- prompts generate predictable code, structured text, or factual continuations;
- acceptance rate is high;
- the MTP head remains on fast memory;
- the runtime efficiently verifies multiple candidates.

vLLM provides an MTP serving configuration for Qwen3.6-27B and explicitly positions it for low-latency, small-batch serving. [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B).

### 5.3 When it helps less or hurts

MTP may provide little benefit or regress performance when:

- batch concurrency is already high;
- creative/high-temperature sampling reduces acceptance;
- each sequence accepts a different number of tokens, increasing scheduler divergence;
- verification kernels are inefficient;
- the draft head increases memory pressure enough to reduce cache capacity;
- multimodal or tool-bound workloads spend little time in decode;
- short outputs do not amortize setup overhead.

Community measurements report approximately 1.5x to 2x for some Qwen3.6 local deployments and 33% or more for some Qwen3.8-27B configurations, but these vary heavily by hardware, quantization, prompt domain, and runtime. Use them only as evidence that MTP is viable, not as a planning constant. [Qwen3.6 MTP artifact](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF), [Qwen3.8 community tests](https://github.com/sudoingX/qwen38-mtp).

### 5.4 Interaction with prefix caching

Prefix caching and MTP optimize different phases:

- Prefix caching reduces repeated **prefill**.
- MTP reduces serial iterations in **decode**.

They are complementary for agentic coding, where a large repository/system prefix is reused and each turn generates a substantial answer. However, accepted multi-token blocks alter sequence advancement. The scheduler and cache allocator must commit only accepted tokens, roll back rejected candidates, and update attention KV, GDN recurrent state, QSA metadata, and n-gram state consistently.

### 5.5 Recommended MTP experiment

For each model and hardware target, sweep:

- speculative tokens: 1, 2, 3, 4;
- batch/concurrency: 1, 4, 16, 64;
- prompt lengths: 1K, 8K, 32K, 128K;
- output lengths: 128, 512, 2K;
- domains: code, math, factual QA, creative text, tool calls;
- sampling: greedy and production temperature;
- prefix-cache hit ratios: 0%, 50%, 90%.

Collect acceptance rate, accepted tokens per verification, time to first token, inter-token latency p50/p95/p99, output tokens/s/GPU, energy/token, and cache memory per active request.

---

## 6. Serving cost comparison

### 6.1 Current hosted prices

The following are QwenCloud international list prices observed on 2026-09-04. Prices can change and should be rechecked before a final cross-family comparison.

| Model/API | Input per 1M tokens | Output per 1M tokens | Cached input/read | Context | Interpretation |
|---|---:|---:|---:|---:|---|
| Qwen3.6-27B | $0.60 | $3.60 | Not listed on model page | 262K | Dense downloadable model |
| Qwen3.6-Flash | $0.25 up to 256K | $1.50 | $0.025 explicit read | 1M | Hosted speed/cost tier |
| Qwen3.6-Plus | $0.50 up to 256K | $3.00 | $0.05 explicit read | 1M | Hosted capability tier |
| Qwen3.8-27B | $0.50 | $3.00 | $0.05 explicit read / $0.10 implicit | 1M service | Dense open model service |
| Qwen3.8-Flash | $0.15 | $0.47 | $0.016 implicit or explicit read | 1M | Cost-efficiency leader |
| Qwen3.8-Max / 2.4T-A95B service | $2.00 | $6.00 | $0.17 explicit / $0.25 implicit | 1M | Frontier capability tier |

Sources: [Qwen3.6-27B](https://www.qwencloud.com/models/qwen3.6-27b), [Qwen3.6-Flash](https://www.qwencloud.com/models/qwen3.6-flash), [Qwen3.6-Plus](https://www.qwencloud.com/models/qwen3.6-plus), [Qwen3.8-27B](https://www.qwencloud.com/models/qwen3.8-27b), [Qwen3.8-Flash](https://www.qwencloud.com/models/qwen3.8-flash), [Qwen3.8-Max](https://www.qwencloud.com/models/qwen3.8-max), [Alibaba pricing reference](https://www.alibabacloud.com/help/en/model-studio/model-pricing).

### 6.2 Example cost

For a coding-agent turn with 100K input tokens and 5K output tokens, no cache:

```text
Qwen3.8-Flash = 0.1 * $0.15 + 0.005 * $0.47
               = $0.01735

Qwen3.8-27B   = 0.1 * $0.50 + 0.005 * $3.00
               = $0.065

Qwen3.8-Max   = 0.1 * $2.00 + 0.005 * $6.00
               = $0.23
```

With a 90K-token reusable prefix and explicit cache read pricing, Qwen3.8-Flash becomes approximately:

```text
cached prefix = 0.09 * $0.016  = $0.00144
new input     = 0.01 * $0.15   = $0.00150
output        = 0.005 * $0.47  = $0.00235
 total                           $0.00529
```

This is about 3.3x cheaper than the uncached request in this example. It also improves time to first token by avoiding most repeated prefill. Cache creation cost and cache lifetime must be amortized over later hits.

### 6.3 Why API price is not hardware cost

Provider prices include utilization, quantization, speculative decoding, caching, routing, cluster amortization, and commercial strategy. They should not be used to infer physical FLOPs directly.

For self-hosting, estimate:

```text
cost per 1M tokens =
  (GPU hourly cost * GPU count + CPU/RAM/storage/network cost)
  / effective tokens per hour
```

Use effective accepted output tokens, not raw model steps. Include idle capacity required to maintain latency SLOs, replica redundancy, model loading time, failed requests, and cache hit rate.

### 6.4 Model-specific self-hosting implications

#### Qwen3.6/3.8 dense 27B

Best fit for simple deployment and moderate hardware. Dense execution avoids expert dispatch and often scales predictably. Weight memory is manageable under FP8/FP4, but long-context state can still dominate. A vLLM recipe estimates Qwen3.6-27B at about 65 GB BF16 and 33 GB FP8 before runtime/cache overhead. [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B).

#### Qwen3.6-35B-A3B

Lower active compute than dense 27B, but total weight memory is higher. It is attractive when expert kernels are mature and the model can remain on one device or a low-latency node. It may underperform at batch 1 if expert GEMMs are too small.

#### Qwen3.8-2.4T-A95B

Self-hosting requires a large multi-node or very high-memory deployment because all 2.4T parameters must be stored. Sparse activation reduces arithmetic, not checkpoint storage or expert-network complexity. It should only be chosen when its quality improves end-to-end task completion enough to offset much higher infrastructure and operational costs.

#### Qwen3.8-Flash-Next

The 6B active backbone is attractive, but its total storage is about 176B parameters including the n-gram table. Its intended cost advantage relies on host-resident embedding lookup, QSA/GDN kernels, and overlap between storage/memory and GPU work. Without those optimizations, naive serving may fail to realize the architecture's advantage.

---

## 7. Measurement plan for comparison with GLM, DeepSeek, and Kimi

### 7.1 Keep quality and systems settings controlled

For a fair family comparison, record:

- exact checkpoint and revision;
- precision and quantization;
- reasoning mode and reasoning budget;
- prompt/chat template;
- tool schema;
- maximum context and rope scaling;
- serving engine and commit;
- attention backend;
- tensor, pipeline, data, and expert parallel sizes;
- prefix-cache policy and measured hit rate;
- speculative-decoding configuration and acceptance rate;
- power limit and hardware clocks.

Do not compare one model in thinking mode with another in direct mode without reporting the extra output/reasoning tokens.

### 7.2 Workload grid

| Dimension | Suggested values |
|---|---|
| Input length | 1K, 8K, 32K, 128K, 256K, 1M where supported |
| Output length | 128, 512, 2K, 8K |
| Concurrency | 1, 4, 16, 64, saturation point |
| Prefix reuse | 0%, 50%, 90%, 99% |
| Modality | text, one image, many images, video/document |
| Task | chat, code completion, repo agent, RAG, summarization, reasoning, tool use |
| Decode | autoregressive, MTP/speculative |
| Precision | BF16, FP8, FP4/int4 where supported |

### 7.3 Metrics

Measure at least:

- time to first token p50/p95/p99;
- inter-token latency p50/p95/p99;
- input tokens/s/GPU;
- accepted output tokens/s/GPU;
- requests/s at latency SLO;
- HBM used by weights, KV/state, activations, and runtime;
- host RAM and storage bandwidth;
- interconnect bytes and all-to-all time;
- expert load imbalance;
- cache hit rate and cache restoration time;
- MTP acceptance rate;
- energy per accepted token;
- dollars per completed task;
- task success rate and retries.

For agentic workloads, **dollars per successful task** is more meaningful than dollars per token. A more expensive model can be cheaper if it needs fewer attempts, fewer tool calls, and less human correction.

### 7.4 Instrumentation questions

For each model, answer:

1. Is prefill compute-bound, HBM-bound, or attention-bound?
2. Is decode weight-bandwidth-bound or KV/state-bandwidth-bound?
3. What fraction of step time is MoE routing and all-to-all?
4. At what concurrency do expert GEMMs become efficient?
5. At what context length does full/sparse attention dominate?
6. How much state is retained per token or per sequence?
7. Does prefix caching preserve exact logits?
8. At what batch size does MTP stop helping throughput?
9. Which quantized tensors are quality-sensitive?
10. Does serving cost track active parameters, total parameters, or communication?

---

## 8. Current conclusions and open questions

### Conclusions

1. **Qwen3.6 to Qwen3.8 is one architectural generation, not three unrelated backbones.** Most improvements before Flash-Next are caused by scaling, data, post-training, multimodality, and agent optimization.
2. **Dense versus MoE is a deployment trade-off.** Dense 27B is simpler and often strong at low concurrency; MoE reduces active arithmetic but adds storage and communication complexity.
3. **Hybrid GDN plus attention changes long-context economics.** Three quarters of layers avoid sequence-growing KV history, but the recurrent state complicates prefix caching.
4. **Prefix cache correctness is harder than in a pure Transformer.** Cache GDN state, attention KV, positional state, QSA metadata, and multimodal state consistently.
5. **MTP is primarily a low-batch decode optimization.** Its value must be measured by acceptance rate and end-to-end latency at production concurrency.
6. **Flash-Next shifts cost from arithmetic toward memory hierarchy and irregular access.** Its n-gram table and sparse indexer require systems co-design.
7. **The newest model is not necessarily trained on more tokens.** Flash-Next reports one-third the tokens and one-ninth the FLOPs of the comparison predecessor.
8. **Serving cost must be task-level.** API price, self-hosting TCO, cache reuse, reasoning-token volume, and task success all matter.

### Open questions requiring experiments or disclosure

- Exact training tokens, GPU-hours, and FLOPs for Qwen3.6, Qwen3.7, and mainline Qwen3.8 remain undisclosed.
- The optimal expert-parallel layout for Qwen3.8-2.4T-A95B depends heavily on hardware topology.
- QSA prefix-cache metadata and incremental-index maintenance need implementation-level validation.
- The best location for Flash-Next's n-gram table, HBM, host DRAM, CXL memory, or NVMe mmap, depends on row locality and interconnect.
- MTP gains at high concurrency and with production sampling remain workload-sensitive.
- Cross-family conclusions require the same engine maturity. Comparing a highly optimized Qwen kernel against a fallback GLM/Kimi/DeepSeek implementation would be misleading.

---

## 9. Primary documents and useful links

### Official Qwen model and architecture sources

- [Qwen3.6-27B official model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3.6-27B official blog](https://qwen.ai/blog?id=qwen3.6-27b)
- [Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8)
- [Qwen3.8-2.4T-A95B official model card](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
- [Qwen3.8-Flash-Next official blog](https://qwen.ai/blog?id=qwen3.8-flash-next)
- [Qwen3.8-Flash-Next repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next technical report, arXiv:2608.30320](https://arxiv.org/abs/2608.30320)

### Serving and pricing

- [vLLM Qwen3.6-27B serving recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- [SGLang Qwen3.5 hybrid/MoE guide](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)
- [NVIDIA Qwen3.8-Flash-Next deployment analysis](https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-on-nvidia-gb300-nvl72-for-agentic-coding/)
- [Alibaba Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
- [Qwen3.8-Flash service and cache pricing](https://www.qwencloud.com/models/qwen3.8-flash)
- [Qwen3.8-Max service and cache pricing](https://www.qwencloud.com/models/qwen3.8-max)
- [Qwen3.8-27B service pricing](https://www.qwencloud.com/models/qwen3.8-27b)
- [Qwen3.6-27B service pricing](https://www.qwencloud.com/models/qwen3.6-27b)

### Implementation case studies, not primary vendor claims

- [Flash-Next on DGX Spark, including hybrid-state prefix-cache bug](https://github.com/blazux/qwen3.8-Flash-DGX)
- [Qwen3.8-27B community MTP measurements](https://github.com/sudoingX/qwen38-mtp)
- [Qwen3.6 MTP GGUF artifact](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)

---

## 10. Notes for the next agent/session

1. Treat this report as the Qwen-specific chapter of a wider GLM/Qwen/DeepSeek/Kimi comparison.
2. Preserve the distinction between **official facts**, **analytical implications**, and **community measurements**.
3. Update prices and latest model identifiers before publication.
4. Do not invent undisclosed training-token, GPU-hour, or FLOP totals.
5. Build the cross-family comparison around the same workload grid and hardware.
6. Prioritize architecture-to-bottleneck mapping over benchmark-score aggregation.
7. Add profiler data when available: per-layer time, HBM traffic, MoE all-to-all, GDN state traffic, QSA indexing, and n-gram prefetch stalls.
8. For prefix caching, demand cached-versus-uncached logit equivalence tests before accepting throughput results.
