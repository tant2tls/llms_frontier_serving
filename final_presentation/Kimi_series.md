# Kimi Forward Architecture and Serving Guide

> **Research cutoff:** 4 September 2026, 18:16 GMT+7  
> **Scope:** Officially documented open Kimi models from Kimi K2 through Kimi K3.  
> **Latest verified flagship:** **Kimi K3**  
> **Reproducibility pin:** `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`

## Document purpose

This report consolidates architecture, inference-state, serving, deployment, and benchmarking guidance for the open Kimi lineage. It distinguishes primary-source facts from engineering estimates and excludes provider aliases or rumored variants unless Moonshot AI documents them in an official repository, model card, technical report, or API page.

The report is intended for:

- inference and serving engineers;
- AI systems researchers;
- capacity and infrastructure planners;
- benchmark designers;
- technical decision-makers comparing Kimi with GLM, DeepSeek, and Qwen.

## Evidence taxonomy

Every important statement should be interpreted under one of these labels:

- **Officially reported:** stated in a Moonshot AI repository, model card, technical report, or official product documentation.
- **Independently measured:** observed in a documented external or vendor-linked deployment with a specific configuration.
- **Analytically estimated:** derived from disclosed dimensions or simplified systems models.
- **Not disclosed:** not recoverable from the primary sources reviewed.
- **Unverified and excluded:** rumor, provider alias, or unsupported architecture claim omitted from the verified lineage.

Vendor benchmark scores without complete engine, hardware, sampling, tool, and trace disclosure are considered **non-reproducible from published information**.

---

## 1. Release verification and scope

### 1.1 Latest verified release

**Officially reported:** Kimi K3 is an open-weight, native-multimodal, 2.8-trillion-parameter mixture-of-experts model with 104 billion activated parameters and a 1,048,576-token context window. The documented backbone includes:

- 93 transformer layers;
- 69 Kimi Delta Attention layers;
- 24 Gated MLA layers;
- Attention Residuals;
- Stable LatentMoE;
- 896 routed experts;
- top-16 expert selection;
- two shared experts;
- MoonViT-V2;
- MXFP4 weights and MXFP8 activations through quantization-aware training.

Moonshot's repository describes K3 as its most capable model at the research cutoff. The repository has no tagged GitHub releases, so `main` should be treated as rolling. For reproducibility, pin an explicitly recorded commit such as:

```text
9f62e4e9fffbd0a83ddd60e1c209d828994b3569
```

Primary sources:

- [Kimi K3 repository](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi K3 pinned Hugging Face tree](https://huggingface.co/moonshotai/Kimi-K3/tree/9f62e4e9fffbd0a83ddd60e1c209d828994b3569)
- [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)

### 1.2 Verified lineage

```text
Kimi K2 Base, July 2025
  -> Kimi K2 Instruct, non-thinking post-training
  -> Kimi K2 Instruct 0905, agentic/coding update with 256K context
  -> Kimi K2 Thinking, reasoning and long tool-use post-training
  -> Kimi K2.5, continual multimodal pretraining on K2 Base plus post-training

Kimi K3, July 2026
  -> new 2.8T/104B native-multimodal backbone
  -> KDA, Gated MLA, Attention Residuals, Stable LatentMoE
  -> MXFP4/MXFP8 QAT and 1M context
```

Kimi K2 has 1 trillion total parameters and approximately 32 billion activated parameters. It was released as separate Base and Instruct checkpoints. The initial Instruct release is a non-thinking model. The 5 September update increased agentic coding capability and extended context to 256K. K2 Thinking is a reasoning and tool-use post-training branch rather than a documented replacement of the K2 backbone.

Kimi K2.5 is not merely a vision adapter. Moonshot describes continual pretraining of K2 Base on approximately 15 trillion mixed visual and text tokens. It retains the K2-class language MoE while adding native multimodality through MoonViT and supporting instant, thinking, and agent-swarm behaviors.

Primary sources:

- [Kimi K2 official page](https://moonshotai.github.io/Kimi-K2/)
- [Kimi K2 Thinking](https://moonshotai.github.io/Kimi-K2/thinking.html)
- [Kimi K2.5 repository](https://github.com/MoonshotAI/Kimi-K2.5/tree/master)
- [Kimi K2.5 technical report](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf)

### 1.3 Official status classes

K2 Base, K2 Instruct, K2 Thinking, K2.5, and K3 have official downloadable weights. Exact API aliases should be considered rolling services unless Moonshot explicitly documents an immutable snapshot.

Product-routing identifiers such as `kimi-for-coding`, `kimi-for-coding-highspeed`, and `k3` are not automatically equivalent to immutable weight hashes. A production record should distinguish:

- public product alias;
- region and endpoint;
- resolved model revision, when exposed;
- tokenizer and processor revision;
- chat template;
- reasoning mode;
- tool parser and schema;
- observation date.

The Kimi CLI documentation distinguishes global and mainland-China providers and allows capability declarations for thinking, always-thinking, image, and video input.

- [Kimi CLI provider configuration](https://moonshotai.github.io/kimi-cli/en/configuration/providers.html)
- [Kimi CLI environment variables](https://moonshotai.github.io/kimi-cli/en/configuration/env-vars.html)

---

## 2. Executive summary

1. **Officially reported:** Kimi K3 is a new backbone, not K2.5 post-training. It increases total parameters from 1T to 2.8T and activated parameters from 32B to 104B.
2. **Officially reported:** K3 has 93 layers, 896 routed experts, top-16 selection, two shared experts, hidden width 7,168, 96 attention heads, a 160K vocabulary, and 1M context.
3. **Officially reported:** KDA is the main long-context mechanism. Its persistent recurrent state is fixed in sequence length, but K3 also has 24 Gated-MLA layers with sequence-growing state. Therefore, the full K3 history cache is not constant-size.
4. **Officially reported:** Attention Residuals alter activation flow and within-layer computation. They do not, by themselves, imply sequence-length-growing decode state.
5. **Officially reported:** Stable LatentMoE routes through a 3,584-dimensional latent representation and activates 16 of 896 experts, with two shared experts.
6. **Officially reported:** K3's MXFP4 weights and MXFP8 activations are part of the trained deployment design rather than only a community post-training conversion.
7. **Officially reported:** K3 uses a 401M-parameter MoonViT-V2 vision encoder. K2.5 uses a 400M MoonViT encoder and approximately 15T mixed visual and text continual-pretraining tokens.
8. **Analytically estimated:** K2.5 retains K2-style sequence-growing MLA cache behavior. K3 combines fixed KDA state, sequence-growing Gated-MLA state, visual tokens, and positional metadata.
9. **Analytically estimated:** Batch-1 K3 decode is likely constrained by weight bandwidth, expert collectives, kernel-launch latency, KDA-state traffic, and the Gated-MLA cache.
10. **Analytically estimated:** At 1M context, KDA prevents all 93 layers from carrying a conventional million-token KV cache, but the 24 MLA layers, media tokens, metadata, and context-parallel communication remain substantial.
11. Reasoning and long tool trajectories increase serialized output, session duration, cache occupancy, and tool-loop exposure.
12. Multimodal caching must include exact media identity and preprocessing. Different resizing, cropping, orientation, frame selection, processor revision, or media bytes invalidate the visual prefix.
13. Raw 4-bit storage for 2.8T parameters is 1.4 TB before scales, metadata, embeddings, higher-precision tensors, cache, recurrent state, and workspaces.
14. API price is not physical serving cost. Compare cost per successful task, including media encoding, reasoning, tools, retries, and wall-clock occupancy.
15. Cross-family comparisons must control checkpoint, quantization, optimized kernels, topology, reasoning budget, context, cache hit rate, tool environment, and success criterion.

---

## 3. Release and architecture matrix

| Model | Status and relation | Parameters | Layers and experts | Attention | Context | Modality and weights |
|---|---|---:|---|---|---:|---|
| Kimi K2 Base | New base, July 2025 | 1T total, 32B active | 61 layers; 384 routed, top-8, one shared | MLA | Initially 128K; later K2 update 256K | Text; open weights |
| Kimi K2 Instruct | Post-trained K2 Base, non-thinking | Same class | Same backbone | MLA | Release-dependent | Text; open weights |
| Kimi K2 Instruct 0905 | Updated agentic/coding post-training | Same class | Same backbone | MLA | 256K | Text; open weights |
| Kimi K2 Thinking | Reasoning and tool-use post-training | Same class unless pinned config differs | No backbone change officially established | MLA | API/checkpoint-dependent | Text; open weights |
| Kimi K2.5 | Multimodal continual pretraining atop K2 Base | 1T/32B | 61; 384 routed, top-8, one shared | MLA | 256K | Image/text and product video paths; MoonViT 400M; open |
| Kimi K3 | New base, July 2026 | 2.8T/104B | 93; 896 routed, top-16, two shared | 69 KDA + 24 Gated MLA | 1,048,576 | Native image/text and product video paths; MoonViT-V2 401M; open |

Candidate names such as `Kimi K2.6`, `Kimi K3 Turbo`, `Kimi K3 Thinking`, or provider-specific “K3 high-speed” checkpoints are **unverified and excluded** as separate architectures unless Moonshot publishes a primary source.

---

## 4. Kimi K2 baseline

### 4.1 Sparse MoE serving behavior

K2 combines a trillion-parameter checkpoint with sparse per-token expert activation. The full expert set must remain available somewhere in the serving system even though each token activates only a subset. A serving engine typically:

1. computes routing scores;
2. selects experts;
3. permutes tokens by destination expert;
4. dispatches token representations;
5. runs grouped expert GEMMs;
6. weights expert outputs;
7. combines and restores token ordering.

At batch 1, expert batches are small and weight traffic can dominate useful arithmetic. Topology-aware expert placement matters because selected routed experts and the shared expert may span devices. At higher concurrency, expert GEMMs become more efficient, but load imbalance, synchronization, dispatch, and all-to-all traffic can become the bottleneck.

Sparse activation does **not** proportionally reduce:

- checkpoint download size;
- aggregate HBM requirement;
- cold-start loading time;
- host RAM and NVMe staging;
- replica cost;
- operational complexity.

### 4.2 MLA and cache growth

A conventional GQA logical KV-cache estimate is:

```text
KV_bytes = L * S * H_kv * D_head * 2 * bytes_per_element
```

where:

- `L` is the number of attention layers;
- `S` is sequence length;
- `H_kv` is the number of KV heads;
- `D_head` is head dimension;
- the factor `2` represents keys and values.

MLA stores a compressed latent for each historical token plus positional components. A generic estimate is:

```text
MLA_bytes(S) ~= L * S * (D_latent + D_rope) * bytes_per_element
```

The exact K2 latent dimensions, packing, page size, quantization, and absorbed projection strategy must be read from the pinned configuration and serving implementation. It is unsafe to estimate K2 by substituting conventional KV-head counts.

MLA remains linear in sequence length. Absorbed MLA can reduce cache bytes and projection traffic, but prefill and decode still consume historical latents.

### 4.3 K2 Thinking operational implications

K2 Thinking changes reasoning and tool-use behavior rather than establishing a new attention architecture. Long reasoning and sequential tool calls imply:

- longer decode residence;
- more generated state;
- repeated tool latency;
- larger retained contexts;
- more opportunities for timeout, retry, and cache invalidation;
- lower requests per second under fixed latency and concurrency limits.

When provider APIs preserve reasoning state across turns, the preserved representation must be part of cache identity. Otherwise identical visible text could incorrectly reuse a prefix created under a different hidden or serialized reasoning history.

---

## 5. Kimi K2.5

K2.5 changes modality and continual pretraining while preserving the K2-class language backbone. Officially summarized dimensions include:

- width: 7,168;
- attention heads: 64;
- transformer layers: 61;
- routed experts: 384;
- selected experts: 8;
- shared experts: 1;
- expert hidden dimension: 2,048;
- vocabulary: 160K;
- context: 256K;
- attention: MLA;
- vision encoder: MoonViT, approximately 400M parameters.

### 5.1 Multimodal pipeline

A multimodal request adds distinct stages:

```text
media validation
  -> media decode
  -> orientation correction
  -> resize/crop or frame sampling
  -> MoonViT encoding
  -> projection into language tokens
  -> language-model prefill
  -> autoregressive decode
```

Visual token count and preprocessing policy affect time to first token, cache size, and language-model prefill. Encoder parallelism can be configured independently of language-model tensor parallelism. Moonshot's deployment guidance includes an H200 TP8 example with:

```text
--mm-encoder-tp-mode data
```

The deployment also requires Kimi-specific reasoning and tool-call parsers for correct structured output.

- [K2.5 deployment guide](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/docs/deploy_guidance.md)

### 5.2 Heterogeneous deployment example

The deployment guide reports an external KTransformers heterogeneous example using eight L20 GPUs and two Intel 6454S CPUs at concurrency 48:

- prefill: 640.12 tokens/s;
- decode: 24.51 tokens/s.

This is **independently measured/vendor-linked** and configuration-specific. It should not be transferred to H200 or pure-GPU serving. It demonstrates that CPU expert offload may increase model capacity while making decode sensitive to host-memory bandwidth, NUMA placement, PCIe traffic, and expert scheduling.

### 5.3 Agent Swarm

Agent Swarm is primarily an orchestration and post-training behavior. Parallel subagents can reduce wall-clock time for decomposable tasks, but they also multiply:

- concurrent model requests;
- duplicated prefixes;
- tool calls;
- total generated tokens;
- merge and synthesis context;
- failure and retry surfaces.

Prefix sharing is useful only until subagent contexts diverge. The benchmark unit should therefore be a completed agent task rather than a single request or raw output token.

---

## 6. Kimi K3 redesigned backbone

### 6.1 Scaling and Stable LatentMoE

K3 scales to 2.8T total and 104B activated parameters with:

- 896 routed experts;
- top-16 routing;
- two shared experts;
- 3,584-dimensional latent MoE representation;
- 3,072 expert hidden dimension;
- 7,168 model hidden dimension.

Stable LatentMoE may reduce routing and communication width relative to sending full hidden states. Exact bytes depend on where projection occurs, whether destinations are deduplicated per device, how shared experts are placed, and how expert outputs are combined.

A simplified dispatch model is:

```text
dispatch_bytes ~= tokens * selected_experts * routed_width * bytes * overhead
```

Some implementations send one token representation per destination device rather than one copy per selected expert. Therefore, the top-k factor cannot be blindly applied without tracing the engine.

At 896 experts, the following become first-order concerns:

- expert placement;
- routing skew;
- cross-node topology;
- shared-expert replication;
- grouped-GEMM shape;
- all-to-all latency and bandwidth;
- stragglers and synchronization.

Top-16 increases active expert fan-out relative to K2's top-8 design.

### 6.2 Kimi Delta Attention

KDA is a gated, delta-rule linear-attention mechanism. A conceptual form is:

```text
prediction_t = State_(t-1) * key_t
error_t = value_t - prediction_t
State_t = decay_t * State_(t-1)
          + learning_rate_t * error_t * key_t^T
output_t = State_t * query_t
```

This is explanatory, not a claim about exact tensor ordering. The technical report and reference implementation define the actual normalization, gating, chunking, tensor dimensions, and numerical behavior.

#### Decode-state implications

KDA's persistent state is fixed in sequence length per layer, but scales with:

- concurrent sequences;
- number of KDA layers;
- heads and state dimensions;
- state precision and packing;
- duplicated or sharded layouts;
- checkpoints required for branching and rollback.

KDA avoids reading every historical token during decode, but it requires specialized kernels and careful state management. Depending on state size and reuse, KDA may be compute-bound or HBM-bandwidth-bound.

#### Prefix caching and branching

For a prefix boundary, the cache can store the final recurrent state rather than every intermediate token state. Arbitrary branching inside a cached prefix requires one of:

- intermediate checkpoints;
- replay from an earlier checkpoint;
- recomputation from tokens;
- an engine-specific persistent scan structure.

Speculative decoding and cancellation require transactional or copy-on-write state so rejected tokens do not corrupt the verified recurrent state.

### 6.3 Gated MLA

K3 includes 24 Gated-MLA layers among 93 total attention layers. These retain sequence-growing compressed history and provide direct content-addressable retrieval.

A logical estimate is:

```text
K3_MLA_cache(S) ~= 24 * S * (D_c + D_rope) * bytes
```

The combined history can be represented as:

```text
K3_total_history(S)
  ~= KDA_fixed_state
   + K3_MLA_cache(S)
   + multimodal_prefix_state
   + positional_state
   + page_tables_and_metadata
```

The hybrid architecture creates a useful systems split:

- 69 layers have fixed-in-`S` KDA recurrent state;
- 24 layers keep linear-in-`S` MLA state.

At one million tokens, the 24 MLA layers can still dominate per-request capacity even though this is much smaller than retaining a conventional million-token cache for all 93 layers.

### 6.4 Attention Residuals

Attention Residuals expose or combine multiple prior attention outputs through learned residual pathways instead of relying only on the immediately preceding residual stream. They can improve information flow but may add:

- projection or mixing operations;
- activation traffic;
- synchronization;
- implementation complexity;
- revision-sensitive cache semantics.

Attention Residuals do not automatically require persistent sequence-length-growing state. Unless the implementation explicitly caches historical residual activations, prefix-cache storage is driven primarily by KDA, Gated MLA, multimodal state, and positional metadata.

### 6.5 Quantization-aware inference

K3 specifies MXFP4 weights and MXFP8 activations through quantization-aware training. The ideal raw weight-storage calculation is:

```text
2.8e12 parameters * 4 bits / 8 = 1.4e12 bytes
```

Therefore, ideal raw storage is approximately **1.4 TB** in decimal units. Actual runtime footprint is larger because of:

- block scales and metadata;
- embeddings and output heads;
- unquantized tensors;
- vision encoder and projector;
- alignment and expert padding;
- routing and permutation buffers;
- KDA state and MLA cache;
- CUDA graphs and workspaces;
- allocator fragmentation and safety margin.

Efficient serving requires native microscaling kernels. Dequantizing weights to BF16 can erase much of the storage and bandwidth advantage.

- [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)
- [Kimi K3 reference implementation](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_k3.py)

---

## 7. Multimodal serving

K3's 401M-parameter MoonViT-V2 encoder produces a visual prefix consumed by the 2.8T language model. Although the encoder is small relative to total model weights, it can dominate time to first token for requests with large images, many images, or numerous video frames followed by a short answer.

A correct multimodal cache key should include:

- original media hash;
- decoded pixel or frame content;
- EXIF orientation correction;
- resize and crop policy;
- frame timestamps and sampling policy;
- processor revision;
- encoder revision;
- projector revision;
- visual token IDs;
- multimodal positional IDs;
- model commit and quantization configuration.

Caching encoded visual features saves vision-encoder computation. Caching the post-projection language prefix also saves language-model prefill. Neither is portable across processor or model revisions without an explicit equivalence test.

Exact patching, pooling, projector behavior, and visual-token scaling should be taken from the current technical report. If absent, they must remain **not disclosed** rather than inferred.

---

## 8. Persistent state and prefix-cache correctness

A logical cache entry should contain enough identity and state to reconstruct the exact prefix computation:

```text
CacheEntry {
  exact text token IDs;
  exact visual token IDs and media identity;
  model weight commit;
  tokenizer and processor revisions;
  chat template revision;
  reasoning mode and preserved reasoning;
  tool schema and serialized tool results;
  adapter identity;
  KDA recurrent state per layer;
  Gated-MLA compressed latent and positional cache;
  K2 or K2.5 MLA cache when applicable;
  multimodal encoder and projection state;
  sequence length and positional state;
  quantization formats and scales;
  page tables and shard ownership;
  tensor, expert, pipeline, and context-parallel layout;
}
```

### 8.1 Correctness rules

- A KDA state is reusable only at the exact prefix boundary that produced it.
- Any token, media, tool, template, reasoning, adapter, or model change before the boundary invalidates downstream state.
- MLA pages are prefix-dependent and appendable.
- Raw device pages are usually engine- and topology-specific.
- Logical caches may be portable only after explicit serialization and layout conversion.
- Attention-residual activations should be treated as ephemeral unless the implementation proves otherwise.

### 8.2 Validation matrix

Compare cached and uncached execution from identical inputs. Test:

- exact prefix hit;
- partial prefix hit;
- one-token edit;
- reasoning-mode toggle;
- altered preserved reasoning;
- tool-schema reordering;
- changed tool result;
- visual-resize or crop change;
- frame-sampling change;
- processor or tokenizer revision;
- model commit change;
- eviction and reload;
- MXFP4/MXFP8 versus fallback precision;
- tensor-, expert-, pipeline-, and context-parallel layouts.

Compare:

- KDA states;
- MLA pages;
- selected internal outputs where practical;
- final logits;
- greedy continuations;
- structured tool output.

Tolerance must be precision-aware. Run repeated deterministic trials to distinguish expected floating-point drift from cache corruption.

---

## 9. Speculative decoding

No official K3 Medusa or EAGLE heads, or a dedicated draft checkpoint, were verified in the reviewed primary sources. A vLLM or SGLang command-line flag does not prove model-native speculative support.

Prompt lookup may exploit repeated n-grams without model-specific heads, but effectiveness depends on workload.

### 9.1 Transactional state requirements

For KDA:

1. proposals update shadow recurrent state;
2. the target model verifies proposals;
3. accepted tokens are committed;
4. rejected tokens are rolled back to the last verified boundary.

For Gated MLA:

- speculative pages need transactional append;
- rejected entries must be discarded;
- accepted entries must become visible atomically.

Visual prefixes are read-only after prefill and can be shared only when media and preprocessing identities match exactly.

### 9.2 Speculative benchmark sweep

Test:

- proposal lengths: 1, 2, 3, 4;
- concurrency: 1, 4, 16, 64;
- input lengths: 1K, 8K, 32K, 128K, 1M;
- output lengths: 128, 512, 2K;
- instant and thinking modes;
- cache-hit rates: 0%, 50%, 90%.

Measure:

- accepted proposals per verification;
- time to first token;
- inter-token latency;
- target and draft compute;
- state-copy bytes;
- all-to-all time;
- energy per accepted token;
- successful-task quality.

Speculation may improve batch-1 latency while reducing saturation throughput because draft work, verification, state copying, and MoE collectives compete for resources.

---

## 10. Bottleneck analysis

A useful latency decomposition is:

```text
latency = queueing
        + media_preprocessing
        + vision_encoding
        + prefill
        + decode
        + tool_latency
        + retries
```

For a phase:

```text
phase_time ~= max(
  FLOPs / compute_throughput,
  bytes / memory_bandwidth,
  communication_bytes / link_bandwidth
)
```

| Phase | K2 and K2 Thinking | K2.5 | K3 |
|---|---|---|---|
| Batch-1 decode | Expert weights, MLA history, all-to-all | Same plus modality and longer reasoning | MXFP4 expert reads, top-16 all-to-all, KDA state, MLA history |
| High concurrency | Expert balance and cache capacity | Expert balance and encoder queue | Expert collectives, KDA batching, 24-layer MLA capacity |
| Long prefill | MLA attention and cache writes | MLA plus vision encoding | KDA scan, Gated-MLA prefill, vision |
| Maximum context | Product/checkpoint-dependent | 256K official | Context parallelism, MLA cache, metadata, KDA scan |
| Agent workloads | Serialized tools and reasoning | Swarm fan-out and media tools | Long sessions, huge context, top-16 communication |

### 10.1 Context regimes

- **1K to 8K:** weight traffic, expert dispatch, and MoE collectives usually dominate.
- **32K to 128K:** prefill and MLA-cache traffic become substantial.
- **256K:** K2.5 capacity and prefill become major constraints.
- **1M:** K3 benefits from fixed KDA state across 69 layers, but 24 MLA layers, visual prefixes, context-parallel traffic, and metadata remain expensive.
- **2K to 8K output:** decode bandwidth, expert scheduling, reasoning occupancy, and tool-loop duration become visible.

A tool-heavy task may be wall-clock-bound by browsing, code execution, or network operations rather than by GPU inference.

---

## 11. Training and compute disclosure

| Model | Tokens and training | Hardware, precision, and compute disclosure |
|---|---|---|
| K2 | Exact pretraining tokens should be taken from the current report; post-training amount not disclosed here | MuonClip is a reported contribution; complete accelerator-hours and wall time not disclosed here |
| K2 Thinking | Reasoning and tool-use post-training; amount not disclosed | Not disclosed |
| K2.5 | Approximately 15T mixed visual/text continual-pretraining tokens atop K2 Base | Hardware, hours, and total FLOPs not disclosed in this summary |
| K3 | Corpus and compute accounting should be extracted from the current report | MXFP4/MXFP8 QAT design reported; accelerator-hours not disclosed in the repository summary |

Analytical bounds, not official compute figures:

```text
K2 dense-style for T tokens:  6 * 1.0e12 * T
K2 active-style:              6 * 32e9 * T
K3 dense-style:               6 * 2.8e12 * T
K3 active-style:              6 * 104e9 * T
```

These omit attention, vision, dense and shared experts, routing, optimizer state, recomputation, communication, imbalance, data pipeline cost, and post-training. Active parameters are not an exact FLOP count. Accelerator-hours are not wall-clock time.

---

## 12. Weight storage and self-hosting

| Model | BF16 analytical | FP8 analytical | Ideal 4-bit analytical |
|---|---:|---:|---:|
| K2/K2.5, 1T | 2.0 TB | 1.0 TB | 0.5 TB |
| K3, 2.8T | 5.6 TB | 2.8 TB | 1.4 TB |

Only officially documented formats should be called official. Community INT4 or FP8 conversions should be labeled by publisher, method, source revision, and calibration strategy.

HBM planning must add:

- quantization scales and metadata;
- embeddings and unquantized tensors;
- KDA recurrent state;
- MLA cache;
- visual encoder and projector;
- router and expert-permutation buffers;
- collective-communication buffers;
- CUDA graphs and workspaces;
- allocator margin;
- redundancy and failover capacity.

Host RAM and NVMe must stage multi-terabyte packages. Expert parallelism needs high-bandwidth fabric. Ethernet-only multi-node deployments are likely to be latency-limited, especially with top-16 routing.

### 12.1 Serving-cost metric

Use accepted output rather than attempted output:

```text
cost_per_1M_accepted_tokens =
  hourly_infrastructure_cost
  / accepted_tokens_per_hour
  * 1,000,000
```

Illustrative example:

```text
$200/hour / 250,000 accepted tokens/hour * 1,000,000
= $800 per 1M accepted tokens
```

This is **analytically estimated**, not K3 performance. Add engineering, idle capacity, storage, networking, observability, retries, and availability targets.

---

## 13. API pricing and alias hygiene

Moonshot operates separate global and mainland-China endpoints, along with Kimi Code plan endpoints. USD global token prices, mainland RMB prices, and subscription quotas must be recorded separately.

Current exact K3 token pricing was not recovered from a stable official public pricing table in this research pass and is therefore **not disclosed here**. Do not copy a third-party number into an “official” table.

For each price observation, record:

- observation date and timezone;
- region and endpoint;
- currency;
- public model alias;
- resolved snapshot, if available;
- input price;
- cached-input price;
- output price;
- reasoning-token treatment;
- media charges;
- batch discount;
- rate and context limits;
- promotion expiry.

A coding-plan quota is not equivalent to a per-token API rate.

- [Kimi CLI provider documentation](https://moonshotai.github.io/kimi-cli/en/configuration/providers.html)
- [Kimi CLI configuration files](https://moonshotai.github.io/kimi-cli/en/configuration/config-files.html)

---

## 14. Cross-family comparison hooks

### 14.1 Kimi versus GLM

Compare:

- K3's 69 KDA and 24 Gated-MLA layers with GLM hybrid linear or sparse attention;
- fixed recurrent state versus sequence-growing cache;
- index and gather overhead;
- million-token retrieval quality;
- Attention Residuals versus mHC;
- Stable LatentMoE versus GLM MoE;
- top-16 versus alternative top-k routing;
- native visual prefix processing;
- forced and preserved reasoning;
- speculative-state transaction semantics.

### 14.2 Kimi versus DeepSeek

Compare:

- Gated MLA with MLA, DSA, CSA, or HCA variants;
- KDA recurrent state with sparse or hybrid attention;
- 896 top-16 experts with DeepSeekMoE;
- Attention Residuals with mHC;
- MXFP4 QAT with FP8 and FP4 deployments;
- all-to-all behavior under identical network topology.

### 14.3 Kimi versus Qwen

Compare:

- KDA with Gated DeltaNet;
- Gated MLA with sparse or global attention layers;
- Stable LatentMoE with ultra-sparse MoE;
- MoonViT-V2 with Qwen vision encoders;
- reasoning and tool-state preservation under equal token budgets.

Do not rank families without controlling:

- checkpoint and revision;
- precision and quantization;
- optimized kernel availability;
- engine and commit;
- GPU and interconnect topology;
- input and output lengths;
- concurrency and scheduling;
- cache-hit rate;
- reasoning budget;
- media preprocessing;
- tools and external environment;
- success criterion.

---

## 15. Controlled benchmark plan

### 15.1 Workload matrix

Cross the following dimensions:

- prompts: 1K, 8K, 32K, 128K, 256K, 1M;
- outputs: 128, 512, 2K, 8K;
- concurrency: 1, 4, 16, 64, saturation;
- prefix hits: 0%, 50%, 90%, 99%;
- reasoning: instant and thinking;
- modality: text, image, video;
- decode: ordinary and speculative.

Unsupported combinations should be marked **N/A**, not forced.

### 15.2 Metrics

Measure:

- time to first token;
- p50, p95, and p99 inter-token latency;
- input tokens/s/GPU;
- accepted output tokens/s/GPU;
- requests/s under SLO;
- HBM usage and utilization;
- KDA state bytes per request;
- MLA bytes per request;
- expert-load distribution;
- all-to-all bytes and time;
- vision-encoding latency;
- speculative acceptance;
- joules per accepted token;
- dollars per completed task;
- task success and retries.

### 15.3 Instrumentation

Instrument:

- media validation and decode;
- image resize/crop and frame sampling;
- MoonViT encoding;
- multimodal projection;
- KDA update and read;
- MLA projection and cache reads;
- Attention Residual mixing;
- routing and permutation;
- grouped expert GEMMs;
- dispatch and combine collectives;
- speculative proposal and verification;
- cache lookup and restoration;
- scheduler and queueing;
- tool execution.

### 15.4 Reproducibility record

Record:

- model commit;
- model-file hashes;
- tokenizer and processor revision;
- chat template;
- engine and commit;
- CUDA, driver, and kernel path;
- GPU model and count;
- interconnect topology;
- quantization format;
- power limit;
- reasoning mode;
- tool schema;
- media transforms;
- sampling settings;
- cache policy.

---

## 16. Conclusions

### 16.1 High-confidence official conclusions

K2 introduced a 1T/32B open agentic MoE with MLA. K2 Thinking is a reasoning and tool-use post-training branch. K2.5 adds native multimodal continual pretraining while preserving the K2-scale language backbone. K3 is the latest verified release at the research cutoff and introduces a new 2.8T/104B backbone with KDA, Gated MLA, Attention Residuals, Stable LatentMoE, native vision, quantization-aware MXFP4/MXFP8 inference, and 1M context.

### 16.2 Analytical implications

K3 replaces sequence-growing attention state in most layers with fixed recurrent KDA state, but its 24 Gated-MLA layers prevent total history from becoming constant-size. The 896-expert, top-16 MoE and multi-terabyte package make self-hosting a cluster-scale problem. MXFP4 reduces capacity and bandwidth only when native kernels avoid expensive dequantization paths.

### 16.3 Undisclosed or incomplete information

Before deployment, re-read the current primary sources for:

- API snapshot guarantees;
- official regional pricing;
- exact K3 training-token accounting;
- optimizer schedule;
- accelerator-hours and wall time;
- precise KDA production-state dimensions and layout;
- visual-token scaling;
- exact media preprocessing and projector behavior.

### 16.4 Engineering risks

Major risks include:

- fallback KDA or MLA kernels;
- expert imbalance;
- top-16 all-to-all saturation;
- transactional recurrent-state bugs;
- false prefix-cache hits;
- media-processor drift;
- API alias advancement;
- hidden reasoning cost;
- tool-loop runaway;
- insufficient multi-node bandwidth;
- incorrect cross-topology cache reuse.

---

## 17. References

### Official technical reports and repositories

- [Kimi K3 repository and architecture summary](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- [Kimi K2 official page](https://moonshotai.github.io/Kimi-K2/)
- [Kimi K2 Thinking](https://moonshotai.github.io/Kimi-K2/thinking.html)
- [Kimi K2.5 repository](https://github.com/MoonshotAI/Kimi-K2.5/tree/master)
- [Kimi K2.5 technical report](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/tech_report.pdf)

### Official model cards and implementations

- [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)
- [Kimi K3 pinned tree](https://huggingface.co/moonshotai/Kimi-K3/tree/9f62e4e9fffbd0a83ddd60e1c209d828994b3569)
- [Kimi K3 reference implementation](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_k3.py)

### Serving and product documentation

- [K2.5 deployment guidance](https://github.com/MoonshotAI/Kimi-K2.5/blob/master/docs/deploy_guidance.md)
- [Kimi Code providers](https://moonshotai.github.io/kimi-cli/en/configuration/providers.html)
- [Kimi Code model and thinking environment](https://moonshotai.github.io/kimi-cli/en/configuration/env-vars.html)
- [Kimi Code configuration](https://moonshotai.github.io/kimi-cli/en/configuration/config-files.html)

---

## 18. Next-session verification checklist

1. Reverify whether K3 remains the latest official flagship and redesigned backbone.
2. Pin Git and Hugging Face commits and record file hashes.
3. Reverify global, mainland-China, and coding-plan pricing separately.
4. Extract exact KDA equations, state dimensions, and layer pattern.
5. Extract exact K3 training tokens, hardware, optimizer, and compute disclosures.
6. Measure KDA recurrent-state and Gated-MLA cache bytes.
7. Test cached versus uncached logit equivalence and KDA rollback.
8. Record optimized and fallback kernel paths.
9. Measure top-16 expert imbalance and all-to-all behavior.
10. Verify visual-token scaling and processor identity.
11. Test reasoning preservation and tool-schema cache invalidation.
12. Measure speculative acceptance with transactional KDA and MLA state.
13. Compare cost per successful agent task rather than raw token price.
14. Extend the controlled matrix to GLM, DeepSeek, and Qwen.
15. Exclude K2.6 or K3 aliases unless officially verified.

---

## Report maintenance rules

When updating this report:

1. change the research cutoff explicitly;
2. pin immutable commits wherever possible;
3. preserve the evidence labels;
4. separate official facts from analytical estimates;
5. never treat a product alias as an immutable checkpoint without evidence;
6. keep regional pricing and subscription plans separate;
7. record complete serving configuration for every benchmark;
8. mark missing information as **not disclosed** rather than inferring it;
9. test cache correctness before reporting cache speedups;
10. evaluate successful-task economics, not only tokens per second.
