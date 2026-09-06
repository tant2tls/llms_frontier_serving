> ARCHIVED SESSION RECORD — not current instructions or conclusions. Original location: `README.md`. See [current report](../../report.md), [debugging guide](../../fix_bug.md), and [session handoff](../../WORKFLOW.md). Historical claims may be superseded.

# Serving the 2026 model generation on one 8×H100 node

**Measured comparison of GLM-5.3-Flash, DeepSeek-V4-Flash, and Qwen3.8-Flash-Next-FP8**

Tan Ngo · 2026-09-02 · for Kan Zhu, UW SyFI · answers [`task.md`](../../task.md)

**115 measured points**, three models, one node, all cold and complete. Plus **12 quarantined** as
evidence of measurement errors caught before they reached a table.

---

## Read in this order

| # | doc | why |
|--:|---|---|
| **1** | **[`report.md`](../../report.md)** | **The deliverable.** Answers all four `task.md` questions with labelled numbers. Start here. |
| 2 | [`WHY.md`](../../archive/notes/WHY.md) | The mechanism behind each number, written for the questions a serving-systems PI would ask |
| 3 | [`fix_bug.md`](../../fix_bug.md) | **13 bugs, each with the error text, the wrong hypotheses, and the fix.** Read before debugging any bring-up |
| 4 | [`REPRODUCE.md`](../../archive/notes/REPRODUCE.md) | Re-run any arm from scratch |
| 5 | Per model | [GLM-5.3-Flash](../../GLM-5.3-Flash/report.md) · [DeepSeek-V4-Flash](../../deepseek_v4_flash/report.md) · [Qwen3.8-Flash-Next-FP8](../../Qwen3.8-Flash-Next-FP8/report.md) |
| — | [`WORKFLOW.md`](../../archive/notes/WORKFLOW.md) · [`CLAUDE.md`](../../archive/notes/CLAUDE.md) · [`plan.md`](../../archive/notes/plan.md) | Working notes: session state, full reference, original plan. Not the deliverable |

**Labels appear on every number and are never mixed silently:**

| label | meaning |
|---|---|
| **[M]** | **Measured** on this hardware, this session |
| **[E]** | **Engine-side estimate** — vLLM computed it from config shapes × measured batch composition. Better than a hand model, *not* a hardware counter |
| **[A]** | **Analytical** — from `config.json`, tensor shapes, or vendor spec |
| **[H]** | **Hypothesis** — not yet tested. Named as such so it cannot be mistaken for a result |

**Nothing here measures accuracy.** Per the assignment, the emphasis is throughput and serving cost.

---

## The claim

> The uniform per-layer cost model that every scheduler and paged-KV allocator assumes — one block
> size, one bytes-per-token, one bottleneck per decode step — **is no longer true of a *single* model.**
> A 2026 decode step touches layers with different memory-growth laws, different bytes-per-parameter,
> and different cost classes, inside one transformer stack. Heterogeneity moved *inside* the layer
> stack, which makes it a **scheduling and memory-management problem, not a kernel problem.**

M\* (Kasikci & Wang, June 2026) argues serving systems break because they model inference as *"a single
autoregressive loop"* on a *"flat DAG"*, and answers with the Walk Graph at the **inter-component**
level. **This report's finding is that M\*'s premise now holds one level lower than M\* addresses it.**

---

## Headline result (ISL 16,384 · OSL 256 · TP8 · base models · same engine · all cold) [M]

| conc | Qwen3.8 | GLM-5.3 | V4-Flash | Qwen /B-act | GLM /B-act | V4 /B-act |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | **106.2** | 96.3 | 85.0 | **14.61** | 5.54 | 6.04 |
| 4 | **256.8** | 225.3 | 219.8 | **35.33** | 12.96 | 15.61 |
| 16 | **420.2** | 359.2 | 330.9 | **57.79** | 20.67 | 23.50 |
| 64 | **517.7** | 447.1 | 389.4 | **71.22** | 25.72 | 27.65 |

Engine effect measured at **0.997×**, so cross-model ratios are architecture, not tooling.

**The metrics disagree, and that disagreement is the answer.** Qwen wins raw, per-GPU, per-dollar, and
per-active-parameter (2.6–2.8×). V4 is the only model with KV headroom at 260K. GLM has the best
long-context TTFT. There is no single winner, and any report claiming one has picked a metric quietly.

**Five findings worth leading with:**

1. **Decode is latency-bound, not bandwidth-bound.** GLM measured at **9.5–18.1% of 3.35 TB/s [E]**, and
   bandwidth **peaks at c≈16–32 and then FALLS, while throughput keeps rising to c=64** — so what binds at high
   concurrency is demonstrably not HBM. Qwen reads the **fewest** bytes/step (6.77 GiB vs GLM's 16.19)
   and is the **fastest**. `B*_MoE = B*_dense·E/k` is not the operative cost model at reachable batch.
2. **The batch knee is c≈4–8, not c=64** — measured on two models, which agree to within 2% on scaling
   shape despite completely different attention stacks. 1→8 buys 3.08×; 8→64 buys 1.51× for 8× the load.
3. **Speculative decoding's sign depends on the axis you measure.** On the batch axis MTP decays to
   ≤1.0× by c=16. On the **context** axis it costs 4–6% throughput but **buys 16.7–24.8% TTFT** — which is
   the metric that matters on a 530:1 ISL:OSL agentic workload.
4. **KV bytes/token spans 3.3× within one generation** [A]: Qwen 24.75 > GLM 11.35 > V4 ~7.4.
5. **The same allocator resolved `block_size` to 4, 640, and 1152** across these models on one engine —
   **288× apart**, all three hybrids. One block size cannot serve them.

---

## Repo layout

```
.
├── README.md               ← you are here
├── report.md               ← THE DELIVERABLE
├── WHY.md  fix_bug.md  REPRODUCE.md      shared analysis + debugging + repro
├── task.md  plan.md  model_list.md       the assignment and the plan
├── WORKFLOW.md  CLAUDE.md  potential_questions.md    working notes
│
├── bench.sh                sweep harness — cold/complete/unqueued guards, MFU scraping
├── normalize.sh            THE ONLY sanctioned way to compare models (never raw tok/s)
├── sending.sh              functional smoke test
│
├── GLM-5.3-Flash/          46 pts · 34 KDA + 11 DSA — the hybrid centrepiece
├── deepseek_v4_flash/      50 pts · 43 DSA + MXFP4 experts
├── Qwen3.8-Flash-Next-FP8/ 19 pts · 36 GDN + 12 QSA, E/k = 51.2×
└── cache/                  HF_HOME (630 GiB of weights, gitignored in spirit)
```

Every model directory has the same shape, so one convention transfers:

```
<model>/
├── README.md      what ran, and WHY each non-default flag is not a free choice
├── report.md      the per-model deliverable
├── run*.sh        one script per arm — the config that actually produced the numbers
├── _common.sh     shared flags + preflight (GPU free? caches writable?)
├── results/<arm>/ one JSON per point + manifest.txt (hardware, versions, server cmdline)
├── results/_*/    QUARANTINED points — kept as evidence, never deleted
└── logs/          server startup logs; where block_size and KV sizing actually appear
```

**A leading underscore on an arm means "evidence, not a result."** `normalize.sh` excludes those
directories by default; pass one explicitly to inspect it.

### Quick start

```bash
./normalize.sh                    # all 115 points, fairly normalized
./normalize.sh --csv > out.csv    # machine-readable
./normalize.sh GLM-5.3-Flash/results/bf16kv     # one arm
```

---

## ⚠️ Docker/container traps when serving these models

**Every bug below cost real debugging time, and most of them named the wrong subsystem in their error
message.** Full diagnosis for each — error text, the hypotheses that failed, the fix — is in
[`fix_bug.md`](../../fix_bug.md). This section is the pre-flight checklist.

### 1. The image entrypoint silently serves a *different model* and squats a GPU

`vllm/vllm-openai:*` declares `ENTRYPOINT ["vllm", "serve"]`. Launch the container with no command and
**PID 1 becomes `vllm serve` with no model argument**, so vLLM falls back to its default — verified in
this image at `entrypoints/cli/serve.py:36`: *"Defaults to Qwen/Qwen3-0.6B if no model is specified."*

That phantom server parks **~77 GiB on GPU 0** (measured: 77,146 MiB used, 3,934 MiB free) and owns
**port 8000**. A TP=8 model then cannot start.

**The failure mode that actually burns you is not the OOM — it is the port.** Smoke-testing
`localhost:8000` hits Qwen3-0.6B and **passes**, so you can "verify" a model you never launched.

```bash
# fix AT SUBMIT TIME (it cannot be fixed from inside the container)
runai submit ... --command -- sleep infinity      # must be LAST; later flags get eaten by sleep

# then verify, do not assume:
ps -p 1 -o args=                                              # want: sleep infinity
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader # want: all 8 near 0
```

🛑 **Do not `kill -9` the squatter — it backfires.** Killing the EngineCore frees the memory but PID 1
dies with it; PID 1 is container init, so the container terminates, the scheduler reschedules, and the
entrypoint comes back with a **fresh** server holding the same 77 GiB. A respawn loop, not a fix.

🛑 **Never `pkill -f "vllm serve"` inside such a container** — PID 1 matches that pattern. Match the
model instead: `pkill -f "GLM-5.3-Flash"`, then re-check `ps -p 1 -o args=`.

### 2. `$HOME` is a quota-full NFS mount, and the JIT caches default to it

This is the highest-value trap in the list, because the failure arrives **~8 minutes into startup** and
its traceback mentions neither quota, nor home, nor NFS:

```
RuntimeError: Worker failed with error '[Errno 122] Disk quota exceeded'
  raised from triton/runtime/cache.py:120
```

**`df` does not show the problem.** Re-verified while writing this README — `df` reports **609 GB
available** on a mount whose per-user quota is exhausted:

```
$ df -h /usr2/tanngo          →  8.4T  7.8T  609G  93%   ← looks fine
$ mkdir -p /usr2/tanngo/.probe →  mkdir: cannot create directory: Disk quota exceeded
$ [ -w /usr2/tanngo ] && echo PASS → PASS      ← the writability test LIES
```

So `-w` passes on a directory you cannot write to. **Only a real write of real bytes is a valid probe:**

```bash
dd if=/dev/zero of=$HOME/.probe bs=1M count=20   # the only check that tells the truth
```

Redirect every JIT/compile cache to node-local `/tmp`, **keyed by CUDA major version** — these dirs are
also shared across containers on a network mount, and a cubin compiled under one CUDA runtime can
segfault when loaded under another:

```bash
TAG=$(id -u)_${CUDA_VERSION%%.*}
export TRITON_CACHE_DIR=/tmp/triton_cache_$TAG
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_$TAG   # assign UNCONDITIONALLY; see below
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$TAG
export XDG_CACHE_HOME=/tmp/xdg_cache_$TAG
export TILELANG_CACHE_DIR=/tmp/tilelang_cache_$TAG
export TRTLLM_DG_CACHE_DIR=/tmp/trtllm_dg_cache_$TAG           # resolved in C++, not Python
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_ws_$TAG
```

Two sharp edges here. **`TORCHINDUCTOR_CACHE_DIR` must be assigned unconditionally** (not `${X:-...}`)
if an inherited profile already points it at `$HOME` — last assignment wins. And **two of these are
resolved by C++/other libraries, not Python**, so they are invisible to a grep of your own code:
`TRTLLM_DG_CACHE_DIR` (DeepGEMM, `compiler.cuh`) and `FLASHINFER_WORKSPACE_BASE`.

### 3. A `CUDA error` that is not a CUDA error

Same root cause as #2, reached through a different door, and it is the most instructive bug in the repo:

```
CUDA error: invalid argument      # at ~69-78% of cudagraph capture, no traceback
```

DeepGEMM JITs an FP8 GEMM into `$HOME/.tensorrt_llm`; the quota write fails; the GEMM then launches an
**unbuilt kernel** and CUDA reports `invalid argument` downstream. The real error is **one frame up**
(`tvm.error.InternalError: cannot create directories`).

**Three plausible fixes were tried and all failed** — `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, capping the
capture ladder, and `--enforce-eager`. **Eager failing was the tell:** if disabling graph capture does
not help, the bug was never in graph capture. A failed disproof is information.

### 4. Image tags differing only by CUDA version are **not** interchangeable

`vllm/vllm-openai:glm53-flash` (**cu130**) and `...:glm53-flash-cu129` carry the same vLLM build, but on
cu129 the TileLang `mhc_post_tilelang` kernel **segfaults in `cuModuleLoadData`** and the model cannot
complete a forward pass. The image's own `/vllm-workspace/torch_lib_versions.txt` declares
`torch==2.13.0+cu130` — **read that file rather than trusting the tag.**

⚠️ Careful with the verification grep: `grep -c "Segfault\|mhc_post_tilelang"` is an **OR** that also
matches *successful* compile lines, so it overcounts failures.

### 5. The model you need may not be registered in the image you have

Different vendor images register different architectures. `Glm5Next*` is **absent** from
`vllm/vllm-openai:qwen38-flash-next` — so the same `vllm serve` command works in one image and dies with
"architecture not supported" in another. **Check before scheduling a multi-hour job:**

```bash
/usr/bin/python3 -c "from vllm import ModelRegistry; \
  print([a for a in ModelRegistry.get_supported_archs() if 'Glm5' in a])"
```

### 6. A conda env on `PATH` can shadow the image's own vLLM

If the shell profile activates a conda env, a bare `vllm` may resolve to that env's older build **inside
the image**, which then cannot parse the new model's config. This bites the **client** as well as the
server: `vllm bench serve` reads the model config to tokenize, so a sweep fails in preflight *even when
the server is perfectly healthy*. Use absolute paths on both sides:

```bash
VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 ./bench.sh batch
```

(In this session's non-interactive shell `which vllm` correctly resolved to `/usr/local/bin/vllm` — so
this trap is **profile-dependent**. Check rather than assume, in both directions.)

### 7. A fresh container means cold JIT caches — which silently **mis-size the KV pool**

Not a crash; a **wrong number**, which is worse. vLLM sizes the KV pool as
`util × total − weights − peak_activation − cudagraph`, and it *measures* `peak_activation` at startup.
If that measurement overlaps a cold `torch.compile`, the transient compile allocation is charged to
activation and the KV pool is reserved too small. Measured on two models:

| | cold compile | warm cache | effect |
|---|--:|--:|---|
| Qwen3.8 peak activation | **17.07 GiB** | **0.99 GiB** | KV pool 2.05 M → **3.20 M tokens**; **c=4 understated 1.58×** |
| GLM-5.3 peak activation | 4.88 GiB | 4.05 GiB | KV pool 1.84 M → 1.92 M tokens |

**The tell is an inversion: *lowering* `--gpu-memory-utilization` gave *more* KV.** Throughput at
mid-concurrency (c=4, c=16) is KV-pressure-sensitive, so a pure capacity artifact is easily mistaken for
an architecture effect — it briefly made Qwen look slower than GLM at c=4, and an architectural
explanation had already been written for it.

**Rules:** warm the compile cache with one throwaway run before any arm that sizes KV (or pin
`--kv-cache-memory`); read the startup line *"Replace gpu_memory_utilization config with
`--kv-cache-memory=…`"* and compare it to the pool actually used; and never attribute a mid-concurrency
delta to architecture without diffing `peak activation` between both arms.

### 8. `/health 200` is **not** benchmark-ready

The server answers health checks while DeepGEMM is still JIT-compiling (measured: **967 kernels** for
GLM; 1,261 and a 538.8 s engine init on another build). Points collected in that window are **cold,
complete, and still wrong by 25–62%** — enough to invert a conclusion.

**Signature: p99 TTFT ≫ median TTFT at low concurrency.** It happened **four times** in this repo. Two
examples from this session, both caught by the guard in `bench.sh` and both re-run warm:

| point | first run | rerun | p99/median TTFT |
|---|--:|--:|--:|
| util-0.85 c=4 | 98.8 tok/s | **223.9** | 7.47× (limit 4×) |
| MTP-context ISL 16K | 143.5 tok/s | **314.8** | 5.13× (vs 1.61× in the base arm) |

The second would have published *"MTP halves throughput at short context"* — the **opposite** of the
true result, and it would have inverted that arm's whole trend. Both first runs are quarantined under
`results/_*_queued_firstrun/` rather than deleted.

### 9. Flags that look equivalent but silently destroy config

Use `--max-cudagraph-capture-size 256`, **never**
`--compilation-config '{"max_cudagraph_capture_size":256}'`. The latter **replaces** the entire
`CompilationConfig`, which silently wiped `pass_config` to `{}` — losing `fuse_norm_quant`,
`fuse_act_quant`, and `fuse_allreduce_rms`. The dedicated flag is *merged*. This was caught only by
diffing the startup banners of two runs.

### 10. Benchmark harnesses can lie in three different ways

Not Docker-specific, but these ride along with every containerized sweep:

- **`vllm bench serve` exits 0 when every request failed.** It warns "All requests failed", prints 0.00
  for every metric, and returns success. A 262,144-token point 400'd on all 16 requests and produced a
  summary row of zeros that looks like a real, terrible datapoint. **Validate outputs, not exit codes**
  (`completed > 0`).
- **`--num-warmups` self-poisons the prefix cache.** The warmup is drawn from the *same seeded prompt
  set* as the measured run, so the run re-reads its own warmup's blocks. Measured: exactly 16,000 new
  hits at every concurrency — **12.2% of the c=1 point but 0.8% of c=64**, so it inflates the
  low-concurrency anchors and **flattens measured concurrency scaling**.
- **A fixed seed plus prefix caching measures your cache, not your model** — a phantom 640 tok/s at c=4
  that collapsed to 214 on a fresh seed. Use a unique seed per point and assert 0 new cache hits.

### 11. A flag you set is not a measurement you made

`--enable-mfu-metrics` was passed on **every arm of all three models for two sessions** and produced
**zero data**, because nothing ever read the gauges it populates. No error, no warning — a missing JSON
field looks exactly like a field you never requested. The report's headline
*"decode runs at ~1% of the memory roofline"* stayed analytical the whole time, and when finally
measured it was **wrong by an order of magnitude** (9.5–18.1%).

Three things to know before citing these counters:

1. They are Prometheus **Counters** (monotonic totals since server start), so only
   **(after − before) ÷ duration** is a rate. A single scrape is meaningless.
2. They are an **engine-side estimate** — config shapes × the step's real batch composition — **not a
   hardware counter.** Label **[E]**.
3. ⚠️ **Coverage is silently partial and differs per model**, so the raw numbers are **not comparable
   across models.** vLLM discards components it cannot build at `debug` level:

   | model | components instantiated | missing |
   |---|---|---|
   | Qwen3.8-Flash-Next | `attn`, `ffn`, `unembed` | — |
   | GLM-5.3-Flash | `ffn`, `unembed` | **`attn`** |
   | DeepSeek-V4-Flash | `unembed` only | **`attn`, `ffn`** |

**Why the attention components drop out is itself evidence for this report's thesis.** Both attention
estimators gate on one whole-model boolean: `AttentionMetrics` raises if `is_deepseek_mla` is true,
`MLAAttentionMetrics` raises if it is false — so a **hybrid stack satisfies neither and loses attention
accounting entirely.** vLLM's source says so directly:

> `# TODO: discern cases where we have mixture of different attention layer types such as SWA, MLA, etc.`

The engine already models per-layer heterogeneity in its **allocator** — there is an explicit guard
preventing that same flag from collapsing across layers, because doing so would make `use_mla` true
model-wide and return 1 KV head for every layer — but **not in its performance accounting.** That
asymmetry, heterogeneity handled for correctness and ignored for cost, is precisely the gap this report
argues about, found in the engine rather than in a config file.

### Pre-flight checklist

```bash
hostname; nvidia-smi -L || echo "NO GPU"        # never assume the hardware
ps -p 1 -o args=                                # want: sleep infinity, NOT vllm serve
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # want: all near 0
dd if=/dev/zero of=$HOME/.probe bs=1M count=20 && rm $HOME/.probe  # real write, not -w
cat /vllm-workspace/torch_lib_versions.txt      # the image's declared CUDA, not the tag
/usr/bin/python3 -c "import vllm; print(vllm.__version__)"
/usr/bin/python3 -c "from vllm import ModelRegistry; print(len(ModelRegistry.get_supported_archs()))"
cat /sys/fs/cgroup/cpu.max                      # thread budget — NOT nproc
```

---

## What is *not* measured — stated rather than hidden

A report that cannot say how its numbers could be wrong should be discounted.

- **No dense control anywhere.** Qwen3.8 was the designated baseline and turned out to be a 512-expert
  MoE, so *"MoE decode is memory-bound"* is **unfalsified against a dense model on this harness** —
  though the 9–18% roofline result suggests the premise is wrong for all three MoEs anyway. **Largest
  single gap.**
- **No TP/EP sweep**, so "EP all-to-all and launch overhead are the real bound" is **[H]** — and it is
  the report's central mechanistic claim. ⚠️ Closed for GLM (TP4 OOMs at 75.36 GiB/GPU; TP=7 fails
  `64 % 7`), so it must run on V4 or Qwen, which fit in 3 GPUs.
- **Achieved bandwidth is [E] for GLM, still [A] for V4 and Qwen**, and not comparable across models
  until the coverage gap above is closed.
- **MTP context axis measured for GLM only** — and it *inverted* the batch-axis conclusion, so the
  missing V4/Qwen context arms are load-bearing, not cosmetic.
- **Kimi-K3 and DeepSeek-V4-Pro were not measured** — 1,453.7 GiB (~21 H100s) and ~740 GiB (~11) exceed
  one node. Weight offloading was deliberately not used: the numbers would be PCIe-dominated and
  architecturally meaningless.
- **Two clean impossibilities, recorded as results:** GLM's `fp8_ds_mla` KV is unreachable on *any* GPU
  (it needs `pe_dim == 64`; GLM is NoPE), and PD-disaggregation needs two weight copies = 612 of 640 GiB.
- **TP=8 handicaps the models unequally.** V4 and Qwen fit in 3 GPUs but were measured at TP=8, so their
  per-GPU numbers are **pessimistic**; GLM genuinely needs ~5.
- **Tokenizers differ** (Qwen 248,320 · GLM 154,880 · V4 129,536), so cross-family tok/s is not strictly
  commensurable. Prefer bytes/s for strict claims.
- **Synthetic random prompts route ~uniformly across experts** — best case for load balance. Real text
  has correlated routing, so measured expert cost may understate imbalance, most of all for Qwen.

---

## Hardware and provenance

All numbers: **8× NVIDIA H100 80GB HBM3**, driver 575.57.08, TP8 + expert parallel, `vllm bench serve`,
`--ignore-eos`, unique seed per point, spec decode **off** on all headline numbers, vision towers
disabled for text-to-text fairness. Roofline constants: **3.35 TB/s** HBM3 and **1,979 TFLOP/s** dense
FP8 per GPU.

Every result directory carries a `manifest.txt` with the hostname, all GPUs, driver, torch/vLLM
versions, the server command line **scraped from `ps`**, the quantization resolved from the **live
server**, and the MFU component coverage — so a number six weeks old is still citable.

⚠️ **One arm selection worth knowing:** Qwen's c=4 and c=16 headline points come from
`results/base-util082/`, **not** `results/base/`, because the original arm's KV pool was mis-sized by a
cold `torch.compile` (trap 7). `normalize.sh` will happily print both; the per-model README says which
is citable.
