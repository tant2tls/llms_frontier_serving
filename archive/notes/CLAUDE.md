> ARCHIVED SESSION RECORD — not current instructions or conclusions. Original location: `CLAUDE.md`. See [current report](../../report.md), [debugging guide](../../fix_bug.md), and [session handoff](../../WORKFLOW.md). Historical claims may be superseded.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> ## 👉 START WITH [`WORKFLOW.md`](../../archive/notes/WORKFLOW.md)
> One page: current state, the next task, the 6 rules that keep numbers publishable, and the
> do-not-re-explore list. **This file (CLAUDE.md) is the full reference** — read WORKFLOW.md
> first and come here for detail.
>
> **State 2026-09-02 (session 3):** **115 measured points, all three models DONE, and GLM's 4 owed
> measurements are COMPLETE.** GLM-5.3-Flash (**46**) · DeepSeek-V4-Flash (50) ·
> Qwen3.8-Flash-Next-FP8 (19). All **engine-bridged** — the `dev20051`↔`dev20073` build effect was
> measured at **0.997×**, so cross-model ratios are architecture, not tooling. `report.md` (root) +
> all three per-model reports are written and updated.
>
> **Qwen wins raw/per-GPU/$-per-token and per-B-active (2.6–2.8×); V4 is the only model with KV
> headroom at 260K; GLM has the best long-context TTFT.** The metrics disagree — that tension is
> the result.
>
> 🛑 **CORRECTION THAT AFFECTS A HEADLINE: the "~1% of memory roofline" claim was too low by an
> order of magnitude.** `--enable-mfu-metrics` was passed on every arm of all three models but
> **nothing ever read its counters**. `bench.sh` now scrapes them: GLM measures **9.5–18.1% of
> 3.35 TB/s**. The conclusion survives (decode is latency-bound, and bandwidth **peaks at c≈16–32 then FALLS
> while throughput rises to c=64**), but **never quote ~1% again.** Label these **[E]** — engine-side
> estimate, and for GLM they **exclude attention entirely** (only `ffn`+`unembed` ComponentMetrics
> instantiated), so they are a lower bound and **not comparable across models**. `fix_bug.md` bug 13.
>
> **Next session: the open work is cross-model, not GLM** — re-poll V4/Qwen with the fixed MFU
> harness, MTP context axis on V4/Qwen (GLM's context arm **inverted** the batch-axis conclusion),
> then the TP/EP sweep and a dense control. See WORKFLOW.md §2.

## What this directory is

A **research report deliverable**, not a software product. It answers [`task.md`](../../task.md) — a task
assigned to Tan Ngo by **Kan Zhu** (PhD student, UW **SyFI** lab; co-advised by Baris Kasikci and
Arvind Krishnamurthy) as part of a PhD-application evaluation. The audience is a serving-systems PI
and their students. The output artifact is `report.md`, backed by reproducible code in `bench/`.

[`plan.md`](../../archive/notes/plan.md) is the authoritative plan — **read it before doing anything substantive.** It
contains the thesis, the derivation, the phase order, and the settled tooling decisions.

Related context lives outside this repo: `../PhD/` (application strategy, target labs, the H1–H4
hooks), `../PhD/ray_learning/CLAUDE.md` (the measurement-honesty conventions this repo inherits),
`../PhD/resource_note.md` (container CPU-limit forensics), `../arp/nano-vllm/` (installed `-e`).

## The thesis (Revision 2 — don't dilute it)

The target is the **2026** generation (DeepSeek-V4-Flash/Pro, Qwen3.8, Kimi-K3, GLM-4.7/5.3-Flash),
not the 2025 one. M\* (Kasikci & Wang, June 2026 — https://m-star.org/) argues serving systems break
because they assume inference is *"a single autoregressive loop"* modeled as a *"flat DAG"*, and
answers with the **Walk Graph** (nodes/edges/Walks) at the *inter-component* level.

> **M\*'s premise now holds one level lower than M\* addresses it. 2026 models are internally
> heterogeneous *within one transformer stack*: a Kimi-K3 decode step touches 69 KDA layers with O(1)
> state and 24 MLA layers with O(ctx) KV; DeepSeek-V4-Flash reads FP4 experts and FP8 attention
> weights under per-layer KV compression ratios differing 32×. The uniform per-layer cost model every
> scheduler and paged-KV allocator assumes — one block size, one bytes-per-token, one bottleneck per
> step — is no longer true of a single model. Heterogeneity moved inside the layer stack: a scheduling
> and memory-management problem, not a kernel problem.**

Supporting result (Revision 1's thesis, now confirmed as a *trend*): `B*_MoE = B*_dense · E/k` from
`D(B) ≈ min(Bk, E)` distinct experts per step, `B*_dense = Compute/(2·MemBW)·bytes_per_param` = 148
tok/step (H100 FP8). E/k rose to 42–64×, so **B\*_MoE is now 8K–19K tok/step — the expert read binds
at any reachable batch.** FP4 experts are the vendors' response.

> ⚠️ **Correction to the FP4 arithmetic (verified 2026-09-01).** FP4 **halves HBM→SMEM bytes**, so the
> *memory* term genuinely improves and `B*_MoE ≈ 12,637` for V4-Flash is right on the memory side. But
> **compute stays FP8** (weights unpack FP4→FP8 before the GEMM), so the **roofline ridge point does
> not move** — do *not* claim FP4 doubles `B*_dense` to 296 as a compute effect. Byte traffic halves;
> FLOP/byte at the tensor core is unchanged.

Every table and figure exists to test or support this. **If a proposed addition doesn't answer a
`task.md` question or serve this thesis, it belongs in the appendix or nowhere.**

## Current priority: measurement has STARTED (updated 2026-09-01, later session)

**Superseded:** the earlier "Phase 1 = architecture report, GPU-free, delivered first" instruction. That
was true when the session had no GPU. **DeepSeek-V4-Flash has now been served and swept on 8×H100** —
see the evaluation-process section below and `deepseek_v4_flash/report.md`.

Still in force from the original priority:
- **No speculative decoding in headline numbers**; base models only (`<model>/run_nomtp.sh`). Spec decode is
  reported *only* as an explicit A/B — and it is now **measured**, not just analyzed.
- Evaluation stays coarse: throughput/TTFT/TPOT + bandwidth fraction, **never accuracy**.

**Next up (revised 2026-09-02, GLM-5.3 session):** **GLM-5.3-Flash on 8×H100 is the immediate task** —
cached and scripted. The GPU-0 squatter is **fixed**; the remaining blocker is the **image tag**:
resubmit with the **CUDA-13** `vllm/vllm-openai:glm53-flash`, not `-cu129`. Two dtype arms were
explicitly requested (BF16 KV headline + `fp8_ds_mla` attempt), then a **DeepSeek-V4-Flash BF16-KV
rerun** in the same image, then Qwen3.8. Report everything in normalized form (per-GPU, per-B-active),
never raw tok/s. GLM-4.7-Flash (29 GiB, 1 GPU) remains the cheapest unrun cross-architecture point.

## ✅ SOLVED: the image entrypoint squats GPU 0 — fix is in `submit_job.sh` (2026-09-01/02)

**Status: FIXED and PROVEN.** `GLM-5.3-Flash/submit_job.sh` adds `--command -- sleep infinity` to the
RunAI submit; the resubmitted workspace came up with PID 1 idle and **all 8 GPUs free (~81 GiB each,
zero compute apps)**. Keep that flag on every future submit of a `vllm/vllm-openai:*` image. The
current blocker is now the **image tag**, not the GPUs — see the GLM-5.3-Flash section.

Verify after any resubmit:

```bash
ps -p 1 -o args=      # WANT: `sleep infinity`.  BAD: `vllm serve` -> squatter is back
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # WANT: all 8 near 0 MiB
```

**The problem it solves.** `vllm/vllm-openai:*` declares `ENTRYPOINT ["vllm", "serve"]`. With no
`--command` on the submit, PID 1 runs `vllm serve` with **no model argument**, so vLLM falls back to its
default (`entrypoints/cli/serve.py:36` — *"Defaults to Qwen/Qwen3-0.6B if no model is specified"*). PID 1
becomes a live Qwen3-0.6B server that parks **~77 GiB on GPU 0** (measured: 77,146 MiB used, only 3,934
MiB free) and owns **port 8000**. GLM-5.3-Flash is 305.8 GiB at TP=8 and needs all eight GPUs.

**⚠️ DO NOT try to kill it — this was tested and it backfires.** `kill -9 <EngineCore pid>` frees the
memory, but **PID 1 dies with it**; PID 1 is container init, so the container terminated, RunAI
rescheduled (`tan-8gpus-glm53-0-0` → `-0-1`), and the entrypoint came back with a **fresh EngineCore
holding the same 77 GiB**. A respawn loop, not a fix, and it costs a reschedule each time. vLLM's
`/sleep` would release weights without killing the process, but it needs `VLLM_SERVER_DEV_MODE=1` **at
container start** and is absent from `/openapi.json` otherwise — unfixable from inside.

Two gotchas baked into `submit_job.sh`: the workspace **delete is what releases an old squatter** (there
is no way to free it from inside), and any `--nfs`/other flag left *after* `--` gets swallowed into
`sleep`'s argv, so `--command -- sleep infinity` must be **last**.

**Dead ends, all checked — do not re-explore:**

| Attempt | Why it fails |
|---|---|
| TP=7 on the 7 free GPUs | `glm5next/nvidia/model.py:669` asserts `num_attention_heads % tp == 0`; 64 % 7 ≠ 0 |
| TP=4 on GPUs 4–7 | 305.8 GiB / 4 = 76.5 GiB/GPU vs ~65 GiB usable at util 0.82 — does not fit |
| Pipeline parallel | `glm5next` does declare `SupportsPP`, but PP changes the latency structure and makes TTFT/TPOT non-comparable to the V4 baseline — which is the whole point |
| Partial fit on GPU 0 | only 3.9 GiB free there |

**Also note:** `runai` **inside** the container cannot reach the control plane
(`x509: certificate signed by unknown authority`), so submit/delete must happen from a login host.
`--preemptible` is a real risk for these runs: weight load alone is minutes across 62 shards and a full
sweep is much longer; `bench.sh` resumes (it skips existing results) but the server reloads from scratch.

## Non-negotiable conventions

These come from `../PhD/ray_learning/CLAUDE.md` and are what make the report credible to a PI:

1. **Label analytical vs. measured on every table cell and figure.** Most numbers in `plan.md` §1 are
   *predictions* computed from `config.json`. Presenting a modeled number as measured invalidates the
   whole report — the reader can't tell which to trust, so they discount all of them.
2. **Every performance claim needs three things:** the number, the hardware it ran on, and how it was
   measured. No exceptions, including in prose.
3. **The dense control run is mandatory.** Without a dense model on the same harness, "MoE is
   memory-bound" is unfalsifiable — it could be the setup. The control also reproduces NanoFlow Fig. 3.
4. **Record quantization per run.** Comparing an FP8 model to a BF16 one and attributing the delta to
   architecture is the easiest way to publish a wrong result.
5. **Report achieved bandwidth / FLOP fraction**, not just latency. "Memory-bound" means achieved GB/s
   is near peak — **3.35 TB/s on H100, ~2.0 TB/s on A100**; state which. Show the fraction.
6. **Don't invent results.** Note what couldn't run and why (see the weight-size and GPU-arch ceilings
   below).

## Hardware: detect, never assume

Per the global `CLAUDE.md`, this runs in a **RunAI-scheduled container whose spec changes between
sessions**. The planning session (2026-09-01) was on `tan-cpu-0-0` with **no GPU**. Re-detect every time:

```bash
hostname; nvidia-smi -L 2>/dev/null || echo "NO GPU"
nproc   # HOST count — NOT the budget
echo $(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) ))
```

Size thread pools from the **cgroup quota** (198 on the planning node), never `nproc` (256).
GPUs come from RunAI: project `gen-opt-vn`, GPU quota 24. `runai node list` is permission-denied;
`runai workload list` works.

**GPU *type* matters as much as count here** — see the GPU-architecture section below. `nvidia-smi -L`
alone is not enough: record the arch (A100/SM80 vs H100/SM90), because DeepSeek-V4 runs on one and not
the other, and the roofline constants differ (2.0 vs 3.35 TB/s HBM).

**Never cache models under `$HOME`** — `/usr2/tanngo` is a fleet-shared NFS export mounted in every
container, at 93% (621 GiB free). Use `/prj/.../vol22-scratch` (12 TiB free) with **absolute**
`HF_HOME`. See the storage section below for the relative-path trap.

## Evaluation process — MEASURED, and the rules that make it fair (established 2026-09-01)

**Status:** DeepSeek-V4-Flash has been served and swept on 8×H100. Deliverable:
[`deepseek_v4_flash/report.md`](../../deepseek_v4_flash/report.md) — one subfolder per model (see the
deliverable-layout section at the end).
Anticipated-question rehearsal: [`potential_questions.md`](../../archive/notes/potential_questions.md). Target models are
listed in `model_list.md`.

### The harness

| Script | Role |
|---|---|
| `<model>/run.sh` | Server, **MTP on**. `--gpu-memory-utilization 0.82`, keep `--block-size 256` |
| `<model>/run_nomtp.sh` | **Base-model control** — identical minus `--speculative-config`. **Primary numbers come from here.** |
| `sending.sh` *(root)* | 8-section functional smoke test (health/models/chat/stream/raw/tools/batch/metrics) |
| `bench.sh` *(root)* | Sweep harness: `batch`, `context`, `prefix`, `sharegpt`, `quick`; `DRY_RUN=1` to inspect |
| **`normalize.sh`** *(root)* | **Turns result JSONs into FAIR cross-model numbers** — always emits per-GPU / per-B-active / per-B-total together so a flattering one can't be cherry-picked, and prints the caveats normalization can't fix. **Use this, never raw tok/s.** |
| **`REPRODUCE.md`** *(root)* | Step-by-step re-measurement guide for a future session with no memory of this one: prerequisites, the 5 load-bearing env overrides, expected numbers, and a dead-ends table |
| **`fix_bug.md`** *(root)* | **Teaching doc: all 8 bring-up bugs** — error text, real cause, how it was localized, fix, transferable lesson. Read before debugging a new model bring-up |

`bench.sh` writes `manifest.txt` per run directory with hostname, all GPUs, driver, torch/vLLM versions,
the **server command line scraped from `ps`**, and the quantization. It polls
`vllm:kv_cache_usage_perc` during each point and folds the peak into the result JSON. Existing results
are skipped, so interrupted sweeps resume.

### Non-negotiable eval rules (each one exists because it was violated once)

1. **Base model first: no speculative decoding in headline throughput.** Comparing an MTP-accelerated
   model to a base model compares *inference tricks*, not architectures, and flatters whichever vendor
   shipped a draft head. Report spec decode only as an explicit A/B. **Measured:** MTP gives 1.25× at
   c=1 but **0.99× at c=64** — the gain decays to nothing as the batch saturates the machine, and it
   costs 2.4% of KV capacity.
2. **Unique seed per sweep point, and verify `newcachehits == 0`.** With `enable_prefix_caching=True`
   (the default), a fixed seed makes every concurrency point replay the *same prompts*, so later points
   read the earlier points' cache. This produced a **phantom 640 tok/s at c=4** that collapsed to 214
   with a fresh seed. **Benchmarking a prefix-caching engine with a fixed prompt set measures your
   cache, not your model.**
3. **`--ignore-eos` on controlled sweeps.** Without it this reasoning model chooses its own output
   length and OSL stops being an independent variable. Drop it only for realism runs (ShareGPT).
4. **Record dtype, TP/EP layout, and spec-decode state with every number.** See the fair-comparison
   section below.
5. **Report achieved bandwidth as a fraction of peak**, and say whether it is measured or modelled.
   Analytical `active_params × bytes × steps/s` is an *upper bound* — label it **[A]**. Prefer
   `--enable-mfu-metrics`, which populates `vllm:estimated_flops_per_gpu_total`,
   `estimated_read_bytes_per_gpu_total` and `estimated_write_bytes_per_gpu_total` (default off → 0.0).
   ✅ **`bench.sh` now scrapes these** (2026-09-02) as a **delta per point** — they are Prometheus
   **counters**, so a single scrape is meaningless. ⚠️ **Label the result [E], not [M]:** vLLM computes
   them from **config shapes × measured batch composition** (`perf.py:488-513`, `:1069-1127`), not from a
   hardware counter. And **coverage is silently partial and differs per model** — Qwen gets
   `attn`+`ffn`+`unembed`, GLM only `ffn`+`unembed`, V4 only `unembed`, because both attention estimators
   gate on the whole-model `is_deepseek_mla` flag and a **hybrid stack satisfies neither**. So the numbers
   are **lower bounds that are NOT comparable across models**. `fix_bug.md` bug 13.
6. **Label every table cell [M] measured or [A] analytical.** Mixing them silently makes the whole
   report undiscountable.
7. **Don't invent results.** State what didn't run and why.

### Engine settings that confound results — check before interpreting

- 🛑 **`--gpu-memory-utilization` is capacity-only — BUT a cold `torch.compile` corrupts the KV sizing,
  and that DOES move throughput (MEASURED 2026-09-02).** vLLM computes
  `kv_pool = util × total − weights − peak_activation − cudagraph` and **measures `peak_activation` at
  startup**. If that measurement overlaps a cold compile, the transient compile allocation is charged to
  activation. Measured on Qwen3.8, same flags apart from util:

  | | cold compile (util 0.85) | warm cache (util 0.82) |
  |---|--:|--:|
  | peak activation **measured** | **17.07 GiB** | **0.99 GiB** |
  | compilation time | 60.41 s | 1.03 s |
  | KV pool | 2,048,645 tok | **3,197,331 tok** |
  | throughput @c=4 | 162.2 | **256.8 (1.58×)** |

  **17× difference on identical weights and identical `max_num_batched_tokens`**, reserving ~14 GiB/GPU
  of KV it never needed. **Lowering util gave MORE KV** — that inversion is the tell. The flag itself is
  capacity-only as predicted (c=1 0.99×, c=64 1.00×, identical TPOT), but **mid-concurrency (c=4, c=16)
  is KV-pressure-sensitive**, so a capacity artifact is easily mistaken for an architecture effect —
  it briefly made Qwen look *slower* than GLM at c=4, and an architectural explanation had already been
  written for it. **Rules: (1) warm the compile cache before any run that sizes KV, or pin
  `--kv-cache-memory`; (2) read the startup line "Replace gpu_memory_utilization config with
  `--kv-cache-memory=…` to fully utilize gpu memory" — if it differs materially from the pool actually
  used, the pool is mis-sized; (3) never attribute a mid-concurrency delta to architecture without
  checking both arms' `peak activation`.** Full writeup: `fix_bug.md` bug 12 and
  `Qwen3.8-Flash-Next-FP8/results/base-util082/RESULT-util-ab.md`.
  ✅ **REPLICATED on DeepSeek-V4-Flash (2026-09-02), warm cache, and it is CLEAN:** util 0.82 → 0.85 gave
  **+7.0% KV** (1,487,070 → 1,590,723 tok) and **no throughput change** (0.98–1.09×, no trend), with peak
  activation **2.32 GiB in both arms**. So the flag really is capacity-only; it only *looks* like a
  throughput knob when a mis-sized pool creates scheduler pressure. **Audited all 16 GLM+V4 startup logs:
  peak activation 2.03–4.27 GiB, all sane — the 17 GiB artifact was isolated to the one Qwen arm**, so no
  other published number is affected. V4 is less exposed because its 43 uniform DSA layers compile to far
  fewer distinct graphs than Qwen's four-cost-class stack. Writeup:
  `deepseek_v4_flash/results/util085-dev20073/RESULT-util-ab-and-8point-curve.md`.
  ⬜ **Still owed:** the same A/B on **GLM-5.3-Flash** — needs the `glm53-flash` image, so a container
  switch (`Glm5Next*` is NOT registered in the qwen38 image, verified). Low priority now that two models
  agree.

- **Chunked prefill is ON by default with `max_num_batched_tokens=8192`** (`scheduler.py:242`). A 131K
  prompt is split into **16 chunks** that interleave with other requests' decode steps. **TTFT is
  therefore not a pure prefill-compute measurement** — it includes scheduler round-trips and queueing.
  Never claim TTFT measures prefill FLOPs. Sweeping this knob is untested and is plausibly the largest
  server-side lever for long-context work.
- **`--max-num-seqs` defaults to 128**, so any concurrency > 128 silently queues. The `B*_MoE ≈ 6,300
  tok/step` prediction may be **untestable** on one node, not merely untested.
- **EP shards experts, TP shards the dense path.** Measured: `Local/global number of experts: 32/256` at
  EP8. With k=6 across 8 ranks, the expected experts touched **per rank per token is 0.75** — most ranks
  contribute 0 or 1, so load imbalance is structural and **all-to-all sits on the critical path of every
  decode step**. This is the leading hypothesis for why achieved HBM stays far below peak (**GLM measured
  at 9.5–18.1% of 3.35 TB/s [E]**, and it **peaks at c≈16–32 then FALLS while throughput rises to c=64**), and it
  is a *network* term invisible to bandwidth accounting.
- **TP=8 on a model that fits in 3 GPUs is a confound.** It inflates per-GPU cost and adds collectives
  a smaller TP wouldn't. Per-GPU numbers taken at TP=8 are **pessimistic**; a TP sweep is unrun.

### Fair comparison across models — the rules

**Raw tok/s is never a valid cross-model comparison.** Minimum deployable footprint spans **21×**
(GLM-4.7-Flash 1 GPU → Kimi-K3 ~21 GPUs), so raw throughput mostly reports how much silicon was used.
Always publish all three normalizations:

| Metric | Answers | Caveat |
|---|---|---|
| **tok/s per GPU** | what an operator pays; **the only one that maps to cost** | depends on TP/EP choice |
| **tok/s per B-active** | is the architecture's FLOP budget used well | ignores the HBM you had to buy |
| **tok/s per B-total** | efficiency per byte of HBM purchased | penalizes sparsity by design |

Also required for fairness:

- **Match context length and concurrency** across models; ISL dominates everything (23× cost spread).
- **Match quantization, or state the mismatch.** An FP8+FP4 model vs a BF16 model is not an
  architecture comparison.
- **Same spec-decode state** (prefer off), same engine, same version.
- **Tokenizers differ**, so tok/s is not commensurable across families — a denser tokenizer does more
  work per token. For cross-family claims use **tokens-per-fixed-corpus or bytes/s**. TraceLab's own
  counts come from Claude/GPT tokenizers, so they anchor grid *design*, not precise claims.
- **Synthetic random prompts route ~uniformly across experts** — best case for expert coverage, so
  measured expert-read cost may exceed real text. Cross-check with trained-router hit counts.

### Measured baseline: DeepSeek-V4-Flash, 8×H100, base model (MTP off)

Anchor for future runs. ISL 16,384 · OSL 256 · unique seeds · `deepseek_v4_flash/results/mtp-off/`:

| conc | tok/s | per B-active | per GPU | TTFT p50 | TPOT p50 |
|--:|--:|--:|--:|--:|--:|
| 1 | 92 | 6.8 | 11.5 | 714 ms | 8 ms |
| 64 | 383 | 28.4 | 47.9 | 3,687 ms | 151 ms |

64× concurrency → **4.2×** throughput, TPOT **19×** worse. Context (c=8): 101 → 42 tok/s from 16K → 131K,
**TTFT 26.0 s at 131K**. Peak KV **30.2%** even at 256K×8 — **KV capacity is not the constraint for this
model**. Prefix sharing (64K shared prefix): **3.7× throughput, 4.6× lower TTFT**, the single largest
lever. Cost spread **23×**, driven by context and sharing, *not* batch size.

### Measured: GLM-5.3-Flash, 8×H100, BF16 KV, base model (MTP off) — 2026-09-02

`GLM-5.3-Flash/results/bf16kv/` · image dev build **cu130** · TP8/EP · `--max-num-seqs 256` ·
unique seed per point · **every point cold (0 new prefix-cache hits)** · normalized with
[`normalize.sh`](../../normalize.sh).

**Param counts are MEASURED [M] from safetensors tensor shapes: 321.34 B total / 17.38 B active.**
Mutually-exclusive buckets that partition the total exactly: routed experts (main) **304.42 B**
(only k/E = 8/288 fire per token) + always-on attn/dense/emb **8.92 B** + MTP layer **7.43 B**
(excluded — base model) + vision tower **0.56 B** (excluded — `--limit-mm-per-prompt 0`).
active = 8.92 + 304.42·8/288 = **17.38 B**.

> ⚠️ **CORRECTED 2026-09-02.** An earlier *analytical* estimate of "310.96 B / 15.01 B" was wrong —
> it under-counted always-on attention (GLM keeps q/k/v/o in BF16 and adds a kpool indexer) and
> mishandled the MTP layer. It made GLM look **better** per-B-active than it is. The 18 B active /
> 320 B total figures on the model card are the correct ones. **Never hand-derive param counts from
> `config.json`; read tensor shapes.**

**Batch** (ISL 16,384 · OSL 256) — all **[M]**:

| conc | tok/s | per B-active | per GPU | TTFT p50 | TPOT p50 | KV peak |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 5.54 | 12.04 | 857 ms | 7.1 ms | 0.014 |
| 4 | 225.3 | 12.96 | 28.16 | 1,742 ms | 10.9 ms | 0.051 |
| 16 | 359.2 | 20.67 | 44.90 | 2,685 ms | 33.6 ms | 0.201 |
| 64 | 447.1 | 25.72 | 55.88 | 2,513 ms | 129.5 ms | 0.809 |

**Context** (conc 8 · OSL 256): 16K **295.9** → 64K **104.3** → 131K **47.2** → 260K **26.5** tok/s.
TTFT 2.6 s → 38.5 s. **Peak KV 89.5% at 260K×8 — unlike V4-Flash (30.2%), KV capacity DOES become
binding for GLM at max context**, because BF16 KV costs ~11.35 KiB/tok where V4 pays fp8.
**Prefix** (64K shared): n=1 265.3 · n=4 **365.9** · n=16 199.4 tok/s.

⚠️ Top context point is **ISL 260,000, not 262,144**. `max_model_len - OSL` (261,888) still 400s —
`--dataset-name random` does not emit exactly `--random-input-len` tokens. Bisected; 99.2% of
max_model_len. See `fix_bug.md` bug 7.

#### GLM-5.3 vs DeepSeek-V4-Flash — MATCHED grid (ISL 16,384 · OSL 256 · TP8 · base, both cold)

| conc | GLM t/s | V4 t/s | GLM/V4 | GLM per B-act [M] | V4 per B-act [A] |
|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 92.3 | 1.04× | 5.54 | 6.84 |
| 4 | 225.3 | 222.0 | 1.01× | 12.96 | 16.46 |
| 16 | 359.2 | 293.7 | **1.22×** | 20.67 | 21.77 |
| 64 | 447.1 | 383.4 | **1.17×** | 25.72 | 28.42 |

Concurrency scaling 1→64: GLM **4.64×** vs V4 **4.15×**.

✅ **SUPERSEDED by the engine-matched rerun — use the table below instead.** The numbers above are
cross-engine (V4 on conda 0.28.0/cu129). Kept only for the record.

#### ★ ENGINE-MATCHED HEADLINE (2026-09-02) — the number to put in the report

V4-Flash rerun **in the GLM image**: same engine `0.1.dev20051+g487ecf187`, CUDA 13.0/torch
2.13.0+cu130, same node, TP8/EP, `--max-num-seqs 256`, same `bench.sh` grid, spec decode off.
`deepseek_v4_flash/results/mtp-off-image/` — 11 points, all cold.

| conc | GLM t/s | V4 t/s | GLM/V4 | GLM /GPU | V4 /GPU | GLM /B-act [M] | V4 /B-act [M] |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 85.0 | **1.13×** | 12.04 | 10.63 | 5.54 | 6.04 |
| 4 | 225.3 | 219.8 | 1.02× | 28.16 | 27.48 | 12.96 | 15.61 |
| 16 | 359.2 | 330.9 | **1.09×** | 44.90 | 41.36 | 20.67 | 23.50 |
| 64 | 447.1 | 389.4 | **1.15×** | 55.89 | 48.67 | 25.72 | 27.65 |

Scaling 1→64: GLM **4.64×**, V4 **4.58×**. **Engine effect measured at mean 1.01×**
(0.92/0.99/1.13/1.02 per conc) — essentially neutral, so the earlier cross-engine ratios were
approximately right, but we could not have known that without running it.

**BOTH param counts are now MEASURED [M]**, so per-B-active is finally a fair column: GLM
321.34 B/17.38 B, V4 **290.91 B/14.08 B**. ⚠️ V4 needs care — its routed experts are `I8` tensors
packing **two MXFP4 values per byte**, so naive shape-summing yields 158.07 B/11.01 B, **wrong by
~2× on the expert term**. Correct: 141.734 B I8 elements ×2 = 283.468 B logical + 7.438 B
non-expert = 290.91 B; active = 7.438 + 283.468·6/256 = 14.08 B (card: 284 B/13 B ✓).

**The honest reading:** GLM wins on raw and per-GPU throughput (1.02–1.15×) but **loses on
per-B-active at every concurrency** — it needs *more* active params for that throughput, so V4 uses
its FLOP budget better. Which is "better" depends on whether you pay for GPUs or for parameters.

⚠️ **A guard gap this rerun exposed.** Three V4 points passed cold+completed yet under-reported by
**25–60%** from scheduler queueing; using them the engine effect computed to **0.64×** ("image is 36%
slower") — a wrong headline. Diagnostic: **p99 TTFT ≫ median**. A concurrency-aware guard is now in
`bench.sh`. Evidence: `deepseek_v4_flash/results/_v4image_firstrun_queued/`.

**⚠️ THROUGHPUT IS NOT LINEAR IN ACTIVE PARAMS — that deviation is a finding, not noise.** If decode
were purely memory-bound, `tok/s × B-active` would be ~constant across models on identical hardware.
Measured spread: **+30.8% at c=4 → +57.6% at c=16** (GLM higher). GLM carries **29% more active
params** (17.38 B [M] vs 13.49 B [A]) yet is **faster** at high concurrency — so it extracts
substantially more throughput per active parameter-byte than a roofline model predicts. Consistent
with (a) 34 of its 45 layers being recurrent KDA that read a **fixed-size state** instead of O(ctx)
KV, and (b) achieved HBM far below peak (GLM **9.5–18.1% [E]**, corrected 2026-09-02 from an earlier
`<5%` estimate) — neither model sits at the memory roofline, so active-param
accounting is the wrong cost model. EP all-to-all on the decode critical path remains the leading
hypothesis for the real bound.

**⚠️ "Both models are FP8" is TOO COARSE for the expert-read term this report is about.**
GLM-5.3-Flash ships **native FP8 weights**: **314.40 of 321.34 B (97.8%) are `F8_E4M3` on disk**,
`quant_method: fp8`, `fmt: e4m3`, `weight_block_size [128,128]`, `activation_scheme: dynamic`, with a
**1,509-entry `modules_to_not_convert`** list. The **6.93 B BF16 remainder** is q/k/v/o_proj (1.14 B
each), embeddings + lm_head (0.63 B each), `kv_b_proj`, the kpool indexer, the MoE gate, and the
vision tower. **DeepSeek-V4-Flash uses MXFP4 experts**, so per expert-parameter GLM reads ~1 byte and
V4 ~0.5 — a 2× difference in exactly the term the thesis is about. Report the **bytes**, not the label.

⚠️ **Caveats that normalization cannot fix, and must be stated with these numbers:** engine mismatch
(V4 on conda 0.28.0/cu129 vs GLM on image dev/cu130 — a **real fairness gap, rerun owed**); forced-
opposite KV dtypes; V4 fits in 3 GPUs but was measured at TP=8, so its per-GPU number is pessimistic
while GLM genuinely needs ~5; differing tokenizers (GLM vocab 154,880); and V4 ran at the H100
default `max_num_seqs=1024` vs GLM's pinned 256 (no published point affected — both grids top out at
conc 64).

### ⚠️ CORRECTED: FP8 KV **DOES** work on H100 — via a different backend (2026-09-02)

**Supersedes the "impossible by architecture" claim below.** That claim was based on the wrong
route. Read the vendor recipe (https://recipes.vllm.ai/zai-org/GLM-5.3-Flash?hardware=h100) — it says
*"Hopper does not support FP8 KV cache for this model and must run BF16 KV."* **That statement is
true of the shipped image, not of Hopper or of the model.**

`--kv-cache-dtype fp8_ds_mla` fails (`pe_dim must be 64`) because that is DeepSeek-V3.2's layout and
GLM is NoPE. But vLLM has a **second FP8-KV path built for NoPE sparse models**:
`FLASHINFER_MLA_SPARSE_SM90`. Its requirements (`flashinfer_mla_sparse_sm90.py:150-158`) are
`kv_lora_rank==512` (GLM: 512 ✅), `qk_rope_head_dim in (0,64)` (GLM: **0** ✅ — NoPE *explicitly*
allowed), `hasattr(index_topk)` (GLM: 2048 ✅). `cuda.py:150-157` even **prefers** it when
`qk_rope_head_dim==0`. The only gate is a FlashInfer feature probe for the `ckv_scale_arr` kwarg,
added in **≥ 0.6.18**; the image ships **0.6.17**.

**MEASURED with FlashInfer 0.6.18 overlaid ([M], `results/fp8kv-fi618/`):**

| | BF16 KV (0.6.17) | FP8 KV (0.6.18) | Δ |
|---|--:|--:|--:|
| KV tokens | 2,099,654 | **3,790,580** | **1.805×** |
| concurrency @256K | 8.01× | **14.46×** | **1.805×** |
| out tok/s @c=1 | 96.3 | 68.9 | **0.72×** |
| out tok/s @c=16 | 359.2 | 208.8 | **0.58×** |
| out tok/s @c=64 | 447.1 | 262.0 | **0.59×** |
| correctness | ✓ | ✓ (`2+2=4`, coherent) | |

**FP8 KV TRADES ~30-42% THROUGHPUT FOR 1.81× KV CAPACITY here.** It is *not* a free win, which is
itself the finding — the recipe's advice to avoid it on Hopper is directionally right for throughput
even though the stated reason (unsupported) is wrong.

**Why 1.805× and not 2.0× [A]:** only the **11 DSA layers'** latent KV is quantized; the **34 KDA
layers' recurrent state stays BF16** (`mamba_cache_dtype=auto`) and is allocated **one block per
sequence regardless of dtype** — a floor FP8 cannot shrink. Per-token KV arithmetic predicts
**1.889×**; measured **1.805×**, residual = the new backend's workspace (23.26 → 22.05 GiB).

⚠️ **CONFOUND, and the control that fixes it.** The FP8 arm changed the KV dtype **and** the
attention backend (FLASH_ATTN_MLA_SPARSE → FLASHINFER_MLA_SPARSE_SM90) **and** forced
`moe_backend=deep_gemm`. So the 30-42% cannot be attributed to the dtype from those two dirs alone.
`run_bf16kv_fi618.sh` is the control: **same overlay, same MoE backend, BF16 KV** → verified it
selects the *same* FLASHINFER_MLA_SPARSE_SM90 backend at 1,990,167 KV tokens. Then
`bf16kv-fi618 vs fp8kv-fi618` = **clean dtype A/B**, and `bf16kv vs bf16kv-fi618` = the
**backend/version effect**. Report the decomposition, not the raw cross-version delta.

⚠️ **Not a production config.** FlashInfer 0.6.18's Python runs against the image's pinned **0.6.17**
compiled artifacts (this mirror has no `flashinfer-cubin` > 0.6.13 and cannot reach
`flashinfer-jit-cache`), so `FLASHINFER_DISABLE_VERSION_CHECK=1` is required, and
`moe_backend=deep_gemm` is required to dodge an ABI mismatch in FlashInfer's fused-MoE
(`init(): Expected 8 but got 9 arguments`, `fused_moe/core.py:693`).

### ❌ TP4 and TP4+TP4 PD-disaggregation: IMPOSSIBLE on 8×H100 (2026-09-02, measured)

The recipe's single-node example is **TP4**, and its PD-disaggregation example is **TP4 prefill +
TP4 decode** bridged by NIXL. **Neither can run on this node**, and the recipe itself says why:
*"306 GiB ... alone exceeds 4×H100-80GB."*

**MEASURED [M]** — TP4, `--gpu-memory-utilization 0.95`, `--max-model-len 32768`:
```
Model loading took 75.36 GiB          (= 305.78 / 4)
torch.OutOfMemoryError: GPU 0 has 79.18 GiB total, 1.60 GiB free
```
Weights alone leave **1.6 GiB** — nothing for KV, activations, or cudagraphs.

**PD disaggregation needs TWO full weight copies** (one per pool) = **612 GiB** against **640 GiB**
of node HBM, before any KV. The recipe's PD example targets a **GB200 tray**, not an H100 node.

| layout | GPUs | GiB/GPU | fits @0.95 |
|---|--:|--:|:--|
| TP8 (this report's baseline) | 8 | 38.2 | ✅ [M] |
| TP4 single pool | 4 | 76.4 | ❌ [M] OOM |
| PD disagg TP4+TP4 | 4+4 | 76.4 | ❌ [A] |

**Say this to a PI as a memory-hierarchy argument, not a config complaint:** PD disaggregation is a
*latency-structure* optimization (long prefills stop blocking decode steps) bought with a **2×
weight footprint**. That trade is unavailable when weights are ~48% of node HBM. It pays off when
weights are small relative to node memory — the opposite of this model.

### Superseded: the original `fp8_ds_mla` attempt (kept for the record)

### FP8 KV arm: IMPOSSIBLE BY ARCHITECTURE (2026-09-02, resolved — don't retry)

`run_fp8kv.sh` died at the first KV write:
`RuntimeError: concat_and_cache_mla, cache_kernels.cu:866, pe_dim must be 64 for fp8_ds_mla`.
`fp8_ds_mla` hardcodes a decoupled-RoPE dim of 64; **GLM-5.3 is NoPE** (`qk_rope_head_dim: 0`,
`mla_use_nope: true`), so `pe_dim = 0 ≠ 64` and no flag can change it — unreachable on **any**
hardware. ⚠️ The predicted block-size conflict did **not** happen (FLASHMLA_SPARSE resolved fine at
the auto-raised 640). Full evidence: `GLM-5.3-Flash/results/fp8kv/RESULT-arm-impossible.md`.

**So the KV-dtype asymmetry is confirmed from both directions and is UNCLOSABLE**, for a stronger
reason than a version gate: V4 can only run fp8_ds_mla (BF16 backend gated to Blackwell), GLM can
only run BF16 (FP8 route needs FlashInfer ≥0.6.18 *and* `fp8_ds_mla` is geometrically inapplicable).

## Model dtype: what is ACTUALLY loaded (verified from weights, 2026-09-01)

**`torch_dtype` in `config.json` is the ACTIVATION dtype and is a decoy. Never report it as the weight
dtype.** DeepSeek-V4-Flash says `torch_dtype: bfloat16` and loads **zero BF16 weights**. Reporting that
run as "BF16" would be wrong, and attributing an FP8-vs-BF16 delta to *architecture* is the easiest way
to publish a false result.

**How to determine the real dtype — in this order:**

1. **`quantization_config` in `config.json`**, not `torch_dtype`. V4-Flash:
   `{quant_method: fp8, fmt: e4m3, scale_fmt: ue8m0, weight_block_size: [128,128]}` plus a top-level
   **`expert_dtype: "fp4"`**.
2. **The safetensors headers** — the ground truth. V4-Flash sampled byte split: ~85% `I8` (FP4 experts,
   packed 2/byte), ~11% `F8_E8M0` (scales), ~4% `F8_E4M3` (attention).
3. **The engine's own resolution.** vLLM logs `quantization=deepseek_v4_fp8` at startup, and
   `quant_config.py:75` emits `DeepSeek V4 expert_dtype resolved to 'fp4'`.

**Verified dtype map for DeepSeek-V4-Flash** (`vllm/models/deepseek_v4/quant_config.py:29`,
`DeepseekV4FP8Config`):

| Component | Storage dtype | vLLM method |
|---|---|---|
| Routed experts (97.4% of params) | **MXFP4** | `Mxfp4MoEMethod` |
| Attention + dense linear | **FP8 e4m3**, block 128×128 | `Fp8Config` |
| Scales | FP8 **ue8m0** (`is_scale_e8m0 → True`) | — |
| KV cache | fp8 (`--kv-cache-dtype fp8`) | `fp8_ds_mla` alias |

The class resolves `expert_dtype` **lazily** on purpose: it is constructed before
`set_current_vllm_config`, so an eager read would always see the `"fp4"` default and silently misroute
`DeepSeek-V4-Flash-Base` (which ships `expert_dtype: "fp8"` with float32 scales). **The two checkpoints
differ in expert dtype — do not treat them as interchangeable.**

**FP4 is a storage format, not a compute format.** `fp4_gemm_kernel` unpacks FP4→FP8 in shared memory
and runs H100's FP8 tensor cores. **Byte traffic halves; FLOP/byte at the tensor core is unchanged** —
so do *not* claim FP4 doubles `B*_dense` to 296. H100 has no FP4 hardware.

**Parameter accounting (measured from 69,187 safetensors tensor headers, FP4 unpacked to logical
width):** 290.9 B total / **13.49 B active** — reconciling with the card's 284B/13B to within 2.4%.

| Component | Logical params | Share of total |
|---|--:|--:|
| Routed experts | 283.47 B | 97.4% |
| Attention | 5.21 B | 1.8% |
| Shared expert + embed + other | 2.23 B | 0.8% |

**Load-bearing consequence: measured sparsity is 21.6×, not `E/k` = 42.7×.** The dense remainder
(attention 5.21 + shared 1.11 + embed 0.53 = **6.85 B**) is **51% of the active budget** while being
2.4% of total. **Expert sparsity does not buy 42.7× because attention does not sparsify** — any
`E/k`-based decode-cost prediction overestimates the MoE contribution, and the dense half sets a floor.
This supports the heterogeneity thesis while correcting the arithmetic.

## The binding constraint: weight sizes (real bytes, safetensors index, 2026-09-01)

8× H100 = 640 GiB HBM, ~576 GiB usable at `gpu_memory_utilization=0.9`. **The "Flash" tier means most
2026 models now fit** — a big change from the 2025 generation:

| Model | Weights | Fits 8×H100 | Min H100s |
|---|--:|:--|--:|
| GLM-4.7-Flash | **29.1 GiB** | ✅ 1 GPU | 1 |
| Qwen3.8-27B | **51.7 GiB** | ✅ 1 GPU | 1 |
| DeepSeek-V4-Flash | **148.6 GiB** | ✅ 3 GPUs | 3 |
| GLM-5.3-Flash | 305.8 GiB | ✅ tight | 5 |
| DeepSeek-V4-Pro | ~740 GiB | ❌ | ~11 |
| **Kimi-K3** | **1453.7 GiB** | ❌ | **~21** |

**"Flash" is a product tier split along the single-node boundary, not a smaller checkpoint** —
GLM-4.7-Flash is 29 GiB with *full MLA*. State the 21× min-GPU spread on page 1 as a finding. For K3
and V4-Pro use the proxy ladder (`plan.md` §2) — **never weight-offload**, the numbers would be
PCIe-dominated and architecturally meaningless.

## Architecture facts (from `config.json`, 2026-09-01 — re-verify, repos change)

| Model | L | Attention pattern | E | k | E/k | KV KiB/tok | MTP |
|---|--:|---|--:|--:|--:|--:|:--|
| GLM-4.7-Flash | 47 | **MLA all layers** (v_head_dim 256) | 64 | 4 | 16× | 52.9 | ✓ 1 |
| Qwen3.8-Flash-Next | 48 | 36 linear + **12 full** (interval 4) | **512** | **10** | **51.2×** | 24.0 | ✓ mtp=1 |
| DeepSeek-V4-Flash | 43 | DSA + `compress_ratios` 4/128 + swin 128 | 256 | **6** | 42.7× | **~7.4** | ✓ 1 |
| GLM-5.3-Flash | 45 | 34 KDA + **11 DSA** (`index_topk` 2048) | 288 | 8 | 36× | 12.4 | ✓ 1 |
| ↳ *GLM-5.3 params* | **321.34 B total / 17.38 B active [M]** (safetensors shapes) — 97.8% native FP8 e4m3 on disk; + 7.43 B MTP, 0.56 B ViT excluded ||||||| |
| DeepSeek-V4-Pro | 61 | DSA + compress + swin | 384 | **6** | **64×** | – | ✓ 1 |
| **Kimi-K3** | 93 | 69 KDA + **24 MLA** (interval 4) | **896** | **16** | **56×** | **27.0** | **✗ 0** |
| *GLM-4.5 (2025)* | 92 | GQA 96:8 all layers | 160 | 8 | 20× | 368.0 | ✓ 1 |
| *DeepSeek-V3 / Kimi-K2 (2025)* | 61 | MLA all layers | 256/384 | 8 | 32/48× | 68.6 | ✓1/✗0 |

**Five load-bearing facts:**
1. **~25% full-attention layers at interval 4** — Kimi-K3 24/93, Qwen3.8 16/64, GLM-5.3 11/45. Three
   vendors converged independently.
2. **KV cost spans 50×** in one generation (368 → ~7.4 KiB/tok). Two independent mechanisms reach it:
   per-layer compression (DeepSeek-V4) vs. hybrid O(1)-state layers (Kimi/GLM/Qwen).
3. **`expert_dtype: "fp4"`** in DeepSeek-V4 while attention weights stay FP8 — **mixed
   bytes-per-param inside one model**. **Verified from the safetensors headers**, not just the config:
   experts are `I8` at half logical width (FP4 packed 2/byte). Halves byte traffic; ridge point unmoved.
4. ⚠️ **CORRECTED 2026-09-02: Qwen3.8 is NOT dense.** It reads `n_routed_experts: None` only because Qwen uses a different key — actual: **`num_experts: 512`, `num_experts_per_tok: 10`, `moe_intermediate_size: 640`** (verified from 150,528 `.experts.` tensors and zero plain `mlp.*_proj`). It is a **fine-grained MoE at E/k = 51.2×**, the report's most extreme sparsity point — not a dense control. **This report has no dense baseline; say so.**
   Also **BF16-native**, so it's the one model that runs on A100 unchanged.
5. **Kimi-K3 has no MTP** (`num_nextn_predict_layers: 0`); GLM-5.3 has
   `index_share_for_mtp_iteration: true` (indexer shared across MTP iterations — sparse-attn/MTP co-design).

Linear/KDA layers hold a **constant recurrent state** (Qwen3.8: 0.023 GiB/seq over 48 layers) that
does *not* grow with context — this is what breaks prefix caching (see below).

**DeepSeek's own names for the V4 attention mechanisms** (from the model card, so cite as a vendor
claim): **CSA = Compressed Sparse Attention** and **HCA = Heavily Compressed Attention** — these are
the `hc_*` config fields. The card claims V4-Pro needs *"only 27% of single-token inference FLOPs and
10% of KV cache"* vs DeepSeek-V3.2 at 1M context, which independently corroborates the ~50× KV-spread
finding.

**Unverified — confirm before publishing:** `compress_ratios` semantics (the ~7.4 KiB/tok figure
assumes a per-layer KV divisor — check `inference/model.py` or **arXiv 2606.19348**, now known to
exist); whether Kimi-K3's `num_experts: 896` is per-layer or global; V4's `o_lora_rank` / `o_groups` /
`num_hash_layers` exact role.

## Layer reduction is the sanctioned throughput proxy

The user's idea, adopted (`plan.md` §3): cut `num_hidden_layers`, keep architecture, random init, no
spec decode. It works here because KV bytes/token, `E/k`, and per-layer-class cost mix are all
**per-layer properties** — reduction preserves exactly what's under test.

**The trap:** hybrid models place full-attn layers at **interval 4**. Reduced `L` must be a multiple
of 4 **and** the layer-type histogram ratio must be preserved. Valid: K3 → 12/24/48, Qwen3.8 → 16/32,
GLM-5.3 → 12/24, V4-Flash → 12/20. Invalid: 23, 46, 11, 22. `bench/reduce.py` must **assert** the
ratio and print before/after histograms; truncate `layer_types`, `full_attn_layers`, `kda_layers`, and
`compress_ratios` consistently.

**Never claim quality/accuracy from reduced random-init models.** Also: random routers route uniformly
= best case for expert coverage, so real expert-read cost may be *lower* than predicted — cross-check
trained-router hit counts on Qwen3-30B-A3B.

Validation gate before trusting any extrapolation: GLM-4.7-Flash runs at full L=47 on **1 GPU** —
measure L ∈ {12,24,36,47}, fit linearity, report fit quality. If non-linear, restrict to
equal-L cross-architecture comparison (which is the cleaner experiment anyway).

## GPU architecture is a HARD gate — H100 yes, A100 no (verified 2026-09-01)

**FP8 tensor-core support decides whether DeepSeek-V4 runs at all.** The shipped kernels terminate
every GEMM in FP8, so an arch without FP8 tensor cores cannot run this model by any supported path.

| Arch | SM | FP8 tensor cores | FP4 | DeepSeek-V4-Flash |
|---|---|:--|:--|:--|
| **A100 (Ampere)** | SM80 | **NO** | NO | ❌ **cannot run** |
| L40S / 4090 (Ada) | SM89 | YES | NO | ✅ in principle |
| **H100 (Hopper)** | SM90 | **YES** | NO | ✅ **works as shipped** |
| B200 (Blackwell) | SM100 | YES | **YES** | ✅ native FP4 |

*(Arch capability table is from general NVIDIA knowledge, not a datasheet checked this session — the
A100-lacks-FP8 fact is well established, but verify if a claim depends on it.)*

**Why H100 works despite having no FP4 hardware** — `inference/kernel.py:442-496`:

> `fp4_gemm_kernel`: *"FP8 act x FP4 weight GEMM … Strategy: load FP4 sub-blocks of size
> [block_N, sub_K] (sub_K=32), cast FP4 to FP8 via float, then do FP8xFP8 GEMM."*

**FP4 is a storage format, not a compute format.** Weights unpack FP4→FP8 in shared memory, then run
on H100's native FP8 tensor cores. There is **no SM gate, no capability check, no Blackwell assert**
anywhere in `kernel.py` or `model.py`. `kernel.py:10-11` even sets `TL_DISABLE_WARP_SPECIALIZED` and
`TL_DISABLE_TMA_LOWER`, so the kernels deliberately avoid Hopper-only TMA/warp-specialization — **the
FP8 dtype is the wall, not the kernel structure.**

**Why A100 fails twice over.** No FP8 compute, and upcasting to BF16 doesn't fit:

| Component | As shipped | → BF16 |
|---|--:|--:|
| Experts (FP4, ~85% of bytes) | 126.6 GiB | **506.6 GiB** (4×) |
| Rest (FP8/BF16/F32) | 22.0 GiB | 44.0 GiB (2×) |
| **Total** | **148.6 GiB** | **550.6 GiB** |

8× A100-80GB usable ≈ **536 GiB** → misses by ~15 GiB **with zero KV cache**. 16× A100-80GB
(~1073 GiB) would fit. And there is no supported route: `convert.py --expert-dtype` accepts only
**`{fp8, fp4}`**, and unlike DeepSeek-V3 this repo ships **no FP8→BF16 cast script** — you would write
dequantization yourself and rebuild all five tilelang kernels for SM80.

**If the session lands on A100**, the report survives without the DeepSeek family:
**Qwen3.8-Flash-Next is BF16-activation-native** so it runs unchanged (⚠️ but it is a **512-expert MoE,
NOT the dense control** — see the corrected fact 4 above), GLM-4.7-Flash likely fits on 1 GPU. Lost axes:
FP4/mixed-precision and DSA/compression. **Also re-derive the roofline** —
A100 HBM ≈ 2.0 TB/s vs H100 3.35 TB/s, so `B*_dense` drops from 148 to ~78 tok/step at BF16. The
`E/k` thesis holds; the crossover numbers change.

## Storage: no per-user quota; you are the 3rd-largest consumer (2026-09-01)

`vol22-scratch` = 99 TiB, **89% full, 12 TiB free** (re-measured 2026-09-01 after the GLM-5.3 and
Qwen3.8 downloads landed; was 91% / 9.3 TiB earlier the same day). **No quota mechanism exists** (`xfs_quota`,
`repquota`, `lfs` all absent); a 2 GiB write test passed at **761 MB/s**. The limit is social, not
technical.

| Consumer | Size |
|---|--:|
| `users/khail/` | ~5.9 TiB |
| `users/ppreetam/` | ~3.2 TiB |
| **`users/tanngo/` (you)** | **~5.1 TiB** — of which `FORCING-SERIES/` alone is **3.4 TiB**, `DROID/` 555 GiB, `arp/` 234 GiB |

Report needs ~685 GiB worst case (V4-Flash 148.6 + its **converted second copy** ~149 + GLM-4.7-Flash
29 + Qwen3.8 52 + GLM-5.3-Flash 306) ≈ **7% of free space** — comfortable, but the volume being at 91%
is a real risk if another user lands a multi-TiB job mid-download.

**`convert.py` doubles the DeepSeek footprint** — delete the HF-format copy once converted.

**Do NOT use the shared `/prj/.../vol22-scratch/huggingface_cache/`** (258 GiB, owned by `phucpham`,
**not writable by you**, and holds only video/vision models — nothing relevant).

⚠️ **`HF_HOME=cache` / `HF_HUB_CACHE=cache/hub` are RELATIVE** — they resolve against the *current
working directory*, so from a `$HOME`-side path a download lands on the 621 GiB fleet-shared NFS
export. Always export absolute paths first:

```bash
export HF_HOME=/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/cache
export HF_HUB_CACHE=$HF_HOME/hub
```

## DeepSeek-V4-Flash: DOWNLOADED and SERVED (updated 2026-09-01, later session)

**Superseded header:** this section previously read "nothing downloaded." The model is now **fully
cached (150 GiB)** at
`cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1`
and has been served on 8×H100 under vLLM 0.28.0. **Do not re-download.** The `convert.py` /
`--expert-dtype fp8` escape hatch below is **not needed** — vLLM handles the FP4 checkpoint natively, so
there is no second on-disk copy and the storage estimate drops by ~149 GiB.

The historical detail below remains accurate as a record of the checkpoint's structure.

**Repo is fully open:** `private: False`, **`gated: False`**, `disabled: False`, **MIT** license, no
token needed. HTTP 206 on a range request confirms weight bytes are fetchable. 1.7M downloads, commit
`60d8d707`, last modified 2026-06-22. **148.6 GiB** (`metadata.total_size` = 159,609,485,896 B) across
**46 shards** + 27 small files. Measured single-stream **23.4 MB/s** → ~1.9 h; ~15 min at 8 workers.

**FP4 storage confirmed from the actual safetensors headers** (not inferred from a config field):

```
layers.18.ffn.experts.0.w1.weight   I8       [2048, 2048]   ← logical [2048, 4096], packed 2/byte
layers.18.attn.wkv.weight           F8_E4M3  [512, 4096]
layers.18.attn.wkv.scale            F8_E8M0  [4, 32]        ← matches scale_fmt: "ue8m0"
```

Sampled byte split: **~85% I8 (FP4 experts) · ~11% F8_E8M0 (scales) · ~4% F8_E4M3 (attention)**. Also
present: `attn.indexer.*`, `attn.compressor.*`, `attn_sink`, `hc_attn_base` — the DSA machinery.

**Dependency gaps** (`inference/requirements.txt` vs conda `nano-vllm`):

| Package | Required | Have | Status |
|---|---|---|:--|
| `torch` | ≥2.10.0 | 2.11.0+cu128 | ✅ |
| `safetensors` | ≥0.7.0 | 0.8.0 | ✅ |
| `torch.float4_e2m1fn_x2` / `float8_e8m0fnu` | needed | both present | ✅ |
| **`transformers`** | **≥5.0.0** | **4.57.1** | ❌ **major bump — use a FRESH env, not `nano-vllm`** |
| **`tilelang`** | **==0.1.8** | – | ❌ on PyPI ✓ |
| **`fast_hadamard_transform`** | any | – | ❌ on PyPI ✓ |

**The shipped `inference/` is `torchrun` offline generation** — no continuous batching, no paged KV, no
server. Requires a `convert.py` step first (`MP=4` recommended). For throughput numbers use
vLLM/SGLang; `sgl-project/DeepSeek-V4-Flash-FP8` exists, suggesting SGLang took the FP8 route.

**Escape hatch** (`inference/README.md`): *"If you want to use fp8, just remove `"expert_dtype": "fp4"`
in `config.json` and specify `--expert-dtype fp8`"* → ~270 GiB, still fits 8×H100, skips the FP4
kernel. Or use **`DeepSeek-V4-Flash-Base`** (FP8 Mixed natively) — **the right checkpoint for
layer-reduction work anyway**.

## GLM-5.3-Flash: COMPLETE (46 pts) — all owed measurements landed 2026-09-02 (session 3)

Weights **already cached — do not re-download**: 308 GiB, snapshot `03eb5366`, 62 shards, under
`cache/hub/models--zai-org--GLM-5.3-Flash`. Runs only in the `glm53-flash` image (see the env table).

**Status: 46 measured points, all cold and complete. `report.md` + `README.md` written and updated.**
Arms on disk: `bf16kv/` (**15** — now an 8-point batch curve) · `fp8kv-fi618/` (11) · `bf16kv-fi618/`
(4, the control) · `bf16kv-mtp-n1/` (4) · `bf16kv-mtp-n5/` (4) · **`bf16kv-mtp-n1-context/` (4)** ·
**`util085/` (4)** · `fp8kv/` (0 JSONs, `RESULT-arm-impossible.md`). Plus 7 quarantined points in
`_discarded-warmup-contaminated/`, `_util085_queued_firstrun/`, `_mtpctx_queued_firstrun/`.

### ✅ ALL 4 OWED GLM MEASUREMENTS ARE DONE — do not re-run them

⚠️ **No container switch is needed** if the session is already in `vllm/vllm-openai:glm53-flash`
(vLLM `0.1.dev20051`). **Verify, do not assume** — the qwen38 image does NOT register `Glm5Next*`:
`/usr/bin/python3 -c "from vllm import ModelRegistry; print([a for a in ModelRegistry.get_supported_archs() if 'Glm5' in a])"`

| # | task | result | writeup |
|--:|---|---|---|
| **1** | util A/B 0.82→0.85 | ✅ **capacity-only**: **+11.2% KV** (1,916,967→2,131,562 tok), throughput **0.994–1.000×**, no trend, TPOT flat. Peak activation **4.05 GiB in both arms** = the control that makes it clean. Third model to confirm; the two clean ones (GLM, V4) agree. | `results/util085/RESULT-util-ab.md` |
| **2** | 8-point batch curve | ✅ **knee at c≈4–8**, not c=64: 1→8 buys **3.08×**, 8→64 only **1.51×**; efficiency 1.00→0.385(c8)→**0.073**(c64). Matches V4 almost exactly (3.03× / 1.37× / 0.378 / 0.065) **despite completely different attention stacks**. ⚠️ **V4's c=48 dip does NOT reproduce** (GLM 1.064× vs V4 0.985×). Also: **TTFT is non-monotonic** (falls c=16→48, chunked-prefill packing). | `results/bf16kv/RESULT-8point-batch-curve.md` |
| **3** | scrape `--enable-mfu-metrics` | ✅ **CORRECTED A HEADLINE BY 10×** — see the status block at the top of this file. `bench.sh` now snapshots the three counters before/after each point and folds rates into the JSON. | `fix_bug.md` **bug 13** |
| **4** | MTP on the context axis | ✅ **INVERTS the batch-axis advice**: −4–6% throughput but **−16.7–24.8% TTFT** up to 131K. On TraceLab's **530:1 ISL:OSL** that is a good trade. ⚠️ The `index_share_for_mtp_iteration` hypothesis was **FALSIFIED** on throughput (1.064× at 16K → **0.839×** at 260K). The 260K ceiling is **KV-bound (96.9%), not accuracy-bound** (acceptance held **69.1%**); MTP costs **13.3% of KV** here, not V4's 2.4%. | `results/bf16kv-mtp-n1-context/RESULT-mtp-context-axis.md` |
| 5 | TP sweep | ❌ **CLOSED, no sweep exists.** TP4 **measured OOM** (75.36 GiB/GPU, 1.60 GiB free); TP=7 fails `64 % 7`. GLM needs ≥5 GPUs → **only TP8 reachable on one node.** Test EP-all-to-all on **V4 or Qwen** (both fit in 3 GPUs) or via Nsight. | — |

### ⚠️ Two hazards this session re-confirmed — budget for both

1. **`/tmp` JIT caches start EMPTY on a freshly scheduled container**, so the first launch is a **cold
   compile** and it **mis-sizes KV** (bug 12). Measured on GLM this session: peak activation
   **4.88 GiB cold vs 4.05 GiB warm** → KV pool **1,838,761 vs 1,916,967 tokens**. **Warm the cache with
   one throwaway point, restart, then measure** anything that sizes KV. Cold start ≈ 350–425 s to
   `/health 200` (DeepGEMM JITs 967 kernels); warm is far quicker.
2. **The queueing guard fired twice more** on first-points-after-cold-start: util-0.85 c=4 read
   **2.3× low** (98.8 vs 223.9) and MTP-context 16K read **2.2× low** (143.5 vs 314.8) — the latter would
   have **inverted that arm's trend** and published "MTP halves throughput at short context." Signature:
   **p99/median TTFT** (7.47× and 5.13× vs ≤2.2× on good points). **Rerun warm, quarantine, never delete.**


**Do NOT re-run these — already resolved, evidence on disk:**
- `fp8kv` via `fp8_ds_mla` → **impossible by geometry** (GLM is NoPE, `qk_rope_head_dim: 0`, needs
  `pe_dim==64`). Fails on **any** GPU including Blackwell. `results/fp8kv/RESULT-arm-impossible.md`.
- FP8 KV **does** work via the FlashInfer 0.6.18 overlay (**1.805× KV**, −25% dtype / −22% backend after
  the control). ⚠️ Not production (`FLASHINFER_DISABLE_VERSION_CHECK=1` + `moe_backend=deep_gemm`).
- TP4 single pool and PD-disaggregation TP4+TP4 → **measured OOM / 612 of 640 GiB**. The recipe's GB200
  TP4 command **cannot run on H100**: 305.8/4 = 76.5 GiB/GPU vs ~75.6 usable.
- MTP `n=5` → **worse than n=1** at every conc ≥4 (0.91× at c=64, acceptance 71.8%→30.6%, KV 99.2%).

**GLM cold-compile audit (2026-09-02): CLEAN.** Both `bf16kv` launches measured **4.05 GiB peak
activation**, and all 16 GLM+V4 startup logs sit in 2.03–4.27 GiB. The 17.07 GiB artifact was isolated to
one Qwen arm, so **no published GLM number is affected.** Still: warm the compile cache before task 1,
since that arm *sizes KV* and is exactly where the artifact would bite.

```bash
# next session, in the glm53-flash container, after preflight
cd GLM-5.3-Flash && ./run.sh                    # BF16 KV, ~226 s cold / ~24 s warm; port 8001
#   then WARM THE CACHE: run one throwaway point before the util arm (task 1)

# task 1 -- util A/B. GPU_UTIL was PARAMETERIZED in _common.sh on 2026-09-02
# (it was hardcoded 0.82 before; default is still 0.82 so existing arms reproduce).
# NOTE: unlike Qwen's script, GLM's run.sh takes NO GPUS=/TP= -- it uses all 8
# GPUs implicitly and hardcodes --tensor-parallel-size 8.
GPU_UTIL=0.85 ./run.sh
cd .. && MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/util085 \
QUANT=fp8-attn-dense+fp8-experts KV_DTYPE=bfloat16 ./bench.sh batch

# task 2 -- finer batch axis into the EXISTING arm (existing files are skipped,
# and each point's seed derives from (isl,conc,osl) so new conc = new seed = no
# cache contamination)
CONCS="1 2 4 8 16 32 48 64" MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/bf16kv \
QUANT=fp8-attn-dense+fp8-experts KV_DTYPE=bfloat16 ./bench.sh batch
```

🛑 **Never `pkill -f "vllm serve"`** — PID 1 matches it and the container restarts. Use
`pkill -f "GLM-5.3-Flash"` then confirm `ps -p 1 -o args=`.
⚠️ **Expect the queueing guard to fire spuriously at c=1 and on the first points after a cold start.**
Five occurrences so far (`fix_bug.md` bugs 9, 10, 12). Signature: **p99 TTFT ≫ median**, or a point that
is non-monotonic vs its neighbours. Re-run warm; median TTFT should be unchanged.

### Historical: how it was unblocked (kept for the record)

**Status: the server is UP on 8×H100 / CUDA 13.0 and sweeps are running.** Both historical blockers
are closed — the GPU-0 squatter (`--command -- sleep infinity`) and the `-cu129` MHC segfault (gone
on the cu130 image, verified). Four further bugs had to be fixed to boot; all are now in the scripts
and are tabulated immediately below, with the full diagnosis method in
[`../fix_bug.md`](../../../fix_bug.md).

### ✅ UNBLOCKED AND SERVING on the CUDA-13 image (2026-09-02, later session)

**GLM-5.3-Flash now boots and answers on `vllm/vllm-openai:glm53-flash` (CUDA 13.0, torch
2.13.0+cu130).** Verified live: `/health` 200, BF16 KV, spec decode OFF, 2,099,654 KV tokens,
23.26 GiB KV/GPU, 8.01x concurrency at 256K, `2+2 = 4` with reasoning. Sweeps are running.

**Full bug-by-bug writeup with diagnosis method: [`../fix_bug.md`](../../../fix_bug.md).** Read that
before debugging a future model bring-up -- four of the six bugs named the wrong subsystem.

**The cu129 MHC/TileLang segfault is GONE on this image** -- verified, not assumed: the latest
logs show **24 successful `mhc_post_tilelang` compiles across 8 workers and zero "Segfault
encountered"**. The `-cu129` diagnosis in the previous session's notes was correct *for that
image* and is now historical. ⚠️ Careful with the grep: `grep -c "Segfault\|mhc_post_tilelang"`
is an OR that also matches the *successful* compile lines and will overcount failures.

**Four separate blockers had to be fixed. All are now in the scripts:**

| # | Symptom | Real cause | Fix |
|---|---|---|---|
| 1 | `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (512)` | GLM-5.3 is a **HYBRID** (34 KDA linear-attention + 11 DSA layers); KDA layers need **one recurrent-state block per decode sequence**. And `--max-num-seqs` does **not** default to 128 on H100 -- `arg_utils.py:2547 get_batch_defaults()` gives any non-A100 GPU ≥70 GiB **1024** | `--max-num-seqs 256` (2x headroom over the 512 ceiling; grid tops out at conc 64) |
| 2 | `RuntimeError: [Errno 122] Disk quota exceeded` from `triton/runtime/cache.py:120`, ~8 min into startup | **NFS `$HOME` is at its ~5 GB per-user quota.** `df` shows 622 G avail -- it reports the volume, not the quota. `-w`/`mkdir`/`touch` all PASS on a quota-full dir | `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` (unconditional), `VLLM_CACHE_ROOT`, `XDG_CACHE_HOME` -> `/tmp`, keyed by uid+CUDA major, plus `preflight_caches()` that writes 4 MB for real |
| 3 | `CUDA error: invalid argument` at ~69-78% of cudagraph profiling, no traceback | **Bug 2 again, via a dir the C++ resolves off `$HOME`**: DeepGEMM JITs the FP8 block-scale GEMM into `$HOME/.tensorrt_llm` (`deep_gemm/compiler.cuh:65`); the quota write fails, the GEMM launches an unbuilt kernel, CUDA reports `invalid argument` downstream. Real error is **one frame up**: `tvm.error.InternalError: cannot create directories ... /.tensorrt_llm/tmp/gemm_swapAB_...` | `TRTLLM_DG_CACHE_DIR` + `FLASHINFER_WORKSPACE_BASE` -> `/tmp` |
| 4 | Every result JSON labelled `quant=...mxfp4-experts kv_cache_dtype=fp8` | `bench.sh` **hardcoded DeepSeek-V4's provenance**. GLM-5.3 has **FP8** (not MXFP4) experts and **BF16** (not fp8) KV | `QUANT`/`KV_DTYPE`/`HW` env vars; manifest now **scrapes `cache_config_info` from the live server** |

**⚠️ Bug 3 cost ~40 min of wrong hypotheses -- don't repeat them.** The error says CUDA during
cudagraph capture, so all three of these were tried and **all three failed**:
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` (verified off, same failure index), capping the capture ladder,
and **`--enforce-eager`** (skips capture entirely). Eager failing was the tell: if disabling
capture doesn't help, the bug was never in capture.

**⚠️ Use `--max-cudagraph-capture-size 256`, NEVER `--compilation-config
'{"max_cudagraph_capture_size":256}'`.** The latter **replaces** the whole `CompilationConfig` and
silently wiped `pass_config` to `{}` (losing `fuse_norm_quant`/`fuse_act_quant`/
`fuse_allreduce_rms`). The dedicated flag is merged (`arg_utils.py:2452`) and is mutually exclusive
with the config key. Caught only by diffing `pass_config` between two runs' startup banners.

**⚠️ `--num-warmups` was REMOVED from `bench.sh` -- it self-poisons the prefix cache.** It draws
the warmup from the **same seeded prompt set** as the measured run, so the run re-reads the
warmup's blocks. Every batch point reported exactly **16,000 new hits = 25 blocks x block_size 640
= one ISL-16384 prompt**, constant across concurrency (so self-contamination, not the cross-point
leakage unique seeds already fix). The bias is **uneven** -- 12.2% of the c=1 point but 0.8% of
c=64 -- so it inflates the low-concurrency anchors and **flattens measured concurrency scaling**.
Contaminated points quarantined in `results/_discarded-warmup-contaminated/` (kept as evidence, not
deleted). Reruns report `cold run confirmed (0 new prefix-cache hits)`.

**Also worth knowing:** `--block-size 128` is the *minimum legal* value (kpool), but vLLM
**auto-raises it to 640** so the attention page >= mamba page, then pads the mamba page by 20.75%.
Live: `block_size=640`, `mamba_block_size=128`, `cache_dtype=auto`->BF16, `num_gpu_blocks=3064`.
640 is still kpool-legal (640/4 = 160, 160 % 32 == 0).

**Fairness note for the V4 comparison:** the V4 baseline passed **no** `--max-num-seqs`, so it ran
at the 1024 H100 default while GLM is pinned to 256. No published point is affected (both sweeps
top out at concurrency 64), but it is an engine-config difference and must be **stated**.

### CONFIRMED WORKING (originally verified on cu129; all still true on cu130)

Everything up to the first forward pass is verified — this is real progress, not a dead end:

| Checked | Result |
|---|---|
| `glm5_next` registration | ✅ `Glm5NextForCausalLM`, `...ForConditionalGeneration`, `...MTPModel` |
| `--block-size 128` | ✅ accepted; vLLM then **auto-raises to 640** ("attention page size ≥ mamba page size"), then pads the mamba page by 20.75%. **640 is legal for kpool** (640/4 = 160, 160 % 32 == 0) |
| Weight load | ✅ all 62 shards, **38.08 GiB/GPU** — matches the predicted 38.2 GiB |
| `kv_cache_dtype` | ✅ `auto` → **BF16**, as intended; `quantization=fp8` |
| `--limit-mm-per-prompt` zeros | ✅ *"All limits of multimodal modalities set to 0, running in text-only mode"* — the ViT-skip fairness flag genuinely works |
| `speculative_config` | ✅ `None` on the headline arm |
| Load time | ~226 s cold (NFS), ~24 s warm (page cache) |


| File | Arm |
|---|---|
| `GLM-5.3-Flash/_common.sh` | shared flags + `preflight_gpus` (detects the squatter, refuses to launch) |
| `run.sh` | **BF16 KV — headline** |
| `run_fp8kv.sh` | `fp8_ds_mla` KV — second dtype arm, **may legitimately fail**, see below |
| `run_mtp.sh` | BF16 KV + MTP `num_speculative_tokens:1` — A/B only |
| `sending.sh` | smoke test, **port 8001** |
| `submit_job.sh` | the RunAI submit with the entrypoint override |

**Port 8001, not 8000** — 8000 was the squatter's. Hitting 8000 by mistake smoke-tests Qwen3-0.6B **and
passes**, which is the worst kind of failure.

### `--block-size 128` is MANDATORY and the default 64 hard-asserts

GLM-5.3 sets `index_kpool: 4`. `Glm5NextIndexerCache` (`models/glm5next/nvidia/attention.py:122`,
`:140-150`) requires **both** `block_size % index_kpool == 0` *and* `(block_size / index_kpool) % 32 == 0`
— i.e. a multiple of `index_kpool * 32` = **128** — so DeepGEMM paged-MQA pool pages (32 or 64 entries)
tile the storage block. The default 64 collapses the storage block to 16 and fails. 128 is the smallest
legal value. ⚠️ **This is a DIFFERENT cause than V4's `--block-size 256`** (`sparse_mla.py:53`, per-layer
`compress_ratios`) — same-looking flag, unrelated reason; don't copy one justification onto the other.

### The KV-dtype asymmetry is UNCLOSABLE on H100 — report it, don't try to fix it

Each model can only run the KV dtype the other cannot. **Verified from source and by runtime probe**, not
from release notes:

| Model on SM90 | KV dtype available | Gate |
|---|---|---|
| DeepSeek-V4-Flash | **`fp8_ds_mla` only** | its BF16 path (`use_fp8_ds_mla_layout=False`) lives on a backend gated to `capability.major in [10, 12]` — Blackwell/SM120 — `models/deepseek_v4/nvidia/flashinfer_sparse.py:113` |
| GLM-5.3-Flash | **BF16 only** | FP8 KV needs `FLASHINFER_MLA_SPARSE_SM90`, which feature-probes FlashInfer for the `ckv_scale_arr` kwarg (≥ 0.6.18). Image ships **0.6.17**; `has_flashinfer_sm90_nope_mla()` → **False** (ran it) |

This matches the vLLM recipe's note: *"Hopper does not support FP8 KV cache for this model and must run
BF16 KV."* **Why it's survivable:** KV is not binding in either run — GLM BF16 KV is **~11.35 KiB/tok**
(11 DSA layers × 512 latent × 2 B = 11.00, + kpool indexer ≈ 0.35) → ~1.42 GiB per 131K seq against
~27 GiB/GPU free; V4 peaked at **30.2%** KV even at 256K×8. And **weights are FP8 e4m3 on both**, so the
expert-read comparison the thesis rests on is intact. The larger, pre-existing dtype gap is
**V4's MXFP4 experts vs GLM's FP8 + a 1509-entry `modules_to_not_convert` BF16 exclusion list** — that
one is load-bearing and must be stated (rule 4).

**`run_fp8kv.sh` may fail by design, and that failure is a result.** `FLASHMLA_SPARSE` advertises
`get_supported_kernel_block_sizes() == [64]` while the kpool indexer demands a multiple of 128 — possibly
unsatisfiable, in which case `v1/worker/utils.py` raises **"No common block size"**. Record the exact
error (rule 7); do **not** retry with `--block-size 64`, which trips the kpool assert with a worse message.

### Other verified GLM-5.3 facts

- **`--trust-remote-code` NOT needed** — `glm5_next` is natively registered
  (`transformers_utils/config.py:96-98`). Passing it grants arbitrary code execution for nothing.
- **`--limit-mm-per-prompt '{"image":0,"video":0}'` is a FAIRNESS flag, and it genuinely works**: zeroing
  every modality makes vLLM **skip constructing** the 24-layer ViT (`interfaces.py:307`), keeping the
  comparison text-to-text against text-only V4.
- Parsers: **`glm45` and `glm47` both exist** and map to the same `glm47_moe_reasoning_parser`
  (`reasoning/__init__.py:55-60`). Tool parser `glm47` confirmed registered.
- Architecture confirmed from `config.json`: **L=45**, `layer_types` = **34 `linear_attention` (KDA) + 11
  `deepseek_sparse_attention`**, `mlp_layer_types` = 42 sparse + 3 dense, E=288, k=8, `kv_lora_rank` 512,
  **`qk_rope_head_dim: 0` (NoPE MLA)**, `index_topk` 2048, `mla_use_nope: true`,
  `num_nextn_predict_layers: 1`, `index_share_for_mtp_iteration: true`. Vision: 24-layer ViT, 448px.
- **KDA state is context-independent**: ~0.017 GiB/seq/GPU at TP8 (34 layers × 8 heads/GPU × 128 × 128 ×
  4 B). This is the O(1)-state half of the heterogeneity thesis, measured.
- **TP=8 is more defensible here than for V4.** V4 fits in 3 GPUs, so its TP=8 per-GPU numbers are
  pessimistic; GLM-5.3 needs ≥5 GPUs (38.2 GiB/GPU at TP8), so TP=8 is closer to a real deployment.
  Note the asymmetry when comparing per-GPU figures.

### Engine-version mismatch — a real fairness threat, fix pending

The V4 baseline ran on **conda vLLM 0.28.0 / cu129**; GLM will run on the **image dev build** (use the
**cu13** tag — see the blocker above).
That violates "same engine, same version" more than the KV dtype does. V4 is still cached (150 GiB) and
runs natively in this image, so a **version-matched V4 re-sweep is cheap and should be done** — it turns
an uncontrolled variable into a controlled one. The user also asked for a **V4 BF16-KV** rerun; per the
table above that is **expected to be impossible on SM90** — attempt it, capture the error, report it.

## Qwen3.8-Flash-Next-FP8: DOWNLOADED and LOADS in this image (2026-09-01 — correction)

**Corrects an earlier assumption that Qwen3.8 would need a different image.** It does not, on the
evidence available: weights are **fully cached (174 GiB, 131 shards, snapshot `236dfdf2`)** under the
shared `cache/hub/models--Qwen--Qwen3.8-Flash-Next-FP8`, and a launch in the `glm53-flash` image got
**past config parsing and into weight loading** at TP=4 on GPUs 4–7 — log shows `model_type` accepted,
FP8 MoE via **TRITON** backend, **FlashAttention 3**, FlashInfer GDN prefill (JIT — first run slow; use
`--gdn-prefill-backend triton` to skip), and EP with **Local/global experts 128/512**. Its
`run.sh` / `run_mtp.sh` / `_common.sh` already exist in `Qwen3.8-Flash-Next-FP8/`.

**Not yet known:** whether it reached `/health 200` — the log ends mid-load at shard 3/131, and **no
results exist**. Its `run.sh` header states Qwen3.8 is **BF16-KV-only** (every QSA backend declares
`supported_kv_cache_dtypes = ["auto","bfloat16"]`), which makes **Qwen3.8 ↔ GLM-5.3 the one KV-dtype-matched
pair** in the comparison. Verify that claim from source before publishing it.

Official spec from the model card: **284B total / 13B active / 1M context / FP4+FP8 Mixed**. Paper:
**arXiv 2606.19348** (*"DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"*).
**No Jinja chat template** — a separate `encoding/` folder ships Python encode/parse scripts.

## Tooling (settled in `plan.md` §5 — don't re-litigate)

- **vLLM** = primary serving engine; **one SGLang cross-check** to show findings aren't engine-specific.
- **nano-vllm** (`../arp/nano-vllm`, installed `-e`) = the **instrumentation vehicle** for prefix-cache
  work. Its `nanovllm/engine/block_manager.py` (xxhash block hashing, `ref_count`, `hash_to_block_id`)
  is the clearest legible prefix-cache implementation available. **Caveat:** it only ships
  `nanovllm/models/qwen3.py` — a Qwen3-*dense* instrument unless extended. Scope work accordingly.
- **Profiling:** CUDA events (timing), `torch.profiler` (operator attribution), Nsight Systems
  (overlap/network). Attribute operators using **NanoFlow's §2.2 taxonomy** — *dense / attention /
  network / other* — so breakdowns are directly comparable to his Fig. 4.
- **Env — ⚠️ DEPENDS ON THE MODEL. `vllm-py12` is NOT universal (corrected 2026-09-01, GLM-5.3 session).**

  | Model | Engine to use | Why |
  |---|---|---|
  | DeepSeek-V4-Flash | conda `vllm-py12` — vLLM **0.28.0**, torch 2.13.0+cu129, transformers 5.16.1 | the env the V4 baseline was measured on |
  | **GLM-5.3-Flash** | **the IMAGE's `/usr/local/bin/vllm`** — `0.1.dev20051+g487ecf187`, transformers 5.15.1 | **`vllm-py12` CANNOT run it** |

  ⚠️ **The image comes in two CUDA flavors and they are NOT equivalent.** Both carry the same vLLM dev
  build, but `vllm/vllm-openai:glm53-flash` is **cu130** and `...:glm53-flash-cu129` is **cu129**.
  **Use the cu13 tag** — on cu129 the TileLang `mhc_post_tilelang` kernel segfaults in
  `cuModuleLoadData` and GLM-5.3 cannot get past its first forward pass. The image's own
  `/vllm-workspace/torch_lib_versions.txt` declares `torch==2.13.0+cu130`, so cu130 is what it was built
  for. Full diagnosis in the GLM-5.3-Flash section.

  `vllm-py12`'s registry has **no `Glm5Next*` entry** (`ModelRegistry.get_supported_archs()` stops at
  `Glm4MoeLiteForCausalLM` / `GlmMoeDsaForCausalLM`), so `vllm serve` dies with "architecture not
  supported." The `vllm/vllm-openai:glm53-flash` image registers all three —
  `Glm5NextForCausalLM`, `Glm5NextForConditionalGeneration`, `Glm5NextMTPModel` — plus a full
  `vllm/models/glm5next/` package (`attention.py`, `kda.py`, `mtp.py`, `multimodal.py`,
  `ops/{fused_eh_norm,kpool_compress}.py`).

  🪤 **The trap that will bite you: the shell profile activates `vllm-py12` and puts it FIRST on PATH,
  so a bare `vllm` resolves to 0.28.0 even inside the image.** Always use the absolute
  `/usr/local/bin/vllm` + `/usr/bin/python3`. This also applies to the **client**: `bench.sh` defaults
  `VLLM=`/`PY=` to the conda env, and `vllm bench serve` reads the model config to tokenize — so the
  sweep fails in preflight on an unknown architecture *even when the server is healthy*. Override both:
  `VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 ./bench.sh ...`

  The older `nano-vllm` env (torch 2.11, transformers 4.57.1) is **too old for V4** — keep it only as the
  nano-vllm instrumentation vehicle. SGLang still not installed.
- **Blockers, RESOLVED — do not re-litigate:**
  (a) **transformers ≥5.0.0** — satisfied, `vllm-py12` has 5.16.1.
  (b) **FP4 on H100** — fine; storage format unpacked to FP8 in-kernel.
  (c) **`tilelang` / `fast_hadamard_transform` are NOT needed.** That was a requirement of DeepSeek's
  shipped `inference/` folder. **vLLM 0.28 supports V4 natively** via a full
  `vllm/models/deepseek_v4/` package (`attention.py`, `compressor.py`, `sparse_mla.py`, `quant_config.py`
  + `nvidia`/`amd`/`xpu` backends). No `convert.py` step, no second on-disk copy. This also supersedes
  the note predicting an SGLang-only FP8 route.
- **`bench/reduce.py`** — `AutoConfig` → truncate `layer_types` / `full_attn_layers` / `kda_layers` /
  `compress_ratios` consistently → `from_config` random init → **assert layer-type ratio preserved**.

## Datasets: settled and MEASURED (updated 2026-09-01)

We measure **throughput/TTFT/TPOT + bandwidth fraction, never accuracy**, so the dataset only needs
realistic token-length distributions and prefix-sharing structure. Tiering, in priority order:

| Tier | Dataset | Answers | Status |
|---|---|---|---|
| 1 primary | **synthetic `random`**, grid anchored to TraceLab percentiles | Q1 regime map, Q4 cost | ✅ used |
| 2 realism | **`UW-SyFI/TraceLab` v0.0.2** — Kan's own lab's trace | Q2 prefix cache, workload shape | ✅ percentiles used; replay not done |
| 3 comparability | ShareGPT, one run, ~500 reqs | matches published numbers | ⬜ not run |
| 4 free | `/metrics` spec-decode + KV counters | Q3 | ✅ used |

**`UW-SyFI/TraceLab` is public, ungated, CC-BY-4.0, 21.7 MiB** (665,453 rounds / 8,058 sessions,
arXiv 2606.30560). **Already downloaded — do not re-fetch.** ⚠️ It landed under a *different* cache root
than the models: `/prj/.../users/tanngo/cache/hub/datasets--UW-SyFI--TraceLab` (the models are under
`LLMs_serving_report/cache/hub/`). Point `HF_HOME` at the former, or pass the parquet path directly:
`.../datasets--UW-SyFI--TraceLab/snapshots/7256bbf7*/data/v0.0.2/rounds/train.parquet`.
Reading it needs `pyarrow` (installed in `vllm-py12`; there is no `pandas`/`polars`/`duckdb` in that env).
**Measured distributions** — use these to anchor any future grid:

| Metric | p50 | p90 | p99 | mean |
|---|--:|--:|--:|--:|
| `input_tokens_total` | **132,092** | 338,662 | 856,464 | 171,576 |
| `output_tokens` | **249** | 1,332 | 5,542 | 589 |
| `prefix_tokens` | 126,336 | 326,527 | 848,886 | 164,053 |

**ISL:OSL ≈ 530:1** at the median; **`prefix_tokens`/`input_tokens_total` = 95.6%** mean, with 98.8% of
rounds having prefix > 0; provider-reported cache reads cover only **57.6%** (a floor — Codex rounds
contribute 0 to the numerator). **Agentic serving is a prefill problem, and ~38 points of reuse are
unrealized.** ISL grid 16K/64K/128K/256K covers ~84% of real rounds.

**Do NOT make ShareGPT the primary dataset.** Its ~1K-token prompts have no prefix structure; at that
context KV cost is negligible and **every model in the comparison looks identical** — it would hide the
50× KV spread that is the report's thesis. Keep it as a Tier-3 comparability run only.

**TraceLab caveats (state them whenever the numbers appear):** token counts come from Claude/GPT
tokenizers, not DeepSeek's; prompt text is sanitized away so only *structure* is real; documented
`prefix_tokens` over-reporting (issue #22) means session-local replay must cap reusable prefix at
`previous.input_tokens_total + previous.output_tokens`; sessions are extremely skewed (rounds/session
p50 = 16, max = 21,351) — **subsample by session, not by round**.

For replay, vLLM's **`timed_trace`** dataset reconstructs prompts from `hash_ids` (chunk hash → seeded
tokens), so identical hashes produce identical tokens and *genuine* cache hits. `prefix_repetition` is
the synthetic control on the same axis (`--prefix-repetition-{prefix,suffix,num-prefixes}`).

## Phase order (updated: real-model measurement came FIRST, ahead of the layer-reduction ladder)

`plan.md` §6 sequenced GPU-free phases first. **That has been overtaken by events:** a full-size model was
available on 8×H100, so serving it directly beat building proxies. Current state:

- ✅ **Real-model measurement on DeepSeek-V4-Flash** — batch, context, prefix, and MTP A/B sweeps.
  ⚠️ on conda vLLM **0.28.0/cu129**, a different engine from the image everything else will use.
- ✅ **GLM-5.3-Flash SERVED and SWEPT on 8×H100 / cu130** — cached (308 GiB), server healthy on :8001,
  **batch + context + prefix sweeps collected** in `GLM-5.3-Flash/results/bf16kv/`. Both historical
  blockers closed (GPU-0 squatter; `-cu129` MHC segfault, gone on cu130). Four further bugs had to be
  fixed to boot and to stop producing wrong numbers — see the GLM-5.3-Flash section and
  [`fix_bug.md`](../../fix_bug.md). The **`fp8kv` arm is RESOLVED as impossible-by-architecture** (GLM is
  NoPE, `fp8_ds_mla` needs `pe_dim==64`) — a recorded result, not a gap. ✅ `report.md`, `README.md`
  and the `mtp` A/B arm (n=1 and n=5) are all **done as of 2026-09-02**.
- ✅ **DeepSeek-V4-Flash re-sweep in the `glm53-flash` image** — DONE, `results/mtp-off-image/`, 11
  points. **Engine effect measured at 1.01×**, so the GLM↔V4 comparison is engine-matched. The
  requested BF16-KV variant remains **impossible on SM90** (backend gated to Blackwell).
- ✅ **Qwen3.8-Flash-Next-FP8 — SERVED AND SWEPT on 8×H100 TP8 (2026-09-02).** 15 points:
  11 base (`results/base/`) + 4 MTP n=1 (`results/mtp-n1/`), all cold. `report.md` + `README.md`
  written. **176.94 B served / 7.27 B active [M]** from 152,089 tensor shapes.
  ⚠️ **NOT the dense control** — it is a 512-expert MoE at E/k = 51.2× (the report's most extreme
  sparsity point). It IS the only **KV-dtype match for GLM-5.3** (both BF16-only). ⚠️ It ran on a
  **different vLLM build** (`0.1.dev20073` in `vllm/vllm-openai:qwen38-flash-next`) — a bridge arm
  via V4-Flash (registered in that image) closes the confound; see WORKFLOW.md §2.
- ⬜ **GLM-4.7-Flash** (29 GiB, 1 GPU, full L=47) — cheapest complete architecture, no layer-reduction
  caveat needed. Not downloaded.
- ⬜ **Layer reduction** (`bench/reduce.py`) — still the sanctioned proxy for K3 / V4-Pro, which cannot
  fit. Phase 5's linearity gate still **gates any extrapolation**: don't trust reduced-L numbers before
  the fit exists. **Prefer full-size measurement whenever the model fits** — it needs no validation gate.

**Deliverables: all written.** ✅ `GLM-5.3-Flash/report.md` + `README.md`,
✅ `Qwen3.8-Flash-Next-FP8/report.md` + `README.md`, ✅ root `report.md`, ✅ `WHY.md`, `REPRODUCE.md`,
`fix_bug.md`, plus a `RESULT-*.md` per non-obvious arm. **The remaining work is measurement, and it is
now cross-model rather than GLM** — re-poll V4/Qwen with the fixed MFU harness, MTP context axis on
V4/Qwen, TP/EP sweep on V4 or Qwen, and a dense control. See WORKFLOW.md §2.

## Existing commands

```bash
export HF_HOME=/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/LLMs_serving_report/cache
export HF_HUB_CACHE=$HF_HOME/hub   # V4-Flash 150 GiB · GLM-5.3-Flash 308 GiB · Qwen3.8 174 GiB
                                   # ALL THREE already cached — do not re-download

# --- DeepSeek-V4-Flash (conda vllm-py12; measured baseline lives here) ---
cd deepseek_v4_flash && ./run_nomtp.sh   # base model (PRIMARY) — ~255 s to /health 200
cd deepseek_v4_flash && ./run.sh         # MTP on (A/B arm only)

# --- GLM-5.3-Flash (IMAGE vllm only; needs the sleep-infinity container) ---
cd GLM-5.3-Flash && ./run.sh             # BF16 KV, headline. serves on :8001
cd GLM-5.3-Flash && ./run_fp8kv.sh       # fp8_ds_mla arm — may fail by design
cd GLM-5.3-Flash && ./run_mtp.sh         # MTP A/B
cd GLM-5.3-Flash && ./sending.sh --wait  # smoke test (polls; 62 shards load slowly)

./sending.sh            # V4 functional smoke test (run from repo ROOT, port 8000)
./bench.sh batch context prefix    # from repo ROOT; auto-routes to <model>/results/

# GLM sweep — BOTH overrides are load-bearing (see the env table: bench.sh defaults
# to conda vLLM 0.28.0, which cannot parse glm5_next and dies in preflight even
# when the server is perfectly healthy)
MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/bf16kv \
  ./bench.sh batch context prefix
```

There is no build/lint/test framework, and none is needed — `sending.sh` is the functional check and
`bench.sh` the measurement harness. **Only one server can hold the GPUs at a time**: `pkill -f "vllm
serve"` and wait ~25 s for HBM to free before switching arms. ⚠️ **In the `glm53-flash` image, never
`pkill -f "vllm serve"` unqualified** — PID 1 matches that pattern and killing it restarts the container
(see the blocked-on-GPU-0 section). Match on the model name instead: `pkill -f "GLM-5.3-Flash"`.

nano-vllm remains available as the prefix-cache instrumentation vehicle:

```bash
cd ../arp/nano-vllm && python bench.py    # edit `path` to a local model dir first
```

`nanovllm.config.Config` asserts `kvcache_block_size % 256 == 0` and `1 <= tensor_parallel_size <= 8`,
and requires `model` to be a local directory (not a HF repo id). **Note:** that `% 256 == 0` assertion is
a *nano-vllm* constraint — do not use it to reason about vLLM's block size, which for V4 is pinned to 256
by `sparse_mla.py:53` for an unrelated reason (see the eval-process section).

## Deliverable layout: ONE SUBFOLDER PER MODEL

**Convention (established 2026-09-01):** each model gets a directory named after a slug of its HF repo
name, holding the launch configs that were actually run, the raw results, the logs, and its report.
Shared tooling stays at the repo root. This keeps "which config produced these numbers" answerable
months later, and scales to the other models without a monolithic results tree.

```
LLMs_serving_report/
├── CLAUDE.md  plan.md  task.md  model_list.md
├── potential_questions.md        # anticipated lab questions (cross-model)
├── bench.sh  sending.sh          # SHARED harness — model-agnostic
├── datasets/                     # shared downloads (ShareGPT etc.)
├── cache/                        # HF_HOME; V4-Flash already here (150 GiB)
└── deepseek_v4_flash/            # ← one of these per model
    ├── README.md                 # what ran, why each non-default flag exists
    ├── report.md                 # the deliverable for THIS model
    ├── run_nomtp.sh              # base model — PRIMARY numbers
    ├── run.sh                    # MTP on — A/B arm only
    ├── results/{mtp-off,mtp-on,mtp-on-noreuse}/
    └── logs/                     # server startup + sweep console output
```

⚠️ **The naming convention drifted, and the ACTUAL dirs win.** The slug rule below describes
`deepseek_v4_flash/`, but the newer model dirs use the **HF repo name verbatim**:
`GLM-5.3-Flash/` and `Qwen3.8-Flash-Next-FP8/` (both exist on disk). Since these do **not** match
`bench.sh`'s auto-slug (`GLM-5.3-Flash` → `glm_5_3_flash`), **`MODELDIR=` must be passed explicitly**
for them or results land in a new wrong-named directory. Still planned: a dir for GLM-4.7-Flash
(29 GiB, 1 GPU) as the cheapest cross-architecture point.

**`bench.sh` is model-aware:** `MODELDIR` defaults to a slug of the served model name
(`deepseek-ai/DeepSeek-V4-Flash` → `deepseek_v4_flash`), so results land in the right per-model folder
automatically. Override with `MODELDIR=` or `OUTDIR=` — **required** for the two verbatim-named dirs
above. Run it **from the repo root**, not from inside a model dir.

**Per-model README must record:** the exact flags run, and for each non-default flag *why it is not a
free choice*. For V4-Flash that is `--block-size 256` (mandatory, `sparse_mla.py:53`),
`--gpu-memory-utilization 0.82` (0.9 OOMs at warmup), `--kv-cache-dtype fp8`, and the reasoning/tokenizer
parsers. For GLM-5.3-Flash: `--block-size 128` (mandatory, `index_kpool=4`), the **absent**
`--kv-cache-dtype` (BF16 forced by FlashInfer 0.6.17), the **absent** `--trust-remote-code` (natively
registered), `--limit-mm-per-prompt` zeros (fairness: skips the ViT), and port 8001. Also record choices
that were **not** deliberate — TP=8 on a model that fits in 3 GPUs makes
per-GPU numbers pessimistic, and that has to travel with the numbers.
