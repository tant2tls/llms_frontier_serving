> ARCHIVED SESSION RECORD — not current instructions or conclusions. Original location: `fix_bug_previous.md`. See [current report](../../report.md), [debugging guide](../../fix_bug.md), and [session handoff](../../WORKFLOW.md). Historical claims may be superseded.

# Historical bug notes — superseded by fix_bug.md; retained for provenance

**Bugs 1–8: GLM-5.3-Flash**, container `tan-8gpus-glm53-0-0`, image
`vllm/vllm-openai:glm53-flash` (CUDA **13.0**, torch 2.13.0+cu130, vLLM
`0.1.dev20051+g487ecf187`), model `zai-org/GLM-5.3-Flash` (306 GiB, TP=8, EP on).

**Bugs 9–11: Qwen3.8-Flash-Next-FP8 + the V4 bridge arm** (2026-09-02, later session),
container `tan-8gpus-qwen38-0-0`, image `vllm/vllm-openai:qwen38-flash-next` (CUDA 13.0,
vLLM **`0.1.dev20073+g8e685d198`**). **Bug 10 is the highest-value one in this file** — it is
the third recurrence of a single failure mode, and it produced a wrong headline twice.

**Bug 12: Qwen3.8** (KV mis-sizing from a cold `torch.compile`). **Bug 13: GLM-5.3-Flash**
(2026-09-02, session 3, back in `vllm/vllm-openai:glm53-flash`) — a flag that was passed on
every arm of all three models and silently measured nothing. **Bug 13 corrected a published
number by an order of magnitude**, so read it before quoting any "% of roofline" figure.

Written as a teaching document. Each bug has: the **error as it appeared**, the
**real cause**, **how it was localized**, the **fix**, and the **transferable lesson**.

Starting state: seven prior launch attempts, zero benchmark points collected.
Ending state: server healthy, **11 measured sweep points** (batch + context +
prefix), the FP8-KV arm resolved as impossible-by-architecture, all bugs below fixed
in the scripts.

**Bugs 1-3 blocked the server from starting. Bugs 4-7 were producing wrong or
mislabelled NUMBERS from a server that looked fine** -- the more dangerous class,
because nothing crashes. **Bug 8 is a failure that was correctly predicted for the
wrong reason.**

---

## The single most important lesson

> **Nine of the twelve bugs presented as an error that named the wrong subsystem,
> as no error at all, or with a plausible-but-wrong cause.**

- A CUDA error that was a **disk quota**.
- A "Mamba cache" error on a model nobody had labelled as a hybrid.
- A CUDA-version bug (from the previous session's notes) that was **already gone**.
- A prefix-cache warning that was the **benchmark harness poisoning itself**.
- A benchmark point that "passed" with **every metric at 0.00** and exit code 0.
- An FP8 arm that failed as predicted, but from a **NoPE checkpoint geometry**
  rather than the block-size conflict that had been reasoned out in advance.
- A guard that fired with the **right instinct and the wrong label** (Bug 9).
- Benchmark points that were **cold, complete, and 62% wrong** (Bug 10) — twice.
- A **mis-sized KV pool that I explained architecturally instead of fixing** (Bug 12).

The generalizable habit: **the error text names where the process DIED, not what
was WRONG.** Always read the frame *above* the one that raised, and prefer the
error that appears *earliest* in the log over the loudest one.

---

## Bug 1 — `max_num_seqs (1024) exceeds available Mamba cache blocks (512)`

### Error
```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (512).
Each decode sequence requires one Mamba cache block, so CUDA graph capture
cannot proceed. Please lower max_num_seqs to at most 512 or increase
gpu_memory_utilization.
```
Raised from `vllm/config/compilation.py:1507` via `determine_available_memory`.

### Two surprises, both load-bearing

**(a) GLM-5.3-Flash is a HYBRID model.** Nothing in the model name says so.
`config.json` → `text_config.layer_types` is 45 entries:

```
34 × "linear_attention"            (KDA — Kimi Delta Attention)
11 × "deepseek_sparse_attention"   (DSA, index_topk 2048)
```
in a repeating 3:1 pattern (`linear_attn_config.kda_layers` lists exactly which).

The 34 KDA layers carry a **recurrent state**, which vLLM manages as "Mamba"
blocks: **one block per decode sequence**, allocated up front, *not* paged like
token KV. So **max concurrency is capped by state blocks, not KV tokens** — a
structurally different constraint from DeepSeek-V4-Flash (pure sparse MLA, no
recurrent state). This is a reportable *architecture* finding, not a config nit.

**(b) `--max-num-seqs` does NOT default to 128 on this hardware.** The dataclass
default is 128, and the project's notes said 128 — but
`arg_utils.py:2547 get_batch_defaults()` overrides it *by device*: any GPU with
≥ 70 GiB that is not an A100 gets **1024**. H100-80GB hits that branch.

```python
elif device_memory >= 70 * GiB_bytes and "a100" not in device_name:
    default_max_num_seqs = {LLM_CLASS: 1024, OPENAI_API_SERVER: 1024}
```

### Fix
`--max-num-seqs 256` in `_common.sh`.

**Why 256, not the 512 the error suggests.** 512 is the measured ceiling *at this
util on this node*, and it is what the cudagraph check compares against — sitting
exactly on the boundary means any small change in free HBM (different node, driver
bump, fragmentation) fails the launch again. 256 gives 2× headroom and still
cannot clip any measured point: the sweep grid tops out at concurrency **64**.

### Lessons
1. **A documented default is not the effective default.** Verify against
   `get_batch_defaults`-style hardware branches, not the dataclass.
2. **Read `layer_types` before assuming a model's attention structure.** "Flash"
   in a name tells you nothing.
3. **Don't set a limit to exactly the value an error reports** — that is the
   boundary that just failed. Leave headroom.

---

## Bug 2 — `[Errno 122] Disk quota exceeded` (Triton cache)

### Error
Died ~8 minutes into startup, *after* loading all 62 shards:
```
RuntimeError: Worker failed with error '[Errno 122] Disk quota exceeded'
  from triton/runtime/cache.py:120  ->  with open(temp_path, mode) as f
```

### Cause
`$HOME=/usr2/tanngo` is NFS with a **~5 GB per-user quota, and it was full.**
Triton defaults to `$HOME/.triton`; the inherited env also pointed Inductor at
`TORCHINDUCTOR_CACHE_DIR=/usr2/tanngo/.torchinductor_cache` (1.2 GB by itself).

### Why it was hard to see
**`df` shows the volume, not the quota.** `df -h /usr2/tanngo` reported **622 G
available** (93% used) — the disk looks fine. Worse, the usual "is it writable"
checks all **pass** on a quota-full directory:

| check | result on quota-full dir |
|---|---|
| `[ -w $dir ]` | ✅ passes |
| `mkdir -p $dir` | ✅ passes |
| `touch $dir/f` | ✅ passes (0 bytes) |
| `dd ... bs=1M count=20` | ❌ **fails** ← the only reliable probe |

### Fix
Node-local `/tmp` (overlay, 2.3 T free), keyed by uid **and CUDA major** so a
cu129 and a cu130 container never share compiled binaries:
`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT`, `XDG_CACHE_HOME`.
Plus a `preflight_caches()` that **actually writes 4 MB** and refuses to launch
otherwise — failing in 2 seconds instead of 8 minutes.

`~/.bashrc` was separately pointed at vol22 for everyday work.

### Lessons
1. **Quota ≠ free space.** `df` cannot see a quota; neither can `-w`.
2. **Probe by doing the real operation.** Any cheaper check gives a false pass.
3. **Fail fast on cheap invariants.** An 8-minute weight load before a
   2-second-checkable error is a waste; preflight it.
4. `TORCHINDUCTOR_CACHE_DIR` is assigned **unconditionally** (no `:-` default) —
   the inherited value is known-bad, so respecting it would reintroduce the bug.

---

## Bug 3 — `CUDA error: invalid argument` — THE INSTRUCTIVE ONE

### Error
Startup died at ~69–78% of `Profiling CUDA graph memory (PIECEWISE)` with a bare,
**tracebackless** line, ×7 workers:
```
CUDA error: invalid argument
Worker proc VllmWorker-1 died unexpectedly (exit code: None)
```

### Three wrong hypotheses (and the ~40 min they cost)

The error says CUDA, during cudagraph capture, so I chased cudagraphs:

| Hypothesis | Test | Result |
|---|---|---|
| Experimental breakable-cudagraph path | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | ❌ Verified OFF, **failed at the same index** |
| Capture ladder walks past `max_num_seqs` | `--max-cudagraph-capture-size 256` | ❌ Still failed, now at index 24 |
| Cudagraph capture broken for this model | `--enforce-eager` (skips capture) | ❌ **Still failed** |

Eager mode disproving it was the key signal: if disabling cudagraph capture
*entirely* doesn't help, **the bug was never in cudagraph capture.**

A misleading clue nearly derailed this: the failing capture size *moved* (352 →
208) between runs, which looks like memory pressure. It was an artifact of the
ladder getting shorter.

### Real cause — one frame above the CUDA error
```
tvm.error.InternalError: filesystem error: cannot create directories:
Disk quota exceeded
[/usr2/tanngo/.tensorrt_llm/tmp/gemm_swapAB_3072_4096_128_16_128_1_81_8_1_...]
  from fp8_blockscale_gemm_sm90 -> run_flashinfer_deepgemm_swapAB
```

**Bug 2 again, through a cache dir I had missed.** DeepGEMM JIT-compiles the FP8
block-scale GEMM into `$HOME/.tensorrt_llm`; the quota write fails, the GEMM
launches an **unbuilt kernel**, and CUDA reports `invalid argument` downstream.

It survived my Bug-2 fix because the path is built **in C++**, so no Python-level
cache setting touches it:
```cpp
// flashinfer/data/csrc/nv_internal/tensorrt_llm/deep_gemm/compiler.cuh:65
char const* cacheDir = getenv("TRTLLM_DG_CACHE_DIR");
if (!cacheDir) userDir = std::filesystem::path(getenv("HOME")) / ".tensorrt_llm";
```

### Fix
`TRTLLM_DG_CACHE_DIR` + `FLASHINFER_WORKSPACE_BASE` (also `Path.home()`-derived,
`flashinfer/jit/env.py:59`) → `/tmp`, and added both to `preflight_caches()`.
**Server booted, cudagraphs and all.** No eager fallback needed.

### Lessons
1. **A CUDA error is often a downstream symptom.** A failed JIT compile surfaces
   as `invalid argument` at launch. Read *up* the stack.
2. **When a fix that should work doesn't, question the diagnosis, not the fix.**
   Eager mode failing was proof the theory was wrong — that is a *result*, not a
   dead end.
3. **grep the earliest error, not the loudest.** `CUDA error` appeared 7× and was
   noise; the real cause appeared once, 1 line earlier.
4. **Env-var overrides can live in C++.** Grep the whole package, `.cuh` included.
5. **One root cause can have many faces.** Bugs 2 and 3 were the same quota.

---

## Bug 4 — the previous session's CUDA-13 diagnosis was already stale

The prior notes named the blocker as image tag `-cu129`, where MHC TileLang
kernels segfaulted in `cuModuleLoadData`.

**Checked instead of assumed.** In the most recent log:
```
Segfault encountered      : 0
mhc_post_tilelang compiles: 24 successful, across 8 workers
```
The cu130 image compiles MHC fine. That blocker was **gone**; the *real* remaining
blocker was Bug 1, visible in the same log.

⚠️ **My own near-miss, worth copying.** My first grep was
`grep -c "Segfault encountered\|mhc_post_tilelang"` — an **OR** that also matched
the *successful* compile lines, reporting "3 of 4 logs segfaulted" when the true
count was 2 of 4. Separating the patterns changed the conclusion.

### Lessons
1. **Notes describe a past container.** Re-verify hardware/image claims first.
2. **A grep with `\|` can silently conflate success and failure.** Count patterns
   separately when the conclusion depends on which matched.

---

## Bug 5 — `bench.sh` mislabelled every non-DeepSeek result

`bench.sh` hardcoded V4-Flash's provenance into every result JSON:
```
quant=fp8-attn-dense+mxfp4-experts   kv_cache_dtype=fp8
```
Both **wrong for GLM-5.3**, which has **FP8** (not MXFP4) experts and **BF16**
(not fp8) KV. Silent: it corrupts the metadata, not the run — and the whole point
of the manifest is that results stay citable months later.

### Fix
`QUANT` / `KV_DTYPE` / `HW` env vars, and the manifest now **scrapes
`cache_config_info` from the live server** rather than asserting from a comment.
That immediately confirmed something worth knowing:

```
block_size = 640          # requested 128; vLLM AUTO-RAISED it
mamba_block_size = 128
cache_dtype = auto -> BF16
num_gpu_blocks = 3064
```

`--block-size 128` is the *minimum legal* value (kpool needs a multiple of 128),
but vLLM raised it to **640** so the attention page ≥ mamba page, then padded the
mamba page by 20.75%. 640 is still kpool-legal (640/4 = 160, 160 % 32 = 0).

### Lesson
**Provenance must be derived, not asserted.** A hardcoded metadata string is a
lie waiting for its second model.

---

## Bug 6 — `--num-warmups 1` poisons the prefix cache (harness self-contamination)

### Symptom
bench.sh's own guard fired on **every** batch point:
```
FAIL  new cache hits : 16000 -- NOT A COLD RUN, throughput is inflated
```

### Cause
`--num-warmups 1` draws its warmup request from the **same seeded prompt set** as
the measured run. With `enable_prefix_caching=True` (default), the measured run
re-reads the warmup's own blocks.

The number proves it: **16,000 = 25 blocks × block_size 640 = exactly one
ISL-16384 prompt**, constant at every concurrency. Constant ⇒ one point
contaminating *itself*, not cross-point leakage (which unique seeds already fix).

### Why it mattered — the bias is UNEVEN
```
c=1   16,000 / 131,072 prompt tokens = 12.2%
c=4   16,000 / 131,072               = 12.2%
c=16  16,000 / 524,288               =  3.1%
c=64  16,000 / 2,097,152             =  0.8%
```
A constant absolute error is a large share of a small point and a rounding error
on a big one — so it inflates exactly the **low-concurrency anchors** of the
"throughput vs batch size" curve, **flattening the measured scaling**. Reporting
that would have understated GLM's concurrency scaling.

### Fix
Removed `--num-warmups`. Contaminated points quarantined to
`results/_discarded-warmup-contaminated/` with a README (not deleted — they
document the effect). Rerun reports **`cold run confirmed (0 new prefix-cache
hits)`** at every point.

### Lessons
1. **Warmup and measurement must not share a prompt set** on a caching engine.
   Either skip warmup or warm with disjoint prompts.
2. **A constant-magnitude error is not a harmless one.** Ask what *fraction* of
   each point it is; uneven relative bias distorts the shape of a curve, which is
   usually the actual finding.
3. **Keep the guardrail that catches you.** The `newcachehits == 0` assertion —
   added after an earlier phantom-throughput incident — paid for itself again.
4. **Quarantine, don't delete, bad data.** It is the evidence for the fix.

---

## Bug 7 — a benchmark point that "succeeded" with every metric at 0.00

### Symptom
The context sweep's top point printed a clean-looking summary row:
```
ctx_isl262144_c8    0.0    0.0    0    0.0    0.000
```
and `bench.sh` **did not flag it**. `vllm bench serve` had **exited 0**.

### Cause
Pure arithmetic: **ISL 262,144 + OSL 256 = 262,400 > `--max-model-len 262144`**, by
exactly 256 tokens. All 16 requests came back `Bad Request`:
```
UserWarning: All requests failed. This is likely due to a misconfiguration...
Error 0: Bad Request      (×16)
Successful requests: 0
```
`vllm bench serve` does **not** pre-validate ISL+OSL against the server's
`max_model_len`, and — the real trap — it **returns exit code 0** after warning.
So `rc != 0` never fired, the JSON was saved with `completed: 0`, and the point
entered the results table as a plausible-looking datapoint.

I first suspected the server had died at long context. A one-line probe disproved
that: a normal chat request answered fine while the sweep was failing.

### Fix
Two changes:
1. `sweep_context` now uses a **verified-working** top ISL instead of a computed one.
2. A **zero-completion guard** in `run_point`: read `.completed` from the result
   JSON, and if it is 0, fail the point, print the errors, rename the file to
   `.failed`, and return nonzero.

⚠️ **My first fix was WRONG, and the guard caught me.** I "fixed" it by computing
`max_model_len - OSL` = 261,888 — arithmetically exact, and it **still 400'd on all
16 requests**. The guard fired again, which is the only reason I noticed instead of
shipping a second all-zero row. Bisecting with 1-prompt probes:

| ISL | result |
|--:|---|
| 262,144 | ❌ 400 (obviously over: +OSL) |
| 261,888 | ❌ 400 — *even though 261,888 + 256 == 262,144 exactly* |
| 260,000 | ✅ works |

`--dataset-name random` does **not** emit exactly `--random-input-len` tokens (it
jitters, and the chat template adds tokens on top), so a point sitting exactly on
the boundary still overflows. 260,000 is still 99.2% of `max_model_len`, which is
all the "256K context" claim needs.

### Lessons
1. **Exit 0 is not success.** A benchmark that warns and returns 0 will silently
   poison a results table. Validate the *output*, not the return code.
2. **All-zeros is a failure signature, not a datapoint.** Assert a minimum of real
   work (`completed > 0`) on every measurement.
3. **Exact arithmetic is not headroom.** A nominal length parameter is often a
   *target*, not a guarantee — leave slack and **verify empirically**.
4. **Bisect with the cheapest probe.** `--num-prompts 1` answers in seconds what a
   16-prompt 256K run takes many minutes to fail at.
5. A cheap independent probe (one chat request) separates "server is broken" from
   "this request is invalid" in seconds.
6. **A guard that fires twice has paid for itself twice.** Both times it stopped a
   fabricated number from entering the report.

---

## Bug 8 — the FP8-KV arm failed for a reason nobody predicted (and that's fine)

Not a bug in our code — a **correctly-predicted failure with an incorrectly-predicted
cause**, which is worth studying because the wrong prediction was well-reasoned.

### Predicted
`run_fp8kv.sh`'s header argued the arm would die on a **block-size conflict**:
`FLASHMLA_SPARSE` advertises kernel block size `[64]`, GLM's kpool indexer requires a
multiple of 128, so "the two constraints may be unsatisfiable" → `No common block size`.

### Actually happened
The backend resolved fine (block size auto-raised to 640, legal for both) and got all
the way to the first KV write:
```
RuntimeError: concat_and_cache_mla, cache_kernels.cu:866,
              pe_dim must be 64 for fp8_ds_mla
```

### Real cause — an architectural mismatch, not a version gate
`fp8_ds_mla` is DeepSeek-V3.2's KV layout and hardcodes a decoupled-RoPE dim of 64.
**GLM-5.3 is a NoPE model**: `qk_rope_head_dim = 0`, `mla_use_nope = true`. So
`pe_dim = 0 ≠ 64` and **no flag can change it** — FP8 KV via this layout is
unreachable for this model on *any* hardware, not just Hopper.

### Lessons
1. **A right conclusion from a wrong premise is still a wrong model of the system.**
   "This arm will fail" was correct; "because of block sizes" was not. Had I only
   recorded the outcome, the note would have taught the wrong lesson.
2. **Prefer the constraint that comes from the checkpoint over the one from the
   toolchain.** Version gates move; attention geometry does not. The NoPE finding is
   permanent and much more useful than "FlashInfer is too old".
3. **Check a config field before blaming a kernel.** `qk_rope_head_dim: 0` was in
   `config.json` the whole time and settles it in one line.
4. **A clean failure is a deliverable.** Recorded in
   `GLM-5.3-Flash/results/fp8kv/RESULT-arm-impossible.md` with the exact error, the
   config evidence, and a do-not-retry list.

---

---

## Bug 9 — the queueing guard fires at concurrency 1, where queueing is impossible

**Error as it appeared** (Qwen3.8, base arm, first point of the sweep):

```
batch_isl16k_c1  ISL=16384 OSL=256 conc=1 prompts=8 seed=32239
  note     QUEUEING SUSPECT: p99/median TTFT = 7.81x at conc=1 (limit 4x).
  note     Throughput is probably UNDERSTATED -- rerun before citing.
  output tok/s   : 84.158604
  ok       cold run confirmed (0 new prefix-cache hits)
```

**Why the label cannot be right.** At `conc=1` the client keeps **one** request in flight, so requests
are served strictly sequentially. **There is no queue to wait in.** The guard's diagnosis was
structurally impossible for this point.

**How it was localized.** The distribution gave it away without needing the server at all:

```
median_ttft_ms = 602      mean_ttft_ms = 1264      p99_ttft_ms = 4702      completed = 8
```

With 8 sequential requests, a mean sitting *between* median and p99 means **exactly one outlier**. The
first request is the only one that can be special — it pays lazy JIT compilation (Triton/DeepGEMM
kernels for shapes not covered by the cudagraph capture ladder).

**The fix — re-run the same point on the now-warm engine** (`bench.sh` skips existing results, so
deleting just that one file re-runs only it):

| | first (cold) | re-run (warm) |
|---|--:|--:|
| output tok/s | 84.2 | **107.7** |
| median TTFT | 602 ms | **602 ms** |
| p99/median | 7.81× | **1.07×** |

**The median TTFT is identical.** Only the first-request tail moved — which confirms JIT warmup and
rules out queueing, contention, or a throttled GPU.

Same thing happened on the MTP arm: 90.4 → **125.9 tok/s**. Had the flagged points been published, the
**MTP gain at c=1 would have read 1.07× instead of 1.17×** — i.e. the wrong conclusion about the *one*
concurrency where MTP actually helps this model.

**Fix applied:** quarantined both cold points as evidence (`results/_base_c1_jitcold/`,
`results/_mtp_c1_jitcold/`) rather than deleting them, and published the warm ones.

**Transferable lesson.** **A guard firing is information, not a verdict — check whether its stated
mechanism is even possible for that point.** Here the guard was *useful* (the number really was 22–28%
low) but its *explanation* was wrong, and acting on the explanation ("scheduler contention") would have
sent me to `max_num_seqs` and chunked-prefill settings instead of to a one-line re-run. Also:
**median-vs-mean-vs-p99 tells you the shape of the outlier for free** — one slow sample pulls the mean
off the median without moving it.

---

## Bug 10 — ⚠️ THE IMPORTANT ONE: cold, complete points that were 62% wrong

**Third recurrence of this failure mode in this project, and it produced a wrong headline twice.**

**Error as it appeared: none.** The V4-Flash bridge arm ran to completion, every point passed the cold
check and the completeness check, exit code 0. The numbers were simply wrong:

| conc | first pass | p99/median TTFT | re-run warm | understated by |
|--:|--:|--:|--:|--:|
| 4 | 89.3 | **7.96×** | **219.4** | **−59%** |
| 16 | 124.0 | **8.25×** | **330.0** | **−62%** |
| 64 | 284.0 | 10.4× | **386.4** | −26% |

**What it would have caused.** This arm exists to measure one thing: the vLLM build effect
(`dev20051` → `dev20073`), which decides whether the Qwen3.8 cross-model ratios are publishable. With
the first-pass numbers the build effect computes to **0.73×** — *"the newer engine is 27% slower"* — a
confident, wrong, load-bearing headline. **Measured correctly it is 0.997×.** The earlier occurrence of
this same bug made a different arm read 0.64× instead of 1.01×.

**Real cause.** This build adds a **DeepGEMM warmup pass that `dev20051` does not have**:

```
DeepGEMM warmup:  47%|████▋     | 594/1261 [03:10<02:52,  3.87it/s]
...
init engine (profile, create kv cache, warmup model) took 538.8 s
```

1,261 kernels, ~5 minutes. **`/health` returns 200 before this finishes.** So a sweep that starts the
moment the server reports healthy benchmarks a **half-warm engine**, and the first several points absorb
JIT compilation *while being timed*.

**How it was localized.** `bench.sh`'s concurrency-aware guard (added after the previous recurrence)
flagged the two worst points via **p99 TTFT ≫ median**. The signature distinguishes this from a genuinely
slow engine: a uniformly slower engine raises median and p99 together, whereas JIT/queueing raises only
the tail. Cross-checking against the same points on `dev20051` (219.8, 330.9) confirmed the *warm*
re-runs matched to within 0.3% while the cold ones did not.

**Fix.** Quarantine (`results/mtp-off-bridge-dev20073_queued/`), re-run the flagged points on the warm
engine, and record the whole episode in the arm's `RESULT-engine-bridge.md` so the correction is
auditable.

**Transferable lesson — the one to actually remember:**

> **`/health 200` is not a readiness signal for benchmarking.** It means the API server is accepting
> requests, not that the engine has reached steady state. Warmup passes (DeepGEMM, FlashInfer autotune,
> lazy Triton compiles) can run for *minutes* afterwards.

Practical guards, in order of value:
1. **Check `p99/median TTFT` on every point before citing it.** Cold + complete is not sufficient.
2. **Discard or re-run the first point of any sweep**, or issue a throwaway warm-up request that is *not*
   drawn from the measured prompt set (drawing it from the same seeded set caused Bug 6).
3. **When a measurement's purpose is to compare two configs, sanity-check the direction against a point
   you already trust.** Here `dev20051`'s existing numbers were the trusted anchor; a 59% gap on an arm
   whose only variable was a minor version bump is not plausible and should trigger suspicion before it
   triggers a writeup.

---

## Bug 11 — `sending.sh` dies on a missing `bc`, and the model looks broken

**Error as it appeared:**

```
vLLM smoke test  http://localhost:8000  model=deepseek-ai/DeepSeek-V4-Flash
missing required tool: bc
```

**Two distinct problems in one line, and the second is the dangerous one.**

1. **`bc` is not installed in the `qwen38-flash-next` image** (it was present in `glm53-flash`).
   `sending.sh` uses it only for float arithmetic in its own reporting, so the dependency check aborts
   the smoke test before a single request is sent. `bench.sh` does **not** need `bc`, so the sweeps were
   unaffected — but at that moment I had no functional confirmation the model could answer.
2. **The defaults point at the wrong server.** `sending.sh` defaults to `PORT=8000` and
   `MODEL=deepseek-ai/DeepSeek-V4-Flash`; Qwen3.8 serves on **8001**. Note what the banner says: it was
   about to smoke-test **a different model on a port this project deliberately avoids** (8000 is where the
   image entrypoint's Qwen3-0.6B squatter lives in the GLM image). **A smoke test that passes against the
   wrong server is worse than one that fails.**

**Fix used:** verified the model directly instead of patching the shared harness mid-run —

```bash
curl -s http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"Qwen/Qwen3.8-Flash-Next-FP8",
 "messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],
 "max_tokens":600,"temperature":0}' | jq -r '.choices[0].message'
# -> content "\n\n4", reasoning "We need to answer...", finish_reason "stop"
```

This confirmed three things at once: the model answers correctly, `--reasoning-parser qwen3` is splitting
reasoning into `message.reasoning`, and 600 max_tokens is enough that reasoning does not consume the whole
budget (it used 32 reasoning tokens of 36).

**Transferable lesson.** **Tooling assumptions do not survive an image change** — a container swap can
remove a coreutil you never thought about. And when a smoke test defaults to a host/port/model triple,
**check the banner it prints against what you intended to test**: the failure mode where it silently
validates the wrong server produces a green check and zero information. Prefer explicit
`PORT=`/`MODEL=`, and prefer a one-line `curl` you can read over a script whose defaults you have to
remember.

---

## Bug 12 — a mis-sized KV pool, and I explained the artifact instead of fixing it

**The worst kind of bug in this file: it produced a wrong number AND a plausible story to justify it.**

**Symptom: none.** Qwen3.8's batch sweep completed, every point cold and complete, no warning. But c=4
read **162.2 tok/s** where GLM read 225.3 — so Qwen appeared to *lose* by 0.72× at exactly one
concurrency while winning at the other three. I wrote an architectural explanation into the report:
*"a prefill effect — 512-expert routing plus a 4-token block size gives the scheduler finer-grained work
to place, and at low concurrency that overhead is not amortized."* It was coherent, it fit the TTFT and
TPOT columns, and it was **wrong**.

**How it was found: by testing an assumption I had only asserted.** Three reports stated that
`--gpu-memory-utilization` "changes KV capacity, not the decode cost model." That was a *prediction*
never measured, and it was the last unmatched provisioning variable across the three models (Qwen 0.85
vs GLM/V4 0.82). Re-running Qwen at 0.82 gave:

| conc | util 0.85 | util 0.82 | ratio |
|--:|--:|--:|--:|
| 1 | 107.7 | 106.2 | 0.99× |
| 4 | 162.2 | **256.8** | **1.58×** |
| 16 | 371.0 | **420.2** | 1.13× |
| 64 | 517.8 | 517.7 | 1.00× |

**The first clue that the flag was not the cause: LOWERING utilization gave 1.56× MORE KV**
(2,048,645 → 3,197,331 tokens). That is backwards, so something other than the flag was setting pool size.

**Real cause.** vLLM computes `kv_pool = util × total − weights − peak_activation − cudagraph`, measuring
`peak_activation` at startup. In the first run that measurement overlapped **cold `torch.compile`**:

| | util 0.85 run | util 0.82 run |
|---|--:|--:|
| weights + non-torch | 25.13 GiB | 24.75 GiB |
| **peak activation** | **17.07 GiB** | **0.99 GiB** |
| compilation | **60.41 s (cold)** | **1.03 s (warm)** |
| engine init | 132.05 s | 46.41 s |

**17× difference on identical weights and identical `max_num_batched_tokens=8192`.** Transient
compile-time allocation was charged to activation, so vLLM reserved ~14 GiB/GPU of KV it never needed.
**vLLM says so in its own log** and I had not read it closely enough:

```
Replace gpu_memory_utilization config with `--kv-cache-memory=24916765696` (23.21 GiB) to fit into
requested memory, or `--kv-cache-memory=35225873408` (32.81 GiB) to fully utilize gpu memory.
```

That second figure — 32.81 GiB against the 25.1 GiB actually used — was the bug, printed at startup,
in every run.

**Why only the middle concurrencies moved.** c=1 has nothing to schedule; c=64 saturates the machine
either way. Only mid-concurrency is KV-pressure-sensitive (preemptions during chunked prefill), which is
**exactly the regime where a capacity artifact is easiest to mistake for an architectural difference.**

**Fix.** Published the `results/base-util082/` arm for c=4 and c=16, corrected both reports, and recorded
the A/B in `results/base-util082/RESULT-util-ab.md`. Practically: **warm the compile cache before any run
whose purpose is to size KV**, or pin `--kv-cache-memory` to remove profiling from the critical path.

**Transferable lessons — two, and the second matters more:**

1. **Read the memory-profiling line, not just the KV total.** vLLM prints its own recommended
   `--kv-cache-memory`; if that differs materially from what it used, the pool is mis-sized. Cross-check
   `peak activation` against `compilation time` — a large activation figure alongside a slow cold compile
   is the signature.
2. 🛑 **A plausible mechanism is not evidence, and writing one down makes the error durable.** I had a
   real architectural story (fine-grained routing, small block size, low-concurrency overhead) that
   *predicted the observed shape*, so it felt confirmed. It survived because the anomaly appeared at one
   point and I explained it instead of re-measuring it. **When a single point breaks an otherwise
   consistent pattern, suspect the measurement before you theorize** — and be most suspicious when the
   theory is one you find satisfying. This is also the fourth appearance of "cold engine, valid-looking
   numbers" in this file (bugs 9, 10, 12), which is itself the signal: the failure mode is systemic in
   this stack, not incidental.

---

## Debugging patterns worth reusing

1. **Read the earliest error, not the loudest.** Sort by log line, not frequency.
2. **Read one frame above the raise.** The raiser reports the *consequence*.
3. **A failed disproof is information.** `--enforce-eager` not helping is what
   cracked Bug 3.
4. **Probe with the real operation.** Quota/permission checks lie otherwise.
5. **Fail fast on cheap invariants** — never after an 8-minute load.
6. **Verify inherited notes before trusting them**; containers are not stable.
7. **Separate grep patterns** when success and failure strings share a substring.
8. **Derive provenance from the live system**, never hardcode it.
9. **Distinguish "constant" from "harmless"** — check the relative share per point.
10. **Correct your own write-ups.** I initially wrote that capping the cudagraph
    ladder fixed the CUDA error; it did not, and the comment now says so. A note
    that records a wrong cause is worse than no note.
11. **Validate outputs, not exit codes.** Bug 7's harness warned and returned 0.
12. **Assert a floor of real work** on every measurement (`completed > 0`,
    `hits == 0`). Both guards in this harness caught a real bug within one session.
13. **A flag you set is not a measurement you made.** `--enable-mfu-metrics` rode
    along on every arm of all three models for two sessions and produced no data,
    because nothing read the gauges it populates (Bug 13). Grep for the *consumer*.
14. **Record which optional components were active alongside the number.** vLLM
    silently skips `ComponentMetrics` it cannot build, at `debug` level; the
    resulting partial coverage differs per model and cost an order of magnitude in
    a headline claim.

---

## Bug 13 — a flag that was passed on every arm and silently measured nothing

**Symptom.** `--enable-mfu-metrics` had been on **every arm of all three models** for two sessions,
specifically so that "achieved bandwidth" could be **[M]** instead of **[A]** (eval rule 5). Yet every
bandwidth number in the report was still analytical. Nothing errored. Nothing warned.

**Cause 1 — the harness never read the gauges.** `bench.sh` polled `vllm:kv_cache_usage_perc` during
each point but never touched `vllm:estimated_flops_per_gpu_total` /
`estimated_read_bytes_per_gpu_total` / `estimated_write_bytes_per_gpu_total`. The flag populated them
correctly; the flag was not the problem. **A flag that enables a metric does nothing if nothing reads
the metric** — and there is no failure mode to notice, because the *absence* of a JSON field looks
exactly like a field you never asked for.

**Cause 2 — they are COUNTERS, so a single scrape is meaningless.** `perf.py:1559` sets
`_counter_cls = prometheus_client.Counter`: these are monotonic totals since server start. Reading one
after a run gives cumulative work across every prior point. The fix takes a **delta** across each
point and divides by the benchmark's own `duration`.

**Cause 3 — ⚠️ THE ONE THAT MATTERS: coverage is model-dependent, and silently partial.** vLLM builds
its byte/FLOP estimate from `ComponentMetrics` classes instantiated per model. Any component that
fails to instantiate is **silently excluded** — `ModelMetrics.__init__` catches `InvalidComponent` and
logs at `debug` (`perf.py:1284`). Measured by grepping each model's startup log:

| model | components instantiated | missing |
|---|---|---|
| Qwen3.8-Flash-Next | `attn`, `ffn`, `unembed` | — (full) |
| **GLM-5.3-Flash** | `ffn`, `unembed` | **`attn`** (34 KDA + 11 DSA) |
| **DeepSeek-V4-Flash** | `unembed` only | **`attn` AND `ffn`** |

So the counters are a **lower bound whose gap differs per model** — i.e. **not comparable across
models** without that table. Comparing GLM's raw GB/s to V4's would compare "FFN + unembed" against
"unembed alone."

**Why the attention components drop out, and why it is the report's thesis in miniature.** Both
attention classes gate on one whole-model boolean: `AttentionMetrics` raises if
`model_config.is_deepseek_mla` is true, `MLAAttentionMetrics` raises if it is false
(`perf.py:403-408`, `:551-554`). A hybrid stack is neither, so **both** refuse. `perf.py:428` says so
outright:

> `# TODO: discern cases where we have mixture of different attention layer types such as SWA, MLA, etc.`

That TODO **is** this report's thesis: the uniform per-layer cost model breaks down inside a single
2026 model. vLLM's own config layer already knows it — `model_arch.py:119-122` carries a comment about
keeping `is_deepseek_mla` from collapsing to `any` across layers, because doing so would make
`use_mla` true model-wide and return 1 KV head for every layer. **The engine models heterogeneity in
its allocator but not in its performance accounting**, which is a concrete instance of the gap the
report argues about. Cite this rather than only citing config files.

**Fix.** `bench.sh` now snapshots all three counters before/after each point, divides by `duration`,
and folds into the result JSON: `achieved_tflops_per_gpu`, `achieved_gbps_per_gpu`,
`frac_peak_hbm_h100`, `frac_peak_flops_h100_fp8`, plus a `mfu_provenance` string. `manifest.txt` gains
an MFU-coverage section (pass `SERVER_LOG=<serve log>` to record the actual per-model component list).

**What it changed in the report.** The headline "decode runs at **~1% of the memory roofline**" was
from a hand-rolled `active_params × bytes × steps/s` estimate. vLLM's per-step accounting measures
GLM at **9.5–18.1% of 3.35 TB/s** — **an order of magnitude higher**, and that with attention excluded,
so the true figure is higher still. The *qualitative* conclusion survives (decode is latency-bound,
nowhere near the roofline) but the number was wrong and is now corrected.

⚠️ **Label these [E], not [M].** They are an **engine-side estimate**: config shapes × the step's real
batch composition (`perf.py:488-513`, `:1069-1127`). The batch/context term is measured; bytes-per-token
is modelled. Better than our own estimate — per-step, tracks the real prefill/decode mix, counts
activations and the unembed — but it is **not** a hardware counter. Only DCGM/Nsight would be, and
`dcgmi` is absent from this container.

**Transferable lessons.**
- **A flag you set is not a measurement you made.** Grep for a consumer of the metric, not just the
  flag. Two sessions of runs carried this flag and produced no data from it.
- **Counter vs gauge changes the arithmetic.** Check the Prometheus type before reading a `_total`.
- **When a library estimates something for you, read how.** "vLLM says 18%" and "the GPU read 18% of
  peak" are different claims; only one is defensible here.
- **Silent partial coverage is worse than a hard failure.** An `InvalidComponent` logged at `debug`
  cost a wrong order of magnitude in a headline. When consuming an optional-component API, **record
  which components were active alongside the number.**

---

## Config that actually boots (8×H100, CUDA 13.0)

```bash
# node-local JIT caches — NFS $HOME is quota-full (Bugs 2 & 3)
TAG=$(id -u)_${CUDA_VERSION%%.*}
export TRITON_CACHE_DIR=/tmp/triton_cache_$TAG
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_$TAG   # unconditional
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$TAG
export XDG_CACHE_HOME=/tmp/xdg_cache_$TAG
export TILELANG_CACHE_DIR=/tmp/tilelang_cache_$TAG
export TRTLLM_DG_CACHE_DIR=/tmp/trtllm_dg_cache_$TAG           # C++-resolved
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_ws_$TAG

/usr/local/bin/vllm serve zai-org/GLM-5.3-Flash \
  --port 8001 \
  --max-cudagraph-capture-size 256 \   # == max-num-seqs; dedicated flag, MERGES
  --block-size 128 \                   # min legal (kpool); vLLM raises to 640
  --max-num-seqs 256 \                 # NOT 1024 (H100 default) — Mamba blocks
  --tensor-parallel-size 8 --enable-expert-parallel \
  --gpu-memory-utilization 0.82 --max-model-len 262144 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --no-enable-flashinfer-autotune --enable-mfu-metrics
```

⚠️ **Use `--max-cudagraph-capture-size`, never
`--compilation-config '{"max_cudagraph_capture_size":256}'`.** The latter
**replaces** the whole `CompilationConfig`, which silently wiped `pass_config` to
`{}` (losing `fuse_norm_quant`, `fuse_act_quant`, `fuse_allreduce_rms`). The
dedicated flag is *merged* (`arg_utils.py:2452`) and is mutually exclusive with
the config key, so it cannot clobber siblings. I hit this and caught it only by
diffing `pass_config` between two runs' startup banners.

Verified live: `/health` 200 · BF16 KV · spec decode OFF · 2,099,654 KV tokens ·
23.26 GiB KV/GPU · 8.01× concurrency at 256K · answers correctly.
