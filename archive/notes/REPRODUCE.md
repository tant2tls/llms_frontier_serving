> ARCHIVED SESSION RECORD — not current instructions or conclusions. Original location: `REPRODUCE.md`. See [current report](../../report.md), [debugging guide](../../fix_bug.md), and [session handoff](../../WORKFLOW.md). Historical claims may be superseded.

# REPRODUCE.md — bring up GLM-5.3-Flash and re-measure it from scratch

Everything needed to reproduce `GLM-5.3-Flash/results/bf16kv/` on a fresh RunAI
container. Written for a future session with **no memory of this one**.

**Read [`fix_bug.md`](../../fix_bug.md) first if anything fails.** All seven bugs hit
during bring-up are documented there with the error text, so a symptom you see is
probably already diagnosed. Four of them named the wrong subsystem.

Total wall time from an empty container: **~10 min server startup + ~2.5 h sweeps**
(the 260K-context point alone is ~40 min).

---

## 0. TL;DR — the whole thing

```bash
# from a LOGIN HOST (runai CLI does not work inside a container)
cd LLMs_serving_report/GLM-5.3-Flash && ./submit_job.sh

# then exec into the container:
cd LLMs_serving_report/GLM-5.3-Flash
ps -p 1 -o args=          # MUST print `sleep infinity`
./run.sh                  # ~10 min. serves on :8001
./sending.sh --wait       # smoke test

cd ..                     # repo root
MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/bf16kv \
QUANT=fp8-attn-dense+fp8-experts KV_DTYPE=bfloat16 \
  ./bench.sh batch context prefix

./normalize.sh GLM-5.3-Flash/results/bf16kv    # fair cross-model numbers
```

**The five env overrides are all load-bearing.** Details in §3.

---

## 1. Prerequisites, and how to check each one

| Requirement | Check | If wrong |
|---|---|---|
| 8× H100-80GB | `nvidia-smi -L` | TP=8 is required — see §6 for why 7 or 4 don't work |
| **CUDA 13.0** image | `nvcc --version` → `release 13.0` | `-cu129` segfaults in TileLang MHC. Resubmit with `vllm/vllm-openai:glm53-flash` |
| PID 1 is idle | `ps -p 1 -o args=` → `sleep infinity` | If it says `vllm serve`, the entrypoint is squatting ~77 GiB on GPU 0. **Do not kill it** — see §6 |
| GPUs free | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` → empty | `run.sh` refuses to launch otherwise (`preflight_gpus`) |
| Weights cached | `du -sh $HF_HUB_CACHE/models--zai-org--GLM-5.3-Flash` → 308 G | Do **not** re-download; set `HF_HOME` per `_common.sh` |
| `/tmp` writable | `df -h /tmp` → TBs free | JIT caches go here; NFS `$HOME` is quota-full (§2) |

Hardware is **not** stable across RunAI sessions — always detect, never assume.

---

## 2. ⚠️ The NFS quota trap (cost the most time — bugs 2 and 3)

`$HOME=/usr2/tanngo` is NFS with a **~5 GB per-user quota that is FULL**.

**`df` cannot see it.** `df -h /usr2/tanngo` reports hundreds of GB available
because it describes the *volume*, not your quota. `[ -w ]`, `mkdir -p`, and
`touch` **all pass** on a quota-full directory. The only reliable probe:

```bash
dd if=/dev/zero of=$HOME/.triton/_probe bs=1M count=20   # → Disk quota exceeded
```

**Symptoms it causes, neither of which mentions "quota" or "home":**
- `RuntimeError: [Errno 122] Disk quota exceeded` from `triton/runtime/cache.py:120`,
  ~8 min into startup.
- A bare **`CUDA error: invalid argument`** during cudagraph profiling. The real
  error is one frame up: a DeepGEMM JIT write to `$HOME/.tensorrt_llm` failed, so
  the GEMM launched an unbuilt kernel.

`_common.sh` already pins **seven** cache dirs to node-local `/tmp`, keyed by uid
and CUDA major, and `preflight_caches()` writes 4 MB to each and refuses to launch
otherwise (2-second failure instead of 8-minute). Two of them are easy to miss
because they resolve off `$HOME` outside Python's reach:

```
TRTLLM_DG_CACHE_DIR         # built in C++: deep_gemm/compiler.cuh:65
FLASHINFER_WORKSPACE_BASE   # Path.home(): flashinfer/jit/env.py:59
```

`~/.bashrc` separately points everyday caches at vol22. **vol22 fixes the quota but
not fleet-shared state** — it is mounted in every container, so prefer `/tmp` for
anything that JIT-compiles CUDA kernels.

---

## 3. The five bench.sh overrides, and why each is load-bearing

```bash
VLLM=/usr/local/bin/vllm     # bench.sh defaults to conda vLLM 0.28.0, which does
PY=/usr/bin/python3          #   NOT register glm5_next -> dies in preflight even
                             #   when the server is perfectly healthy
MODEL=zai-org/GLM-5.3-Flash
PORT=8001                    # 8000 is the image entrypoint's Qwen3-0.6B. Hitting
                             #   it smoke-tests the WRONG MODEL and PASSES.
QUANT=fp8-attn-dense+fp8-experts   # bench.sh's defaults are DeepSeek-V4's
KV_DTYPE=bfloat16                  #   (mxfp4 experts, fp8 KV) -- both WRONG for
                                   #   GLM and they get stamped into every JSON
```

Omitting `QUANT`/`KV_DTYPE` does not fail — it silently mislabels every result.
That is worse.

---

## 4. Expected results (measured 2026-09-02, 8×H100, cu130, base model)

If your numbers differ by more than ~10%, something changed. Compare against
`GLM-5.3-Flash/results/bf16kv/manifest.txt`, which records the live server config.

**Server, verified at startup:**
```
block_size = 640        # requested 128; vLLM AUTO-RAISES it (attn page >= mamba page)
mamba_block_size = 128
cache_dtype = auto -> BF16
num_gpu_blocks = 3064
KV cache = 23.26 GiB/GPU, 2,099,654 tokens, 8.01x concurrency at 256K
weights = 38.08 GiB/GPU
spec decode = OFF
```

**Batch sweep** (ISL 16,384 · OSL 256 · TP8 · unique seed per point):

| conc | out tok/s | per GPU | per B-active | TTFT p50 | TPOT p50 | KV peak |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 12.04 | 6.42 | 857 ms | 7.1 ms | 0.014 |
| 4 | 225.3 | 28.16 | 15.01 | 1,742 ms | 10.9 ms | 0.051 |
| 16 | 359.2 | 44.90 | 23.93 | 2,685 ms | 33.6 ms | 0.201 |
| 64 | 447.1 | 55.88 | 29.78 | 2,513 ms | 129.5 ms | 0.809 |

**Context sweep** (conc 8 · OSL 256):

| ISL | out tok/s | per GPU | TTFT p50 | TPOT p50 | KV peak |
|--:|--:|--:|--:|--:|--:|
| 16,384 | 295.9 | 36.99 | 2,629 ms | 16.7 ms | 0.102 |
| 65,536 | 104.3 | 13.03 | 7,212 ms | 48.0 ms | 0.366 |
| 131,072 | 47.2 | 5.91 | 12,891 ms | 116.7 ms | 0.725 |

**Prefix sweep** (64K shared prefix, conc 8): n=1 → 265.3 · n=4 → **365.9** ·
n=16 → 199.4 out tok/s.

**Every point must print `cold run confirmed (0 new prefix-cache hits)`.** If it
says `NOT A COLD RUN`, the throughput is inflated — discard it (see §5).

---

## 5. Two guards that will fire on you, and what they mean

`bench.sh` asserts two invariants. Both caught real bugs in this session; do not
disable them.

**`FAIL new cache hits : N -- NOT A COLD RUN`.** The point read a prefix cache, so
its throughput is inflated. Caused by `--num-warmups` (removed — the warmup drew
from the *same seeded prompt set* as the measured run, so the run re-read the
warmup's own blocks) or by a repeated seed. A constant hit count across
concurrencies means self-contamination; it biases small points far more than large
ones (12.2% of c=1 but 0.8% of c=64) and **flattens measured concurrency scaling**.

**`FAIL <point> completed 0 of N requests -- NOT A RESULT`.** Usually
`ISL+OSL > max_model_len`. ⚠️ `vllm bench serve` **exits 0** when every request
fails — it warns, prints `0.00` for every metric, and returns success. Without this
guard an all-zero row enters the table looking like a real datapoint.

**Note the context grid tops out at ISL 260,000, not 262,144.** `max_model_len - OSL`
= 261,888 is *not* enough headroom — it still 400s, because `--dataset-name random`
does not emit exactly `--random-input-len` tokens. 260,000 was found by bisection
and is 99.2% of `max_model_len`. Re-bisect with `--num-prompts 1` if you change
`--max-model-len`.

---

## 6. Dead ends — all tested, do not re-explore

| Attempt | Why it fails |
|---|---|
| `kill -9` the GPU-0 squatter | Frees the memory, but **PID 1 dies with it** → container terminates, RunAI reschedules, entrypoint respawns holding the same 77 GiB. A respawn loop. Fix at submit time with `--command -- sleep infinity` |
| TP=7 on the 7 free GPUs | `glm5next/nvidia/model.py:669` asserts `num_attention_heads % tp == 0`; 64 % 7 ≠ 0 |
| TP=4 on GPUs 4–7 | 305.8 GiB / 4 = 76.5 GiB/GPU vs ~65 GiB usable at util 0.82 |
| Pipeline parallel | Works, but changes the latency structure and makes TTFT/TPOT non-comparable to the V4 baseline |
| `--max-num-seqs 512` (the value the error suggests) | That is the boundary that just failed. Use 256 |
| `--compilation-config '{"max_cudagraph_capture_size":256}'` | **Replaces** the whole CompilationConfig, silently wiping `pass_config`. Use `--max-cudagraph-capture-size` |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` / `--enforce-eager` to fix the CUDA error | Neither helps — the CUDA error was a quota bug (§2) |
| `--kv-cache-dtype bfloat16` | Rejected by vLLM for models that don't default to fp8. Omit the flag; `auto` → BF16 |
| FP8 KV on H100 | Gated off: needs FlashInfer ≥ 0.6.18 for `ckv_scale_arr`; image ships 0.6.17 |

---

## 7. Fair comparison — use `normalize.sh`, never raw tok/s

```bash
./normalize.sh                                   # every result dir it can find
./normalize.sh --csv > normalized.csv
```

Minimum deployable footprint spans **21×** across this report's models, so raw
tok/s mostly reports how much silicon was used. `normalize.sh` always emits all
three normalizations (per GPU / per B-active / per B-total) so a flattering one
cannot be cherry-picked. It also prints the caveats normalization *cannot* fix —
differing tokenizers, the forced-opposite KV dtypes on H100, and TP=8 on a model
that fits in 3 GPUs.

**★ THE HEADLINE — engine-matched** (same engine/CUDA/node/grid, both base, all cold).
Reproduce V4's side with `deepseek_v4_flash/run_nomtp_image.sh`:

| conc | GLM-5.3 t/s | V4-Flash t/s | GLM/V4 | GLM per B-act [M] | V4 per B-act [M] |
|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 85.0 | **1.13×** | 5.54 | 6.04 |
| 4 | 225.3 | 219.8 | 1.02× | 12.96 | 15.61 |
| 16 | 359.2 | 330.9 | **1.09×** | 20.67 | 23.50 |
| 64 | 447.1 | 389.4 | **1.15×** | 25.72 | 27.65 |

**Engine effect measured at mean 1.01×** (V4 conda 0.28.0/cu129 → image dev/cu130,
identical model+flags), so this is an architecture result, not an engine artifact.

⚠️ **GLM wins raw/per-GPU but LOSES per-B-active at every concurrency** — it needs more
active params to get there. V4 uses its FLOP budget better. Report both.

Concurrency scaling 1→64: GLM **4.64×** vs V4 **4.15×**.

**Throughput is NOT linear in active params, and the deviation is the finding.**
If decode were purely memory-bound, `tok/s × B-active` would be constant across
models on the same hardware. Measured spread: **+30.8% at c=4 rising to +57.6% at
c=16**. GLM carries **29% more** active params (17.38 B [M] vs 13.49 B [A]) yet
wins at high concurrency — consistent with its 34 recurrent KDA layers reading a
fixed-size state instead of O(ctx) KV, and with achieved HBM bandwidth measured at
<5% of peak (so neither model sits at the memory roofline; the bound is elsewhere,
and EP all-to-all on the decode critical path is the leading hypothesis).

⚠️ **GLM's counts are MEASURED [M] from safetensors tensor shapes: 321.34 B total /
17.38 B active** = always-on 8.92 B + 8/288 of 304.42 B routed experts; the 7.43 B
MTP layer and 0.56 B vision tower are excluded (base model, text-only). V4's
13.49 B is still **[A]** — re-derive it from tensor shapes before publishing a
per-B-active ratio between the two.

⚠️ **V4's param count needs a packing correction.** Its routed experts are stored as
`I8` tensors holding **two MXFP4 values per byte**. Naive shape-summing gives
158.07 B/11.01 B — **wrong by ~2× on the expert term**. Correct: 141.734 B I8 elements
× 2 = 283.468 B logical experts + 7.438 B non-expert = **290.91 B total**; active =
7.438 + 283.468·6/256 = **14.08 B**. Reconciles with the card's 284 B/13 B.

⚠️ **GLM ships NATIVE FP8 weights** — 314.40 of 321.34 B (97.8%) are `F8_E4M3` on
disk (`quant_method: fp8`, block `[128,128]`, dynamic activations, 1,509-entry
`modules_to_not_convert`). The 6.93 B BF16 remainder is q/k/v/o_proj, embeddings/
lm_head, `kv_b_proj`, the indexer, the MoE gate, and the vision tower. V4-Flash
uses **MXFP4** experts, so per expert-param GLM reads ~1 byte vs V4's ~0.5. Do not
flatten this to "both FP8".

---

## 8. The other five arms — commands and expected results

All six arms live in `GLM-5.3-Flash/results/`. **34 measured points total.** Each arm is one
server launch + one `bench.sh` call with a different `OUTDIR`.

| arm | script | dir | points | headline |
|---|---|---|--:|---|
| BF16 KV (headline) | `./run.sh` | `bf16kv/` | 11 | 96.3 → 447.1 tok/s (c=1→64) |
| FP8 KV | `./run_fp8kv_fi618.sh` | `fp8kv-fi618/` | 11 | **1.805× KV capacity, −15.5% throughput** |
| BF16 on 0.6.18 (**control**) | `./run_bf16kv_fi618.sh` | `bf16kv-fi618/` | 4 | isolates backend from dtype |
| MTP n=1 | `NSPEC=1 ./run_mtp.sh` | `bf16kv-mtp-n1/` | 4 | 1.24× @c=1 → 0.99× @c=64 |
| MTP n=5 (recipe) | `NSPEC=5 ./run_mtp.sh` | `bf16kv-mtp-n5/` | 4 | 1.25× @c=1 → **0.91×** @c=64 |
| `fp8_ds_mla` | `./run_fp8kv.sh` | `fp8kv/` | 0 | fails by architecture (NoPE) — a result |
| **V4 engine-matched** | `../deepseek_v4_flash/run_nomtp_image.sh` | `../deepseek_v4_flash/results/mtp-off-image/` | 11 | **closes the fairness gap**; engine effect 1.01× |

### 8a. FP8 KV — needs a FlashInfer overlay, and a CONTROL

The recipe says *"Hopper does not support FP8 KV cache for this model."* **That is true of the
shipped image, not of Hopper.** GLM meets every requirement of the
`FLASHINFER_MLA_SPARSE_SM90` backend (`kv_lora_rank=512`, `qk_rope_head_dim=0` — NoPE is
*explicitly* allowed, `index_topk` present); only a FlashInfer feature probe for
`ckv_scale_arr` (**≥ 0.6.18**) gates it, and the image ships 0.6.17.

```bash
# build the overlay once (keeps the image's 0.6.17 intact)
pip download flashinfer-python==0.6.18 -d /tmp/fi_check --no-deps
/usr/bin/python3 -m pip install --no-deps --target /tmp/fi618 \
    /tmp/fi_check/flashinfer_python-0.6.18-py3-none-any.whl

cd GLM-5.3-Flash && ./run_fp8kv_fi618.sh          # then bench with OUTDIR=.../fp8kv-fi618
cd GLM-5.3-Flash && ./run_bf16kv_fi618.sh         # THE CONTROL — same overlay, BF16 KV
```

⚠️ **Two things make this non-production, both handled in the scripts:**
`FLASHINFER_DISABLE_VERSION_CHECK=1` (this mirror has no `flashinfer-cubin` > 0.6.13 and cannot
reach `flashinfer-jit-cache`, so 0.6.18 Python runs against 0.6.17 binaries), and
`--kernel-config '{"moe_backend":"deep_gemm"}'` to dodge an ABI mismatch in FlashInfer's
fused-MoE (`init(): Expected 8 but got 9 arguments`, `fused_moe/core.py:693`). **Always validate
with `./sending.sh chat` before trusting a number from this overlay.**

⚠️ **RUN THE CONTROL OR PUBLISH NOTHING.** Comparing `bf16kv/` to `fp8kv-fi618/` directly changes
dtype **and** backend **and** MoE kernel:

| conc | A bf16/0.6.17 | B bf16/0.6.18 (control) | C fp8/0.6.18 | backend A→B | **dtype B→C** | naive A→C |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 71.5 | 68.9 | 0.74× | **0.96×** | 0.72× |
| 4 | 225.3 | 147.5 | 128.7 | 0.65× | **0.87×** | 0.57× |
| 16 | 359.2 | 263.0 | 208.8 | 0.73× | **0.79×** | 0.58× |
| 64 | 447.1 | 348.8 | 262.0 | 0.78× | **0.75×** | 0.59× |

**Backend = −27.3%. Dtype = −15.5%.** The naive read would have claimed "FP8 KV costs 39%".

### 8b. MTP — run BOTH depths

```bash
cd GLM-5.3-Flash && NSPEC=1 ./run_mtp.sh    # matches the V4 A/B -> architecture comparison
cd GLM-5.3-Flash && NSPEC=5 ./run_mtp.sh    # the recipe's setting
```
Read acceptance from the live server (it is not in the result JSON):
```bash
curl -s localhost:8001/metrics | awk '/spec_decode_num_draft_tokens_total/{d=$2}
  /spec_decode_num_accepted_tokens_total/{a=$2} END{printf "%.1f%%\n",100*a/d}'
```
Expected: **n=1 → 71.8%**, **n=5 → 30.6%**. n=5 is *worse* than n=1 at c=64 (0.91× vs 0.99×)
because GLM has one MTP layer run recurrently 5×, so drafts 2–5 condition on the draft head's
own output and error compounds. See `WHY.md` §7b.

⚠️ **A single anomalous point can be a scheduling artifact, not a result.** The first c=4 n=1 run
gave 94.7 tok/s (0.42×) with p99 TTFT 12.3 s vs median 1.7 s; the rerun gave 239.8 tok/s with
p99 2.4 s. Both passed the cold-run guard and completed 8/8 — **the guards do not catch
queueing**. Reproduce anything that breaks the trend.

### 8c. TP4 and PD-disaggregation — do not attempt on 8×H100

The recipe's single-node example is TP4 and its PD example is TP4+TP4. **Both are impossible
here**, and the recipe says why: *"306 GiB ... alone exceeds 4×H100-80GB."* Measured:
```
TP4, --gpu-memory-utilization 0.95:  Model loading took 75.36 GiB
torch.OutOfMemoryError: GPU 0 has 79.18 GiB total, 1.60 GiB free
```
PD disaggregation needs **two** weight copies = 612 GiB vs 640 GiB of node HBM, before any KV.
The recipe's PD example targets a **GB200 tray**. This is a memory-hierarchy fact, not a config
problem — see `WHY.md` §6.

---

## 9. What is still owed

- `GLM-5.3-Flash/report.md` and `README.md`
- `./run_fp8kv.sh` — the `fp8_ds_mla` arm; **expected to fail by design** on H100
  (a clean failure with the exact error is a valid result)
- `./run_mtp.sh` — the spec-decode A/B (never a headline number)
- A **DeepSeek-V4-Flash rerun in this image** — the baseline ran on conda vLLM
  0.28.0/cu129 while GLM ran on the image's dev build/cu130. That engine mismatch
  is a real fairness gap in the comparison above and should be closed.
- ⚠️ **Fairness note to carry:** the V4 baseline passed no `--max-num-seqs`, so it
  ran at the H100 default of 1024 while GLM is pinned to 256. No published point is
  affected (both grids top out at concurrency 64), but it must be stated.
