> ARCHIVED SESSION RECORD — not current instructions or conclusions. Original location: `WORKFLOW.md`. See [current report](../../report.md), [debugging guide](../../fix_bug.md), and [session handoff](../../WORKFLOW.md). Historical claims may be superseded.

# WORKFLOW.md — start here

Entry point for a new session. **Read this first, then the one doc you need.**

| doc | read it when |
|---|---|
| **WORKFLOW.md** (this) | always — state, next task, the 6 rules |
| [`WHY.md`](../../archive/notes/WHY.md) | explaining a number, or prepping for Kan / SyFI questions |
| [`REPRODUCE.md`](../../archive/notes/REPRODUCE.md) | re-running any arm |
| [`fix_bug.md`](../../fix_bug.md) | **anything fails** — 8 bugs, error text searchable |
| [`CLAUDE.md`](../../archive/notes/CLAUDE.md) | full reference: thesis, architecture facts, conventions |

---

## 1. State (2026-09-02, session 3 — GLM's owed tasks are DONE)

**115 measured points, all cold + complete** (+12 quarantined as evidence). Three models done and
**engine-bridged**. All reports written. **All 4 owed GLM measurements are complete.**

| model | pts | status |
|---|--:|---|
| **GLM-5.3-Flash** | **46** | ✅ **complete** — BF16 KV, FP8 KV (+control), MTP n=1/n=5, **util A/B**, **8-point curve**, **MTP context axis** |
| **DeepSeek-V4-Flash** | 50 | ✅ done — + engine bridge (0.997×), util A/B, 8-point batch curve |
| **Qwen3.8-Flash-Next** | 19 | ✅ done — base (11) + MTP n=1 (4) + util A/B (4) · reports written |

**★ New this session (4 results + 1 harness fix):**
1. **`--gpu-memory-utilization` is capacity-only — confirmed on the 3rd and most KV-constrained model.**
   GLM 0.82→0.85: **+11.2% KV, 0.994–1.000× throughput**, peak activation 4.05 GiB in *both* arms.
2. **GLM's batch knee is c≈4–8, matching V4 almost exactly** (1→8 = 3.08× vs 3.03×; efficiency at c=8
   0.385 vs 0.378) despite completely different attention stacks. **V4's c=48 dip does NOT generalize**
   (GLM 1.064× vs V4 0.985×).
3. **⚠️ MTP on the context axis INVERTS the batch-axis advice** — costs 4–6% throughput but buys
   **~17–25% TTFT** to 131K. On TraceLab's 530:1 ISL:OSL that is a *good* trade. The
   `index_share_for_mtp_iteration` prediction was **falsified** on throughput; the 260K ceiling is
   **KV-bound (96.9%), not accuracy-bound** (acceptance held 69.1%).
4. **🛑 THE "~1% OF ROOFLINE" HEADLINE WAS WRONG BY AN ORDER OF MAGNITUDE.** `--enable-mfu-metrics` was
   set on every arm of all three models but **nothing ever read its counters**. Now scraped: GLM measures
   **9.5–18.1% of 3.35 TB/s [E]**. Conclusion survives (decode is latency-bound, and **bandwidth plateaus
   at c≈16–32 while throughput rises to c=64**) but **never quote ~1% again.** `fix_bug.md` bug 13.

**★ The headline** (same engine/CUDA/node/grid, both base, all cold):

| conc | GLM t/s | V4 t/s | GLM/V4 | GLM /B-act | V4 /B-act |
|--:|--:|--:|--:|--:|--:|
| 1 | 96.3 | 85.0 | **1.13×** | 5.54 | 6.04 |
| 16 | 359.2 | 330.9 | **1.09×** | 20.67 | 23.50 |
| 64 | 447.1 | 389.4 | **1.15×** | 25.72 | 27.65 |

Engine effect measured at **1.01×** — so this is architecture, not tooling.
**GLM wins raw/per-GPU; V4 wins per-B-active.** Report both — that tension *is* the result.

**★ Qwen3.8 added a third point** (⚠️ **different vLLM build** — `dev20073` vs `dev20051`; bridge
arm owed, see §2):

| conc | Qwen t/s | GLM t/s | V4 t/s | Qwen /B-act | GLM /B-act | V4 /B-act |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 106.2 | 96.3 | 85.0 | **14.61** | 5.54 | 6.04 |
| 4 | **256.8** | 225.3 | 219.8 | **35.32** | 12.96 | 15.61 |
| 16 | **420.2** | 359.2 | 330.9 | **57.79** | 20.67 | 23.50 |
| 64 | **517.7** | 447.1 | 389.4 | **71.21** | 25.72 | 27.65 |

Qwen **wins at every concurrency** (1.10–1.17× GLM), is **2.6–2.8× better per active param**
(7.27 B active [M], E/k = 51.2×), and wins the prefix axis by **1.71×** at n=1.
⚠️ c=4/c=16 are from `results/base-util082/` — see the util finding in §1's finding 6.

**Five findings worth leading with:**
1. **Decode runs far below the memory roofline — GLM measures 9.5–18.1% of 3.35 TB/s [E]**, and
   **bandwidth peaks at c≈16–32 and then FALLS, while throughput keeps rising to c=64**, so what binds at high
   concurrency is demonstrably not HBM. Decode is latency-bound, and **active-param × bytes mis-ranks
   the models**: Qwen reads the *fewest* bytes/step (6.77 GiB vs GLM 16.19) and is the *fastest* at c=64.
   `B*_MoE = B*_dense·E/k` is not the operative cost model at reachable batch.
   ⚠️ **The old "~1%" figure was a hand-rolled estimate and was too low by ~10×** — corrected 2026-09-02
   from vLLM's own per-step counters. Label them **[E]**, not [M]: engine-side estimate, and for GLM they
   **exclude attention entirely** (only `ffn`+`unembed` ComponentMetrics), so 18.1% is a lower bound.
2. **FP8 KV works on H100** (recipe says it doesn't): +1.805× KV capacity, and the **control arm**
   decomposes the cost as backend −22.0%, dtype −24.9% (naive read would say −41.4%).
3. **MTP: helps at c=1 only, and hurts Qwen everywhere else.** GLM 1.24×→0.99×; Qwen
   **1.17× at c=1 → 0.87–0.93×** above it (57.6% acceptance, and it costs **12% of KV**).
   Qwen's draft head is *full-attention* over a 36/48 linear-attention target — no
   `index_share_for_mtp_iteration` as GLM has.
4. **KV bytes/token spans 3.3× within this generation** [A]: Qwen **24.75** > GLM 11.35 > V4 ~7.4.
   Qwen and GLM both hit KV exhaustion at 260K×8 (**97.3%** / 89.5%); V4 has headroom (36.9%).
5. **The same allocator resolved `block_size` to 4 for Qwen and 640 for GLM** — 160× apart, both
   hybrids, same engine family. One block size cannot serve both.
6. ⚠️ **`--gpu-memory-utilization` is capacity-only (confirmed [M]) — but cold `torch.compile` corrupts
   the KV sizing.** vLLM measured peak activation as **17.07 GiB during a cold compile vs 0.99 GiB warm**,
   reserving ~14 GiB/GPU of KV it never needed. That understated Qwen c=4 by **1.58×** and briefly made
   Qwen look *slower* than GLM there. **Warm the compile cache before any KV-sizing run**, or pin
   `--kv-cache-memory`. Full story: `fix_bug.md` bug 12.


---

## 2. Next task: GLM is DONE — the open work is now cross-model

**All 4 owed GLM measurements landed 2026-09-02 (session 3).** No container switch is needed for GLM
work any more, and the `glm53-flash` image is the one that has `Glm5Next*` registered — verify with
`/usr/bin/python3 -c "from vllm import ModelRegistry; print([a for a in ModelRegistry.get_supported_archs() if 'Glm5' in a])"`.

### ✅ Completed this session — evidence on disk

| # | task | result | writeup |
|--:|---|---|---|
| 1 | util A/B 0.82→0.85 | **capacity-only**: +11.2% KV, 0.994–1.000× tok/s, peak activation 4.05 GiB both arms | `GLM-5.3-Flash/results/util085/RESULT-util-ab.md` |
| 2 | 8-point batch curve | **knee at c≈4–8**, matches V4 (3.08× vs 3.03× for 1→8); **no c=48 dip** | `GLM-5.3-Flash/results/bf16kv/RESULT-8point-batch-curve.md` |
| 3 | scrape `--enable-mfu-metrics` | **corrected the roofline headline ~1% → 9.5–18.1%**; harness fix in `bench.sh` | `fix_bug.md` bug 13 |
| 4 | MTP on the context axis | **inverts the advice**: −4–6% tok/s but **−16.7–24.8% TTFT** to 131K; 260K ceiling is KV-bound | `GLM-5.3-Flash/results/bf16kv-mtp-n1-context/RESULT-mtp-context-axis.md` |

### ⬜ What is actually open now, highest value first

| # | task | why it matters | cost |
|--:|---|---|---|
| **1** | **Re-poll V4 + Qwen with the fixed MFU harness** | GLM's bandwidth is now **[E]**; the other two are still **[A]**, so §1's cross-model bandwidth table mixes label classes. ⚠️ **Raw counters are NOT comparable across models** — coverage differs (Qwen `attn`+`ffn`+`unembed`, GLM `ffn`+`unembed`, V4 `unembed` only). Either report per-model with the coverage table, or reconstruct the missing components by hand. | ~8 pts/model |
| **2** | **MTP context axis on V4 + Qwen** | GLM's context axis **reversed** the conclusion drawn from three models' batch axes. A finding that flips on an untested axis needs the other two models before it can be stated generally. | 8 pts/model |
| **3** | **TP/EP sweep on V4 or Qwen** (both fit in 3 GPUs) | The report's **central mechanistic claim** — that EP all-to-all + launch overhead is the real bound — is still **[H]**. ⚠️ Closed for GLM (TP4 OOM, 64 % 7); must run on V4/Qwen or via Nsight. | unscoped |
| **4** | **A dense control** | Largest single gap. Qwen was the designated baseline and is a 512-expert MoE, so *"MoE decode is memory-bound"* is unfalsified against a dense model on this harness. | unscoped |
| **5** | **Sweep `max_num_batched_tokens`** | Chunked prefill is on by default at 8192; plausibly the largest untested server-side lever for long context, and TTFT non-monotonicity (§4c) suggests real headroom. | unscoped |

```bash
# GLM is complete; this is the reproduce path, not new work.
cd GLM-5.3-Flash && ./run.sh                     # BF16 KV, util 0.82, port 8001
GPU_UTIL=0.85 ./run.sh                           # util arm
NSPEC=1 ./run_mtp.sh                             # MTP arm

# ⚠️ WARM THE COMPILE CACHE FIRST on a freshly scheduled container. /tmp caches start
# EMPTY after a reschedule, and a cold torch.compile mis-sizes KV (bug 12). Measured
# this session on GLM: peak activation 4.88 GiB cold vs 4.05 GiB warm -> KV pool
# 1,838,761 vs 1,916,967 tokens. Run one throwaway point, restart, then measure.

# sweeps -- SERVER_LOG is new: it records per-model MFU component coverage in the manifest
MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/<arm> \
QUANT=fp8-attn-dense+fp8-experts KV_DTYPE=bfloat16 \
SERVER_LOG=GLM-5.3-Flash/logs/<the serve log> ./bench.sh batch
```

⚠️ **Expect the queueing guard to fire on the first points after a cold start.** It happened **twice more
this session** (util-0.85 c=4 read **2.3× low**; MTP-context 16K read **2.2× low** and would have inverted
that arm's trend). Signature: **p99 TTFT ≫ median**. Rerun warm, quarantine the first run, never delete it.

🛑 **Never `pkill -f "vllm serve"`** — PID 1 matches it and the container restarts. Use
`pkill -f "GLM-5.3-Flash"`, then confirm `ps -p 1 -o args=` still says `sleep infinity`.


### ❌ Closed for GLM — do not spend time on these

| attempt | why it is closed |
|---|---|
| **TP sweep** | **No sweep exists.** TP4 is measured OOM (75.36 GiB/GPU, 1.60 GiB free); TP=7 fails `64 % 7`. GLM needs ≥5 GPUs, so **only TP8 is reachable on one node.** Test the EP-all-to-all hypothesis on V4 or Qwen (both fit in 3 GPUs) or with Nsight. |
| `--kv-cache-dtype fp8_ds_mla` | Impossible by geometry — GLM is NoPE (`qk_rope_head_dim: 0`), needs `pe_dim==64`. Fails on **any** GPU, Blackwell included. |
| The recipe's **GB200 TP4 + `--kv-cache-dtype fp8`** command | Cannot run on H100: 305.8/4 = **76.5 GiB/GPU** vs ~75.6 usable, and the FlashInfer MLA backends that give Blackwell free FP8 KV are gated `capability.major == 10`. |
| MTP `n=5` for throughput | Worse than n=1 at every conc ≥4 (0.91× at c=64, acceptance 71.8%→30.6%, KV 99.2%). |
| FP8 KV as a **production** config | Works (1.805× KV, −25% dtype / −22% backend) but needs `FLASHINFER_DISABLE_VERSION_CHECK=1` + `moe_backend=deep_gemm`. Capability demo only. |

## 3. The 6 rules that keep numbers publishable

Each exists because it was violated once. Details + evidence in `fix_bug.md` / `WHY.md`.

1. **One variable per arm.** Our worst near-miss: an FP8-KV arm changed dtype + attention
   backend + MoE kernel at once and read as *"FP8 costs 39%"*. The control showed
   **backend −27.3%, dtype −15.5%.** Always run the control arm.
2. **Cold, complete, AND unqueued.** `bench.sh` asserts all three. The third is newest:
   3 V4 points passed cold+completed yet were **25–60% low** from scheduler queueing;
   they made the engine effect read 0.64× instead of 1.01×. Signal = **p99 TTFT ≫ median**.
3. **Never raw tok/s across models.** `./normalize.sh` — per-GPU + per-B-active + per-B-total,
   always all three, or the table is marketing.
4. **Param counts from tensor shapes, never `config.json`.** Two corrections so far: GLM
   15.01 → **17.38 B**; V4 needs an I8-packing fix (2 MXFP4/byte) or you get 11.01 instead of
   **14.08 B**.
5. **Label every cell [M] / [A] / [H].** Mixing them silently makes the whole report
   discountable.
6. **A clean failure is a result.** State what didn't run and why, with the exact error —
   e.g. TP4 OOM (75.36 GiB/GPU), `fp8_ds_mla` NoPE mismatch.

---

## 4. Do not re-explore (all measured)

| attempt | why it fails |
|---|---|
| **TP4 / TP4+TP4 PD-disagg** (the recipe's own examples) | weights = 75.36 GiB/GPU → **OOM at util 0.95**, 1.6 GiB free. PD needs 2 copies = 612 of 640 GiB. Recipe's PD targets a **GB200 tray** |
| TP=7 | `num_attention_heads % tp != 0` (64 % 7) |
| `--kv-cache-dtype fp8_ds_mla` on GLM | needs `pe_dim==64`; GLM is **NoPE** (`qk_rope_head_dim: 0`) |
| `--compilation-config '{"max_cudagraph_capture_size":N}'` | **replaces** the whole config, wipes `pass_config`. Use `--max-cudagraph-capture-size` |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` / `--enforce-eager` for a CUDA error | the "CUDA error" was an **NFS quota** write failure |
| `kill -9` the GPU-0 squatter | PID 1 dies with it → container reschedules → squatter returns |
| MTP `n=5` for throughput | worse than `n=1` (0.91× vs 0.99× at c=64) |

---

## 5. Deliverables — all written; measurement gaps remain

1. ✅ `GLM-5.3-Flash/report.md` + `README.md`
2. ✅ `Qwen3.8-Flash-Next-FP8/report.md` + `README.md`
3. ✅ **`report.md` at the root** — answers all four `task.md` questions
4. ✅ **V4-Flash bridge arm** — engine effect **0.997×**, confound closed
5. ✅ **The 4 GLM measurements** — DONE this session, each with a `RESULT-*.md` writeup (§2)
6. ✅ **`bench.sh` MFU scraping** — every point now carries `achieved_gbps_per_gpu`,
   `frac_peak_hbm_h100`, and a per-model MFU-coverage table in `manifest.txt`

**Known gaps to state rather than hide:**
- **No dense control anywhere.** Qwen3.8 was the designated baseline and is a 512-expert MoE,
  so "MoE decode is memory-bound" is still unfalsified against a dense model on this harness —
  and the measured 9–18%-of-roofline result suggests the premise is wrong for all three MoEs anyway.
- **No TP/EP sweep**, so "collectives/launch overhead are the bound" is motivated but unproven.
  This is the report's central mechanistic claim, so the gap matters.
- ⚠️ **Achieved bandwidth is [E] for GLM, still [A] for V4 and Qwen** — and the raw counters are
  **not comparable across models** because vLLM's component coverage differs per model
  (Qwen `attn`+`ffn`+`unembed` · GLM `ffn`+`unembed` · V4 `unembed` only). Re-poll owed.
- ✅ **The Qwen engine build difference is CLOSED** — bridge arm measured it at **0.997×**.
- ⚠️ **MTP context axis measured for GLM only** — and it **inverted** the batch-axis conclusion, so
  the missing V4/Qwen context arms are load-bearing, not cosmetic. No prefix-axis MTP anywhere.
- GLM's FP8-KV arm needs `FLASHINFER_DISABLE_VERSION_CHECK=1` + `moe_backend=deep_gemm`
  (**not production**).
- **Tokenizers differ** (Qwen 248,320 · GLM 154,880 · V4 129,536) so cross-family tok/s is not
  strictly commensurable; use bytes/s for strict claims.
