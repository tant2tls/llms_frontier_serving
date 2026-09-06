# Learning ML serving through debugging

GLM-5.3-Flash · DeepSeek-V4-Flash · Qwen3.8-Flash-Next-FP8 · 8 × H100

Refined 6 September 2026. Read alongside [report.md](report.md) and [slides.md](slides.md). Original incident numbers are preserved so existing references still work. [Historical notes](archive/notes/fix_bug_previous.md) retain the original narrative, including superseded explanations.

**The aim:** learn to connect a symptom to evidence, a discriminating experiment, a fix, and a systems lesson. A server that answers requests is not automatically ready for benchmarking, and a completed benchmark is not automatically a controlled experiment.

This revision audits existing scripts, logs, and results. It does not rerun GPU experiments or repair the serving harness. “Implemented” below means present in the saved scripts; “observed” means recorded evidence; “proposed” means remaining work. Analytical explanations are distinguished from proven causes.

## 1. Start with the system, then locate the failure

```text
Deployment:  environment → model/backend selection → weights → JIT → memory pools
Request:     client → tokenization → validation → cache lookup → prefill → decode
Measurement: response + server counters → result JSON → comparison → conclusion
```

A failure can propagate across these boundaries. A filesystem error during JIT can surface at kernel launch. A client validation failure can produce a zero-throughput result. A cache-pool difference can change a model ranking without changing the checkpoint.

| Architectural feature | Runtime obligation | Incident / result | What you learn |
|---|---|---|---|
| GLM/Qwen recurrent layers | Allocate state per active sequence as well as token history | Bug 1: Mamba-state capacity blocks startup | Context length is not the only memory axis |
| FP8 MoE kernels | Compile and load specialized kernels | Bugs 2–3: quota failures through multiple JIT caches | Disk and toolchain state affect GPU availability |
| Sparse attention with different layouts | Match checkpoint geometry to supported cache layout | Bug 8: NoPE versus a layout requiring 64 positional dimensions | “FP8” is a dtype label, not a complete compatibility specification |
| Prefix caching | Reuse only intended prefixes and compatible layer state | Bug 6: warmup creates 16,000 cache-hit tokens | Warm kernels and cold prefixes are separate conditions |
| Chunked prefill and lazy initialization | Stabilize runtime before timing | Bugs 9–10: early runs have large latency tails | TTFT includes more than attention compute |
| Automatic memory profiling | Budget activations, graph buffers, and history pools | Bug 12: Qwen pool and throughput change with startup state | One flag change need not mean one variable changed |
| Hybrid layers and MoE | Account for each operation/state class in metrics | Bug 13: partial FLOP/byte estimates | An enabled estimator is not hardware evidence |

**Memory model to remember:**

```text
HBM = weights + workspace + graph buffers
    + sequence-growing history + per-sequence recurrent state + padding/metadata
```

**Metric model to remember:** throughput counts completed work over time; TTFT includes queueing/prefill; TPOT summarizes output spacing; cache occupancy refers to an allocated pool, not all GPU memory.

## 2. A study route through the incidents

For a first reading, spend about 40 minutes: architecture map (5), Bugs 1 and 3 (8), Bugs 6 and 7 (7), Bugs 8 and 12 (10), Bug 13 and the remaining gaps (10). Then use the exercises in §7 without looking at the answers.

| Bug | Symptom | Decisive evidence or check | Resolution / scope |
|---:|---|---|---|
| 1 | Insufficient Mamba blocks | Effective sequence cap and hybrid layer list | Pin a feasible sequence cap; implemented |
| 2 | Triton disk quota error | Real allocation fails despite free volume space | Move JIT caches; preflight write; implemented |
| 3 | CUDA invalid argument | Earlier TensorRT/DeepGEMM filesystem failure | Redirect overlooked caches; startup succeeds |
| 4 | Old CUDA diagnosis persists | Current logs show successful compilation | Update the diagnosis for the current image |
| 5 | Wrong precision labels | Compare live config with JSON metadata | Overrides and manifests added; validation incomplete |
| 6 | 16,000 prefix-hit tokens at every point | Warmup reuses measured prompt set | Remove overlapping warmup; bad runs retained |
| 7 | Exit 0, zero completed requests | Actual request lengths exceed limit | Lower top input target; zero-completion guard added |
| 8 | FP8 KV layout fails | NoPE has zero positional dimensions | Alternate layout/backend runs; do not claim universal impossibility |
| 9 | Large tail at concurrency 1 | Early versus warm rerun comparison | Warm rerun; mechanism plausible, not uniquely proven |
| 10 | Valid-looking runs far below reruns | Startup/warmup evidence plus bridge reruns | Quarantine; rerun; timing attribution incomplete |
| 11 | Smoke test fails before a request | Missing bc; wrong endpoint defaults | Direct request workaround; shared script still needs attention |
| 12 | Qwen c4 appears unexpectedly slow | Startup activation/pool differs across arms | Correct batch baseline; other comparisons remain confounded |
| 13 | Enabled metrics absent/partial | Check counter consumer, type, and component coverage | Collection added; hardware attribution remains open |

## 3. Startup: learn to distinguish architecture from infrastructure

### Bug 1 — Mamba-state capacity blocks a hybrid model

**Symptom.** `max_num_seqs (1024) exceeds available Mamba cache blocks (512)`.

**Initial assumption.** A large HBM budget should be enough for the weights and KV, and the configured default was believed to be 128 sequences.

**Discriminating check.** Read the effective startup configuration and the layer list. GLM has 34 recurrent KDA layers and 11 sparse-attention layers. The saved development build selected 1024 sequences through a hardware-dependent default. The error describes the recurrent-state allocator, not a requirement to run a model named Mamba.

**Fix and result.** The launch script pins `--max-num-seqs 256`; startup proceeds after the other independent blockers are resolved. This is below the reported 512-state-block limit and above the client grid's maximum concurrency of 64. It avoids directly clipping that grid, but changing this cap can still change pool allocation and graph capture below the cap.

**Architectural lesson.** Recurrent state is fixed in context length but scales with active sequences. It is an additional admission constraint alongside attention history. “Fits in memory” must include both.

**Evidence:** [GLM launch scaffold](GLM-5.3-Flash/_common.sh), [launch log](GLM-5.3-Flash/logs/launch_bf16kv_console.log), [retained main manifest](GLM-5.3-Flash/results/bf16kv/manifest.txt). Hardware-dependent defaults are statements about this saved build, not permanent vLLM defaults.

### Bug 2 — Free disk space does not imply available user quota

**Symptom.** After loading weights, Triton fails with `[Errno 122] Disk quota exceeded`.

**Initial assumption.** The NFS volume reports hundreds of gigabytes free, so storage cannot be the issue.

**Discriminating check.** A permissions check or zero-byte file can succeed while a real allocation fails. The incident records a full per-user quota. Test the operation the compiler needs: create and flush a small nonempty file in each resolved cache directory. Also inspect quota and inode limits when available.

**Fix and result.** The launch scaffolds redirect Triton, Inductor, vLLM, and XDG caches to node-local storage and perform a bounded write preflight. This makes this class of failure visible before an expensive weight load. The next incident exposed additional cache directories that the first fix missed.

**Architectural lesson.** JIT compilation is part of serving startup. GPU computation depends on a working compilation/storage path, not just CUDA availability.

**Evidence:** [first cu130 run](GLM-5.3-Flash/logs/launch_cu130_run1.log), [preflight implementation](GLM-5.3-Flash/_common.sh). A uid/CUDA-major cache directory reduces some sharing hazards; it is not complete isolation across compiler, library, driver, and GPU-architecture versions.

### Bug 3 — A CUDA error caused by an overlooked compiler cache

**Symptom.** `CUDA error: invalid argument` appears during graph profiling; several workers then exit.

**Hypotheses tested.**

| Hypothesis | Intervention | Recorded outcome / inference |
|---|---|---|
| Breakable graphs are responsible | Disable that path | Failure remains; this setting is not sufficient to resolve it |
| Capture ladder exceeds useful sequence sizes | Cap graph capture size | Failure remains; cap is not the crash fix |
| Failure is specific to graph capture | Run eager | Failure remains; inspect work shared by eager and graph execution |

**Decisive evidence.** An earlier error reports failure to create directories under `.tensorrt_llm` for an FP8 GEMM. The compiler resolves its own cache path, including a C++ environment-variable lookup; changing only Python/Triton caches missed it.

**Fix and result.** Redirect `TRTLLM_DG_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE`, include them in preflight, then launch successfully with CUDA graphs. The documented causal chain is **compiler-cache write failure → failed kernel preparation → downstream CUDA failure**. The logs do not establish every internal transition, such as exactly what invalid kernel handle was launched.

**Architectural lesson.** Optimized FP8 MoE serving uses multiple compiler/runtime layers. A symptom at the GPU boundary can originate in host-side preparation. Eager mode is a diagnostic intervention, not a universal remedy.

**Evidence:** [eager run](GLM-5.3-Flash/logs/launch_cu130_run6_eager.log), [cache-fix run](GLM-5.3-Flash/logs/launch_cu130_run7_trtllmfix.log), [cache configuration](GLM-5.3-Flash/_common.sh).

### Bug 4 — A correct historical diagnosis becomes a wrong current diagnosis

**Symptom.** Notes continue to blame the older cu129 MHC/TileLang crash after moving to cu130.

**Check.** Search failure and success patterns separately in the current log. Matching `Segfault encountered` OR `mhc_post_tilelang` also counts successful compilations and can create a false failure summary.

**Fix and result.** Current cu130 logs record successful MHC compilation; the remaining blocker must be localized independently. This does not prove CUDA 13 cures every similar error or identify which version change fixed the historical crash.

**Architectural lesson.** Layer implementations depend on toolchain versions. Record the image/build with every diagnosis; a model name alone is not an execution environment.

**Evidence:** [cu130 compilation and later quota errors](GLM-5.3-Flash/logs/launch_cu130_run1.log), [older investigation](archive/notes/fix_bug_previous.md).

## 4. Measurement: learn why successful requests can give misleading results

### Bug 5 — Metadata describes the wrong model

**Symptom.** Shared benchmark defaults label GLM results as DeepSeek-style MXFP4 experts and FP8 KV.

**Check.** Compare model identity, effective KV dtype, selected backends, and requested flags with result metadata. `dtype=bfloat16` in a model config does not by itself describe quantized weight storage. Similarly, `cache_dtype=auto` requires backend/config resolution.

**Fix and result.** `QUANT`, `KV_DTYPE`, and `HW` overrides and live configuration capture were added. The GLM manifest records attention block size 640 despite a requested 128, plus recurrent-state block size 128.

**Architectural lesson.** Precision, storage layout, allocator grouping, and logical block size are different concepts. GLM and Qwen resolve different cache organizations; their block-size ratio is not a measured efficiency ratio.

**Remaining gap.** Metadata overrides can still be wrong. The harness warns on a model mismatch and then adopts the served ID, which can leave the output directory or labels inconsistent. It does not fully validate precision against actual runtime kernels.

**Evidence:** [bench.sh preflight](bench.sh), [GLM manifest](GLM-5.3-Flash/results/bf16kv/manifest.txt), [Qwen manifest](Qwen3.8-Flash-Next-FP8/results/base-util082/manifest.txt).

### Bug 6 — Benchmark warmup creates the cache hits being measured

**Symptom.** Every batch point reports 16,000 new prefix-hit tokens.

**Check.** The harness warmup uses the same seeded prompt set as the measured workload. `16,000 = 25 × 640` aligns with complete cached blocks from a 16,384-token target prompt. The cached portion is not the entire prompt.

**Fix and result.** Remove the overlapping benchmark warmup; retain contaminated results and rerun with fresh-prefix workloads. The retained main batch JSONs report zero new prefix hits. For a stronger future protocol, warm representative kernel shapes with disjoint prompts and retain successful raw metric scrapes.

| Workload | Reused-token share of nominal input |
|---|---:|
| 8 × 16,384 input tokens | 16,000 / 131,072 = 12.2% |
| 32 × 16,384 | 3.1% |
| 128 × 16,384 | 0.8% |

These are **input reuse fractions, not measured throughput inflation percentages**. Actual token counts include templates. Unequal reuse can distort the concurrency curve because low-concurrency points contain fewer requests.

**Architectural lesson.** Kernel/JIT warmth and prefix-cache warmth are independent. A fair cold-prefix test wants warm kernels with fresh prefixes. A reuse test intentionally wants cache hits and needs an explicit cache-state protocol.

**Evidence:** [quarantined warmup data](GLM-5.3-Flash/results/_discarded-warmup-contaminated/), [main batch results](GLM-5.3-Flash/results/bf16kv/), [bench.sh](bench.sh). A saved zero is a recorded counter result, not proof that telemetry could not have failed; see §6.

### Bug 7 — Exit code zero with zero useful work

**Symptom.** The top context point exits successfully but completes zero requests and reports zero throughput.

**Check.** Read the response errors and `completed`, not just the process status. A 262,144-token input plus 256 outputs exceeds a 262,144 limit. A smaller nominal input of 261,888 still fails because actual tokenization/template overhead can exceed the nominal target. A simple shorter request works, separating invalid workload from server failure.

**Fix and result.** A one-request boundary probe finds 260,000 works. The harness uses this target and rejects zero-completion result files, retaining them with `.failed` rather than treating them as performance points.

**Architectural lesson.** The request limit applies to the model's actual token sequence, not the prompt-length label. This was request validation, not evidence of long-context OOM or slow attention.

**Remaining gap.** `completed > 0` does not establish full completion. The current guard does not itself enforce `completed == num_prompts` and `failed == 0`, and the separate prefix path does not inherit every batch/context check.

**Evidence:** [bench.sh zero-completion guard and context target](bench.sh), [retained failed historical top point](deepseek_v4_flash/results/mtp-on/ctx_isl262144_c8.json), [successful GLM top point](GLM-5.3-Flash/results/bf16kv/ctx_isl260000_c8.json).

### Bug 9 — A large latency tail at concurrency 1

**Symptom.** Qwen's first c1 run reports 84.2 output tokens/s and a p99/median TTFT ratio of 7.81. A later warm run reports 107.7 tokens/s and a much smaller tail; median TTFT remains about 602 ms.

**Check.** Compare early/warm runs and startup activity. This supports a startup-related effect. Median, mean, and p99 alone do **not** identify exactly one slow request or prove that the first request was the outlier.

**Fix and result.** Preserve the original base/MTP c1 JSONs and use warm reruns. The old “queueing is impossible at c1” explanation is too strong: one client request rules out queueing behind another request from that same client, but not waiting for initialization, other clients, resources, or scheduler activity.

**Architectural lesson.** TTFT crosses the client, scheduler, prefill, and runtime boundaries. A tail warning is a symptom detector, not a causal diagnosis.

**Evidence:** [cold base c1](Qwen3.8-Flash-Next-FP8/results/_base_c1_jitcold/), [cold MTP c1](Qwen3.8-Flash-Next-FP8/results/_mtp_c1_jitcold/), [warm original base c1](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c1.json). The original base arm was later superseded for the main batch comparison by Bug 12's corrected arm.

### Bug 10 — Cold-prefix, complete runs disagree with warm reruns

**Symptom and result.** DeepSeek's dev20073 bridge produces:

| Concurrency | First-pass tok/s | Warm-rerun tok/s | First-pass shortfall |
|---:|---:|---:|---:|
| 4 | 89.3 | 219.4 | about 59% |
| 16 | 124.0 | 330.0 | about 62% |
| 64 | 284.0 | 386.4 | about 26% |

**Check.** Startup logs record substantial kernel warmup, and early runs have large latency tails. That supports incomplete runtime warmup as a likely contributor. The saved summaries do not uniquely attribute all elapsed time to JIT versus scheduler waiting. A large engine-version effect is possible in principle; its size alone is not grounds to reject data.

**Fix.** Quarantine first-pass measurements, stabilize the runtime, and rerun. Apply the same readiness and retention criteria to both sides of an A/B; do not keep only reruns that agree with the desired ranking. Future experiments should randomize arm order and preserve all repeats.

**Architectural lesson.** API health, successful functional output, kernel readiness, prefix state, and measurement stability are separate conditions. A throwaway request may not exercise every shape in a sweep.

**Scope.** The retained DeepSeek bridge ratios are 0.992–0.998 between builds on its batch grid. This does not establish universal engine equivalence for GLM, Qwen, or long-context/prefix workloads.

**Evidence:** [first-pass bridge data](deepseek_v4_flash/results/mtp-off-bridge-dev20073_queued/), [retained bridge and audit](deepseek_v4_flash/results/mtp-off-bridge-dev20073/), [bridge startup log](deepseek_v4_flash/logs/serve_bridge_dev20073.log).

### Bug 11 — The smoke-test script fails before testing the model

**Symptom.** `missing required tool: bc`; the banner also names the wrong model and port.

**Check.** Inspect client dependencies and endpoint identity. An error before any HTTP request cannot diagnose the model. An answer from another server cannot validate the intended deployment.

**Fix and result.** The incident uses a direct request to the explicit Qwen model on port 8001, obtaining a simple arithmetic response. This is a functional smoke test, not a quality evaluation. The shared [sending.sh](sending.sh) still depends on `bc`; the workaround did not fix that dependency.

**Architectural lesson.** The measurement client and server are separate systems. Match the client tokenizer/config support, model ID, and port with the server before investigating GPU kernels.

**Evidence:** [historical direct-request record](archive/notes/fix_bug_previous.md), [Qwen launch scaffold](Qwen3.8-Flash-Next-FP8/_common.sh), [shared smoke test](sending.sh).

## 5. Controlled comparisons: learn when a fix changes the scientific conclusion

### Bug 8 — One incompatible FP8 layout is not universal FP8 failure

**Symptom.** The first GLM FP8 route fails with `pe_dim must be 64 for fp8_ds_mla`.

**Initial hypothesis.** Requested page sizes might be incompatible. The run actually reaches the cache write; GLM's NoPE configuration has zero positional dimensions while that implementation expects 64.

**Correction.** This identifies a mismatch with the selected layout/kernel in the tested build. It does not establish that all FP8 KV layouts are impossible for GLM or H100. Changing a checkpoint field just to satisfy an assertion would not be a correctness-preserving fix.

**Alternative and result.** The FlashInfer 0.6.18 overlay exposes a NoPE-compatible SM90 path with `--kv-cache-dtype fp8`. The saved run also routes MoE through DeepGEMM after an overlay/binary API mismatch (`Expected 8 but got 9 arguments`) on the other MoE path. A matching BF16 control is required because version, attention backend, and MoE path differ from the original stack.

| GLM at c64 | Output tok/s | What this comparison estimates |
|---|---:|---|
| Original BF16 | 447.1 | Original deployment |
| Alternate-stack BF16 | 348.8 | −22.0% combined software/backend-path effect |
| Alternate-stack FP8 | 262.0 | −24.9% dtype-associated effect on that stack |

The full 41.4% drop combines both steps; percentage losses multiply, not add. The paired dtype experiment includes the backend's different BF16/FP8 execution paths; it is not a measurement of storage bytes in isolation. Prior capacity accounting reports about 1.805× pool growth across its named startup comparison; session-specific capacities must not be mixed.

**Important execution detail.** These saved overlay scripts set `FLASHINFER_DISABLE_VERSION_CHECK=1` and use mixed Python/compiled package versions. Treat them as experimental evidence, not a validated production recipe. Completed synthetic requests do not establish numerical parity or full ABI compatibility.

**Architectural lesson.** Compatibility is a tuple: **checkpoint geometry + storage layout + kernel + GPU architecture + software version**. Record the failed route and search for a compatible alternative; then isolate the variables changed by that alternative.

**Evidence:** [failed FP8 log](GLM-5.3-Flash/logs/serve_fp8kv_20260902-025357.log), [successful alternate log](GLM-5.3-Flash/logs/serve_fp8kv-fi618_20260902-040441.log), [FP8 launch](GLM-5.3-Flash/run_fp8kv_fi618.sh), [paired BF16 launch](GLM-5.3-Flash/run_bf16kv_fi618.sh), [BF16 control results](GLM-5.3-Flash/results/bf16kv-fi618/), [FP8 results](GLM-5.3-Flash/results/fp8kv-fi618/).

### Bug 12 — A plausible architecture story explains a startup confound

**Symptom.** Original Qwen c4 throughput is 162.2 tokens/s, below GLM's 225.3. A fine-grained-routing explanation was written before validating startup state.

**Discriminating check.** The utilization rerun changes more than the intended flag:

| Startup / result | Original Qwen base | Corrected batch arm |
|---|---:|---:|
| GPU memory utilization | 0.85 | 0.82 |
| Recorded peak activation | 17.07 GiB | 0.99 GiB |
| Compilation time | 60.41 s | 1.03 s |
| Reported pool tokens | 2,048,645 | 3,197,331 |
| c4 output tokens/s | 162.2 | 256.8 |

Lower utilization accompanies a **larger** pool, showing that other startup accounting changed. The logs support a cold/warm profiling confound. They do not prove that every saved byte was actually recoverable, that specific preemptions caused the c4 slowdown, or that all other variables were identical.

**Fix and result.** Use the corrected arm for the main batch grid. Qwen now leads that grid. Its original context/prefix data still use the earlier pool; its MTP comparison also retains pool/startup differences. Those claims need matched reruns, not a correction factor transferred from c4.

**What would isolate the cause?** Repeat cold/warm startup at the same utilization, capture the full memory budget, then compare matched explicit pool sizes if supported by that pinned build. Record preemptions and request timelines. A suggested maximum-memory setting in a log is not itself proof of a sizing bug.

**Architectural lesson.** Checkpoint architecture is unchanged, but runtime memory partitioning changes how the system serves it. “Only one CLI flag changed” is weaker than “only one experimental variable changed.”

**Evidence:** [Qwen startup audit](Qwen3.8-Flash-Next-FP8/results/base-util082/RESULT-util-ab.md), [corrected c4 JSON](Qwen3.8-Flash-Next-FP8/results/base-util082/batch_isl16k_c4.json), [original c4 JSON](Qwen3.8-Flash-Next-FP8/results/base/batch_isl16k_c4.json). Some causal wording in the historical audit is stronger than the evidence; the interpretation above supersedes it.

### Bug 13 — Enabled counters, absent collection, incomplete accounting

**Symptom.** Launches enable MFU metrics, but early JSONs have no collected FLOP/byte estimates. Later collection reveals partial component coverage.

**Check the whole measurement chain:**

```text
metric enabled → component implemented → counter emitted → scrape succeeds
→ labels/ranks selected → before/after delta → aligned interval → saved provenance
```

The script previously enabled the producer without consuming its values. Cumulative counters require deltas; a single final scrape mixes current and prior work. The saved implementation now divides byte/FLOP deltas by benchmark duration. Scrape endpoints and duration must cover the same work; unrelated requests or setup work in the interval would bias that rate.

| Recorded model coverage | What the names establish | What they do not establish |
|---|---|---|
| Qwen: attn, ffn, unembed | Three estimators instantiated | Every hybrid/PLE/state operation correctly modeled |
| GLM: ffn, unembed | Attention absent from these estimates | Full-model bytes or bandwidth |
| DeepSeek: unembed | Much of the model is omitted | Cross-model bandwidth comparability |

**Fix and result.** The harness stores estimates, provenance, and available coverage. GLM values are roughly 319–608 GB/s/GPU, about 9.5–18.1% of an assumed 3,350 GB/s peak, across recorded arms. These are **engine estimates [E]**, not hardware counters. Missing work makes them partial, but imperfect estimation means they are not guaranteed numerical lower bounds on real HBM traffic.

**Correction to the old conclusion.** Neither the older “about 1%” arithmetic nor the partial newer estimate proves that decode is latency-bound or not bandwidth-bound. Whole-request averages can hide bandwidth-bound kernels. Instantiating all named estimators does not prove full coverage. A whole-model Boolean alone also cannot explain both complementary estimator gates rejecting: the exact exceptions and other checks must be inspected.

**Architectural lesson.** Observability needs the same understanding of mixed layers, routing, and state that execution needs. Trust an estimator only for the operations and interval it actually covers.

**Evidence:** [collector and formulas](bench.sh), [GLM coverage manifest](GLM-5.3-Flash/results/bf16kv/manifest.txt), [example with counters](GLM-5.3-Flash/results/bf16kv/batch_isl16k_c32.json), [historical coverage investigation](archive/notes/fix_bug_previous.md). Operator traces and hardware counters remain outstanding.

## 6. Remaining gaps in the saved harness — not fixes performed here

This audit found that the prose had sometimes called a warning an assertion. Read the control flow, not the printed label.

| Current behavior in bench.sh | Why it matters | Proposed improvement |
|---|---|---|
| Cache scrape/arithmetic defaults can become zero | Missing data can appear as `cold_run=true` | Preserve “unknown”; require successful scrapes, expected labels, and no counter reset |
| Positive cache-hit delta prints FAIL but does not reject the result in that branch | A contaminated JSON can remain in the result set | Explicit invalid status and nonzero return for cold-prefix tests |
| Completion guard rejects only zero completions | Partially failed points can survive | Require expected completion count and zero failures |
| Prefix path has separate, lighter validation | Main-path guards do not cover every workload | Shared result validation plus prefix-specific cache protocol |
| p99/median tail check only warns | Flagged data can be reported without review | Record review status; preserve repeats and justification |
| Model mismatch warns, then adopts served ID | Directory/metadata may still identify intended model | Require explicit identity match before measuring |
| Existing JSONs are skipped; manifests can be rewritten | A later session may describe earlier files | Immutable run IDs and per-point config/startup fingerprints |
| MFU numerator spans scrapes; denominator uses benchmark duration | Windows may include different work | Timestamp scrapes and validate interval/rank/traffic alignment |

The prior result audit confirmed completion and throughput arithmetic for 95 selected files. It did not independently reconstruct all raw telemetry or prove every `cold_run` field. Keep that distinction when defending the report.

Before a future GPU run, use this sequence:

1. **Identity:** verify executable, package versions, revision, served model, port, GPU allocation, and actual backend.
2. **Resources:** inspect GPU holders and cache paths; exercise small writes before loading weights. Do not terminate unrelated processes as part of a diagnostic probe.
3. **Function:** send one explicit short request and validate its status/content.
4. **Readiness:** warm representative kernel shapes with disjoint prompts; preserve startup logs and memory accounting.
5. **Cache protocol:** specify cold-prefix, cold-fill, or prewarmed reuse; verify telemetry availability before interpreting zeroes.
6. **Measurement:** validate exact completed work, errors, token totals, durations, and latency distributions; save all repeats and manifests.
7. **Interpretation:** compare equivalent arms; label confounds, uncertainty, and hypotheses before writing a mechanism.

Historical Linux commands belong to the recorded GPU containers. Use [REPRODUCE.md](REPRODUCE.md) and the model scripts as environment-specific references, not universal local commands. In shell examples, a continuation backslash must be the final character on the line; the old annotated launch example violated that rule. Do not copy its inline comments after backslashes.

## 7. Practice explaining the diagnosis

Try answering the middle column before reading the last one.

| Case | Your next discriminating check | Answer / reasoning |
|---|---|---|
| CUDA fails even in eager mode | What shared dependency should you inspect? | Earlier compilation/filesystem errors; graph capture is no longer a sufficient explanation |
| 16,000 cache-hit tokens at every batch point | Cross-point leakage or self-contamination? | Inspect warmup prompt identity; constant block-aligned hits fit self-contamination |
| All requests fail with exit 0 | GPU OOM or request validation? | Read response errors; probe a valid shorter request and compare actual token length to limit |
| c1 p99 is high but median stable | Can you prove exactly one first-request outlier? | No; need per-request timings. Startup is a hypothesis supported by reruns |
| FP8 route says positional dimension must be 64 | Abandon all FP8 experiments? | No; inspect checkpoint geometry and alternate supported layouts/backends |
| Lower utilization produces more cache | Did the utilization flag increase memory? | Compare activation/workspace/graph budgets and startup state; more than one variable changed |
| MFU enabled, byte estimate absent | Is GPU bandwidth zero? | Inspect metric producer, component coverage, scrape, and JSON consumer |
| Same selected backend, different dtype | Pure memory-byte experiment? | No; dtype can change dequantization and kernel paths even within one backend |

### Two calculations to rehearse

**Backend decomposition:** `348.8 / 447.1 ≈ 0.780`; `262.0 / 348.8 ≈ 0.751`; `0.780 × 0.751 ≈ 0.586`. The total loss is about 41.4%, not the sum of 22.0% and 24.9%.

**Serving cost:** at eight GPUs and $2.50/GPU-hour, the node costs $20/hour. `20 × 1e6 / (3600 × 447.1) ≈ $12.43/million output tokens`. This includes input processing in the measured runtime. A higher-throughput setting can still be worse for interactive latency.

### A reusable incident record

```text
Environment and exact arm:
Symptom and earliest relevant error:
Hypothesis:
Cheapest experiment that could disprove it:
Observed result and retained evidence:
Change made (and any other variable that changed):
Verification: function / readiness / valid measurement / reproducibility:
Architectural implication:
What remains uncertain:
Guard implemented versus guard still proposed:
```

In the presentation, spend the debugging time on **three linked stories**: recurrent layers require sequence-state capacity (Bug 1); warmup interacts with prefix caching (Bug 6); runtime memory accounting can impersonate architecture (Bug 12). Use Bugs 3, 8, and 13 for follow-up questions about diagnosis, controlled experiments, and observability.
