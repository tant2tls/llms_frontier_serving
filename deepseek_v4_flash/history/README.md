> HISTORICAL MODEL NOTES — use the [current report](../../report.md), [experiment index](../../experiments.md), and [debugging guide](../../fix_bug.md) for corrected conclusions. Original location: `deepseek_v4_flash/README.md`. Some causal claims and configurations below are superseded.

# DeepSeek-V4-Flash — served configs and measured results

Per-model directory. **One of these per model** (`glm_5_3_flash/`, `qwen3_8_flash_next_fp8/`, …), each
holding the launch configs that were actually run, the raw results they produced, and the report.

Shared tooling stays at the repo root: `bench.sh` (sweep harness), `sending.sh` (smoke test),
`plan.md`, `potential_questions.md`.

## Status: measurement CLOSED (2026-09-01)

The server has been shut down and the GPUs released; work moved on to the next model. Everything in this
directory is final and self-contained — the configs, the 22 result JSONs, the logs, and the report.

**To pick this model back up:** re-launch `./run_nomtp.sh` (weights are still cached, ~255 s to
`/health` 200). Nothing needs re-downloading. The open gaps worth returning for are listed in
[`report.md`](../report.md) §8 — highest value first: `--enable-mfu-metrics` (one flag, replaces the
analytical bandwidth model), a `max_num_batched_tokens` sweep, and a TP sweep (2/4/8) to fix the
pessimistic per-GPU number.

## Contents

| Path | What it is |
|---|---|
| [`report.md`](../report.md) | **The deliverable** — measured throughput, normalized cost, bottleneck analysis |
| [`run_nomtp.sh`](../run_nomtp.sh) | **Base model, spec decode OFF — produced the PRIMARY numbers** |
| [`run.sh`](../run.sh) | Same, MTP on (`num_speculative_tokens=1`) — the §5 A/B arm only |
| `results/mtp-off/` | Base-model batch + context sweeps → report §1–§3 |
| `results/mtp-on-noreuse/` | MTP-on batch sweep, unique seeds → report §5 A/B |
| `results/mtp-on/` | Context + prefix sweeps, MTP on. **Batch points here are seed-contaminated** (report §2) — retained deliberately so the error is auditable |
| `logs/server_nomtp.log` | Startup log of the base-model server. Source for the chunked-prefill and EP-sharding facts |
| `logs/results_sweep.log` | Console output of the first (MTP-on) sweep |

Each `results/*/` has a `manifest.txt` with hostname, all 8 GPUs, driver, torch/vLLM versions, the
server command line scraped from `ps`, and the quantization.

## The config that ran

Both scripts are identical except for `--speculative-config`. Settings that are **not** free choices:

| Flag | Why |
|---|---|
| `--block-size 256` | **Mandatory.** `sparse_mla.py:53` returns `[256]`, a single-element list. `storage_block_size = block_size // compress_ratio` with ratios {4,128}: at 128 the ratio-128 layers degenerate to 1, at 64 they floor to **0**. Do not change or remove. |
| `--gpu-memory-utilization 0.82` | `0.9` OOMs. KV is allocated *before* warmup, leaving nothing for the fp32 logits copy (`sampler.py:196`, 506 MiB = 1,024 × 129,536 × 4 B). MTP doubles that buffer. |
| `--kv-cache-dtype fp8` | Aliases to `fp8_ds_mla`, the UE8M0 paged layout V4 expects. |
| `--tokenizer-mode deepseek_v4` | No Jinja chat template ships with the model; this supplies it. |
| `--reasoning-parser deepseek_v4` | Output splits into `message.reasoning` + `message.content`. Budget ≥600 output tokens or you get `content: null` + `finish_reason: "length"`. |
| `--enable-expert-parallel` | 32 of 256 experts per rank at EP8. |

**TP=8 was NOT a considered choice** — weights are 18.6 GiB/GPU and the model fits on 3 H100s. Per-GPU
throughput here is therefore **pessimistic**; a TP sweep is an open gap (report §8).

## Reproducing

```bash
export HF_HOME=/prj/corp/airesearch/lasvegas/vol22-scratch/users/tanngo/LLMs_serving_report/cache
export HF_HUB_CACHE=$HF_HOME/hub          # 150 GiB already cached; do not re-download

cd deepseek_v4_flash && ./run_nomtp.sh    # ~255 s to /health 200
cd .. && ./sending.sh                     # functional check
./bench.sh batch context prefix           # writes to deepseek_v4_flash/results/<timestamp>/
```

`bench.sh` derives its output dir from the served model name, so it lands in the right per-model folder
automatically. Override with `MODELDIR=` or `OUTDIR=`.

**Only one server can hold the GPUs.** To switch arms: `pkill -f "vllm serve"`, wait ~25 s for HBM to
free, then launch the other script.

## Headline numbers (base model, MTP off, 8×H100)

| conc | tok/s | per B-active | per GPU | TTFT p50 | TPOT p50 |
|--:|--:|--:|--:|--:|--:|
| 1 | 92 | 6.8 | 11.5 | 714 ms | 8 ms |
| 64 | 383 | 28.4 | 47.9 | 3,687 ms | 151 ms |

290.9 B total / 13.49 B active params (measured from safetensors headers) → **21.6× sparsity**, not the
42.7× that `E/k` suggests, because attention doesn't sparsify. Prefix sharing is the dominant lever:
**3.7× throughput, 4.6× lower TTFT**. Spec decode gives 1.25× at c=1 and **0.99× at c=64**.

See [`report.md`](../report.md) for the full analysis and the list of what is *not* measured.
