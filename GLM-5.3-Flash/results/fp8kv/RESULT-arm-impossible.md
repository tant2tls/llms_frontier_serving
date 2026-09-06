> HISTORICAL ARM NOTE — numbers describe this arm, but causal claims may be superseded. Read the [current report](../../../report.md), [corrected debugging guide](../../../fix_bug.md), and [arm index](../../../experiments.md) before citing. In particular: no proven hardware bottleneck, universal engine equivalence, universal FP8 impossibility, or universal capacity-only effect is established.

# FP8 KV arm (`fp8_ds_mla`): IMPOSSIBLE for GLM-5.3-Flash — a clean, structural result

Ran `./run_fp8kv.sh` on 8×H100 / CUDA 13.0, image `vllm/vllm-openai:glm53-flash`,
vLLM `0.1.dev20051+g487ecf187`, on **2026-09-02**. Log:
`../../logs/serve_fp8kv_20260902-025357.log`.

**Zero benchmark points. This is a real finding, not a gap** — per this project's
rule 7 (state what didn't run and why, with the exact error).

## What happened

Weights loaded fine (21.28 s warm), backend resolved to `FLASHMLA_SPARSE`, then
every worker died at the first KV write:

```
RuntimeError: concat_and_cache_mla,
  /workspace/csrc/libtorch_stable/cache_kernels.cu:866,
  pe_dim must be 64 for fp8_ds_mla
```

## Why it is STRUCTURAL, not a tuning problem

`fp8_ds_mla` is DeepSeek-V3.2's 656-byte-per-token KV layout. It assumes a
**decoupled RoPE** dimension of exactly 64 — the kernel hardcodes `pe_dim == 64`.

**GLM-5.3-Flash is a NoPE model.** From `config.json` → `text_config`:

| field | value |
|---|--:|
| `qk_rope_head_dim` (= `pe_dim`) | **0** |
| `qk_nope_head_dim` | 256 |
| `mla_use_nope` | `true` |
| `kv_lora_rank` | 512 |

`pe_dim = 0 ≠ 64`, and no flag can change it — it is the checkpoint's attention
geometry. GLM-5.3 has **no RoPE component in its KV at all**, so a layout defined
around one cannot represent it. **FP8 KV via `fp8_ds_mla` is unreachable for this
model on any hardware, not just SM90.**

## This is a DIFFERENT failure than `run_fp8kv.sh` predicted

The script's header predicted a **block-size conflict** (`FLASHMLA_SPARSE`
advertises kernel block size `[64]`, GLM's kpool indexer requires a multiple of
128 → "No common block size"). That is **not** what happened: the backend resolved
without complaint and got as far as the first `concat_and_cache_mla` call. Worth
recording — the predicted-but-wrong hypothesis is itself information, and the
block-size constraint evidently resolves (vLLM auto-raises to 640, which is legal
for both).

## Consequence for the report

The **KV-dtype asymmetry against DeepSeek-V4-Flash is now confirmed from both
directions and is UNCLOSABLE** — and the reason is stronger than "kernel not
compiled for Hopper":

| model | KV dtype available | why |
|---|---|---|
| DeepSeek-V4-Flash | **fp8_ds_mla only** on H100 | its BF16-KV backend is gated to Blackwell (`capability.major in [10,12]`) |
| GLM-5.3-Flash | **BF16 only** | (a) FP8-with-in-kernel-dequant needs `FLASHINFER_MLA_SPARSE_SM90`, gated off — image ships FlashInfer 0.6.17 < 0.6.18, no `ckv_scale_arr`; **(b) `fp8_ds_mla` needs `pe_dim == 64`, GLM is NoPE (`pe_dim = 0`) — architectural, not a version gate** |

Each model can only run the KV dtype the other cannot. **Report it; do not try to
resolve it.** Mitigating context: KV is not the binding resource in either run
(GLM BF16 KV ≈ 11.35 KiB/token; peak KV usage 89.5% only at ISL 260K × 8), and the
**weight** dtype — which drives the expert-read term the report is actually about —
is FP8 e4m3 on both, so that comparison stays intact.

## Do not retry

- `--block-size 64` → violates the kpool assert (`block_size % 128 == 0`) and fails
  deeper with a less legible error.
- `--kv-cache-dtype fp8` (the alias) → resolves to the same `fp8_ds_mla` path.
- Upgrading FlashInfer to ≥ 0.6.18 → would unlock the *other* FP8 route
  (`FLASHINFER_MLA_SPARSE_SM90`), which is untested here and is a separate
  experiment. It would **not** fix `fp8_ds_mla`; `pe_dim` is still 0.
