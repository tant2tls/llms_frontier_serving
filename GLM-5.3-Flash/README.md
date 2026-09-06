# GLM-5.3-Flash — evidence index

Current scientific interpretation: [root report](../report.md). Arm roles and caveats: [experiment index](../experiments.md). Debugging: [fix_bug.md](../fix_bug.md).

34 recurrent KDA + 11 sparse-attention layers; 288 routed experts, 8 selected per token.

GLM main base is BF16 KV on dev20051 at utilization 0.82. MTP batch/context and alternate-stack KV controls are recorded separately. The original fp8_ds_mla route failed; fp8kv-fi618 ran through an alternate NoPE-compatible path with an experimental package overlay.

- [Main result arm](results/bf16kv/)
- [Main manifest](results/bf16kv/manifest.txt)
- [Raw logs](logs/)
- [Historical launch notes](history/README.md)
- [Historical detailed analysis and tensor accounting](history/report.md)

Launch scripts are recorded Linux configurations with historical paths and package assumptions. Read [REPRODUCE.md](../REPRODUCE.md) before any authorized rerun. RESULT notes inside result folders document what was believed during an experiment; the root report and refined debugging guide supersede their causal overclaims. Preserve all raw files and quarantined arms.
