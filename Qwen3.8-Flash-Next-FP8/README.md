# Qwen3.8-Flash-Next-FP8 — evidence index

Current scientific interpretation: [root report](../report.md). Arm roles and caveats: [experiment index](../experiments.md). Debugging: [fix_bug.md](../fix_bug.md).

36 recurrent GDN + 12 QSA layers; 512 routed experts, 10 selected per token.

Main batch is base-util082 on dev20073 with warm compile cache and utilization 0.82. Context/prefix remain in base at utilization 0.85 with an earlier smaller pool. MTP retains startup/pool confounds. The run scaffold defaults to TP4; the measured TP8 manifest, not script defaults, identifies the headline deployment.

- [Main result arm](results/base-util082/)
- [Main manifest](results/base-util082/manifest.txt)
- [Raw logs](logs/)
- [Historical launch notes](history/README.md)
- [Historical detailed analysis and tensor accounting](history/report.md)

Launch scripts are recorded Linux configurations with historical paths and package assumptions. Read [REPRODUCE.md](../REPRODUCE.md) before any authorized rerun. RESULT notes inside result folders document what was believed during an experiment; the root report and refined debugging guide supersede their causal overclaims. Preserve all raw files and quarantined arms.
