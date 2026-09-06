# DeepSeek-V4-Flash — evidence index

Current scientific interpretation: [root report](../report.md). Arm roles and caveats: [experiment index](../experiments.md). Debugging: [fix_bug.md](../fix_bug.md).

43 sparse-attention layers with compressed history; 256 routed experts, 6 selected per token.

Main base is mtp-off-image on dev20051. The finer grid is util085-dev20073; the build bridge is mtp-off-bridge-dev20073. Do not combine them into one curve. Legacy mtp-off/mtp-on-noreuse supports only the qualified historical MTP comparison. Context-16K discrepancies remain unresolved.

- [Main result arm](results/mtp-off-image/)
- [Main manifest](results/mtp-off-image/manifest.txt)
- [Raw logs](logs/)
- [Historical launch notes](history/README.md)
- [Historical detailed analysis and tensor accounting](history/report.md)

Launch scripts are recorded Linux configurations with historical paths and package assumptions. Read [REPRODUCE.md](../REPRODUCE.md) before any authorized rerun. RESULT notes inside result folders document what was believed during an experiment; the root report and refined debugging guide supersede their causal overclaims. Preserve all raw files and quarantined arms.
