#!/usr/bin/env bash
# Qwen3.8-Flash-Next-FP8 -- MTP ON. **A/B ARM ONLY, NEVER THE HEADLINE NUMBER.**
#
#   ./run_mtp.sh                  # serve on :8001, GPUs 4,5,6,7, TP4, MTP k=1
#   NSPEC=3 ./run_mtp.sh          # the recipe's k=3 -- a DIFFERENT experiment
#
# Then, from the REPO ROOT:
#   MODEL=Qwen/Qwen3.8-Flash-Next-FP8 PORT=8001 \
#   VLLM=/usr/local/bin/vllm PY=/usr/bin/python3 \
#   MODELDIR=Qwen3.8-Flash-Next-FP8 OUTDIR=Qwen3.8-Flash-Next-FP8/results/mtp-tp4 \
#     ./bench.sh batch
#
# Identical to run.sh plus --speculative-config, so the ONLY difference from the
# headline arm is spec decode. Comparing an MTP-accelerated model against a base
# model measures inference tricks, not architecture, and flatters whichever vendor
# shipped a draft head -- hence a separate arm (rule 1).
#
# ---------------------------------------------------------------------------
# WHY NSPEC DEFAULTS TO 1 AND NOT THE RECIPE'S 3
# ---------------------------------------------------------------------------
# k=1 matches the V4-Flash and GLM-5.3 A/B arms, so the three MTP results are
# comparable to each other. The recipe's k=3 would be a better-tuned deployment
# but a WORSE experiment: it changes acceptance-rate dynamics and the draft cost
# simultaneously, so a k=3-vs-base delta cannot be compared to V4's k=1 delta.
# Run k=3 as a bonus point AFTER k=1 if there is time; do not substitute it.
#
# The checkpoint ships mtp_num_hidden_layers=1 with a HYBRID draft layer:
#   "mtp": {"hybrid": true, "layer_types": ["full_attention"], "num_hidden_layers": 1,
#           "rope_theta": 10000000}
# The draft head is a FULL-ATTENTION layer even though 36 of the 48 target layers
# are linear-attention. So the draft does not pay the GDN state cost but does pay
# KV -- an asymmetry neither V4 (uniform DSA) nor GLM-5.3 (shared indexer across
# MTP iterations) has. That is the interesting question for this arm.
#
# ⚠️ MTP TIGHTENS A REAL CONSTRAINT, and this is why the arm can fail where
# run.sh succeeds. The QSA side cache asserts (common/qsa_cache.py:773-790):
#     span     = compress_ratio + num_speculative_tokens
#     capacity = compress_ratio * ceil(span / compress_ratio)
#     assert block_size % capacity == 0
# With indexer_compress_ratio=4:
#     k=0  -> capacity 4  -> block_size % 4 == 0
#     k=1  -> capacity 8  -> block_size % 8 == 0
#     k=3  -> capacity 8  -> block_size % 8 == 0
#     k=5  -> capacity 12 -> block_size % 12 == 0   <- awkward, avoid
# The comment explains why: the ring must hold the open group's committed keys
# PLUS every row a speculative step writes before acceptance is known, or a
# rejected draft row overwrites a committed key the next step needs. block_size
# is left unpinned so vLLM resolves the LCM across all four spec kinds; if it
# picks a value the assert rejects, THAT IS THE FINDING -- record the error, do
# not hand-pin a block size to force it through.
#
# Also: the recipe warns to drop k below 3 under MTP memory pressure. At k=1 with
# 256 concurrency slots the GDN state is already ~6.8 GiB/rank at TP4, and MTP
# adds a second logits buffer. If startup OOMs, lower --max-num-seqs before
# lowering utilization, and say so in the report.
#
# Measured for reference: on V4-Flash, MTP gave 1.25x at c=1 but 0.99x at c=64 and
# cost 2.4% of KV capacity -- the gain decays to nothing once the batch saturates
# the machine. The question here is whether a full-attention draft head over a
# mostly-linear-attention target changes that crossover.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

NSPEC=${NSPEC:-1}

preflight_gpus || exit 1
preflight_caches || exit 1

printf '\n  MTP arm: num_speculative_tokens=%s (QSA ring needs block_size %% %s == 0)\n' \
  "$NSPEC" "$("$PY_BIN" -c "
import math
cr=4; span=cr+$NSPEC; print(cr*math.ceil(span/cr))")"

launch "mtp${NSPEC}-tp${TP}" "${COMMON_ARGS[@]}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}"
