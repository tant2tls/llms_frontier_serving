#!/usr/bin/env bash
# GLM-5.3-Flash -- BF16 KV + MTP ON. **A/B ARM ONLY, NEVER THE HEADLINE NUMBER.**
#
#   ./run_mtp.sh
#
# Then, from the REPO ROOT:
#   MODEL=zai-org/GLM-5.3-Flash PORT=8001 \
#   MODELDIR=GLM-5.3-Flash OUTDIR=GLM-5.3-Flash/results/bf16kv-mtp \
#     ./bench.sh batch
#
# Identical to run.sh plus --speculative-config, so the ONLY difference from the
# headline arm is spec decode. Comparing an MTP-accelerated model against a base
# model measures inference tricks, not architecture, and flatters whichever vendor
# shipped a draft head -- hence a separate arm.
#
# GLM-5.3 ships num_nextn_predict_layers=1 AND index_share_for_mtp_iteration=true:
# it SHARES THE SPARSE-ATTENTION INDEXER across MTP iterations, which DeepSeek-V4
# does not. So this is a genuinely different MTP design, not just "MTP on", and the
# interesting question is whether indexer sharing changes the high-concurrency
# crossover.
#
# Measured on V4-Flash for reference: MTP gave 1.25x at c=1 but 0.99x at c=64, and
# cost 2.4% of KV capacity -- the gain decays to nothing once the batch saturates
# the machine. If GLM's shared indexer changes that shape, it is a real finding;
# if it does not, that is also worth stating.
#
# num_speculative_tokens: 1 matches the checkpoint's single MTP layer AND the V4
# A/B. The recipe's GB200 example uses 5, which would need the draft head run
# recurrently -- a different experiment, and not comparable to the V4 A/B.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

preflight_gpus 8 || exit 1
preflight_caches || exit 1

# BF16 KV (no --kv-cache-dtype), matching run.sh so the A/B isolates spec decode.
# NSPEC selects the draft depth. BOTH values are worth measuring and they answer
# DIFFERENT questions -- see ../WHY.md §7.
#
#   NSPEC=1 (default here)  matches the DeepSeek-V4-Flash A/B exactly, so the
#                           GLM-vs-V4 spec-decode comparison is apples-to-apples.
#                           V4 measured: 1.25x at c=1 -> 0.99x at c=64.
#   NSPEC=5                 what the vLLM recipe specifies for this model. GLM ships
#                           ONE MTP layer (num_nextn_predict_layers=1), so 5 tokens
#                           means running that head RECURRENTLY 5x. Deeper drafts
#                           compound acceptance -- expected accepted length goes like
#                           (1-p^5)/(1-p) instead of p -- so more upside AND more
#                           wasted verify when acceptance is low.
#
# ⚠️ Run BOTH. n=1 isolates the ARCHITECTURE difference (GLM's shared indexer vs
# V4's unshared) at matched draft depth; n=5 measures the CONFIGURATION the vendor
# actually recommends. Reporting only n=5 against V4's n=1 would conflate the two
# and is the kind of comparison that gets a paper rejected.
#
# The falsifiable prediction (WHY.md §7): because GLM sets
# index_share_for_mtp_iteration=true and V4 does not, each extra draft token is
# cheaper for GLM, so GLM's BREAK-EVEN CONCURRENCY should be HIGHER than V4's --
# GLM should still show gain where V4 has already dropped below 1.0x.
# Counter-mechanism to watch: GLM's 34 KDA layers must advance a RECURRENT STATE per
# accepted token, which does not parallelize across draft positions the way attention
# does. If GLM underperforms the prediction, probe that first.
NSPEC=${NSPEC:-1}
printf '\n  \033[1mMTP draft depth: %s\033[0m  (NSPEC=5 for the recipe config)\n' "$NSPEC"

launch "bf16kv-mtp-n${NSPEC}" "${COMMON_ARGS[@]}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NSPEC}}"
