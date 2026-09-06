#!/usr/bin/env bash
# HISTORICAL UTILITY — superseded by tools/summarize_results.py.
# Original formulas, filters, and scientific comments are retained for provenance,
# not as current guidance. Do not use this archived file for new comparisons.
#
#   ./normalize.sh                                  # all models it can find
#   ./normalize.sh GLM-5.3-Flash/results/bf16kv     # one result dir
#   ./normalize.sh --csv > normalized.csv           # machine-readable
#
# ---------------------------------------------------------------------------
# WHY THIS SCRIPT EXISTS
# ---------------------------------------------------------------------------
# **Raw tok/s is never a valid cross-model comparison.** The minimum deployable
# footprint across this report's models spans 21x (GLM-4.7-Flash 1 GPU -> Kimi-K3
# ~21 GPUs), so raw throughput mostly reports HOW MUCH SILICON WAS USED. Every
# claim in the report must carry all three normalizations below, and this script is
# the single place they are computed so the report and the JSONs cannot drift.
#
#   tok/s per GPU        what an operator pays -- THE ONLY ONE THAT MAPS TO COST.
#                        Depends on the TP/EP choice, so it is only comparable at
#                        equal TP, or with the TP stated.
#   tok/s per B-active   is the architecture's FLOP budget used well? Ignores the
#                        HBM you had to buy to hold the idle experts.
#   tok/s per B-total    efficiency per byte of HBM purchased. PENALIZES SPARSITY
#                        BY DESIGN -- a 21x-sparse MoE looks bad here and that is
#                        the honest reading of "you bought 306 GiB to use 15 B".
#
# Report all three or none. Picking the flattering one per model is how benchmark
# tables become marketing.
#
# ---------------------------------------------------------------------------
# ⚠️ WHAT NORMALIZATION CANNOT FIX -- STATE THESE, DON'T BURY THEM
# ---------------------------------------------------------------------------
# 1. TOKENIZERS DIFFER. tok/s is not commensurable across families: a denser
#    tokenizer does more work per token. GLM-5.3 vocab=154,880 vs V4-Flash's
#    ~129k. For strict cross-family claims use bytes/s or a fixed corpus.
# 2. KV DTYPE IS FORCED AND OPPOSITE on H100 (see CLAUDE.md): V4-Flash can only
#    run fp8_ds_mla, GLM-5.3 can only run BF16. Each model can only run the dtype
#    the other cannot. Hardware/kernel availability, not a choice.
# 3. TP=8 ON A MODEL THAT FITS IN 3 GPUS IS A CONFOUND. V4-Flash needs 3 GPUs but
#    was measured at TP=8, so its per-GPU number is PESSIMISTIC. GLM-5.3 genuinely
#    needs ~5, so TP=8 is closer to a real deployment for it. Do not present
#    per-GPU numbers as if the two were equally handicapped.
# 4. LINEAR-ATTENTION MODELS DON'T SCALE LIKE ATTENTION MODELS. GLM-5.3 is 34 KDA
#    (recurrent state, O(1)/token) + 11 DSA. Its cost-vs-context curve is
#    STRUCTURALLY different from a pure-attention model's, so a single
#    "tok/s per B-active" at one ISL hides the interesting part. Always pair the
#    batch table with the context table.
# 5. BOTH models' param counts are now MEASURED [M] from safetensors headers, so
#    per-B-active IS comparable between them. GLM 321.34/17.38 B, V4 290.91/14.08 B.
#    Watch the packing: V4's I8 expert tensors hold 2 fp4 values per byte.
# 6. GLM-5.3 SHIPS NATIVE FP8 WEIGHTS, V4-Flash DOES NOT. 314.40 of GLM's 321.34 B
#    are F8_E4M3 on disk (97.8%); the 6.93 B BF16 remainder is q/k/v/o_proj, the
#    embeddings/lm_head, kv_b_proj, the indexer, the MoE gate, and the vision tower
#    (a 1,509-entry modules_to_not_convert list). V4-Flash uses MXFP4 experts. So
#    "both FP8" is TOO COARSE for the expert-read term the report is about:
#    GLM reads ~1 byte/expert-param, V4 ~0.5. State the bytes, not the label.
#
# ---------------------------------------------------------------------------
# IS THROUGHPUT LINEAR IN ACTIVE PARAMS? (the question this script sets up)
# ---------------------------------------------------------------------------
# Naive expectation: decode is memory-bound, so tok/s should go like
# 1 / (active_params x bytes_per_param) -- i.e. tok/s x B-active should be roughly
# CONSTANT across models on the same hardware. It is not, and the deviation is the
# finding. Reasons it breaks, all visible in this report's data:
#   - EP all-to-all sits on the critical path of every decode step and is a NETWORK
#     term that active-param accounting cannot see.
#   - Recurrent (KDA) layers read a fixed-size state, not O(ctx) KV, so their cost
#     does not grow with context the way MLA/DSA layers do.
#   - Achieved HBM bandwidth measured <5% of peak, so the models are NOT actually
#     sitting at the memory roofline -- the bound is elsewhere.
# So: report tok/s-per-B-active, then say explicitly how far from constant it is.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CSV=0
[[ ${1:-} == --csv ]] && { CSV=1; shift; }

# ---------------------------------------------------------------------------
# MODEL FACTS. B_total / B_active are ANALYTICAL [A] from config.json; GiB is the
# MEASURED safetensors total. Sources in CLAUDE.md; re-derive if a repo changes.
#
#   V4-Flash : 290.91 B total / 14.08 B active -- **MEASURED from safetensors shapes**
#              (2026-09-02). ⚠️ THE PACKING BITES HERE: V4's routed experts are stored
#              as `I8` tensors holding TWO MXFP4 values per byte, so raw element counts
#              UNDERCOUNT by 2x. Naively summing shapes gives 158.07 B / 11.01 B, which
#              is WRONG. Correct: 141.734 B I8 elements x2 = 283.468 B logical experts
#              + 7.438 B non-expert = 290.91 B total; active = 7.438 + 283.468*6/256
#              = 14.08 B. Reconciles with the card's 284 B / 13 B and with the earlier
#              290.9 B analytical estimate.
#              => bytes per expert-param: V4 0.5 (fp4) vs GLM 1.0 (fp8). State BYTES.
#   GLM-5.3  : 321.34 B total / 17.38 B active -- **MEASURED from safetensors tensor
#              shapes**, not estimated from config.json. Buckets (mutually exclusive,
#              they partition the 321.34 B exactly):
#                  routed experts (main) 304.42 B   <- only k/E = 8/288 fire per token
#                  always-on (attn+dense+emb) 8.92 B
#                  MTP / nextn layer      7.43 B    <- EXCLUDED (base model, MTP off)
#                  vision tower           0.56 B    <- EXCLUDED (--limit-mm-per-prompt 0)
#              active = 8.92 + 304.42*8/288 = 8.92 + 8.46 = 17.38 B
#              ⚠️ CORRECTED 2026-09-02. An earlier ANALYTICAL estimate of
#              "310.96 B / 15.01 B" was wrong: it under-counted always-on attention
#              (GLM keeps q/k/v/o BF16 and adds a kpool indexer) and mis-handled the
#              MTP layer. It made GLM look BETTER per-B-active than it is. Always
#              read tensor shapes; never hand-derive param counts from config.json.
# ---------------------------------------------------------------------------
facts() {                       # facts <dirpath> -> "name B_total B_active gpus gib"
  case $1 in
    *GLM-5.3*|*glm_5_3*)  echo "GLM-5.3-Flash 321.34 17.38 8 305.8" ;;
    *deepseek_v4_flash*)  echo "DeepSeek-V4-Flash 290.91 14.08 8 148.6" ;;
    # ✅ MEASURED 2026-09-02 from 152,089 safetensors tensor shapes across 131 shards
    # (was a placeholder that wrongly assumed "dense"; Qwen uses `num_experts`, not
    # `n_routed_experts`, which is why the latter reads None). It is a FINE-GRAINED
    # MoE: E=512, k=10, moe_intermediate_size=640 -> E/k = 51.2x.
    #
    # B_total = 176.94 B  = 180.01 B on disk - 2.61 B MTP - 0.45 B ViT (both excluded:
    #                       base arm has spec decode off, and --limit-mm-per-prompt 0
    #                       makes vLLM skip constructing the vision tower)
    # B_active = 7.27 B   = 4.92 B always-on (attn + GDN + shared expert + gate +
    #                       embed/lm_head) + 120.80 B routed x 10/512 = 2.36 B
    #
    # ⚠️ THE ACTIVE FIGURE IS AMBIGUOUS FOR THIS MODEL AND THE AMBIGUITY IS THE FINDING.
    # 51.23 B (28.5% of the total) is a PLE n-gram EMBEDDING TABLE at layer 2 --
    # 128 shards x 2,500,012 rows x 160 dim, FP8 e4m3. It is read by
    # F.embedding(hash(ngram)) (nvidia/ple_layer.py:184), i.e. a GATHER of a few rows
    # per token, NOT a GEMM. So:
    #     active on the GEMM path        =  7.27 B   <- used here, comparable to GLM/V4
    #     active incl. the whole PLE table = 58.51 B <- meaningless (never all read)
    #     PLE bytes actually touched/token = ~2 rows x 160 x 1 B x ngram_heads(16) = ~5 KiB
    # Reconciles with the model card ("125B with 6B activated, plus 51B n-gram embedding
    # and 4B MTP"): measured main = 125.71 B, active-excluding-embed/lm_head = 6.00 B.
    # ⚠️ Card says 4 B MTP; MEASURED is 2.61 B. Trust the tensors (rule 4).
    *Qwen3.8*|*qwen3_8*)  echo "Qwen3.8-Flash-Next-FP8 176.94 7.27 8 172.8" ;;
    *)                    echo "unknown 0 0 8 0" ;;
  esac
}

emit_header() {
  if [[ $CSV == 1 ]]; then
    echo "model,arm_point,conc,isl,osl,out_tok_s,total_tok_s,per_gpu,per_B_active,per_B_total,ttft_p50_ms,tpot_p50_ms,kv_peak,gbps_per_gpu,frac_peak_hbm,cold"
  else
    printf '\n\033[1m%-20s %-34s %8s %9s %8s %9s %9s %8s %9s %6s\033[0m\n' \
      MODEL "ARM:POINT" "out t/s" "per GPU" "per B-act" "per B-tot" "TTFTp50" "TPOTp50" "GB/s[E]" cold
  fi
}

# One row per result JSON. Cold = the point's own guard (0 new prefix-cache hits);
# a point that is not cold is NOT a valid number -- see bench.sh rule 2.
row() {
  local f=$1 name=$2 btot=$3 bact=$4 gpus=$5 arm=$6
  /usr/bin/python3 - "$f" "$name" "$btot" "$bact" "$gpus" "$CSV" "$arm" <<'PY'
import json,sys
f,name,btot,bact,gpus,csv = sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),sys.argv[6]=="1"
arm = sys.argv[7] if len(sys.argv)>7 else ""
d=json.load(open(f))
if int(d.get("completed",0))==0: sys.exit(0)          # zero-completion = not a result
out=d.get("output_throughput",0.0); tot=d.get("total_token_throughput",0.0)
md=d.get("metadata") or {}
def g(k,dflt=""):
    v=md.get(k,dflt)
    return v if v!="" else dflt
label=f.split("/")[-1][:-5]
# ARM is load-bearing: without it, rows from bf16kv / fp8kv-fi618 / mtp-n1 / mtp-n5 are
# indistinguishable, and mixing arms is exactly how a confound gets published (rule 1).
if arm: label = f"{arm}:{label}"
cold = "yes" if float(d.get("new_prefix_cache_hits",0) or 0)==0 else "NO"
pg, pa, pt = out/gpus, out/bact, out/btot
# Achieved HBM bandwidth, folded in by bench.sh from vLLM's --enable-mfu-metrics
# counters. ⚠️ LABEL [E], NOT [M]: this is an ENGINE-SIDE ESTIMATE (config shapes x
# measured batch composition, perf.py), not a hardware counter -- and its COMPONENT
# COVERAGE DIFFERS PER MODEL (Qwen attn+ffn+unembed / GLM ffn+unembed / V4 unembed
# only), so it is a per-model LOWER BOUND and NOT comparable across models. That is
# why it is printed but deliberately NOT normalized into a cross-model ratio here.
gbps = d.get("achieved_gbps_per_gpu")
frac = d.get("frac_peak_hbm_h100")
gbps_s = f"{gbps:.0f}" if isinstance(gbps,(int,float)) else "-"
frac_s = f"{frac:.4f}" if isinstance(frac,(int,float)) else ""
if csv:
    print(f'{name},{label},{g("conc")},{g("isl")},{g("osl")},{out:.1f},{tot:.1f},'
          f'{pg:.2f},{pa:.2f},{pt:.3f},{d.get("median_ttft_ms",0):.0f},'
          f'{d.get("median_tpot_ms",0):.1f},{d.get("peak_kv_cache_usage_perc",0)},'
          f'{gbps_s if gbps_s!="-" else ""},{frac_s},{cold}')
else:
    print(f'{name:<20} {label:<34} {out:8.1f} {pg:9.2f} {pa:8.2f} {pt:9.3f} '
          f'{d.get("median_ttft_ms",0):9.0f} {d.get("median_tpot_ms",0):8.1f} '
          f'{gbps_s:>9} {cold:>6}')
PY
}

main() {
  local -a dirs=("$@")
  if [[ ${#dirs[@]} -eq 0 ]]; then
    # ⚠️ EXCLUDE EVERY QUARANTINE DIR, not just '*_discarded*'. The convention in this
    # repo is that a leading underscore on an arm name means "kept as evidence, NOT a
    # result" (bad points are quarantined rather than deleted, so the error stays
    # auditable). This filter used to match only '_discarded', which silently let
    # _v4image_firstrun_queued, _util085_queued_firstrun, _mtpctx_queued_firstrun and
    # the two _*_jitcold dirs into the normalized table -- i.e. the tool whose whole
    # job is to stop bad numbers being cited was emitting the known-bad numbers.
    # Pass such a dir explicitly as an argument if you really want to inspect it.
    mapfile -t dirs < <(find . -type d -path '*/results/*' -not -name '_*' \
                          -exec sh -c 'ls "$1"/*.json >/dev/null 2>&1' _ {} \; -print | sort)
  fi
  [[ ${#dirs[@]} -gt 0 ]] || { echo "no result dirs found"; return 1; }

  emit_header
  local d f name btot bact gpus gib
  for d in "${dirs[@]}"; do
    read -r name btot bact gpus gib <<<"$(facts "$d")"
    [[ $name == unknown ]] && { echo "  (skipping unrecognized dir: $d -- add it to facts())" >&2; continue; }
    local arm; arm=$(basename "$d")
    for f in "$d"/*.json; do [[ -f $f ]] && row "$f" "$name" "$btot" "$bact" "$gpus" "$arm"; done
  done

  [[ $CSV == 1 ]] && return 0
  cat <<'EOS'

  per GPU     = out tok/s / 8            <- the only cost-relevant number
  per B-act   = out tok/s / B-active [A]  <- FLOP-budget efficiency
  per B-tot   = out tok/s / B-total  [A]  <- HBM-purchase efficiency (penalizes sparsity)
  GB/s[E]     = achieved HBM read+write per GPU, from vLLM's --enable-mfu-metrics

  ⚠️ cold=NO means the point read a prefix cache and its throughput is INFLATED.
     Do not cite it. Re-run with a fresh seed and no --num-warmups.
  ⚠️ Compare only at MATCHED isl/osl/conc. ISL dominates everything (23x spread).
  ⚠️ Tokenizers differ across families -> tok/s is not strictly commensurable.
     GLM-5.3 vocab 154,880. For cross-family claims prefer bytes/s.
  ⚠️ KV dtype is FORCED and OPPOSITE on H100: V4=fp8_ds_mla only, GLM=BF16 only.
  ⚠️ GB/s is [E], NOT [M], and is NOT COMPARABLE ACROSS MODELS. It is an engine-side
     ESTIMATE (config shapes x measured batch composition), and vLLM's component
     coverage differs per model -- Qwen attn+ffn+unembed, GLM ffn+unembed (no attn),
     V4 unembed only -- so each is a per-model LOWER BOUND with a different gap.
     "-" means the point predates the harness fix (2026-09-02); re-poll to fill it.
     Divide by 3350 GB/s for the H100 HBM3 fraction. See fix_bug.md bug 13.
EOS
}

main "$@"
