#!/bin/bash
#SBATCH --job-name=proxc_finish
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.log
# Completion gate for the resiliparse + ProX-C corpus, run after the inference array (afterany).
#   1. backfill: every Dripper step-1 shard without a ProX-C output shard is re-run (one more inference array,
#      at most 28 nodes, followed by this script again in `nobf` mode);
#   2. shard gate: accept the corpus when it has >= 10,315 shards (EXPECT_SHARDS overrides), else stop;
#   3. hand-off to the shared corpus steps: corpus/dedup_bff.sh then corpus/tokenize.sh for corpus name $CORPUS.
# No post-filter is applied: the student post-filter (inference/postfilter.py) removes leaked operation syntax,
# which ProX-C cannot emit, and applying it would change a third-party baseline.
# usage: sbatch --dependency=afterany:<inference array> finish_proxc.sbatch [CORPUS=prox_c] [bf|nobf]
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
CORPUS=${1:-prox_c}; MODE=${2:-bf}
TEXT=${INFER_OUT_DIR:-$WORK_DIR/dclm_pipeline/prox_c/text}
ST1=${DRIPPER_STEP1_DIR:-$WORK_DIR/dclm_pipeline/dripper/step1_inference}
PY=${REFINER_PY:-python}
# Two pool shards are excluded from every corpus in this repository (one page in each makes the HTML renderer
# spin inside C code, where the per-page timeout cannot interrupt it), so all corpora cover the same shards.
POISON=${POISON_STEMS-CC-MAIN-20200807231820-20200808021820-00542,CC-MAIN-20200812053726-20200812083726-00050}
LOGP=$(dirname "$TEXT")/logs_pipeline; mkdir -p "$LOGP"
STEMS=$LOGP/backfill_stems.txt
find "$TEXT" -maxdepth 1 -name "*.jsonl.gz.tmp*" -delete 2>/dev/null
$PY - "$TEXT" "$STEMS" "$POISON" "$ST1" <<'PYEOF'
import os, glob, sys
text, out, poison, st1 = sys.argv[1:5]
s1 = {os.path.basename(f)[:-6] for f in glob.glob(st1 + "/*.jsonl")}
have = {os.path.basename(f).replace("_processed.jsonl.gz", "") for f in glob.glob(text + "/*.jsonl.gz")}
miss = sorted(s1 - have)
bf = [m for m in miss if m not in set(poison.split(chr(44)))]
open(out, "w").write("\n".join(bf) + ("\n" if bf else ""))
print("FINISH step1=%d have=%d missing=%d backfill=%d poison_missing=%s" % (len(s1), len(have), len(miss), len(bf), len(miss)-len(bf)), flush=True)
PYEOF
N=$(grep -c . "$STEMS" || true)
echo "[$(date -u)] proxc finish $CORPUS mode=$MODE backfill_shards=$N"
if [ "$MODE" = bf ] && [ "$N" -gt 0 ]; then
  # Cap the backfill at 28 node tasks; WORLD must follow the task count because proxc_pool.py slices
  # stems[rank::WORLD], so WORLD=min(N, 8*(K+1)) keeps every stem assigned.
  K=$(( (N + 7) / 8 - 1 ))
  if [ "$K" -gt 27 ]; then K=27; fi
  BFW=$(( (K + 1) * 8 )); if [ "$BFW" -gt "$N" ]; then BFW=$N; fi
  BF=$(sbatch --parsable -p ${GPU_PARTITION:-gpu} --array=0-$K --time=01:00:00 -J ${CORPUS}_bf \
       --export=ALL,INFER_OUT_DIR=$TEXT,INFER_STEMS_FILE=$STEMS,WORLD=$BFW,PROXC_MODEL=${PROXC_MODEL:-} \
       $ROOT/baselines/prox_c/infer_pool_proxc.sbatch)
  NX=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} --dependency=afterany:$BF \
       --export=ALL,EXPECT_SHARDS=${EXPECT_SHARDS:-},INFER_OUT_DIR=$TEXT,PROXC_MODEL=${PROXC_MODEL:-} \
       $ROOT/baselines/prox_c/finish_proxc.sbatch "$CORPUS" nobf)
  echo "BACKFILL=$BF NEXT_FINISH=$NX"
  exit 0
fi
HAVE=$(ls "$TEXT"/*.jsonl.gz 2>/dev/null | wc -l)
if [ -z "${EXPECT_SHARDS:-}" ]; then
  if [ "$HAVE" -ge 10315 ] && [ "$HAVE" -le 10320 ]; then EXPECT_SHARDS=$HAVE; else EXPECT_SHARDS=10318; fi
fi
echo "[$(date -u)] shard gate: have=$HAVE expect=$EXPECT_SHARDS"
if [ "$HAVE" -lt "$EXPECT_SHARDS" ]; then
  echo "FATAL incomplete corpus: $HAVE of $EXPECT_SHARDS shards in $TEXT - refusing to hand over a partial corpus."
  exit 1
fi
# shared downstream steps (BFF dedup 13-gram / 0.8 / old-both, then GPT-NeoX-20B tokenization), see corpus/README.md
S4=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} $ROOT/corpus/dedup_bff.sh "$CORPUS")
TK=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} --dependency=afterok:$S4 $ROOT/corpus/tokenize.sh "$CORPUS")
echo "CHAIN_SUBMITTED $CORPUS DEDUP=$S4 TOKENIZE=$TK EXPECT=$EXPECT_SHARDS"
