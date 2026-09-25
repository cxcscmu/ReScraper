#!/bin/bash
#SBATCH --job-name=finish_chain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/finish_chain_%j.log
# Everything after pool inference, as one dependency chain:
#   (optional) backfill of missing shards -> post-filter (shard-count gate) -> BFF dedup -> tokenize
# Submit it with --dependency=afterany:<inference array>. In `bf` mode it lists the step-1 shards that have no
# output yet (minus the shards listed in POISON_STEMS, whose rendering never terminates), submits a backfill
# array for them and re-submits itself in `nobf` mode behind it. In `nobf` mode it accepts the shard count if
# it is within 5 of the full pool (10,320 shards in ours; the released corpus has 10,318) unless EXPECT_SHARDS
# is set, then chains post-filter -> dedup -> tokenize.
# usage: sbatch --dependency=afterany:<ARRAY> finish_chain.sbatch <corpus> <model> <system prompt> [bf|nobf]
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
LEG=${1:?corpus}; MODEL=${2:?model}; SP=${3:?system prompt}; MODE=${4:-bf}
CPU_P=${CPU_PARTITION:-cpu}; GPU_P=${GPU_PARTITION:-gpu}
source $ROOT/corpus/corpora.sh; corpus_vars "$LEG" || exit 2
mkdir -p "$CDIR/logs_pipeline"
STEMS=$CDIR/logs_pipeline/backfill_stems.txt
find "$CDIR/text" -maxdepth 1 -name "*.jsonl.gz.tmp*" -delete 2>/dev/null
${REFINER_PY:-python} - "$CDIR/text" "$STEMS" "${POISON_STEMS:-}" "$ROOT/lib" <<'PYEOF'
import os, glob, sys
text, out, poison, lib = sys.argv[1:5]
sys.path.insert(0, lib)
from pool_join import ST1
s1 = {os.path.basename(f)[:-6] for f in glob.glob(ST1 + "/*.jsonl")}
have = {os.path.basename(f).replace("_processed.jsonl.gz", "") for f in glob.glob(text + "/*.jsonl.gz")}
miss = sorted(s1 - have)
bf = [m for m in miss if m not in set(p for p in poison.split(",") if p)]
open(out, "w").write("\n".join(bf) + ("\n" if bf else ""))
print("FINISH step1=%d have=%d missing=%d backfill=%d poison_missing=%d" % (len(s1), len(have), len(miss), len(bf), len(miss) - len(bf)), flush=True)
PYEOF
N=$(grep -c . "$STEMS" || true)
echo "[$(date -u)] finish $LEG mode=$MODE backfill_shards=$N"
if [ "$MODE" = bf ] && [ "$N" -gt 0 ]; then
  K=$(( (N + 7) / 8 - 1 ))
  BF=$(sbatch --parsable -p $GPU_P --array=0-$K --time=01:00:00 -J ${LEG}_bf \
       --export=ALL,INFER_MODEL_PATH=$MODEL,INFER_SYSTEM_PROMPT_FILE=$SP,INFER_OUT_DIR=$CDIR/text,INFER_STEMS_FILE=$STEMS,WORLD=$N \
       $ROOT/inference/infer_pool.sbatch)
  NX=$(sbatch --parsable -p $CPU_P --dependency=afterany:$BF --export=ALL,EXPECT_SHARDS=${EXPECT_SHARDS:-} $ROOT/corpus/finish_chain.sbatch "$LEG" "$MODEL" "$SP" nobf)
  echo "BACKFILL=$BF NEXT_FINISH=$NX"
  exit 0
fi
HAVE=$(ls "$CDIR/text"/*.jsonl.gz 2>/dev/null | wc -l)
if [ -z "${EXPECT_SHARDS:-}" ]; then
  if [ "$HAVE" -ge 10315 ] && [ "$HAVE" -le 10320 ]; then EXPECT_SHARDS=$HAVE; else EXPECT_SHARDS=10318; fi
fi
echo "[$(date -u)] shard gate: have=$HAVE expect=$EXPECT_SHARDS"
PF=$(sbatch --parsable -p $CPU_P --time=00:20:00 -J ${LEG}_pf --export=ALL,EXPECT_SHARDS=$EXPECT_SHARDS $ROOT/inference/postfilter.sbatch "$LEG")
S4=$(sbatch --parsable -p $CPU_P --time=01:00:00 -J ${LEG}_dedup --dependency=afterok:$PF $ROOT/corpus/dedup_bff.sh "$LEG")
TK=$(sbatch --parsable -p $CPU_P --time=06:00:00 -J ${LEG}_tok --dependency=afterok:$S4 $ROOT/corpus/tokenize.sh "$LEG")
echo "CHAIN_SUBMITTED $LEG PF=$PF DEDUP=$S4 TOK=$TK READ=$READ EXPECT=$EXPECT_SHARDS"
