#!/bin/bash
#SBATCH --job-name=dripper_ultrax_rebuild
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.log
# Dripper -> UltraX, stage 4: post-processed parquet (6 parts) -> corpus text/<stem>.jsonl.gz + rebuild_stats.json.
# Fails unless rc=0 AND the number of text shards equals the number of post parquets. On success, run the shared
# corpus steps for corpus name `dripper_ultrax`: corpus/dedup_bff.sh, then corpus/tokenize.sh.
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
W=${DRIPPER_UX_WORK:-$WORK_DIR/dclm_pipeline/dripper_ultrax/work}
OUT=$WORK_DIR/dclm_pipeline/dripper_ultrax/text
npost=$(ls "$W"/post/part*/*.parquet | wc -l)
echo "[$(date -u)] rebuild start, post parquet=$npost"
${REFINER_PY:-python} "$ROOT/baselines/ultrax/dripper_ultrax/dripper_ultrax_rebuild.py" \
    --post_glob "$W/post/part*/*.parquet" --out "$OUT" --workers 48
rc=$?
ntext=$(ls "$OUT"/*.jsonl.gz 2>/dev/null | wc -l)
echo "post=$npost text=$ntext rc=$rc"
if [ $rc -ne 0 ]; then
    echo "[$(date -u)] rebuild FAILED rc=$rc"
    exit $rc
fi
if [ "$ntext" -ne "$npost" ]; then
    echo "[$(date -u)] rebuild COUNT MISMATCH post=$npost text=$ntext"
    exit 3
fi
echo "[$(date -u)] rebuild complete, $ntext corpus shards"
