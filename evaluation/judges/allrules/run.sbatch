#!/bin/bash
#SBATCH --job-name=allrules
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.log
# Figure 1(a) rule flags (CPU): every RefinedWeb-rule / FineWeb-rule check evaluated independently on the 5,000 held-out
# pages (both stacks in parallel), then the per-page table rule_flags_5000.jsonl.
# usage: sbatch run.sbatch <heldout5k.jsonl> <heldout_pipeline work dir of the release run> <keep_judge_5000.jsonl> <out_dir>
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
PT=$1; WK=$2; KJ=$3; OUT=$4; D=$EV/judges/allrules; mkdir -p $OUT
$PYDCLM $D/rw_allrules.py "$PT" "$WK" "$OUT" &
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HELDOUT_PIPELINE_DIR/cache} HF_HUB_OFFLINE=0 $PYDT $D/fw_allrules.py "$PT" "$WK" "$OUT"
wait
$PY $D/build_rule_flags.py "$WK" "$KJ" "$OUT" || exit 1
echo ALLRULES_DONE
