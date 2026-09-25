#!/bin/bash
# Release-decoding variant of the held-out pipeline (tag <tag>_rel; the variant behind the paper's figures):
#   bash evaluation/heldout_pipeline/run_release.sh $WORK_DIR/eval/heldout5k/heldout5k.jsonl
# Needs the finished greedy run (run_all.sh: work/<tag>/ours_raw.jsonl for plen/tlen, the four baseline outputs and the
# DataMan/Edu scores, reused by exact text) and the operation ablation's full-arm raw rows ($ABLATION_DIR/raw, ablations/).
# Submits: R1 rel_chain (CPU: ablation rows -> ours.jsonl, fidelity, decision_flow_5k_rel.json, gather, score reuse)
#          R2 s3_scores (GPU: only the new ReScraper texts)   R3 rel_build (CPU: figure data *_5k_rel.json)
set -euo pipefail
export RESCRAPER_ROOT=${RESCRAPER_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
source $RESCRAPER_ROOT/evaluation/env.sh
PT=$(readlink -f "$1"); S=$EV/heldout_pipeline/stages
TAG=$(basename "$PT"); TAG=${TAG%.gz}; TAG=${TAG%.jsonl}
GW=$HELDOUT_PIPELINE_DIR/work/$TAG; W=$HELDOUT_PIPELINE_DIR/work/${TAG}_rel; mkdir -p $W logs
[ -s $GW/ours_raw.jsonl ] && [ -s $GW/sys_proxc.jsonl ] && [ -d $GW/score_out ] || { echo "run run_all.sh first ($GW incomplete)"; exit 1; }
G="-p ${GPU_PARTITION:-gpu}"; C="-p ${CPU_PARTITION:-cpu}"
sb() { sbatch --parsable "$@"; }
R1=$(sb $C $S/rel_chain.sbatch "$PT" $W $GW)
R2=$(sb $G --dependency=afterok:$R1 --export=ALL,HP_WORK=$W $S/s3_scores.sbatch "$PT")
R3=$(sb $C --dependency=afterok:$R2 $S/rel_build.sbatch "$PT" $W)
echo "$(date '+%F %T %Z') ${TAG}_rel chain=$R1 scores=$R2 build=$R3" | tee -a logs/heldout_pipeline_submissions.txt
