#!/bin/bash
#SBATCH --job-name=hp_rel_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
# Release-decoding variant (work dir <W>, e.g. work/heldout5k_rel), CPU part: extract the operation-ablation full-arm rows
# of the held-out pages, exec + fidelity + Sankey data, gather, score reuse. Baseline outputs are linked from the greedy
# work dir <greedy_W> unchanged. usage: sbatch rel_chain.sbatch <page_table> <W> <greedy_W>
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
set -o pipefail
PT=$1; export HP_WORK=$2 HP_SUF=${HP_SUF:-5k_rel}; GW=$3; mkdir -p $HP_WORK
S=$EV/heldout_pipeline/stages
for f in sys_ultrax.jsonl sys_proxc.jsonl sys_refinedweb_rule.jsonl sys_fineweb_rule.jsonl ux_out.jsonl; do
  [ -e $HP_WORK/$f ] || ln -s $GW/$f $HP_WORK/$f      # baseline outputs reused unchanged
done
[ -s $HP_WORK/ablation_rows.jsonl ] || $PY $S/s1_rel_extract.py "$PT" $HP_WORK/ablation_rows.jsonl || exit 1
$PY $S/s1_rel_exec.py "$PT" "$GW" 2>&1 | tail -80 || exit 1
$PY $S/s3_gather.py "$PT" || exit 1
$PY $S/s3_reuse.py "$PT" "$GW" || exit 1
echo "REL_CPU_OK"
