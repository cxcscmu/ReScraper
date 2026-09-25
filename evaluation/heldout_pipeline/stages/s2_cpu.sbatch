#!/bin/bash
#SBATCH --job-name=hp_cpu2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.log
# Stage 2a/2b/2d (CPU, 15-30 min for 5,000 pages): RefinedWeb-rule (DCLM env), FineWeb-rule (datatrove 0.2.0 env),
# UltraX per-page join. Each step is skipped when its output exists. usage: sbatch s2_cpu.sbatch <page_table>
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
PT=$1; TAG=$(basename "$PT"); TAG=${TAG%.gz}; TAG=${TAG%.jsonl}; WK=${HP_WORK:-$HELDOUT_PIPELINE_DIR/work/$TAG}; mkdir -p $WK
S=$EV/heldout_pipeline/stages
if [ ! -s $WK/sys_refinedweb_rule.jsonl ]; then $PYDCLM $S/s2_rw_rule.py "$PT" || exit 1; fi
[ -s $WK/sys_refinedweb_rule.jsonl ] || exit 1
if [ ! -s $WK/sys_fineweb_rule.jsonl ]; then
  # URLFilter fetches datatrove's URL block lists through the HF hub on first use
  XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HELDOUT_PIPELINE_DIR/cache} HF_HUB_OFFLINE=0 $PYDT $S/s2_fw_rule.py "$PT" || exit 1
fi
if [ ! -s $WK/sys_ultrax.jsonl ]; then bash $S/s2_ultrax.sh "$PT" || exit 1; fi
echo "S2_CPU_OK"
