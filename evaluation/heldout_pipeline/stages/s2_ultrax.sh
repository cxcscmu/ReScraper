#!/bin/bash
# Stage 2d: per-page UltraX output. keys (gid, stem, md5 of the resiliparse text) -> exact-text join against the UltraX
# run over the raw resiliparse pool ($ULTRAX_RERUN_DIR, baselines/) -> sys_ultrax.jsonl. CPU, about a minute.
# usage: s2_ultrax.sh <page_table>   (sourced environment: evaluation/env.sh)
set -uo pipefail
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
PT=$1; TAG=$(basename "$PT"); TAG=${TAG%.gz}; TAG=${TAG%.jsonl}; WK=${HP_WORK:-$HELDOUT_PIPELINE_DIR/work/$TAG}; mkdir -p $WK
S=$EV/heldout_pipeline/stages
$PY $S/s2_ultrax_finish.py keys "$PT" || exit 1
$PY $S/s2_ultrax_join.py $WK/ux_keys.jsonl $WK/ux_out.jsonl || exit 1
$PY $S/s2_ultrax_finish.py finish "$PT" || exit 1
