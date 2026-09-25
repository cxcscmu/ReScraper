#!/bin/bash
#SBATCH --job-name=heldout_s4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
# Held-out set step 4 (CPU, a few minutes): assemble heldout5k.jsonl (rows 0..959 asserted byte-identical to heldout960),
# attach the pool resiliparse text, write html5k.jsonl and the summary tables.
set -uo pipefail
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
W=$HELDOUT5K_DIR/work; B=$HELDOUT5K_DIR/build; mkdir -p $B
$PY $EV/heldout_set/s4_build.py $W $B > $B/build.log 2>&1 || { cat $B/build.log; echo HELDOUT_S4_FAIL build; exit 1; }
cat $B/build.log
$PY $EV/heldout_set/s5_resiliparse_join.py $B/heldout5k_nores.jsonl $HELDOUT5K_DIR/heldout5k.jsonl > $B/resi.log 2>&1 || { cat $B/resi.log; echo HELDOUT_S4_FAIL resiliparse; exit 1; }
cat $B/resi.log
$PY $EV/heldout_set/s6_html_table.py $HELDOUT5K_DIR/heldout5k.jsonl $W $HELDOUT5K_DIR/html5k.jsonl || { echo HELDOUT_S4_FAIL html; exit 1; }
$PY $EV/heldout_set/s7_stats.py $HELDOUT5K_DIR/heldout5k.jsonl > $B/stats.txt || { echo HELDOUT_S4_FAIL stats; exit 1; }
cat $B/stats.txt
echo "HELDOUT_S4_OK"
