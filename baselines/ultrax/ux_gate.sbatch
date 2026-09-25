#!/bin/bash
#SBATCH --job-name=ux_gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# Self-healing gate after the UltraX array. Run with afterany on ux_inf (or on a ux_fix).
# ux_validate.py --clean; if any shard lacks a complete inf/post output: submit ux_fix + a next gate (ROUND+1, max 3
# rounds, then write $UX_WORK/ALERTS and stop). Else submit the finishing chain:
#   ux_leg (8 tasks) -> corpus/dedup_bff.sh $UX_CORPUS (only if UX_DEDUP=1) -> corpus/tokenize.sh $UX_CORPUS
# UltraX row (10.78B, input = raw resiliparse extraction): UX_CORPUS=ultrax UX_DEDUP=1 (the defaults).
# Earlier 5.99B run (input already rule-filtered + deduplicated): UX_CORPUS=ultrax_rwrule UX_DEDUP=0.
# usage: sbatch --dependency=afterany:<ux_inf array> --export=ALL,ROUND=1 ux_gate.sbatch
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
UX=$ROOT/baselines/ultrax
export UX_WORK=${UX_WORK:-$WORK_DIR/dclm_pipeline/resiliparse_raw_ultrax}; W=$UX_WORK
ROUND=${ROUND:-1}; CORPUS=${UX_CORPUS:-ultrax}; DEDUP=${UX_DEDUP:-1}
PY=${REFINER_PY:-python}
echo "[$(date -u)] gate round $ROUND host=$(hostname) corpus=$CORPUS dedup=$DEDUP"
if grep -q "^CHAIN_SUBMITTED" $W/CHAIN.log 2>/dev/null; then echo "finishing chain already submitted:"; grep "^CHAIN_SUBMITTED" $W/CHAIN.log; exit 0; fi
if squeue -u "$USER" -h -o %j | grep -q -E "^ux_(inf|fix)$"; then
  echo "ux_inf/ux_fix still queued or running - refusing to validate with --clean"; squeue -u "$USER" -h -o "%i %j %T" | grep ux_; exit 2
fi
$PY $UX/ux_validate.py --clean || exit 1
MI=$(grep -c . $W/counts/missing_inf.txt || true); MP=$(grep -c . $W/counts/missing_post.txt || true)
echo "[$(date -u)] missing_inf=$MI missing_post=$MP"
if [ "$MI" -gt 0 ] || [ "$MP" -gt 0 ]; then
  if [ "$ROUND" -ge 3 ]; then
    echo "ALERT $(date -u) round $ROUND still missing inf=$MI post=$MP - stopping, see counts/missing_*.txt" | tee -a $W/ALERTS
    exit 3
  fi
  FX=$(sbatch --parsable -p ${GPU_PARTITION:-gpu} $UX/ux_fix.sbatch)
  NG=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} --dependency=afterany:$FX --export=ALL,ROUND=$((ROUND+1)) $UX/ux_gate.sbatch)
  echo "GATE_FIX_SUBMITTED FIX=$FX NEXT_GATE=$NG" | tee -a $W/CHAIN.log
  exit 0
fi
LG=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} --array=0-7 --export=ALL,NT=8 $UX/ux_leg.sbatch)
DEP=$LG; S4=none
if [ "$DEDUP" = 1 ]; then
  S4=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} --dependency=afterok:$LG $ROOT/corpus/dedup_bff.sh "$CORPUS"); DEP=$S4
fi
TK=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} --dependency=afterok:$DEP $ROOT/corpus/tokenize.sh "$CORPUS")
echo "CHAIN_SUBMITTED $(date -u) LEG=$LG DEDUP=$S4 TOKENIZE=$TK CORPUS=$CORPUS" | tee -a $W/CHAIN.log
