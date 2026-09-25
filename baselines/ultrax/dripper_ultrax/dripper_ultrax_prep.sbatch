#!/bin/bash
#SBATCH --job-name=dripper_ultrax_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.log
# Dripper -> UltraX, stage 1: Dripper step2_text shards -> parquet parts 0-5 for UltraX.
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
W=${DRIPPER_UX_WORK:-$WORK_DIR/dclm_pipeline/dripper_ultrax/work}
${REFINER_PY:-python} "$ROOT/baselines/ultrax/dripper_ultrax/dripper_ultrax_prep.py" \
    --out "$W/in" --parts 6 --workers 48
rc=$?
if [ $rc -ne 0 ]; then
    echo "[$(date -u)] prep FAILED rc=$rc"
    exit $rc
fi
echo "[$(date -u)] prep complete"
