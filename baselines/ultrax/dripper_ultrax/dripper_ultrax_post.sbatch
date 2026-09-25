#!/bin/bash
#SBATCH --job-name=dripper_ultrax_post
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.log
# Post-process-only pass for one Dripper -> UltraX part (used when inference ran with POSTPROC=0).
# usage: sbatch dripper_ultrax_post.sbatch <part 0..5>
set -uo pipefail
K=${1:?part}
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
W=${DRIPPER_UX_WORK:-$WORK_DIR/dclm_pipeline/dripper_ultrax/work}
UXD=${ULTRAX_DIR:?set ULTRAX_DIR to a checkout of https://github.com/OpenBMB/UltraX}
cd "$UXD"
${REFINER_PY:-python} stage2_large_scale_execution/post_processing/post_process_and_execute.py \
    --input_dir "$W/inf/part$K" --output_dir "$W/post/part$K" --num_workers 64
nin=$(ls "$W/in/part$K" | wc -l); npost=$(ls "$W/post/part$K" 2>/dev/null | wc -l)
echo "part$K in=$nin post=$npost"
