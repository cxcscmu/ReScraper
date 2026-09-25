#!/bin/bash
# Build the 5,000-page held-out table heldout5k.jsonl (+ html5k.jsonl) from scratch; submits the SLURM chain and returns.
#   heldout960 (GPU, 1)  |  s1 (CPU) -> s1b (CPU) -> s2 (CPU) -> s3 (8 GPUs) -> s4 (CPU, after heldout960 and s3)
# Every script is resumable; outputs land in $HELDOUT960_DIR and $HELDOUT5K_DIR. Partitions: GPU_PARTITION / CPU_PARTITION.
# usage: bash evaluation/heldout_set/run_all.sh
set -euo pipefail
export RESCRAPER_ROOT=${RESCRAPER_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
source $RESCRAPER_ROOT/evaluation/env.sh
H=$EV/heldout_set; mkdir -p logs $HELDOUT5K_DIR/work $HELDOUT960_DIR
G="-p ${GPU_PARTITION:-gpu}"; C="-p ${CPU_PARTITION:-cpu}"
sb() { sbatch --parsable "$@"; }
A=$(sb $G $H/heldout960/heldout960.sbatch)
S1=$(sb $C $H/s1_old_rows.sbatch)
S1B=$(sb $C --dependency=afterok:$S1 $H/s1b_two_stage_shards.sbatch)
S2=$(sb $C --dependency=afterok:$S1B $H/s2_new_shards.sbatch)
S3=$(sb $G --dependency=afterok:$S2 $H/s3_teachers.sbatch)
S4=$(sb $C --dependency=afterok:$A:$S3 $H/s4_assemble.sbatch)
echo "$(date '+%F %T %Z') heldout960=$A s1=$S1 s1b=$S1B s2=$S2 s3=$S3 s4=$S4" | tee -a logs/heldout_set_submissions.txt
