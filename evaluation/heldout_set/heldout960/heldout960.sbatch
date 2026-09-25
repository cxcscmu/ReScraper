#!/bin/bash
#SBATCH --job-name=heldout960
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.log
# edu1 relabel of the first 960 held-out rows: pool join -> FineWeb-Edu recompute (CPU) -> need list -> T=1 RePro-1B
# paraphrases for the pages without a pool paraphrase (1 GPU, ~1 min) -> staged 960 table.
# Outputs in $HELDOUT960_DIR; sft_e2eC_heldout960_edu1_tagged.jsonl is rows 0..959 of heldout5k (asserted by s4_build.py).
set -uo pipefail
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
H=$EV/heldout_set/heldout960; D=$HELDOUT960_DIR; mkdir -p $D
STEM=${HELDOUT960_STEM:-CC-MAIN-20140707234035-00041-ip-10-180-212-248.ec2.internal}
T=${TWO_STAGE_POOL_DIR:-$WORK_DIR/dclm_pipeline/two_stage_pool}
$PY $H/join_pool960.py || exit 1
[ -s $D/edu_heldout960_recomputed.jsonl ] || $PY $H/edu_score960.py || exit 1
$PY $H/need960.py || exit 1
[ -s $D/gen_repro_t1_out.jsonl ] || $PY $H/gen_repro_t1_960.py ${RW_MODEL:?set RW_MODEL} $D/gen_need.jsonl $T/input/$STEM.jsonl.gz $D/gen_repro_t1_out.jsonl || exit 1
$PY $H/build_heldout960.py $D/sft_e2eC_heldout960_edu1_tagged.jsonl $D/sft_e2eC_heldout960_edu1_val_idx.json $D/audit_edu1.jsonl || exit 1
echo "HELDOUT960_OK"
