#!/bin/bash
#SBATCH --job-name=heldout_s3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=1400G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# Held-out set step 3 (one 8-GPU node, ~10 min on 8x H200): (a) Qwen3.8-27B strict-subset labels for the new-shard
# candidates with the SFT teacher script (data_construction/teacher_refine/label_qwen27b.py, greedy, 8 ranks, one
# resumable part per rank), (b) the RePro need list, (c) T=1 RePro-1B paraphrases (1 GPU). Every phase skips work whose
# output exists, so a requeue resumes.
set -uo pipefail
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
W=$HELDOUT5K_DIR/work
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/vllm_heldout_${SLURM_JOB_ID}}
export T27_SRC=$W/cand_new.jsonl T27_DST=$W/t27
srun --ntasks=8 --gpus-per-task=1 bash -c "echo rank=\$SLURM_PROCID gpu=\$CUDA_VISIBLE_DEVICES; $PYTEACHER $RESCRAPER_ROOT/data_construction/teacher_refine/label_qwen27b.py \$SLURM_PROCID 8"
n=$(ls $W/t27/part-*-of-008.jsonl 2>/dev/null | wc -l); echo "[$(date)] 27B parts $n/8"
[ "$n" -eq 8 ] || { echo "HELDOUT_S3_FAIL 27B parts missing"; exit 1; }
[ -s $W/repro_need.jsonl ] || $PY $EV/heldout_set/s3_need.py $W || { echo "HELDOUT_S3_FAIL need"; exit 1; }
srun --ntasks=1 --gpus-per-task=1 $PY $EV/heldout_set/s3_gen_repro.py ${RW_MODEL:?set RW_MODEL} $W/repro_need.jsonl $W/repro_out.jsonl || { echo "HELDOUT_S3_FAIL repro"; exit 1; }
echo "HELDOUT_S3_OK"
