#!/bin/bash
#SBATCH --job-name=hp_ours
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=112G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
# Stage 1a: released student, greedy, on every page (1x 48 GB GPU, ~25 min for 5,000 pages). usage: sbatch s1_ours_infer.sbatch <page_table>
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
$PY $EV/heldout_pipeline/stages/s1_ours_infer.py "$1" || exit 1
