#!/bin/bash
#SBATCH --job-name=hp_proxc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log
# Stage 2c: ProX-C on the resiliparse text of every page (1 GPU, ~10 min for 5,000 pages). usage: sbatch s2_proxc.sbatch <page_table>
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
$PY $EV/heldout_pipeline/stages/s2_proxc.py "$1" || exit 1
