#!/bin/bash
#SBATCH --job-name=heldout_s1b
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
# Held-out set step 1b (CPU, ~10 min on 16 cores): map the two-stage refiner's SFT-source WARC ids to pool shards by
# scanning the metadata of all resiliparse_extract shards (18.0M records) -> two_stage_refiner_stems.txt (excluded).
set -uo pipefail
source ${RESCRAPER_ROOT:?set RESCRAPER_ROOT}/evaluation/env.sh
W=$HELDOUT5K_DIR/work
$PY $EV/heldout_set/s1b_two_stage_shards.py $W/two_stage_src_warc.tsv $W/two_stage_refiner 16 || exit 1
