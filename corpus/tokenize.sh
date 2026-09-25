#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tokenize_%j.log
# GPT-NeoX-20B tokenization + shuffling into webdataset shards with DCLM's tokenize_shuffle.py.
# Writes $DCLM_DIR/exp_data/datasets/tokenized/<READ>.json (num_tokens = the "#Unique Tokens" of the paper tables),
# which is the --data-config of pretraining/pretrain_*.sh.
# MUST run with cwd=$DCLM_DIR: tokenize_shuffle writes exp_data/... relative to the working directory.
# usage: sbatch tokenize.sh <corpus name from corpora.sh>
set -uo pipefail
set -x
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
source $ROOT/corpus/corpora.sh
corpus_vars "${1:?corpus}" || exit 2
export PYTHONUNBUFFERED=1
cd "$CODE"
export PYTHONPATH=$CODE
exec $PY ray_processing/tokenize_shuffle.py \
  --input "$TOK_IN" \
  --output "$TOK_OUT" \
  --readable_name "$READ" --content_key text \
  --tokenizer EleutherAI/gpt-neox-20b \
  --ray_spill_location "/tmp/rs_${SLURM_JOB_ID:-local}" --overwrite
