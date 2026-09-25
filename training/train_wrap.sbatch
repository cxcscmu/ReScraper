#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
# Thin wrapper: computes save_steps from the prepack (one checkpoint per epoch, so every epoch checkpoint
# survives) and hands off to the launcher. --nodes comes from the submit line; ACC must be 24/NODES
# (4/6, 8/3, 12/2, 24/1) so that the global batch stays 192.
# Held-out evaluation of the checkpoints is in evaluation/ (not submitted from here).
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}
DATA=${1:?}; CK=${2:?}; PORT=${3:?}; MODEL=${4:?}; EP=${5:?}; MICRO=${6:?}; ACC=${7:?}; SP=${8:?}; PK=${9:?}
TAGW=${TAGW:-5}
PY=${REFINER_PY:-python}
STEPS=$($PY -c "import json,math; print(math.ceil(json.load(open('$PK/stats.json'))['bins']/192))")
echo "bins -> steps per epoch = $STEPS"
PREPACK=$PK bash $ROOT/training/train_multinode.sbatch "$DATA" "$CK" "$PORT" "$MODEL" "$EP" "$MICRO" "$ACC" \
  "--gen_eval_every 100000 --system_prompt_file $SP --tag_loss_weight $TAGW --save_steps $STEPS --save_total_limit $((EP+1))" || exit 1
[ -s $CK/final/config.json ] || { echo "no final checkpoint yet, will resume on requeue"; exit 1; }
echo train_wrap done
