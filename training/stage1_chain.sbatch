#!/bin/bash
#SBATCH --job-name=stage1_chain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage1_chain_%j.log
# Stage 1: turn the rescue decisions into student labels and chain everything up to the training submit:
#   data_construction/sft_sets/build_stage1_set.py (relabel the <delete> rows of the 1.38M no-rewrite base set)
#   data_construction/sft_sets/finalize_prompt.py  (fill the frequency line from the file that will actually be trained)
#   prepack (8 CPU shards + merge)  ->  training (8 nodes x 8 GPUs, accum 3, global batch 192, 3 epochs, tag weight 5)
# Nothing here touches GPUs until the training submit.
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; R=${WORK_DIR:?set WORK_DIR}
D=${SFT_DIR:-$R/sft_data}; CKD=${CKPT_DIR:-$R/ckpts}
PY=${REFINER_PY:-python}; GPU_P=${GPU_PARTITION:-gpu}; CPU_P=${CPU_PARTITION:-cpu}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
BASE=${BASE:-$D/base_set/base_set_norw.jsonl}
OUTJ=${OUTJ:-$D/stage1_set.jsonl}
SP=${SP:-$ROOT/prompts/student_system_stage1.txt}
PK=${PK:-$D/prepacked/stage1}
EP=${EP:-3}; NODES=${NODES:-8}; ACC=${ACC:-3}; PORT=${PORT:-29984}
CK=${CK:-$CKD/stage1}
JOIN=${JOIN:-"$D/base_set/half1/join_*.jsonl,$D/base_set/half2/join_*.jsonl"}
DEC=${DEC:-$R/dclm_pipeline/rescue_add/text_decisions}

# Optionally wait until the rescue pass (data_construction/rescue/rescue_build.sbatch) has covered every shard
# this SFT set draws from (SFT_STEMS = file with one shard stem per line).
if [ -n "${SFT_STEMS:-}" ]; then
  WAIT_H=${WAIT_H:-20}; DEADLINE=$(( $(date +%s) + WAIT_H * 3600 ))
  while :; do
    NEED=$(comm -23 <(sort $SFT_STEMS) <(ls $DEC | sed "s/\.jsonl$//" | sort) | wc -l)
    echo "[$(date -u +%H:%M)] shards still undecided for this SFT set: $NEED"
    [ "$NEED" -eq 0 ] && break
    [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "FATAL: still $NEED undecided after ${WAIT_H}h"; exit 1; }
    sleep 300
  done
fi

[ -s "$OUTJ" ] || $PY $ROOT/data_construction/sft_sets/build_stage1_set.py "$BASE" "$JOIN" "$DEC" "$OUTJ" || exit 1
[ -s "$SP" ]   || $PY $ROOT/data_construction/sft_sets/finalize_prompt.py $ROOT/prompts/student_system.tmpl "$OUTJ" "$SP" || exit 1

if [ ! -s $PK/stats.json ]; then
  SH=$(sbatch --parsable --array=0-7 -p $CPU_P --export=ALL,DATA=$OUTJ,OUT=$PK,SP=$SP,NT=8 -J pk_stage1 $ROOT/training/prepack_shard.sbatch)
  MG=$(sbatch --parsable --dependency=afterok:$SH -p $CPU_P --export=ALL,OUT=$PK,NT=8 -J pkmerge_stage1 $ROOT/training/prepack_merge.sbatch)
  echo "PREPACK $SH MERGE $MG"
  DEP="--dependency=afterok:$MG"
else
  DEP=""
fi
# save_steps = one epoch, so every epoch checkpoint survives (Stage 2 starts from the epoch-2 checkpoint)
TR=$(sbatch --parsable $DEP -p $GPU_P --nodes=$NODES -J stage1_${NODES}n \
  --export=ALL,TAGW=5 \
  $ROOT/training/train_wrap.sbatch "$OUTJ" "$CK" "$PORT" Qwen/Qwen3-0.6B "$EP" 1 "$ACC" "$SP" "$PK")
echo "TRAIN $TR"
echo STAGE1_CHAIN_OK
