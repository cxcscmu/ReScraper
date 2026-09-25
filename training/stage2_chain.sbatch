#!/bin/bash
#SBATCH --job-name=stage2_chain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2_chain_%j.log
# Stage 2: build the <rewrite>-heavy SFT set and continue training the Stage-1 student on it.
#   data_construction/sft_sets/build_stage2_set.py -> finalize_prompt.py -> prepack (8 cpu shards + merge) -> train
# INIT is a checkpoint path, not a base model: the run starts from the Stage-1 epoch-2 checkpoint
# (checkpoint-2244 = 2 x 1,122 steps). Defaults below are the paper setting (released model):
#   SCHEME=edu1 (every teacher-deleted page with FineWeb-Edu >= 1.0 and a rewrite -> <rewrite>), RWFRAC=0.30,
#   rewrite source = RePro 1B at T=1.0 (two_stage_pool/rwrepro_olmo_temp1), 3 epochs, 4 nodes x accum 6.
# usage: sbatch [--export=ALL,SCHEME=...,RWDIR=...,RWFRAC=...,EP=...,NODES=...,ACC=...] stage2_chain.sbatch
set -uo pipefail
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT}; R=${WORK_DIR:?set WORK_DIR}
D=${SFT_DIR:-$R/sft_data}; CKD=${CKPT_DIR:-$R/ckpts}
PY=${REFINER_PY:-python}; GPU_P=${GPU_PARTITION:-gpu}; CPU_P=${CPU_PARTITION:-cpu}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
SCHEME=${SCHEME:-edu1}
RWDIR=${RWDIR:-$R/dclm_pipeline/two_stage_pool/rwrepro_olmo_temp1}
RWFRAC=${RWFRAC:-0.30}
BASE=${BASE:-$D/base_set/base_set_norw.jsonl}
DEC=${DEC:-$R/dclm_pipeline/rescue_add/text_decisions}
JOIN=${JOIN:-"$D/base_set/half1/join_*.jsonl,$D/base_set/half2/join_*.jsonl"}
INIT=${INIT:-$CKD/stage1/checkpoint-2244}
OUTJ=${OUTJ:-$D/stage2_set.jsonl}
SP=${SP:-$ROOT/prompts/student_system_stage2.txt}
PK=${PK:-$D/prepacked/stage2}
CK=${CK:-$CKD/stage2}
EP=${EP:-3}; NODES=${NODES:-4}; ACC=${ACC:-6}; PORT=${PORT:-29871}; TAGW=${TAGW:-5}

[ -s "$OUTJ" ] || $PY $ROOT/data_construction/sft_sets/build_stage2_set.py "$BASE" "$JOIN" "$DEC" "$RWDIR" "$SCHEME" "$RWFRAC" "$OUTJ" || exit 1
[ -s "$SP" ]   || $PY $ROOT/data_construction/sft_sets/finalize_prompt.py $ROOT/prompts/student_system.tmpl "$OUTJ" "$SP" || exit 1

if [ ! -s $PK/stats.json ]; then
  SH=$(sbatch --parsable --array=0-7 -p $CPU_P --export=ALL,DATA=$OUTJ,OUT=$PK,SP=$SP,NT=8 -J pk_stage2 $ROOT/training/prepack_shard.sbatch)
  MG=$(sbatch --parsable --dependency=afterok:$SH -p $CPU_P --export=ALL,OUT=$PK,NT=8 -J pkm_stage2 $ROOT/training/prepack_merge.sbatch)
  echo "PREPACK $SH MERGE $MG"
  DEP="--dependency=afterok:$MG"
else
  DEP=""
fi
TR=$(sbatch --parsable $DEP -p $GPU_P --nodes=$NODES -J stage2 \
  --export=ALL,TAGW=$TAGW \
  $ROOT/training/train_wrap.sbatch "$OUTJ" "$CK" "$PORT" "$INIT" "$EP" 1 "$ACC" "$SP" "$PK")
echo "SCHEME=$SCHEME RWFRAC=$RWFRAC INIT=$INIT TRAIN $TR"
echo STAGE2_CHAIN_OK
