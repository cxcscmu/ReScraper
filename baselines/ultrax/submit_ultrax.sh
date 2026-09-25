#!/bin/bash
# Submit the UltraX chain up to the gate:
#   ux_prep (16 CPU tasks) -> ux_inf (28 x 8 GPU, afterok prep) -> ux_gate (afterany)
# The gate then submits fix rounds or the finishing chain (ux_leg -> corpus/dedup_bff.sh -> corpus/tokenize.sh).
# usage: bash submit_ultrax.sh            (env: WORK_DIR, RESCRAPER_ROOT, ULTRAX_DIR, HF_HOME, optional UX_SRC, UX_WORK,
#                                          UX_CORPUS, UX_DEDUP, EXPECT_SHARDS)
# UltraX row (10.78B): defaults (UX_SRC = raw resiliparse extraction of the pool, 10,319 shards).
# Earlier 5.99B run:   UX_SRC=$WORK_DIR/dclm_pipeline/resiliparse/step4_dedup_oldboth_65b \
#                      UX_WORK=$WORK_DIR/dclm_pipeline/resiliparse_ultrax UX_CORPUS=ultrax_rwrule UX_DEDUP=0
set -euo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository}
UX=$ROOT/baselines/ultrax
export UX_WORK=${UX_WORK:-$WORK_DIR/dclm_pipeline/resiliparse_raw_ultrax}; W=$UX_WORK
export UX_SRC=${UX_SRC:-$WORK_DIR/dclm_pipeline/resiliparse/resiliparse_extract/resiliparse_extract/processed_data}
mkdir -p "$W" logs
N=$(ls $UX_SRC | grep -c "jsonl.gz$"); EXP=${EXPECT_SHARDS:-10319}
[ "$N" -eq "$EXP" ] || { echo "src has $N shards, expected $EXP"; exit 1; }
rm -f $W/in/*.parquet.tmp*
P=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} --array=0-15 $UX/ux_prep.sbatch)
A=$(sbatch --parsable -p ${GPU_PARTITION:-gpu} --array=0-27 --dependency=afterok:$P --export=ALL,NT=28 $UX/ux_inf.sbatch)
G=$(sbatch --parsable -p ${CPU_PARTITION:-cpu} --dependency=afterany:$A --export=ALL,ROUND=1 $UX/ux_gate.sbatch)
echo "SUBMITTED $(date -u) PREP=$P INF=$A GATE=$G SRC=$UX_SRC WORK=$W" | tee -a $W/CHAIN.log
