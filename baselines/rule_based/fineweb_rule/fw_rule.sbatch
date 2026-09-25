#!/bin/bash
#SBATCH --job-name=fw_rule60
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log
# FineWeb-rule text on 60 random pool shards (lang+len-filtered HTML -> resiliparse -> datatrove 0.2.0 FineWeb
# filters), plus a 5k-document sample. Output: $FW_OUT/{shards/,fineweb_rule_5k.jsonl,fw_counts.json}.
# Needs $DATATROVE_PY = a Python env with datatrove==0.2.0, resiliparse and fasttext; datatrove's URLFilter downloads
# its block lists on first use (HF_HOME), and the Gopher filters need the nltk punkt data (NLTK_DATA).
set -uo pipefail
: ${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}
ROOT=${RESCRAPER_ROOT:?set RESCRAPER_ROOT to this repository (sbatch copies the script elsewhere)}
OUTD=${FW_OUT:-$WORK_DIR/dclm_pipeline/fineweb_rule/out60}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "[$(date -u)] start host=$(hostname)"
${DATATROVE_PY:-python} $ROOT/baselines/rule_based/fineweb_rule/fw_rule.py $OUTD 16
rc=$?; echo "[$(date -u)] fw_rule rc=$rc"; exit $rc
