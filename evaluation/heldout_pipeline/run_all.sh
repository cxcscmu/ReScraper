#!/bin/bash
# Page-aligned held-out pipeline, greedy student run + all baselines, one command per page table:
#   bash evaluation/heldout_pipeline/run_all.sh $WORK_DIR/eval/heldout5k/heldout5k.jsonl
# Submits with dependencies:
#   A s1_ours_infer (GPU)   B s2_proxc (GPU)   C s2_cpu = RefinedWeb-rule, FineWeb-rule, UltraX join (CPU)
#   D s3_gather = s1_ours_exec + s3_gather (CPU, afterok A:B:C)   E s3_scores = DataMan, FineWeb-Edu (GPU, afterok D)
#   F s4_build = figure data (CPU, afterok E)
# Every stage skips work whose output exists, so re-running this script after a failure resumes (delete a stage's
# outputs in work/<tag>/ to force it). The release-decoding variant (the one the paper's figures use) runs on top of
# this run's outputs: run_release.sh. Partitions: GPU_PARTITION / CPU_PARTITION (default gpu / cpu).
set -euo pipefail
export RESCRAPER_ROOT=${RESCRAPER_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
source $RESCRAPER_ROOT/evaluation/env.sh
PT=$(readlink -f "$1"); S=$EV/heldout_pipeline/stages
[ -s "$PT" ] || { echo "no page table $PT"; exit 1; }
TAG=$(basename "$PT"); TAG=${TAG%.gz}; TAG=${TAG%.jsonl}
mkdir -p logs
$PY - "$PT" <<'PYEOF'
import json, sys
n = 0; need = {"gid", "stem", "url", "warc_id", "input", "output", "resiliparse"}
for i, l in enumerate(open(sys.argv[1], encoding="utf-8")):
    r = json.loads(l); assert r["gid"] == i, ("gid order", i); miss = need - set(r)
    assert not miss, ("missing fields", i, miss); n += 1
print("page table OK:", n, "pages")
PYEOF
# Resume safety: work/<tag>/ outputs are reused by name, so they must belong to this exact page table.
WK=$HELDOUT_PIPELINE_DIR/work/$TAG; mkdir -p $WK
H=$(md5sum < "$PT" | cut -c1-32)
if [ -s $WK/PAGE_TABLE_MD5 ] && [ "$(cat $WK/PAGE_TABLE_MD5)" != "$H" ]; then
  echo "REFUSED: $PT changed since work/$TAG was built ($(cat $WK/PAGE_TABLE_MD5) -> $H)."
  echo "Move the old outputs away first: mv $WK $WK.old_$(date +%m%d%H%M)"; exit 1
fi
echo $H > $WK/PAGE_TABLE_MD5
G="-p ${GPU_PARTITION:-gpu}"; C="-p ${CPU_PARTITION:-cpu}"
sb() { sbatch --parsable "$@"; }
dep() { local d=""; for j in "$@"; do [ -n "$j" ] && d="$d:$j"; done; [ -n "$d" ] && echo "--dependency=afterok$d"; true; }
# stages whose final output already exists are not resubmitted (saves a GPU queue wait on resume)
A=""; [ -s $WK/ours_raw.jsonl ] || A=$(sb $G $S/s1_ours_infer.sbatch "$PT")
B=""; [ -s $WK/sys_proxc.jsonl ] || B=$(sb $G $S/s2_proxc.sbatch "$PT")
Cj=""; { [ -s $WK/sys_refinedweb_rule.jsonl ] && [ -s $WK/sys_fineweb_rule.jsonl ] && [ -s $WK/sys_ultrax.jsonl ]; } || \
  Cj=$(sb $C $S/s2_cpu.sbatch "$PT")
D=$(sb $C $(dep $A $B $Cj) $S/s3_gather.sbatch "$PT")
E=$(sb $G --dependency=afterok:$D $S/s3_scores.sbatch "$PT")
F=$(sb $C --dependency=afterok:$E $S/s4_build.sbatch "$PT")
echo "$(date '+%F %T %Z') $TAG md5=$H ours=${A:-skip} proxc=${B:-skip} cpu=${Cj:-skip} gather=$D scores=$E build=$F" | tee -a logs/heldout_pipeline_submissions.txt
