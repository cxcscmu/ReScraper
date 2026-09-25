#!/bin/bash
# Extraction quality (Figure 8) on one page table: token/line F1 against Dripper and the gpt-oss-120b extraction judge.
# Submits everything with dependencies and returns.
# usage: bash evaluation/extraction/run_all.sh <page_table.jsonl> <student_outputs.jsonl> [out_dir]
#   page table: heldout5k schema (gid, input, drip, gfinal, tag, edu, resiliparse, stem, warc_id, ...)
#   student outputs: one row per gid with the raw program and/or final text / decision / pre-op extraction
#                    (lib/common.py load_student); the paper uses heldout_pipeline/work/heldout5k_rel/ours.jsonl
# env: HTML      raw-html table {gid, html} (default: html5k.jsonl next to a 5000-row page table, else html<N>.jsonl there)
#      SHARD_N   judge array size, one gpt-oss-120b load per task (default 1 if N<=1000 else 2)
#      EXT_PROMPT  extraction-judge prompt file (default prompts/judge_extraction.txt)
# Jobs: A  CPU: trafilatura/jusText + resiliparse fallback -> ext_metrics + ext_texts
#       J  GPU array (1x 96 GB GPU per task, requeue, after A): judge sanity controls + every page x 5 systems
#       G  CPU (after J): agg_extract + extraction_quality.json
set -euo pipefail
export RESCRAPER_ROOT=${RESCRAPER_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
source $RESCRAPER_ROOT/evaluation/env.sh
X=$EV/extraction
PT=$(readlink -f "$1"); SO=$(readlink -f "$2"); N=$(wc -l < "$PT")
OUT=$(readlink -m "${3:-$EVAL_DIR/extraction/runs/n$N}"); mkdir -p "$OUT/logs"
if [[ -z ${HTML:-} ]]; then
  if [[ $N -eq 5000 && -f $(dirname "$PT")/html5k.jsonl ]]; then HTML=$(dirname "$PT")/html5k.jsonl; else HTML=$(dirname "$PT")/html$N.jsonl; fi
fi
[[ -s $HTML ]] || { echo "no html table: $HTML (set HTML=...)"; exit 1; }
SHARD_N=${SHARD_N:-$([[ $N -le 1000 ]] && echo 1 || echo 2)}
# fail fast: every gid has a student row, a drip text and html
$PY - "$PT" "$SO" "$HTML" "$X/lib" <<'PYEOF'
import sys, json
sys.path.insert(0, sys.argv[4])
from common import load_pages, load_student
pages = load_pages(sys.argv[1]); stu, src = load_student(sys.argv[2], pages)
hg = {json.loads(l)["gid"] for l in open(sys.argv[3], encoding="utf-8")}
miss_s = [p["gid"] for p in pages if p["gid"] not in stu]; miss_h = [p["gid"] for p in pages if p["gid"] not in hg]
nodrip = sum(1 for p in pages if not (p.get("drip") or "").strip())
noext = sum(1 for g, s in stu.items() if s["extraction"] is None)
print("pages", len(pages), "student rows", len(stu), "missing student", len(miss_s), "missing html", len(miss_h), "empty drip", nodrip, "student w/o extraction", noext, "fields", src)
if miss_s or miss_h: sys.exit("coverage check failed")
PYEOF
SB="sbatch --parsable"; C="-p ${CPU_PARTITION:-cpu}"; G="-p ${GPU_PARTITION:-gpu}"
A=$($SB $C -c 16 --mem=48G -t 2:00:00 -J extA_$N -o "$OUT/logs/A_%j.log" --wrap="$PY $X/extract_baselines.py $HTML $OUT/baselines_$N.jsonl 16 && $PYDCLM $X/resiliparse_fill.py $HTML $OUT/resiliparse_refill_$N.jsonl 16 && $PY $X/ext_metrics.py $PT $SO $OUT/baselines_$N.jsonl $OUT $OUT/resiliparse_refill_$N.jsonl")
J=$($SB $G --array=0-$((SHARD_N-1)) --export=ALL,SHARD_N=$SHARD_N --dependency=afterok:$A -J extJ_$N -o "$OUT/logs/J_%A_%a.log" $X/judge.sbatch "$OUT/ext_texts_$N.jsonl" "$OUT/judge")
Gj=$($SB $C -c 2 --mem=16G -t 1:00:00 --dependency=afterok:$J -J extG_$N -o "$OUT/logs/G_%j.log" --wrap="$PY $X/agg_extract.py $OUT/judge $N $OUT/ext_texts_$N.jsonl && $PY $X/make_extraction_quality.py $OUT $N $OUT/extraction_quality.json")
echo "$(date) N=$N out=$OUT judge x$SHARD_N prompt=${EXT_PROMPT:-judge_extraction.txt} A=$A J=$J(x$SHARD_N) G=$Gj" | tee -a "$OUT/JOBS.txt"
