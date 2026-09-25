"""Per-stage retention of the pool inference run, from its logs (Table 9, tab:stage-retention, upper block).

Dumps every per-shard counter of the pool inference run (inference/infer_pool.py), so the stage-retention
table can be built with the script's own semantics:
records -> (empty input, no main_html skipped; NOT in `pages`) -> pages -> (render fail, too long)
-> routed -> tags + parse_fail.
Each per-shard log line carries the rank's running counter dict, so a shard's own counts are the difference
from the previous line of the same rank in the same log; shards are then deduplicated by stem (a requeued
attempt overwrites the earlier one) and summed.
usage: stage_retention_from_logs.py '<log glob>' ['<log glob>' ...]
       (the logs of the main inference array and of any backfill runs, e.g. 'logs/rescraper_infer_*.log')"""
import glob, re, ast, collections, json, sys
PATS = sys.argv[1:]
if not PATS:
    sys.exit(__doc__)
LINE = re.compile(r"rank (\d+) (\S+): pages (\d+) kept (\d+) in \d+s \| (\{.*\})\s*$")
per = {}
for p in PATS:
    for f in sorted(glob.glob(p)):
        prev = collections.defaultdict(collections.Counter)
        for line in open(f, errors="ignore"):
            m = LINE.search(line.rstrip())
            if not m: continue
            try: cum = collections.Counter({k: v for k, v in ast.literal_eval(m.group(5)).items() if isinstance(v, int)})
            except Exception: continue
            d = cum - prev[m.group(1)]; prev[m.group(1)] = cum
            d["pages"] = int(m.group(3)); d["written"] = int(m.group(4)); per[m.group(2)] = d
T = collections.Counter()
for d in per.values(): T.update(d)
print("shards", len(per)); print(json.dumps(dict(sorted(T.items())), indent=1))
known = ["tag<keep>", "tag<edit>", "tag<extract>", "tag<refine>", "tag<rewrite>", "tag<delete>"]
tags = sum(v for k, v in T.items() if k.startswith("tag<"))
print("sum of all tag<> =", tags, "| + parse_fail =", tags + T["parse_fail"],
      "| pages - render_fail - too_long =", T["pages"] - T["render_fail_or_empty"] - T["too_long"])
