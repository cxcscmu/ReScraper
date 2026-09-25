"""Stage 1b (CPU): execute the student programs of the greedy run and score them against the teacher.

Execution and operation classes: lib/ours_exec.py (execute / classify_ours; lib/rescraper_ops.py + lib/editops.py, the
executor of the pool writer). Teacher comparison: evaluation/fidelity/teacher_metrics.py (decision accuracy over pages
sent to the model; token P/R/F1 = corpus-level \\w+ token-mass overlap of pfinal vs gfinal, deleted -> "", on pages with
plen + tlen <= 32768; 95% CIs from 2,000 page-bootstrap resamples).
Outputs: <work>/ours.jsonl, <work>/ours_validation.json
usage: s1_ours_exec.py <page_table>
"""
import collections, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib")); sys.path.insert(0, os.path.join(HERE, "..", "..", "fidelity"))
from common import *
from ours_exec import execute
from teacher_metrics import teacher_metrics, rows_for
X, E = e2e()
assert not X.OPS_TOLERANT
PT = sys.argv[1]; W = workdir(PT)
MAXLEN = 32768

pages = load_pages(PT, ["gid", "input", "output", "tag", "gfinal"])
raws = list(read_jsonl(W + "/ours_raw.jsonl"))
assert [r["gid"] for r in raws] == [p["gid"] for p in pages], "ours_raw.jsonl does not cover the page table"
outs = [execute(p, r) for p, r in zip(pages, raws)]
write_jsonl_atomic(W + "/ours.jsonl", outs)

V = {"pages": len(pages), "status": dict(collections.Counter(o["status"] for o in outs)),
     "decision_line1": dict(collections.Counter(o["decision"] for o in outs)),
     "opmix_cls": dict(collections.Counter(o["opmix_cls"] for o in outs)),
     "finish_reason": dict(collections.Counter(o["finish_reason"] for o in outs))}
labelled = [p for p in pages if p.get("output")]
V["pages_with_teacher_output"] = len(labelled)
if labelled:
    rmap = {r["gid"]: r for r in raws}
    ours_pred = lambda g: None if rmap[g]["status"] != "ok" else (rmap[g]["raw"], rmap[g]["plen"] + rmap[g]["tlen"] <= MAXLEN)
    V["vs_teacher_all"] = teacher_metrics(rows_for(labelled, ours_pred))
    if len(pages) > 960:
        V["vs_teacher_gid_ge_960"] = teacher_metrics(rows_for([p for p in labelled if p["gid"] >= 960], ours_pred))
        V["vs_teacher_first960"] = teacher_metrics(rows_for([p for p in labelled if p["gid"] < 960], ours_pred))
json.dump(V, open(W + "/ours_validation.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in V.items()}, indent=1, ensure_ascii=False)[:6000])
print("OURS_EXEC_OK")
