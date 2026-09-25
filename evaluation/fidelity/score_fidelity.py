"""Teacher fidelity of an executed ReScraper run on the held-out table (stand-alone scorer; the pipeline stages
s1_ours_exec.py / s1_rel_exec.py compute the same numbers).

Reads the page table (teacher program `output`, tag) and an ours.jsonl of evaluation/heldout_pipeline (status, op, pt,
text, plen, tlen) and writes
  vs_teacher_all / vs_teacher_first960  teacher_metrics(): decision accuracy (4 operations) and keep-or-delete accuracy
                                        over the pages sent to the model, pooled \\w+ token P/R/F1 of the text each page
                                        contributes (deleted -> "") on pages that fit the 32,768-token context, 95% CIs
  decision_flow                         teacher operation x student operation over all pages (skipped / unparseable
                                        pages count as delete): the data of the decision-flow figure
The per-length-quartile table of the appendix is computed from the same two files by
analysis/analyze_fidelity_by_length_5k.py.
usage: score_fidelity.py <page_table> <ours.jsonl> <out.json>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_metrics import teacher_metrics, rows_from_executed, decision_flow, OPS

PT, OURS, OUT = sys.argv[1:4]
pages = [json.loads(l) for l in open(PT, encoding="utf-8") if l.strip()]
outs = [json.loads(l) for l in open(OURS, encoding="utf-8") if l.strip()]
assert [p["gid"] for p in pages] == list(range(len(pages))) == [o["gid"] for o in outs], "page table and outputs must be gid-aligned"
rows = rows_from_executed(pages, outs)
flow, n_skip, n_unp = decision_flow(pages, outs)
R = {"pages": len(pages), "sent_to_model": len(rows),
     "vs_teacher_all": teacher_metrics([r for _, r in rows]),
     "vs_teacher_first960": teacher_metrics([r for g, r in rows if g < 960]),
     "decision_flow": {"operations": OPS, "teacher_to_student": flow, "skipped_as_delete": n_skip, "unparseable_as_delete": n_unp}}
json.dump(R, open(OUT, "w"), indent=1)
a = R["vs_teacher_all"]
print("pages %d sent %d | acc %.2f %s | keep-or-delete %.2f | token P/R/F1 %.2f / %.2f / %.2f %s" % (
    len(pages), len(rows), a["acc"], a["acc_ci95"], a["keep_or_delete_acc"], a["tok_P"], a["tok_R"], a["tok_F1"], a["tok_F1_ci95"]))
