"""Stage 1b-rel (CPU): the RELEASE-decoding student outputs for the page table, from the operation ablation's full arm.

Input <W>/ablation_rows.jsonl (stages/s1_rel_extract.py). Output <W>/ours.jsonl in the schema of the greedy run's ours.jsonl:
  status   "ok" | "skip:<reason>" (skipped pages were never sent to the model; they count as delete / not_processed)
  decision line 1 of the raw program; op = its operation name (None if not a decision tag)
  pt       the executor tag (ablation raw `tag`); text = executor `final` ("" for <delete>); extracted = ablation `extracted`
  emits    ablation `in_full` (the page produced a corpus row); finish_reason / gen_tokens = ablation finish / n_gen_tok
  opmix_cls/opmix_det = classify_ours() (lib/ours_exec.py) on (page input, raw program) - the same code as the greedy run
  plen/tlen from the greedy run's ours_raw.jsonl (same prompt and tokenizer; used for the `full` flag of token F1)
Fidelity vs teacher: evaluation/fidelity/teacher_metrics.py on (gt, gfinal) x (pt, text).
Also writes the Sankey data $HP_OUT/decision_flow_<suf>.json (teacher tag x student decision; skipped / unparseable ->
delete), the rescue branch (teacher-deleted pages the student rewrites, by FineWeb-Edu band) and length-cap / loop counts.
usage: HP_WORK=<W> s1_rel_exec.py <page_table> <greedy_workdir>"""
import collections, json, os, random, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib")); sys.path.insert(0, os.path.join(HERE, "..", "..", "fidelity"))
from common import *
from ours_exec import classify_ours
from teacher_metrics import teacher_metrics, rows_from_executed, decision_flow, TOP, OPS
X, E = e2e()
PT, GW = sys.argv[1], sys.argv[2]; W = workdir(PT)
assert os.path.abspath(W) != os.path.abspath(GW), "set HP_WORK to the new workdir"
norm = lambda t: " ".join((t or "").split()); TOK = re.compile(r"\w+"); MAXLEN = 32768

pages = load_pages(PT, ["gid", "input", "output", "tag", "gfinal", "edu", "stem", "step1_idx", "in_md5"])
A = load_by_gid(W + "/ablation_rows.jsonl"); G = load_by_gid(GW + "/ours_raw.jsonl")
assert sorted(A) == list(range(len(pages)))
S = collections.Counter(); outs = []
for p in pages:
    a = A[p["gid"]]; g = G[p["gid"]]
    assert a["in_md5"] == p["in_md5"] and a["idx"] == p["step1_idx"]
    o = {"gid": p["gid"], "status": "ok", "decision": None, "pt": None, "op": None, "text": None, "extracted": None,
         "opmix_cls": None, "opmix_det": None, "emits": bool(a.get("in_full")), "gen_tokens": a.get("n_gen_tok"),
         "finish_reason": a.get("finish"), "plen": g.get("plen"), "tlen": g.get("tlen"),
         "in_clean": a.get("in_clean"), "f3_dropped": a.get("f3_dropped"), "raw": a.get("gen")}
    if "skip" in a:
        o["status"] = "skip:" + a["skip"]; o["opmix_cls"] = "omitted"; o["opmix_det"] = {"reason": o["status"]}
        S[o["status"]] += 1; outs.append(o); continue
    raw = a["gen"]; lines = raw.strip().split("\n"); d = lines[0].strip() if lines else ""
    c, det, ext_re, text_re, emits_re = classify_ours(p["input"], raw)
    pt = a.get("tag")
    o.update(decision=d, op=OPNAME.get(d) if d in X.DFIRST_TAGS else None, pt=pt,
             text="" if pt == "<delete>" else (a.get("final") or ""), extracted=a.get("extracted"), opmix_cls=c, opmix_det=det)
    # does re-executing the program on the page table's rendering reproduce the ablation executor output?
    pt_re, body_re = X.body_from_prediction_dfirst(p["input"], raw)
    S["reexec_tag_same"] += pt_re == pt
    S["reexec_final_same"] += norm("" if pt_re == "<delete>" else body_re) == norm(o["text"])
    S["reexec_extracted_same"] += (ext_re is None) or norm(ext_re) == norm(o["extracted"])
    S["emits_eq_nonempty_nondelete"] += o["emits"] == (pt != "<delete>" and bool((o["text"] or "").strip()))
    S["ok"] += 1; outs.append(o)
write_jsonl_atomic(W + "/ours.jsonl", outs)

V = {"source": "operation-ablation full arm raw rows, T=1.0 top_p=1.0 max_tokens=3072", "pages": len(pages),
     "status": dict(collections.Counter(o["status"] for o in outs)), "checks": dict(S),
     "decision_line1": dict(collections.Counter(o["decision"] for o in outs)),
     "opmix_cls": dict(collections.Counter(o["opmix_cls"] for o in outs)),
     "finish_reason": dict(collections.Counter(o["finish_reason"] for o in outs)),
     "emits": sum(o["emits"] for o in outs), "in_clean": sum(bool(o["in_clean"]) for o in outs),
     "emits_but_dropped_by_postfilter": sum(o["emits"] and not o["in_clean"] for o in outs),
     "f3_dropped_heldout_pages": sum(bool(o["f3_dropped"]) for o in outs)}
# ---- fidelity, same code as the greedy run (teacher_metrics); skipped pages are not sent -> excluded, as before
rows = rows_from_executed(pages, outs)
V["vs_teacher_all"] = teacher_metrics([r for _, r in rows])
V["vs_teacher_first960"] = teacher_metrics([r for g, r in rows if g < 960])
V["not_sent_excluded_from_fidelity"] = sum(o["status"] != "ok" for o in outs)
# ---- decision flow (Sankey): teacher tag x student decision; skipped / unparseable -> delete
flow, n_skip, n_unp = decision_flow(pages, outs)
FLOW = {"description": ("Teacher operation -> ReScraper operation on the %d held-out pages (heldout5k: teacher keep/edit/delete; teacher-deleted "
                        "pages with FineWeb-Edu >= 1.0 and a T=1 RePro-1B paraphrase -> rewrite). ReScraper = the RELEASE decoding (T=1.0, "
                        "top_p=1.0, max_tokens=3072), per-page records of the operation ablation's full arm. The %d pages the release run skipped "
                        "and the %d whose first line is not a decision tag count as delete." % (len(pages), n_skip, n_unp)),
        "source": "heldout5k.jsonl (tag) x heldout_pipeline/work/heldout5k_rel/ours.jsonl (decision; operation-ablation raw rows)",
        "pages": len(pages), "operations": OPS, "teacher_to_student": flow}
# ---- rescue branch, caps, loops
edu = {p["gid"]: p.get("edu") for p in pages}
rw_of_del = [p["gid"] for p, o in zip(pages, outs) if TOP[p["tag"]] == "delete" and o["op"] == "rewrite"]
def rep8(t):   # the post-filter's repetition(): share of repeated word 8-grams (it counts > 0.30, never drops)
    w = (t or "").lower().split()
    if len(w) < 40: return None
    g = [tuple(w[i:i + 8]) for i in range(len(w) - 7)]
    return 1.0 - len(set(g)) / len(g)
def loops(sel):
    r = [rep8(o["text"]) for o in sel if o["emits"]]
    return {"emitted": sum(o["emits"] for o in sel), "rep8_gt_0.30": sum(1 for x in r if x is not None and x > 0.30),
            "rep8_gt_0.50": sum(1 for x in r if x is not None and x > 0.50)}
capped = [o for o in outs if o["finish_reason"] == "length"]
V["rescue"] = {"teacher_rewrite_student_rewrite": flow["rewrite"]["rewrite"], "teacher_rewrite_pages": sum(flow["rewrite"].values()),
               "teacher_delete_student_rewrite": len(rw_of_del),
               "of_which_edu_in_[0.5,1.0)": sum(1 for g in rw_of_del if edu[g] is not None and 0.5 <= edu[g] < 1.0),
               "of_which_edu_ge_1.0": sum(1 for g in rw_of_del if edu[g] is not None and edu[g] >= 1.0),
               "of_which_edu_lt_0.5": sum(1 for g in rw_of_del if edu[g] is not None and edu[g] < 0.5),
               "of_which_edu_missing": sum(1 for g in rw_of_del if edu[g] is None),
               "student_rewrites_total": sum(o["op"] == "rewrite" for o in outs)}
V["cap_and_loops"] = {"finish_length": len(capped), "finish_length_by_decision": dict(collections.Counter(o["decision"] for o in capped)),
                      "finish_length_emitted": sum(o["emits"] for o in capped), "loops_all_emitted": loops(outs),
                      "loops_among_finish_length": loops(capped),
                      "loop_definition": "post-filter repetition(): 1 - distinct/total word 8-grams (docs >= 40 words); the post-filter counts > 0.30 (never drops)"}
V["decision_flow"] = flow
json.dump(V, open(W + "/ours_validation.json", "w"), indent=1, ensure_ascii=False)
OUTD = os.environ.get("HP_OUT") or os.path.join(HP, "data"); SUF = os.environ.get("HP_SUF", "5k_rel"); os.makedirs(OUTD, exist_ok=True)
json.dump(FLOW, open(OUTD + "/decision_flow_%s.json" % SUF, "w"), indent=1)
print(json.dumps({k: v for k, v in V.items()}, indent=1, ensure_ascii=False)[:5000]); print("REL_EXEC_OK")
