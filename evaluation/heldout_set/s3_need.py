"""Held-out set step 3b: which pages need a newly generated T=1 RePro-1B paraphrase (same rule as
heldout960/build_heldout960.py, edu1): teacher-deleted (Qwen3.8-27B <delete>; for the existing staged rows: staged
decision <delete> or <rewrite>), edu >= 1.0 where edu = text_decisions value of the joined pool page if present, else
the recomputed value, and no paraphrase in two_stage_pool/rwrepro_olmo_temp1 for the joined pool page. Rows 0..959 are
excluded: they reuse heldout960's generations (gen_repro_t1_out.jsonl), so the first 960 rows stay byte-identical to
the heldout960 table.
usage: s3_need.py <workdir>   (reads old_prov.jsonl, cand_new_meta.jsonl, t27/part-*.jsonl; writes repro_need.jsonl)"""
import json, os, sys, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib")); sys.path.insert(0, os.path.join(HERE, ".."))
from eval_paths import HELDOUT3161_DIR as GD
import rescraper_ops as X
W = sys.argv[1]
gold = [json.loads(l) for l in open(GD + "/sft_e2eC_gold_tagged.jsonl", encoding="utf-8")]
rows = [json.loads(l) for f in sorted(glob.glob(GD + "/render_*.jsonl")) for l in open(f, encoding="utf-8")]
drip = [r["drip"] for r in rows if X.to_staged_row(r["input"], r["drip"], r["output"])[0]]
old = [json.loads(l) for l in open(W + "/old_prov.jsonl")]
assert len(gold) == len(old) == len(drip) == 3161
def edu_rule(p):
    if p.get("pool_idx") and p["td_edu"][0] is not None: return p["td_edu"][0], "text_decisions"
    return p["edu_recomputed"], "recomputed"
def has_rw(p): return bool(p.get("pool_idx")) and p["rw_t1"][0] is not None
C = collections.Counter(); need = []
for k in range(960, 3161):
    tag = X.body_from_prediction_dfirst(gold[k]["input"], gold[k]["output"])[0]
    if tag not in ("<delete>", "<rewrite>"): continue
    e, src = edu_rule(old[k]); C["old_teacher_deleted"] += 1
    if e >= 1.0:
        C["old_edu>=1"] += 1
        if has_rw(old[k]): C["old_pool_rw"] += 1
        else: need.append({"key": "old:%d" % k, "gold_idx": k, "edu": e, "edu_source": src, "text": drip[k]})
lab = {}
for f in sorted(glob.glob(W + "/t27/part-*.jsonl")):
    for l in open(f, encoding="utf-8"): r = json.loads(l); lab[r["i"]] = r
for l in open(W + "/cand_new_meta.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["drop"] is not None: continue
    C["new_cands"] += 1; L = lab.get(r["cid"])
    if L is None: C["new_no_27b_label"] += 1; continue
    if not L["deleted"]: continue
    C["new_27b_deleted"] += 1; e, src = edu_rule(r)
    if e >= 1.0:
        C["new_edu>=1"] += 1
        if has_rw(r): C["new_pool_rw"] += 1
        else: need.append({"key": "new:%d" % r["cid"], "cid": r["cid"], "edu": e, "edu_source": src, "text": r["drip"]})
with open(W + "/repro_need.jsonl", "w", encoding="utf-8") as f:
    for x in need: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("need:", len(need), dict(C)); print("NEED_OK")
