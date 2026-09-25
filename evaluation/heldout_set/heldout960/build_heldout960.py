#!/usr/bin/env python
"""Rebuild the 960-page held-out prefix with the stage-2 SFT's rewrite rule (edu1).

Teacher cascade labels keep/edit stay byte-identical. Every teacher-deleted page (earlier label <delete>, and earlier
<rewrite>, which an older rule carved out of teacher deletes) is re-decided:
    FineWeb-Edu >= 1.0 and a T=1 RePro-1B paraphrase exists  -> <rewrite> + that paraphrase
    otherwise                                                 -> <delete>
FineWeb-Edu = the value the stage-2 SFT build reads (rescue_add/text_decisions `edu`: fineweb-edu-classifier raw
logit on the two_stage_pool/input Dripper text, 3 dp). Pages text_decisions does not cover (the pool's two-stage
refiner kept them, so the SFT build never scored them) get the same classifier on the same text, recomputed
(edu_heldout960_recomputed.jsonl; matches text_decisions on 373/374 overlapping pages, max |diff| 0.001).
Paraphrase = two_stage_pool/rwrepro_olmo_temp1 (the SFT's rewrite source) at the page's pool index; pages with none
there use gen_repro_t1_out.jsonl (same script/settings, generated for just those pages).
The row format is e2e_ops.to_staged_row's: <extract> head unchanged, then "<delete>" or "<rewrite>\\n<text>".
Rows are joined to the pool by exact Dripper text (render `drip` == two_stage_pool/input text); the render `row`
field does not index two_stage_pool/input.

usage: build_heldout960.py <out_staged.jsonl> <out_val_idx.json> <audit.jsonl>
"""
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "lib")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import e2e_ops as X
import editops as E
from eval_paths import HELDOUT3161_DIR, TWO_STAGE_POOL_DIR, HELDOUT960_DIR, HELDOUT960_STEM

S = HELDOUT960_DIR + "/"
OUT, VOUT, AUD = sys.argv[1:4]
gold = [json.loads(l) for l in open(HELDOUT3161_DIR + "/sft_e2eC_gold_tagged.jsonl", encoding="utf-8")][:960]
J = [json.loads(l) for l in open(S + "heldout960_pooljoin.jsonl")]
EDU = [json.loads(l) for l in open(S + "edu_heldout960_recomputed.jsonl")]
rw = {}
for o in map(json.loads, open(f"{TWO_STAGE_POOL_DIR}/rwrepro_olmo_temp1/{HELDOUT960_STEM}.jsonl", encoding="utf-8")):   # SFT build: text stripped, non-empty
    t = (o.get("text") or "").strip()
    if t:
        rw[int(o["i"])] = t
gen = {}
for o in map(json.loads, open(S + "gen_repro_t1_out.jsonl", encoding="utf-8")):
    gen[o["pool_idx"]] = o

def split_head(out):
    """-> (head, decision, body) with head = '<extract>' + stage-1 rm lines, exactly as to_staged_row wrote it."""
    L = out.split("\n")
    assert L[0] == "<extract>", out[:80]
    k = 1
    while k < len(L) and E.OPS_RM.match(L[k]):
        k += 1
    head = "\n".join(L[:k]); dec = L[k]; body = "\n".join(L[k + 1:])
    return head, dec, body

C = collections.Counter(); out_rows = []; audit = []
for k, (g, j, e) in enumerate(zip(gold, J, EDU)):
    assert j["gold_idx"] == k == e["gold_idx"] and j["pool_idx"][0] == e["pool_idx"]
    old, old_body = X.body_from_prediction_dfirst(g["input"], g["output"])
    rec = {"row": k, "old_label": old, "new_label": old, "edu": None, "edu_source": None, "pool_idx": j["pool_idx"][0],
           "pool_student_kept": j["pool_student_kept"][0], "paraphrase_source": None, "text_changed": False}
    if old not in ("<delete>", "<rewrite>"):
        out_rows.append(g); audit.append(rec); C[(old, old)] += 1
        continue
    head, dec, body = split_head(g["output"])
    assert (dec, old) in (("<delete>", "<delete>"), ("<rewrite>", "<rewrite>")), (k, dec, old)
    assert (head + "\n" + dec + ("\n" + body if dec == "<rewrite>" else "")) == g["output"], k
    i = j["pool_idx"][0]
    td = j["td_edu"][0]
    if td is not None:
        edu, src = td, "text_decisions"
    else:
        edu, src = e["edu"], "recomputed(fineweb-edu-classifier, same text)"
    rec["edu"], rec["edu_source"] = edu, src
    text = None
    if edu is not None and edu >= 1.0:
        if i in rw:
            text, rec["paraphrase_source"] = rw[i], "two_stage_pool/rwrepro_olmo_temp1 i=%d" % i
        elif i in gen and gen[i]["status"] == "ok":
            text, rec["paraphrase_source"] = gen[i]["text"].strip(), "generated (gen_repro_t1_out.jsonl, pool i=%d)" % i
        else:
            rec["paraphrase_source"] = "missing (generation status %s)" % gen.get(i, {}).get("status")
    if text:
        o = dict(g); o["output"] = head + "\n<rewrite>\n" + text
        rec["new_label"] = "<rewrite>"
        rec["text_changed"] = (old != "<rewrite>") or (text != old_body)
    else:
        o = dict(g); o["output"] = head + "\n<delete>"
        rec["new_label"] = "<delete>"
    nt, nb = X.body_from_prediction_dfirst(o["input"], o["output"])
    assert nt == rec["new_label"] and (nt == "<delete>" or nb == text), k
    out_rows.append(o); audit.append(rec); C[(old, rec["new_label"])] += 1

with open(OUT, "w", encoding="utf-8") as f:
    for o in out_rows:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
json.dump({"source": OUT, "rows": len(out_rows), "val_idx": list(range(len(out_rows))),
           "note": "first 960 rows of the 3,161-page staged held-out table, teacher-deleted pages relabeled with "
                   "the stage-2 SFT rule edu1 (FineWeb-Edu >= 1.0 -> <rewrite> + T=1 RePro-1B paraphrase); keep/edit untouched"},
          open(VOUT, "w"))
with open(AUD, "w", encoding="utf-8") as f:
    for r in audit:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
lab = collections.Counter(r["new_label"] for r in audit)
print("new label counts:", dict(lab))
print("old -> new:", {"%s->%s" % kv: v for kv, v in sorted(C.items())})
print("paraphrase sources:", dict(collections.Counter((r["paraphrase_source"] or "-").split(" ")[0] for r in audit if r["new_label"] == "<rewrite>")))
print("edu sources (teacher-deleted):", dict(collections.Counter(r["edu_source"] for r in audit if r["edu_source"])))
print("unchanged bytes:", sum(1 for o, g in zip(out_rows, gold) if o["output"] == g["output"]), "/", len(gold))
