#!/usr/bin/env python
"""Build heldout5k: a 5,000-page held-out page table with teacher labels under the released student's rule (edu1).

rows 0..959     heldout960 (heldout960/build_heldout960.py), rebuilt here with its exact logic and inputs
                (heldout960_pooljoin, edu_heldout960_recomputed, gen_repro_t1_out) -> must be byte-identical to
                $HELDOUT960_DIR/sft_e2eC_heldout960_edu1_tagged.jsonl
rows 960..3160  rest of the 3,161-page staged table ($HELDOUT3161_DIR/sft_e2eC_gold_tagged.jsonl), same relabel: keep/edit
                copied; teacher-deleted pages (staged <delete> or <rewrite>) -> <rewrite> + T=1 RePro-1B paraphrase if
                FineWeb-Edu >= 1.0 and a passing paraphrase exists (pool rwrepro_olmo_temp1 first, else generated in
                step 3c), else <delete>. edu = rescue_add/text_decisions value of the pool page joined by exact Dripper
                text, else recomputed.
rows 3161..4999 two new held-out pool shards: Qwen3.8-27B strict-subset label (the SFT teacher's prompt/decoding) -> tag
                by the SFT builder's rule -> e2e_ops.to_staged_row -> same edu1 relabel; taken in (shard draw order,
                step-2 row) order until 5,000.
usage: s4_build.py <workdir> <outdir>"""
import sys, json, re, collections, hashlib, os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib")); sys.path.insert(0, os.path.join(HERE, ".."))
import e2e_ops as X
import editops as E
from render import webkit_post     # the SFT builder's webkit_post, verbatim
from eval_paths import HELDOUT3161_DIR, HELDOUT960_DIR, TWO_STAGE_POOL_DIR, HELDOUT960_STEM
W, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
GF = HELDOUT960_DIR + "/"
N_TOTAL = int(os.environ.get("HELDOUT_NTOTAL", "5000"))   # override only for dry runs
md5 = lambda s: hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()
TAGMAP = {"<extract>": "keep", "<refine>": "edit", "<delete>": "delete", "<rewrite>": "rewrite"}
def split_head(out):   # heldout960/build_heldout960.split_head, verbatim
    L = out.split("\n")
    assert L[0] == "<extract>", out[:80]
    k = 1
    while k < len(L) and E.OPS_RM.match(L[k]):
        k += 1
    head = "\n".join(L[:k]); dec = L[k]; body = "\n".join(L[k + 1:])
    return head, dec, body
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

gold = load(HELDOUT3161_DIR + "/sft_e2eC_gold_tagged.jsonl"); assert len(gold) == 3161
ref960 = load(GF + "sft_e2eC_heldout960_edu1_tagged.jsonl"); assert len(ref960) == 960
J960 = load(GF + "heldout960_pooljoin.jsonl"); EDU960 = load(GF + "edu_heldout960_recomputed.jsonl")
rw960 = {}
for o in load(f"{TWO_STAGE_POOL_DIR}/rwrepro_olmo_temp1/{HELDOUT960_STEM}.jsonl"):
    t = (o.get("text") or "").strip()
    if t: rw960[int(o["i"])] = t
gen960 = {}
for o in load(GF + "gen_repro_t1_out.jsonl"): gen960[o["pool_idx"]] = o      # rule edu1 uses only this file
prov = load(W + "/old_prov.jsonl"); assert len(prov) == 3161 and all(p["gold_idx"] == k for k, p in enumerate(prov))
rows_r = [json.loads(l) for f in sorted(glob.glob(HELDOUT3161_DIR + "/render_*.jsonl")) for l in open(f, encoding="utf-8")]
rprov = [r for r in rows_r if X.to_staged_row(r["input"], r["drip"], r["output"])[0]]; assert len(rprov) == 3161
gen = {o["key"]: o for o in load(W + "/repro_out.jsonl")} if os.path.getsize(W + "/repro_out.jsonl") else {}

def relabel(g, edu, edu_src, rw_text, rw_src, gen_o, gen_src):
    """heldout960/build_heldout960.py (rule edu1) decision for one staged row; returns (row, audit fields)."""
    old, old_body = X.body_from_prediction_dfirst(g["input"], g["output"])
    rec = {"old_label": old, "new_label": old, "edu_rule": None, "edu_source": None, "paraphrase_source": None, "text_changed": False}
    if old not in ("<delete>", "<rewrite>"):
        return g, rec
    head, dec, body = split_head(g["output"])
    assert (dec, old) in (("<delete>", "<delete>"), ("<rewrite>", "<rewrite>"))
    assert (head + "\n" + dec + ("\n" + body if dec == "<rewrite>" else "")) == g["output"]
    rec["edu_rule"], rec["edu_source"] = edu, edu_src
    text = None
    if edu is not None and edu >= 1.0:
        if rw_text is not None:
            text, rec["paraphrase_source"] = rw_text, rw_src
        elif gen_o is not None and gen_o["status"] == "ok":
            text, rec["paraphrase_source"] = gen_o["text"].strip(), gen_src
        else:
            rec["paraphrase_source"] = "missing (generation status %s)" % (gen_o or {}).get("status")
    if text:
        o = dict(g); o["output"] = head + "\n<rewrite>\n" + text
        rec["new_label"] = "<rewrite>"; rec["text_changed"] = (old != "<rewrite>") or (text != old_body)
    else:
        o = dict(g); o["output"] = head + "\n<delete>"; rec["new_label"] = "<delete>"
    nt, nb = X.body_from_prediction_dfirst(o["input"], o["output"])
    assert nt == rec["new_label"] and (nt == "<delete>" or nb == text)
    return o, rec

table = []; staged = []; C = collections.Counter()
# ---------- rows 0..3160 ----------
for k in range(3161):
    g = gold[k]; p = prov[k]; r = rprov[k]
    if k < 960:
        j, e = J960[k], EDU960[k]
        assert j["gold_idx"] == k == e["gold_idx"] and j["pool_idx"][0] == e["pool_idx"]
        i = j["pool_idx"][0]; td = j["td_edu"][0]
        edu, src = (td, "text_decisions") if td is not None else (e["edu"], "recomputed(fineweb-edu-classifier, same text)")
        rwt = rw960.get(i); gen_o = gen960.get(i)
        o, rec = relabel(g, edu, src, rwt, "two_stage_pool/rwrepro_olmo_temp1 i=%d" % i, gen_o, "generated (gen_repro_t1_out.jsonl, pool i=%d)" % i)
        assert p["pool_idx"] and p["pool_idx"][0] == i, (k, p.get("pool_idx"), i)
        source = "heldout960"
    else:
        js = p.get("pool_idx") or []
        td = p["td_edu"][0] if js else None
        edu, src = (td, "text_decisions") if td is not None else (p["edu_recomputed"], "recomputed(fineweb-edu-classifier, same text)")
        i = js[0] if js else None
        rwt = p["rw_t1"][0] if js else None
        o, rec = relabel(g, edu, src, rwt, "two_stage_pool/rwrepro_olmo_temp1 i=%s" % i, gen.get("old:%d" % k), "generated (repro_out.jsonl, old:%d)" % k)
        source = "relabel2201"
    tag, fin = X.body_from_prediction_dfirst(o["input"], o["output"])
    edu_page = rec["edu_rule"] if rec["edu_rule"] is not None else ((p["td_edu"][0] if p.get("pool_idx") and p["td_edu"][0] is not None else p["edu_recomputed"]))
    table.append({"gid": k, "source": source, "stem": p["stem"], "step1_idx": p.get("step1_idx"), "in_md5": p.get("in_md5"), "url": p.get("url"),
                  "warc_id": p.get("warc_id"), "input": o["input"], "drip": r["drip"], "output": o["output"], "tag": TAGMAP[tag], "gfinal": fin,
                  "edu": edu_page, "edu_source": rec["edu_source"] or ("text_decisions" if (p.get("pool_idx") and p["td_edu"][0] is not None) else "recomputed"),
                  "edu_recomputed": p["edu_recomputed"], "old_tag": TAGMAP[rec["old_label"]], "paraphrase_source": rec["paraphrase_source"],
                  "pool_idx": (p.get("pool_idx") or [None])[0], "pool_student_kept": (p.get("pool_student_kept") or [None])[0],
                  "simp_row": p.get("simp_row"), "render_i": p.get("render_i"), "rerender_eq_input": p.get("rerender_eq_input")})
    staged.append(o); C[(source, TAGMAP[tag])] += 1
# byte-identity of rows 0..959
for k in range(960):
    assert json.dumps(staged[k], ensure_ascii=False) == json.dumps(ref960[k], ensure_ascii=False), ("row differs from heldout960", k)
print("rows 0..959 byte-identical to heldout960: OK", flush=True)
# ---------- new shards ----------
meta = [m for m in load(W + "/cand_new_meta.jsonl") if m["drop"] is None] if N_TOTAL > 3161 else []
meta.sort(key=lambda m: (m["shard_order"], m["step2_row"]))
full = {o["cid"]: o["full"] for o in load(W + "/cand_new_full.jsonl")} if meta else {}
lab = {}
for f in sorted(glob.glob(W + "/t27/part-*.jsonl")):
    for o in load(f): lab[o["i"]] = o
NC = collections.Counter(); newrows = []
for m in meta:
    L = lab.get(m["cid"])
    if L is None: NC["no_27b_label(empty_or_too_long)"] += 1; continue
    drip = m["drip"]
    if L["deleted"]: t27tag, body = "<delete>", ""
    else:
        b = webkit_post(L.get("refined") or ""); t27tag, body = ("<extract>" if " ".join(b.split()) == " ".join(drip.split()) else "<refine>"), b
    out = t27tag if not body.strip() else t27tag + "\n" + body
    st, kind = X.to_staged_row(full[m["cid"]], drip, out)
    if st is None: NC["stage1_" + kind] += 1; continue
    NC["kind_" + kind] += 1; NC["finish_" + str(L.get("finish"))] += 1
    js = m.get("pool_idx") or []
    td = m["td_edu"][0] if js else None
    edu, src = (td, "text_decisions") if td is not None else (m["edu_recomputed"], "recomputed(fineweb-edu-classifier, same text)")
    i = js[0] if js else None
    rwt = m["rw_t1"][0] if js else None
    o, rec = relabel(st, edu, src, rwt, "two_stage_pool/rwrepro_olmo_temp1 i=%s" % i, gen.get("new:%d" % m["cid"]), "generated (repro_out.jsonl, new:%d)" % m["cid"])
    newrows.append((m, o, rec, t27tag, kind, L))
print("new-shard rows available", len(newrows), dict(NC), flush=True)
need_new = N_TOTAL - len(table)
assert len(newrows) >= need_new, ("NOT ENOUGH NEW ROWS", len(newrows), need_new)
for m, o, rec, t27tag, kind, L in newrows[:need_new]:
    gid = len(table); tag, fin = X.body_from_prediction_dfirst(o["input"], o["output"])
    js = m.get("pool_idx") or []
    edu_page = rec["edu_rule"] if rec["edu_rule"] is not None else (m["td_edu"][0] if js and m["td_edu"][0] is not None else m["edu_recomputed"])
    table.append({"gid": gid, "source": "new_shard", "stem": m["stem"], "step1_idx": m["step1_idx"], "in_md5": m["in_md5"], "url": m["url"],
                  "warc_id": m["warc_id"], "input": o["input"], "drip": m["drip"], "output": o["output"], "tag": TAGMAP[tag], "gfinal": fin,
                  "edu": edu_page, "edu_source": rec["edu_source"] or ("text_decisions" if (js and m["td_edu"][0] is not None) else "recomputed"),
                  "edu_recomputed": m["edu_recomputed"], "old_tag": None, "paraphrase_source": rec["paraphrase_source"],
                  "pool_idx": js[0] if js else None, "pool_student_kept": (m.get("pool_student_kept") or [None])[0],
                  "step2_row": m["step2_row"], "cid": m["cid"], "t27_tag": TAGMAP[t27tag], "staged_kind": kind, "t27_finish": L.get("finish")})
    staged.append(o); C[("new_shard", TAGMAP[tag])] += 1
assert len(table) == len(staged) == N_TOTAL
last = table[-1]; print("last row: gid", last["gid"], "stem", last["stem"], "step2_row", last.get("step2_row"), "cid", last.get("cid"), flush=True)
with open(OUT + "/sft_e2eC_heldout5k_edu1_tagged.jsonl", "w", encoding="utf-8") as f:
    for o in staged: f.write(json.dumps(o, ensure_ascii=False) + "\n")
json.dump({"source": "sft_e2eC_heldout5k_edu1_tagged.jsonl", "rows": N_TOTAL, "val_idx": list(range(N_TOTAL)),
           "note": "heldout5k: rows 0..959 = heldout960 byte for byte; 960..3160 = rest of the 3,161-page staged table relabelled with edu1; "
                   "3161..4999 = 2 new held-out pool shards, same teacher cascade (27B greedy + FineWeb-Edu >= 1.0 -> T=1 RePro-1B)"},
          open(OUT + "/sft_e2eC_heldout5k_edu1_val_idx.json", "w"))
with open(OUT + "/heldout5k_nores.jsonl", "w", encoding="utf-8") as f:
    for r in table: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("label mix:", {"%s/%s" % k: v for k, v in sorted(C.items())})
print("BUILD_OK")
