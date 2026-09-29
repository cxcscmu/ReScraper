#!/usr/bin/env python3
"""Join the 5,000 held-out pages with every system's output + judge/score signals."""
import json, os, collections, re, sys

H = os.path.expanduser("~/judge_babel")
W = f"{H}/sec5_5k/work/gold5k_rel"


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


gold = {int(r["gid"]): r for r in rd(f"{H}/gold5k/gold5k.jsonl")}
ours = {int(r["gid"]): r for r in rd(f"{W}/ours.jsonl")}
sysd = {}
for k in ("ultrax", "proxc", "refinedweb_rule", "fineweb_rule"):
    sysd[k] = {int(r["gid"]): r for r in rd(f"{W}/sys_{k}.jsonl")}
judge = {int(r["gid"]): r["parsed"] for r in rd(f"{H}/stu_teacher_why/runs/keep_judge_5000.jsonl")}
qb = {p["gid"]: p for p in json.load(open(os.path.expanduser(
    "~/ReScraper-ICLR-2027/data/quality_buckets.json")))["pages"]}
faith = {}
for r in rd(f"{H}/appendix_0925/faithfulness/runs/metrics.jsonl"):
    if r["arm"] == "rel":
        faith[int(r["gid"])] = r

print("gold", len(gold), "ours", len(ours), "judge", len(judge), "qb", len(qb), "faith_rel", len(faith))
print("ops", collections.Counter(o.get("op") for o in ours.values()))
print("status", collections.Counter(o.get("status") for o in ours.values()))
g = gold[0]
print("faith keys sample gid", list(faith)[:5])

# ---- sanity: are rewrite gids in faith == ours rewrites?
rw = {g for g, o in ours.items() if o.get("op") == "rewrite"}
print("rewrites", len(rw), "with faith", len(rw & set(faith)))

out = []
for gid, g in gold.items():
    o = ours.get(gid) or {}
    rec = {
        "gid": gid,
        "url": g.get("url"),
        "rendered": g["input"],
        "resiliparse": g.get("resiliparse"),
        "drip": g.get("drip"),
        "ours_op": o.get("op"), "ours_status": o.get("status"),
        "ours_text": o.get("text"), "ours_raw": o.get("raw"),
        "ours_extracted": o.get("extracted"),
        "judge": judge.get(gid),
        "pre": (qb.get(gid) or {}).get("pre"), "post": (qb.get(gid) or {}).get("post"),
        "faith": {k: faith[gid][k] for k in ("bs_F", "ent_novel", "ent_novel_n", "ent_novel_page_n",
                                            "num_novel", "num_novel_page_n", "num_novel_ex",
                                            "ent_novel_ex", "src_words", "rw_words")} if gid in faith else None,
    }
    for k, d in sysd.items():
        r = d.get(gid) or {}
        rec[k] = {"text": r.get("text"), "status": r.get("status"), "reason": r.get("reason"),
                  "program": r.get("program")}
    out.append(rec)

with open(sys.argv[1], "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", sys.argv[1], len(out))
