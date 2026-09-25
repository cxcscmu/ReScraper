"""Stage-1 SFT set: the no-rewrite base set relabelled with the Stage-1 selective-rescue decisions.

Reuses the per-page decisions rescue_build.py wrote (keep / rewrite / drop for every page the teacher
chain deleted) and rewrites the <delete> rows of the 1.38M whole-page set accordingly:
    action keep    -> <keep>    + the row's own <extract> ops, no refinement payload
    action rewrite -> <rewrite> + the row's own <extract> ops + the accepted paraphrase
    action drop    -> unchanged <delete>
Every other row is untouched, so the set size and the rest of the labels are identical to the baseline.
In our run: 522,312 keep / 359,292 edit / 499,924 delete / 1,587 rewrite (8,588 delete -> keep flips).
usage: build_stage1_set.py <sft_in.jsonl> <join_glob[,join_glob...]> <decisions_dir> <sft_out.jsonl>
The 1.38M set is the union of two halves, so <join_glob> takes a comma-separated list of patterns (the
join_*.jsonl files of data_construction/base_set/, which map each rendered input back to (shard, row)).
"""
import json, sys, glob, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import editops as E
src, joinpat, decdir, dst = sys.argv[1:5]
dec = {}
for f in sorted(glob.glob(os.path.join(decdir, "*.jsonl"))):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r["action"] != "drop": dec[(r["stem"], int(r["row"]))] = r
print("rescue decisions loaded:", len(dec), flush=True)
joins = []
for pat in joinpat.split(","):
    pat = pat.strip()
    if pat: joins += sorted(glob.glob(pat))
assert joins, ("no join files matched", joinpat)
print("join files:", len(joins), flush=True)
key = {}
for i, f in enumerate(joins):
    for l in open(f, encoding="utf-8"):
        try: r = json.loads(l)
        except ValueError: continue
        d = dec.get((r.get("stem"), int(r.get("row", -1))))
        if d: key[E.number_lines(r["input"])] = d
    print("  %3d/%d %-24s matched so far %d" % (i + 1, len(joins), os.path.basename(f), len(key)), flush=True)
print("decisions matched to training inputs:", len(key), flush=True)
S = collections.Counter(); n = 0
with open(src, encoding="utf-8") as f, open(dst + ".tmp", "w", encoding="utf-8") as g:
    for line in f:
        o = json.loads(line); n += 1; t = o["output"]
        if t.startswith("<delete>"):
            d = key.get(o["input"])
            if d:
                parts = t.split("<delete>", 2)          # ['', '\n<extract>\n<ops>\n', '']
                ops = parts[1] if len(parts) > 2 else "\n<extract>\n"
                if d["action"] == "keep":
                    o["output"] = "<keep>" + ops + "<keep>"; S["flip_keep"] += 1
                elif d["action"] == "rewrite" and d.get("text", "").strip():
                    o["output"] = "<rewrite>" + ops + "<rewrite>\n" + d["text"].strip(); S["flip_rewrite"] += 1
                else: S["unchanged"] += 1
            else: S["unchanged"] += 1
        g.write(json.dumps(o, ensure_ascii=False) + "\n")
os.replace(dst + ".tmp", dst)
print("rows %d | %s" % (n, dict(S)), flush=True)
assert S["flip_keep"] + S["flip_rewrite"] > 0
print("BUILD_STAGE1_SET_OK")
