"""Teacher targets for the SFT seed pages, grouped by shard -> <out_dir>/wanted.jsonl for the raw-HTML join.

Each seed page carries the refining teacher's label (data_construction/teacher_refine/, key i). The label rule
(the same as for every other set built from these labels):
    teacher deleted                                  -> <delete>
    webkit_post(refined) == webkit_post(dripper text) up to whitespace -> <extract>   (Dripper output kept as is)
    otherwise                                        -> <refine> + the refined text
i.e. exactly the three no-rewrite targets; <rewrite> targets are added later from the rescue step.

Gates built in:
  * held-out shards (HELDOUT_SHARDS, one stem per line) are excluded BY SHARD STEM, never by content;
  * the base set was built in two halves (~1.0M and ~0.62M seed pages). When PREV_WANTED points at the
    wanted.jsonl of an earlier half, only the complement is selected, and before anything is written the
    construction is verified against the targets of that half on a spread-out sample of pages: every
    non-<rewrite> sample row must come out byte-identical (rows that half had turned into <rewrite> are
    counted separately; build_norw.py folds them back into <delete>). Abort if agreement < 0.95.
env:   WORK_DIR, SFT_DIR, HELDOUT_SHARDS, PREV_WANTED (optional)
usage: select_pages.py <out_dir> [sample=4000]
"""
import json, glob, sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import render as B

R = os.environ["WORK_DIR"]
D = os.environ.get("SFT_DIR", R + "/sft_data")
OUT = sys.argv[1]
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
PREV = os.environ.get("PREV_WANTED", "")

GOLD = set(l.strip() for l in open(os.environ["HELDOUT_SHARDS"], encoding="utf-8") if l.strip())
print("held-out stems", len(GOLD), sorted(GOLD), flush=True)

used = set(); ver = {}
if PREV:
    for l in open(PREV, encoding="utf-8"):
        for r in json.loads(l)["rows"]:
            used.add(r["i"])
            if len(ver) < SAMPLE and r["i"] % 251 == 0: ver[r["i"]] = r["output"]
print("seed pages already used by the earlier half", len(used), "verification sample", len(ver), flush=True)

need = {}; tot = 0; goldskip = 0
for l in open(D + "/seed/seed_pages.jsonl", encoding="utf-8"):
    d = json.loads(l); tot += 1
    if d["stem"] in GOLD: goldskip += 1; continue
    if d["i"] in used and d["i"] not in ver: continue
    need[d["i"]] = (d["stem"], d["row"], d["output"] or "")
print("seed rows", tot, "skipped as held-out stem", goldskip, "candidates (complement + sample)", len(need), flush=True)

by = collections.defaultdict(list); v = collections.Counter(); s = collections.Counter()
for p in sorted(glob.glob(D + "/teacher_refine/part-*.jsonl")):
    for l in open(p, encoding="utf-8"):
        try: r = json.loads(l)
        except ValueError: s["bad_json_line"] += 1; continue
        got = need.pop(r["i"], None)
        if got is None: continue
        stem, row, raw = got
        drip = B.webkit_post(raw)
        if not drip.strip(): s["empty_drip"] += 1; continue
        if r.get("deleted"): tag, body = "<delete>", ""
        else:
            b = B.webkit_post(r.get("refined") or "")
            tag, body = ("<extract>" if " ".join(b.split()) == " ".join(drip.split()) else "<refine>"), b
        out = tag if not body.strip() else tag + chr(10) + body
        if r["i"] in ver:
            ex = ver[r["i"]]; et = ex.split(chr(10), 1)[0].strip()
            if et == "<rewrite>": v["rescued_to_rewrite"] += 1
            elif ex == out: v["match"] += 1
            elif et == tag: v["tag_ok_body_differs"] += 1
            else: v["tagdiff_" + et.strip("<>") + "_to_" + tag.strip("<>")] += 1
            continue
        s[tag] += 1
        by[stem].append({"i": r["i"], "stem": stem, "row": row, "tag": tag, "output": out})
print("no teacher label for", len(need), "candidate pages", flush=True)
print("verification vs the earlier half:", dict(v), flush=True)
comp = sum(n for k, n in v.items() if k != "rescued_to_rewrite")
rate = v["match"] / max(1, comp)
print("comparable (non-rescued) sample rows %d, byte-identical %.4f" % (comp, rate), flush=True)
if PREV:
    assert comp >= 200 and rate >= 0.95, "constructed targets disagree with the earlier half - refusing to build"

n = sum(len(x) for x in by.values())
assert not (set(by) & GOLD)
os.makedirs(OUT, exist_ok=True)
with open(OUT + "/wanted.jsonl.tmp", "w", encoding="utf-8") as f:
    for stem in sorted(by): f.write(json.dumps({"stem": stem, "rows": by[stem]}, ensure_ascii=False) + chr(10))
os.replace(OUT + "/wanted.jsonl.tmp", OUT + "/wanted.jsonl")
print("selected rows", n, "stems", len(by), dict(s), flush=True)
print("SELECT_PAGES_OK")
