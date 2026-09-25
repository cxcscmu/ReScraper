"""Join the first 960 rows of the 3,161-page staged held-out table to the two-stage pool run of their shard by exact
Dripper text (render `drip` == two_stage_pool/input text), since the render `row` field does NOT index
two_stage_pool/input. Pulls, per pool index j: text_decisions edu/action (the SFT build's edu source),
two_stage_pool/edu edu/int_edu, the T=1 paraphrase in rwrepro_olmo_temp1 (keyed i=j), and whether the pool's
two-stage refiner kept the page.
Output: $HELDOUT960_DIR/heldout960_pooljoin.jsonl
usage: join_pool960.py"""
import sys, os, json, glob, gzip, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "lib")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import rescraper_ops as X
from eval_paths import HELDOUT3161_DIR, TWO_STAGE_POOL_DIR, TEXT_DECISIONS_DIR, HELDOUT960_DIR, HELDOUT960_STEM
T, STEM = TWO_STAGE_POOL_DIR, HELDOUT960_STEM
os.makedirs(HELDOUT960_DIR, exist_ok=True)
rows = [json.loads(l) for f in sorted(glob.glob(HELDOUT3161_DIR + "/render_*.jsonl")) for l in open(f, encoding="utf-8")]
prov = [r for r in rows if X.to_staged_row(r["input"], r["drip"], r["output"])[0]][:960]
assert all(r["stem"] == STEM for r in prov), "the 960 rows are expected to come from one shard"
gold = [json.loads(l) for l in open(HELDOUT3161_DIR + "/sft_e2eC_gold_tagged.jsonl", encoding="utf-8")][:960]
inp = [json.loads(l)["text"] for l in gzip.open(f"{T}/input/{STEM}.jsonl.gz", "rt", encoding="utf-8")]
best = [json.loads(l)["text"] for l in gzip.open(f"{T}/best/{STEM}.jsonl.gz", "rt", encoding="utf-8")]
pos = collections.defaultdict(list)
for j, t in enumerate(inp): pos[t].append(j)
td = {o["row"]: o for o in map(json.loads, open(f"{TEXT_DECISIONS_DIR}/{STEM}.jsonl"))}
te = {o["i"]: o for o in map(json.loads, open(f"{T}/edu/{STEM}.jsonl"))}
rw = {}
for o in map(json.loads, open(f"{T}/rwrepro_olmo_temp1/{STEM}.jsonl", encoding="utf-8")):
    t = (o.get("text") or "").strip()
    if t: rw[o["i"]] = t
out = open(HELDOUT960_DIR + "/heldout960_pooljoin.jsonl", "w")
C = collections.Counter()
for k, (r, g) in enumerate(zip(prov, gold)):
    tag, _ = X.body_from_prediction_dfirst(g["input"], g["output"])
    js = pos.get(r["drip"], [])
    C["n_match_%d" % min(len(js), 3)] += 1
    rec = {"gold_idx": k, "old_tag": tag, "render_row": r["row"], "pool_idx": js,
           "td_edu": [td[j]["edu"] if j in td else None for j in js], "td_action": [td[j]["action"] if j in td else None for j in js],
           "pool_edu": [te[j]["edu"] if j in te else None for j in js], "pool_int_edu": [te[j]["int_edu"] if j in te else None for j in js],
           "pool_student_kept": [bool(best[j].strip()) for j in js], "has_rw_t1": [j in rw for j in js]}
    out.write(json.dumps(rec) + "\n")
    if tag in ("<delete>", "<rewrite>"):
        C["teacher_deleted"] += 1
        C["td_has_edu=%s" % any(e is not None for e in rec["td_edu"])] += 1
        C["multi_match"] += len(js) > 1
print(dict(C))
