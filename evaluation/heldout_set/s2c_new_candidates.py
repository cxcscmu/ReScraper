"""Held-out set step 2c (CPU): final candidate set for the 2 new shards + pool join + edu.
Filters, in order: seed filter of the SFT seed sampler (empty / MARKUP / LID en >= 0.65), stage-1 feasibility, then
exact-duplicate Dripper text (the held-out preparation drops texts that occur on more than one held-out page; here the
key set is the new candidates plus the 3,161 existing staged rows, so no page text enters the 5k table twice).
Pool join = heldout960/join_pool960.py (exact Dripper text -> two_stage_pool/input index), edu recompute =
heldout960/edu_score960.py (CPU fp32, batches of 64 in candidate order).
Writes cand_new.jsonl ({"i": cid, "output": drip} = the input format of the Qwen3.8-27B labeling script) and
cand_new_meta.jsonl.
usage: s2c_new_candidates.py <outdir>"""
import json, gzip, os, sys, hashlib, collections, time, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib")); sys.path.insert(0, os.path.join(HERE, ".."))
from eval_paths import TWO_STAGE_POOL_DIR as T, TEXT_DECISIONS_DIR as DEC, HELDOUT3161_DIR as GD
import e2e_ops as X
OUT = sys.argv[1]; t0 = time.time()
md5 = lambda s: hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()
lid = {o["cid"]: o for o in map(json.loads, open(OUT + "/lid.jsonl"))}
old = [json.loads(l) for l in open(OUT + "/old_prov.jsonl")]
rows = [json.loads(l) for f in sorted(glob.glob(GD + "/render_*.jsonl")) for l in open(f, encoding="utf-8")]
olddrip = [r["drip"] for r in rows if X.to_staged_row(r["input"], r["drip"], r["output"])[0]]
assert len(olddrip) == 3161
C = collections.Counter(); keep = []
for l in open(OUT + "/cand_pre.jsonl", encoding="utf-8"):
    r = json.loads(l); L = lid[r["cid"]]; r["lid"], r["lid_prob"] = L["lid"], L["lid_prob"]
    if r["empty"]: r["drop"] = "empty"
    elif r["markup"]: r["drop"] = "markup"
    elif r["lid"] != "__label__en" or r["lid_prob"] < 0.65: r["drop"] = "non_en"
    elif not r["stage1_ok"]: r["drop"] = "stage1_not_line_subset"
    else: r["drop"] = None
    C[r["drop"] or "pass_filters"] += 1; keep.append(r)
cnt = collections.Counter(md5(r["drip"]) for r in keep if r["drop"] is None)
oldk = set(md5(t) for t in olddrip)
for r in keep:
    if r["drop"] is None:
        h = md5(r["drip"])
        if cnt[h] > 1: r["drop"] = "dup_text_within_new"; C["dup_text_within_new"] += 1
        elif h in oldk: r["drop"] = "dup_text_of_existing_heldout"; C["dup_text_of_existing_heldout"] += 1
print("filters:", dict(C), flush=True)
# pool join
by = collections.defaultdict(list)
for r in keep: by[r["stem"]].append(r)
for stem, rs in by.items():
    inp = [json.loads(l)["text"] for l in gzip.open(f"{T}/input/{stem}.jsonl.gz", "rt", encoding="utf-8")]
    best = [json.loads(l)["text"] for l in gzip.open(f"{T}/best/{stem}.jsonl.gz", "rt", encoding="utf-8")]
    pos = collections.defaultdict(list)
    for j, t in enumerate(inp): pos[t].append(j)
    td = {o["row"]: o for o in map(json.loads, open(f"{DEC}/{stem}.jsonl"))}
    te = {o["i"]: o for o in map(json.loads, open(f"{T}/edu/{stem}.jsonl"))}
    rw = {}
    for o in map(json.loads, open(f"{T}/rwrepro_olmo_temp1/{stem}.jsonl", encoding="utf-8")):
        t = (o.get("text") or "").strip()
        if t: rw[int(o["i"])] = t
    J = collections.Counter()
    for r in rs:
        js = pos.get(r["drip"], [])
        r.update({"pool_idx": js, "td_edu": [td[j]["edu"] if j in td else None for j in js], "td_action": [td[j]["action"] if j in td else None for j in js],
                  "pool_edu": [te[j]["edu"] if j in te else None for j in js], "pool_int_edu": [te[j]["int_edu"] if j in te else None for j in js],
                  "pool_student_kept": [bool(best[j].strip()) for j in js], "rw_t1": [rw.get(j) for j in js]})
        J["joined" if js else "unjoined"] += 1; J["pool_idx_eq_step2_row"] += bool(js) and js[0] == r["step2_row"]
    print("pool join", stem, "two_stage_pool/input rows", len(inp), dict(J), flush=True)
# edu recompute on every candidate that passed the filters
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(int(os.environ.get("EDU_THREADS", "8")))
etok = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
emod = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier").eval()
@torch.no_grad()
def escore(ts):
    out = []
    for i in range(0, len(ts), 64):
        e = etok(ts[i:i+64], return_tensors="pt", padding=True, truncation=True, max_length=512)
        out += emod(**e).logits.squeeze(-1).float().cpu().tolist()
    return out
ok = [r for r in keep if r["drop"] is None]
es = escore([r["drip"] for r in ok])
for r, s in zip(ok, es): r["edu_raw_recomputed"] = s; r["edu_recomputed"] = round(float(s), 3)
print("edu recomputed", len(es), "%.0fs" % (time.time() - t0), flush=True)
with open(OUT + "/cand_new_meta.jsonl", "w", encoding="utf-8") as f, open(OUT + "/cand_new.jsonl", "w", encoding="utf-8") as g:
    for r in keep:
        f.write(json.dumps({k: v for k, v in r.items() if k not in ("full",)}, ensure_ascii=False) + "\n")
        if r["drop"] is None: g.write(json.dumps({"i": r["cid"], "stem": r["stem"], "row": r["step2_row"], "output": r["drip"]}, ensure_ascii=False) + "\n")
with open(OUT + "/cand_new_full.jsonl", "w", encoding="utf-8") as f:
    for r in ok: f.write(json.dumps({"cid": r["cid"], "full": r["full"]}, ensure_ascii=False) + "\n")
print("27B input rows", len(ok), "by shard", dict(collections.Counter(r["stem"] for r in ok)))
print("STEP2C_OK")
