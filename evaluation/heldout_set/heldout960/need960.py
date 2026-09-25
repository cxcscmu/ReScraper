"""Pages of the 960 that need a newly generated T=1 RePro-1B paraphrase under the edu1 rule: teacher-deleted (staged
decision <delete> or <rewrite>), FineWeb-Edu >= 1.0 (text_decisions value of the joined pool page if present, else the
recomputed value) and no paraphrase in two_stage_pool/rwrepro_olmo_temp1 for the joined pool page.
Output: $HELDOUT960_DIR/gen_need.jsonl {gold_idx, pool_idx, edu, edu_source, pool_student_kept} (9 pages)
usage: need960.py"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "lib")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import rescraper_ops as X
from eval_paths import HELDOUT3161_DIR, TWO_STAGE_POOL_DIR, HELDOUT960_DIR, HELDOUT960_STEM
gold = [json.loads(l) for l in open(HELDOUT3161_DIR + "/sft_e2eC_gold_tagged.jsonl", encoding="utf-8")][:960]
J = [json.loads(l) for l in open(HELDOUT960_DIR + "/heldout960_pooljoin.jsonl")]
EDU = [json.loads(l) for l in open(HELDOUT960_DIR + "/edu_heldout960_recomputed.jsonl")]
rw = set()
for o in map(json.loads, open(f"{TWO_STAGE_POOL_DIR}/rwrepro_olmo_temp1/{HELDOUT960_STEM}.jsonl", encoding="utf-8")):
    if (o.get("text") or "").strip(): rw.add(int(o["i"]))
out = []
for k, (g, j, e) in enumerate(zip(gold, J, EDU)):
    assert j["gold_idx"] == k == e["gold_idx"] and j["pool_idx"][0] == e["pool_idx"]
    old, _ = X.body_from_prediction_dfirst(g["input"], g["output"])
    if old not in ("<delete>", "<rewrite>"): continue
    i = j["pool_idx"][0]; td = j["td_edu"][0]
    edu, src = (td, "text_decisions") if td is not None else (e["edu"], "recomputed")
    if edu >= 1.0 and i not in rw:
        out.append({"gold_idx": k, "pool_idx": i, "edu": edu, "edu_source": src, "pool_student_kept": j["pool_student_kept"][0]})
with open(HELDOUT960_DIR + "/gen_need.jsonl", "w") as f:
    for x in out: f.write(json.dumps(x) + "\n")
print("need", len(out))
