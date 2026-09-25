"""FineWeb-Edu score of the 960 pages, computed exactly as the rescue builder's escore() (the source of
text_decisions' `edu`, which the stage-2 SFT build thresholds): HuggingFaceFW/fineweb-edu-classifier, raw regression
logit, batches of 64, padding=True, truncation max_length=512, on the two_stage_pool/input (Dripper) text at the
joined pool index. Run on CPU fp32. Used for the teacher-deleted pages that text_decisions does not cover.
Output: $HELDOUT960_DIR/edu_heldout960_recomputed.jsonl
usage: edu_score960.py"""
import json, gzip, os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from eval_paths import TWO_STAGE_POOL_DIR, HELDOUT960_DIR, HELDOUT960_STEM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(4)
etok = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
emod = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier").eval()
@torch.no_grad()
def escore(ts):
    out = []
    for i in range(0, len(ts), 64):
        e = etok(ts[i:i+64], return_tensors="pt", padding=True, truncation=True, max_length=512)
        out += emod(**e).logits.squeeze(-1).float().cpu().tolist()
    return out
inp = [json.loads(l)["text"] for l in gzip.open(f"{TWO_STAGE_POOL_DIR}/input/{HELDOUT960_STEM}.jsonl.gz", "rt", encoding="utf-8")]
t0 = time.time()
J = [json.loads(l) for l in open(HELDOUT960_DIR + "/heldout960_pooljoin.jsonl")]
gi = [r["pool_idx"][0] for r in J]
es2 = escore([inp[j] for j in gi])
with open(HELDOUT960_DIR + "/edu_heldout960_recomputed.jsonl", "w") as f:
    for k, (j, s) in enumerate(zip(gi, es2)): f.write(json.dumps({"gold_idx": k, "pool_idx": j, "edu_raw": s, "edu": round(float(s), 3)}) + "\n")
print("held-out pages scored", len(es2), "%.1fs" % (time.time() - t0), flush=True)
