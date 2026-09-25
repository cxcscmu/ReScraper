"""Held-out set step 2b (DCLM data-processing env with fasttext): fastText LID exactly as the SFT seed sampler
(lid.176, text.replace("\\n"," ")[:2000], k=1).
usage: s2b_lid.py <outdir>"""
import json, os, sys, fasttext
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from render import LID_PATH
lid = fasttext.load_model(LID_PATH)
OUT = sys.argv[1]; n = 0
with open(OUT + "/lid.jsonl", "w") as f:
    for l in open(OUT + "/cand_pre.jsonl", encoding="utf-8"):
        r = json.loads(l); t = r["drip"]
        if t.strip():
            lab, prob = lid.predict(t.replace("\n", " ")[:2000], k=1); lab, prob = lab[0], float(prob[0])
        else: lab, prob = None, None
        f.write(json.dumps({"cid": r["cid"], "lid": lab, "lid_prob": prob}) + "\n"); n += 1
print("LID rows", n); print("STEP2B_OK")
