"""FineWeb-Edu over the pool pages the two-stage refiner deleted, one shard-slice per single-GPU task.
For every shard: input/<f> holds the dripper text, best/<f> the two-stage refiner output ("" = deleted).
Writes edu/<f>.jsonl with {i, edu, int_edu} for the deleted rows only, so the merge step can pick
the ones worth sending to the rephraser. bf16 + batch 128; the classifier truncates at 512 tokens.
int_edu = round(clip(edu, 0, 5)) gates which pages get a rewrite generated (rewrite_pool.py).
usage: score_edu_pool.py <task_id> <n_tasks>
"""
import gzip, json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
T = os.environ["WORK_DIR"] + "/dclm_pipeline/two_stage_pool"
tid, nt = int(sys.argv[1]), int(sys.argv[2])
os.makedirs(T + "/edu", exist_ok=True)
M = "HuggingFaceFW/fineweb-edu-classifier"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForSequenceClassification.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
shards = sorted(os.listdir(T + "/best"))[tid::nt]
BS = int(os.environ.get("EDU_BS", "128"))
done = tot = 0
for f in shards:
    out = T + "/edu/" + f.replace(".jsonl.gz", "") + ".jsonl"
    if os.path.exists(out): done += 1; continue
    a = [json.loads(l)["text"] for l in gzip.open(T + "/input/" + f, "rt", encoding="utf-8")]
    b = [json.loads(l)["text"] for l in gzip.open(T + "/best/" + f, "rt", encoding="utf-8")]
    if len(a) != len(b): print("SKIP length mismatch", f, len(a), len(b), flush=True); continue
    idx = [i for i, (x, y) in enumerate(zip(a, b)) if not y.strip() and x.strip()]
    res = []
    with torch.no_grad():
        for j in range(0, len(idx), BS):
            ch = idx[j:j + BS]
            enc = tok([a[i] for i in ch], return_tensors="pt", padding="longest", truncation=True, max_length=512).to("cuda")
            sc = model(**enc).logits.squeeze(-1).float().cpu().tolist()
            if isinstance(sc, float): sc = [sc]
            for i, v in zip(ch, sc): res.append({"i": i, "edu": round(v, 4), "int_edu": int(round(max(0.0, min(v, 5.0))))})
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fo:
        for r in res: fo.write(json.dumps(r) + chr(10))
    os.replace(tmp, out); tot += len(res); done += 1
    if done % 20 == 0: print("task", tid, "shards", done, "/", len(shards), "deleted scored", tot, flush=True)
print("task", tid, "DONE shards", done, "of", len(shards), "deleted pages scored", tot, flush=True)
