"""FineWeb-Edu classifier over a set of documents (the quality scores of the held-out analyses).

HuggingFaceFW/fineweb-edu-classifier (a regression head on arctic-embed-m), 512-token truncation, raw regression
output rounded to 4 decimals. Writes one score per document and a
summary per corpus: mean, and the share at or above the 1.0 / 1.5 / 2.0 / 3.0 marks
(FineWeb-Edu itself keeps int_score >= 3; our rescue gate used >= 1.0).
usage: score_edu.py --inputs name=path.jsonl ... --output-dir OUT
"""
import argparse, json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--batch-size", type=int, default=64)
a = ap.parse_args(); os.makedirs(a.output_dir, exist_ok=True)
M = "HuggingFaceFW/fineweb-edu-classifier"
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "16")))
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForSequenceClassification.from_pretrained(M).to(dev).eval()
print("device", dev, flush=True)
summary = []
for spec in a.inputs:
    name, path = spec.split("=", 1)
    texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8") if l.strip()]
    scores = []
    with torch.no_grad():
        for i in range(0, len(texts), a.batch_size):
            enc = tok(texts[i:i + a.batch_size], return_tensors="pt", padding="longest", truncation=True, max_length=512).to(dev)
            scores += model(**enc).logits.squeeze(-1).float().cpu().tolist()
            if (i // a.batch_size) % 20 == 0:
                print("[%s] %d/%d" % (name, i, len(texts)), flush=True)
    with open(os.path.join(a.output_dir, name + ".edu.jsonl"), "w") as f:
        for s in scores: f.write(json.dumps({"edu": round(s, 4)}) + "\n")
    n = len(scores)
    rec = {"name": name, "n": n, "mean": sum(scores) / n,
           "ge1.0": sum(s >= 1.0 for s in scores) / n, "ge1.5": sum(s >= 1.5 for s in scores) / n,
           "ge2.0": sum(s >= 2.0 for s in scores) / n, "ge3.0": sum(s >= 3.0 for s in scores) / n}
    summary.append(rec); print(json.dumps(rec), flush=True)
json.dump(summary, open(os.path.join(a.output_dir, "edu_summary.json"), "w"), indent=1)
print("EDU_SCORING_DONE", flush=True)
