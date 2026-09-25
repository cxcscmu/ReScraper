"""Score docs with DataMan-1.5B-EN (all_rating prompt, overall score 1-5), aggregate per dataset.

Usage: python score_dataman.py --inputs name1=path1.jsonl.gz name2=path2.jsonl.gz \
         --output-dir OUT
Each input: jsonl(.gz) with a "text" field. Writes per-doc scores jsonl + summary.tsv.
"""
import argparse
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simple_dataman import DataManInference


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dm = DataManInference(model_type="all_rating", use_server=False, max_tokens=64)

    summary = []
    for spec in args.inputs:
        name, path = spec.split("=", 1)
        texts = [d["text"] for d in read_jsonl(path) if d.get("text", "").strip()]
        print(f"[{name}] {len(texts)} docs from {path}", flush=True)
        results = []
        for i in range(0, len(texts), args.batch_size):
            results.extend(dm.score_texts(texts[i : i + args.batch_size]))
            print(f"[{name}] scored {len(results)}/{len(texts)}", flush=True)
        out_path = os.path.join(args.output_dir, f"{name}.scores.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        valid = [r for r in results if isinstance(r.get("overall_score"), float)
                 and 1.0 <= r["overall_score"] <= 5.0]
        n = len(valid)
        mean = sum(r["overall_score"] for r in valid) / max(n, 1)
        dist = {s: sum(1 for r in valid if r["overall_score"] == s) / max(n, 1)
                for s in (1.0, 2.0, 3.0, 4.0, 5.0)}
        crit_means = {}
        for k in ("accuracy", "coherence", "semantic_density", "knowledge_novelty",
                  "topic_focus", "professionalism", "structural_standardization",
                  "originality"):
            vals = [r[k] for r in valid if isinstance(r.get(k), float) and 1 <= r[k] <= 5]
            crit_means[k] = sum(vals) / max(len(vals), 1)
        row = {"name": name, "n_scored": n, "n_total": len(texts),
               "mean_overall": round(mean, 4),
               "dist": {str(int(k)): round(v, 4) for k, v in dist.items()},
               "criteria": {k: round(v, 4) for k, v in crit_means.items()}}
        summary.append(row)
        print(json.dumps(row), flush=True)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "summary.tsv"), "w") as f:
        f.write("name\tn\tmean_overall\tpct_ge4\tpct_le2\n")
        for r in summary:
            ge4 = r["dist"].get("4", 0) + r["dist"].get("5", 0)
            le2 = r["dist"].get("1", 0) + r["dist"].get("2", 0)
            f.write(f"{r['name']}\t{r['n_scored']}\t{r['mean_overall']:.4f}"
                    f"\t{ge4:.4f}\t{le2:.4f}\n")
    print("DATAMAN_SCORING_DONE", flush=True)


if __name__ == "__main__":
    main()
