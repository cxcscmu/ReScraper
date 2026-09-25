"""Merge $DIV_DIR/results/<corpus>.json (376-shard runs of ngram_scale.py) into the figure data file
(= diversity_ngrams_pool376.json in analysis/data).
usage: build_json.py <out path>"""
import json, os, sys, time
WORK = os.environ.get("WORK_DIR") or sys.exit("set WORK_DIR")
DIV = os.environ.get("DIV_DIR") or os.path.join(os.environ.get("EVAL_DIR") or os.path.join(WORK, "eval"), "diversity")
R = DIV + "/results"
SPEC = [("rescraper", "ReScraper (Ours)", "pre-dedup", True), ("ultrax", "UltraX", "pre-dedup (no dedup applied)", True),
        ("proxc", "ProX-C", "pre-dedup (no dedup applied)", True), ("refinedweb_rule", "RefinedWeb-rule", "pre-dedup", True),
        ("fineweb_rule", "FineWeb-rule", "no dedup", False), ("resiliparse_raw", "Resiliparse raw (input of UltraX / ProX-C / FineWeb-rule)", "no dedup", False)]
corpora, order, ns = {}, None, None
for key, label, stage, plotted in SPEC:
    try:
        r = json.load(open("%s/%s.json" % (R, key)))
    except FileNotFoundError:
        print("missing", key); continue
    if order is None: order, ns = r["order"], r["ns"]
    assert r["order"] == order and r["ns"] == ns
    cum = r["cumulative"]
    corpora[key] = {"label": label, "plotted": plotted, "dedup": stage, "source": r["source"], "field": r["field"],
                    "shards": list(range(1, len(order) + 1)), "distinct": cum["distinct"], "tokens": cum["tokens"], "docs": cum["docs"],
                    "ngram_occurrences": cum["ngram_occurrences"], "per_doc_distinct": cum["per_doc_distinct"], "per_shard": r["per_shard"]}
D = {"description": "Cumulative number of distinct token n-grams in the text each corpus produces from the first k of the same %d DCLM "
                    "pool shards (k = 1..%d), for n in {%s}; also cumulative tokens, documents, n-gram occurrences and the sum of per-document "
                    "distinct n-grams (within-document vs cross-document repeats)." % (len(order), len(order), ", ".join(map(str, ns))),
     "settings": {"ns": ns, "tokenizer": "EleutherAI/gpt-neox-20b (HF AutoTokenizer, fast), add_special_tokens=False",
                  "ngram_scope": "within documents only (no n-gram crosses a document boundary); empty / whitespace-only documents dropped",
                  "distinct_counting": "exact set union over 64-bit keys: n<=4 packs the token ids into one uint64 (collision-free); n>4 chained splitmix64 hash. No sketching.",
                  "shard_set": "the 376 pool shards on which the pool-scale ProX-C run finished; all four corpora exist on each (stems376.txt)",
                  "shard_order": "random.Random(0).shuffle(sorted(stems)), the same order for every corpus",
                  "code": "evaluation/diversity/ngram_scale.py (+ build_json.py)",
                  "created": time.strftime("%Y-%m-%d %H:%M %Z")},
     "order": order, "plot_order": ["proxc", "refinedweb_rule", "ultrax", "rescraper"], "corpora": corpora}
json.dump(D, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], "corpora:", list(corpora))
