"""Token-level retention across the materialization stages of the released ReScraper corpus
(Table 9, tab:stage-retention, "Token axis" block).

Documents are not the only thing a stage removes; it also removes text. For the same set of
shard stems we sum characters at each stage that is stored on disk, then convert with the
corpus's own measured characters-per-token so the numbers line up with the final tokenizer
count. The final row is exact (the tokenizer's own count), the rest are char-derived.
Stages (corpus/corpora.sh layout): text/ (executor output of inference/infer_pool.py),
text_clean/ (inference/postfilter.py), step4_dedup_oldboth_65b/ (corpus/dedup_bff.sh).
env: CORPUS_DIR (default $WORK_DIR/dclm_pipeline/rescraper_corpus), NSH (sampled shards, default 60).
CPU only (the paper run was allocated 16 CPUs, 100 GB, a 2 h limit).
"""
import glob, gzip, json, os, random, sys

if not (os.environ.get("CORPUS_DIR") or os.environ.get("WORK_DIR")):
    sys.exit("set WORK_DIR or CORPUS_DIR (see configs/paths.env.example)")
L = os.environ.get("CORPUS_DIR") or os.path.join(os.environ["WORK_DIR"], "dclm_pipeline", "rescraper_corpus")
NSH = int(os.environ.get("NSH", "60"))
STAGES = [("inference output", L + "/text"),
          ("after post-filter", L + "/text_clean"),
          ("after dedup", L + "/step4_dedup_oldboth_65b")]

# pick stems present in every stage so the comparison is like for like
def stems(d):
    out = set()
    for f in os.listdir(d):
        if f.endswith(".jsonl.gz") or f.endswith(".jsonl"):
            out.add(f.replace("_processed", "").replace(".jsonl.gz", "").replace(".jsonl", ""))
    return out

common = None
for _, d in STAGES:
    if not os.path.isdir(d):
        print("MISSING", d); sys.exit(1)
    common = stems(d) if common is None else (common & stems(d))
common = sorted(common)
print("shards present in all stages:", len(common), flush=True)
random.Random(11).shuffle(common)
pick = set(common[:NSH])
print("sampling", len(pick), "shards", flush=True)

res = {}
for name, d in STAGES:
    chars = docs = 0
    for f in sorted(os.listdir(d)):
        stem = f.replace("_processed", "").replace(".jsonl.gz", "").replace(".jsonl", "")
        if stem not in pick:
            continue
        op = gzip.open if f.endswith(".gz") else open
        try:
            for line in op(os.path.join(d, f), "rt", encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line).get("text", "")
                if t:
                    docs += 1
                    chars += len(t)
        except Exception as e:
            print("  read error", f, str(e)[:60], flush=True)
    res[name] = (docs, chars)
    print("%-22s docs %10d  chars %14d" % (name, docs, chars), flush=True)

base = res[STAGES[0][0]][1]
CPT = 4.2601   # measured chars per gpt-neox-20b token for this corpus
print("\n=== token-level retention (chars / %.4f) ===" % CPT)
for name, _ in STAGES:
    d, c = res[name]
    print("%-22s %12.0f tok  %6.2f%% of inference output  (%d docs)" % (
        name, c / CPT, 100.0 * c / base, d))
