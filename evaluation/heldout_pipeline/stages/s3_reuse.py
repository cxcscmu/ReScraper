"""Stage 3a' (CPU): reuse the DataMan / FineWeb-Edu scores of an earlier run of the SAME page table by exact text hash
(md5 of the string). Used by the release-decoding run, which shares every baseline text with the greedy run.

After s3_gather.py has written <W>/score_in (every unique text of the new run), this collects cached DataMan /
FineWeb-Edu per text hash from <cache_W>/score_in + score_out (only shards whose INPUT_MD5 matches the input they were
scored on), writes <W>/score_cache.jsonl {h, dataman, fineweb_edu} for the new run's texts that have a cached score,
and rewrites <W>/score_in/uniq_*.jsonl with ONLY the texts that have none.
Same scorer code and settings for cached and new texts; s4_build.py asserts that every text has a score.
usage: HP_WORK=<W> s3_reuse.py <page_table> <cache_W>"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from common import *
PT, CW = sys.argv[1], sys.argv[2]; W = workdir(PT); SHARD = int(os.environ.get("SCORE_SHARD", "10000"))
assert os.path.abspath(W) != os.path.abspath(CW)
cache = {}
for f in sorted(os.listdir(CW + "/score_in")):
    b = f[:-len(".jsonl")]; path = CW + "/score_in/" + f
    h_in = hashlib.md5(open(path, "rb").read()).hexdigest()
    ok = all(open("%s/score_out/%s_%s/INPUT_MD5" % (CW, d, b)).read().strip() == h_in for d in ("dataman", "edu"))
    assert ok, ("cached scores were not computed on this input shard", path)
    ins = list(read_jsonl(path))
    dmr = list(read_jsonl("%s/score_out/dataman_%s/%s.scores.jsonl" % (CW, b, b)))
    edr = list(read_jsonl("%s/score_out/edu_%s/%s.edu.jsonl" % (CW, b, b)))
    assert len(ins) == len(dmr) == len(edr), b
    for r, x, y in zip(ins, dmr, edr):
        o = x.get("overall_score")
        cache[r["h"]] = (float(o) if isinstance(o, float) and 1.0 <= o <= 5.0 else None, float(y["edu"]))
new = {}
for f in sorted(os.listdir(W + "/score_in")):
    for r in read_jsonl(W + "/score_in/" + f):
        new[r["h"]] = r["text"]
hit = sorted(h for h in new if h in cache); miss = sorted(h for h in new if h not in cache)
write_jsonl_atomic(W + "/score_cache.jsonl", [{"h": h, "dataman": cache[h][0], "fineweb_edu": cache[h][1]} for h in hit])
for f in os.listdir(W + "/score_in"):
    os.remove(W + "/score_in/" + f)
for i in range(0, len(miss), SHARD):
    write_jsonl_atomic("%s/score_in/uniq_%03d.jsonl" % (W, i // SHARD), [{"h": h, "text": new[h]} for h in miss[i:i + SHARD]])
if not miss:
    write_jsonl_atomic(W + "/score_in/uniq_000.jsonl", [])
R = {"unique_texts": len(new), "score_cache_hits": len(hit), "to_score": len(miss)}
json.dump(R, open(W + "/reuse_report.json", "w"), indent=1); print(json.dumps(R)); print("REUSE_OK")
