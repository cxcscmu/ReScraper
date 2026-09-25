"""Stage 2d (CPU, needs pyarrow): per-page UltraX output from the UltraX run over the raw resiliparse pool (baselines/).

The run's parquets carry no id/url column (inf=[original, model_output], post=[original, cleaned, processed_functions];
rows equal across stages), so pages are joined by EXACT `original` text within the page's stem:
md5(original.encode("utf-8", "surrogatepass")) must equal the md5 of the page's resiliparse text; first occurrence wins
(duplicates counted); the char length is re-checked.
Input: $ULTRAX_RERUN_DIR/{post,inf}/<stem>_processed.parquet
usage: s2_ultrax_join.py <keys.jsonl {gid, stem, md5, nchar}> <out.jsonl>
"""
import collections, hashlib, json, os, sys
import pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from eval_paths import ULTRAX_RERUN_DIR as W
keys = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = collections.defaultdict(list)
for k in keys:
    if k.get("stem") and k.get("md5"): by[k["stem"]].append(k)
S = collections.Counter(); res = {}
for stem, ks in sorted(by.items()):
    st = stem if stem.endswith("_processed") else stem + "_processed"
    pp, ip = "%s/post/%s.parquet" % (W, st), "%s/inf/%s.parquet" % (W, st)
    if not (os.path.exists(pp) and os.path.exists(ip)):
        S["stem_missing"] += 1
        for k in ks: res[k["gid"]] = {"gid": k["gid"], "found": False, "why": "stem_missing"}
        continue
    p = pq.read_table(pp, columns=["original", "cleaned", "processed_functions"])
    a = pq.read_table(ip, columns=["original", "model_output"])
    po, pc, pf = p.column("original").to_pylist(), p.column("cleaned").to_pylist(), p.column("processed_functions").to_pylist()
    ao, am = a.column("original").to_pylist(), a.column("model_output").to_pylist()
    assert len(po) == len(ao)
    first, cnt = {}, collections.Counter()
    for j, t in enumerate(po):
        h = hashlib.md5((t or "").encode("utf-8", "surrogatepass")).hexdigest()
        cnt[h] += 1
        first.setdefault(h, j)
    S["stems"] += 1
    for k in ks:
        j = first.get(k["md5"])
        if j is None:
            res[k["gid"]] = {"gid": k["gid"], "found": False, "why": "text_not_in_stem"}; S["missing"] += 1; continue
        assert ao[j] == po[j] and len(po[j]) == k["nchar"], (stem, j)
        res[k["gid"]] = {"gid": k["gid"], "found": True, "stem": st, "row": j, "n_same_text": cnt[k["md5"]],
                         "cleaned": pc[j], "processed_functions": pf[j], "model_output": am[j]}
        S["found"] += 1; S["dup_text_hits"] += cnt[k["md5"]] > 1
with open(sys.argv[2] + ".tmp", "w", encoding="utf-8") as f:
    for k in keys:
        f.write(json.dumps(res.get(k["gid"], {"gid": k["gid"], "found": False, "why": "no_key"}), ensure_ascii=False) + "\n")
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
print(json.dumps(dict(S))); print("UX_JOIN_OK")
