"""Stage 1a-rel (CPU): the RELEASE-decoding records of the held-out pages, from the `full` arm of the operation ablation.

The released corpus was decoded at T=1.0, top_p=1.0, max_tokens=3072 (inference/infer_pool.py). Its text shards carry
no page key, so the per-page record comes from the operation ablation's full arm (same model, prompt, decoding and
executor; the control arm of the ablation table; its corpus equals the released one within 0.002% of rows, see
ablations/): $ABLATION_DIR/raw/<stem>_raw.jsonl.gz, one row per Dripper step-1 record sorted by idx (0-based line of
step-1 <stem>.jsonl); fields stem, idx, in_md5, skip | gen, finish, n_gen_tok, tag, final, extracted, in_full.
Join: exact (stem, idx == step1_idx) with the page table; in_md5 asserted equal for every page.
Extra checks per page: `in_full_text` = md5(final) is among the rows of $ABLATION_DIR/full/text/<stem>_processed.jsonl.gz
(the full arm's corpus rows), `in_clean` = md5(final) is among $ABLATION_DIR/full/text_clean (after the post-filter,
the release's pre-dedup filter); `f3_dropped` = the page is one of the malformed <extract> generations the ablation
build dropped from every arm (F3_DROP lines of its log, optional: F3_DROP_LOG).
usage: s1_rel_extract.py <page_table> <out ablation_rows.jsonl>"""
import collections, gzip, hashlib, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from eval_paths import ABLATION_DIR as ABL
DROPLOG = os.environ.get("F3_DROP_LOG", "")
md5 = lambda s: hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()
keys = []
for l in open(sys.argv[1], encoding="utf-8"):
    if l.strip():
        r = json.loads(l); keys.append({"gid": r["gid"], "stem": r["stem"], "step1_idx": r["step1_idx"], "in_md5": r["in_md5"]})
raws = sorted(f[:-len("_raw.jsonl.gz")] for f in os.listdir(ABL + "/raw") if f.endswith("_raw.jsonl.gz"))
def resolve(s):
    s = s[:-len("_processed")] if s.endswith("_processed") else s
    if s in raws: return s
    c = [r for r in raws if r.startswith(s)]
    assert len(c) == 1, ("stem not uniquely resolvable", s, c[:5]); return c[0]
f3 = set()
if DROPLOG:
    for l in open(DROPLOG):
        m = re.match(r"F3_DROP (\S+):(\d+) ", l)
        if m: f3.add((m.group(1), int(m.group(2))))
by = collections.defaultdict(list)
for k in keys: by[resolve(k["stem"])].append(k)
S = collections.Counter(); out = {}
for stem, ks in sorted(by.items()):
    want = {k["step1_idx"]: k for k in ks}
    assert len(want) == len(ks), ("duplicate step1_idx", stem)
    rows = {}
    with gzip.open("%s/raw/%s_raw.jsonl.gz" % (ABL, stem), "rt", encoding="utf-8") as g:
        prev = -1
        for l in g:
            r = json.loads(l); assert r["idx"] > prev; prev = r["idx"]
            if r["idx"] in want: rows[r["idx"]] = r
    def md5set(path):
        c = collections.Counter()
        if os.path.exists(path):
            with gzip.open(path, "rt", encoding="utf-8") as g:
                for l in g: c[md5(json.loads(l)["text"])] += 1
        return c
    full = md5set("%s/full/text/%s_processed.jsonl.gz" % (ABL, stem))
    clean = md5set("%s/full/text_clean/%s_processed.jsonl.gz" % (ABL, stem))
    S["stems"] += 1; S["full_rows"] += sum(full.values()); S["clean_rows"] += sum(clean.values())
    for i, k in want.items():
        r = rows.get(i)
        assert r is not None, ("held-out page missing in the ablation raw rows", stem, i)
        assert r.get("in_md5") == k["in_md5"], ("in_md5 mismatch", stem, i, r.get("in_md5"), k["in_md5"])
        fin = r.get("final")
        o = dict(r, gid=k["gid"], stem=stem, f3_dropped=(stem, i) in f3,
                 in_full_text=bool(fin) and full[md5(fin)] > 0, in_clean=bool(fin) and clean[md5(fin)] > 0)
        out[k["gid"]] = o; S["found"] += 1; S["skip_" + str(r.get("skip"))] += "skip" in r
        S["in_full"] += bool(r.get("in_full")); S["in_full_text"] += o["in_full_text"]; S["in_clean"] += o["in_clean"]
        S["f3_dropped"] += o["f3_dropped"]; S["finish_" + str(r.get("finish"))] += 1
assert sorted(out) == sorted(k["gid"] for k in keys)
with open(sys.argv[2] + ".tmp", "w", encoding="utf-8") as f:
    for g in sorted(out): f.write(json.dumps(out[g], ensure_ascii=False) + "\n")
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
print(json.dumps(dict(S))); print("REL_EXTRACT_OK")
