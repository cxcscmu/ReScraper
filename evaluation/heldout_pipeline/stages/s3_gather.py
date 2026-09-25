"""Stage 3a (CPU): collect every text that needs a score, dedupe by exact string, write the scoring inputs.

Texts (all keyed back to gids through <work>/texts_index.jsonl):
  pre                resiliparse text of the page (the common input of the baselines)
  post:<system>      non-empty output of rescraper (final text), ultrax, proxc, refinedweb_rule, fineweb_rule
  ours_extracted     ReScraper's pre-op extraction (apply_ops(input, stage-1 ops))
  op_before/op_after operation-score texts: before = extracted.strip(), after = final.strip() (usually the same strings)
Only non-empty strings (s.strip() != "") are kept - the same test scorers/score_dataman.py applies silently - so the
scorer outputs line up 1:1 with the unique-text list; the counts are asserted in s4_build.py.
Outputs: <work>/score_in/uniq_XXX.jsonl {"h", "text"} shards of <=SHARD texts, <work>/texts_uniq.jsonl {h, n_chars},
         <work>/texts_index.jsonl {gid, field, h}
usage: s3_gather.py <page_table>
"""
import collections, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from common import *
PT = sys.argv[1]; W = workdir(PT); SHARD = int(os.environ.get("SCORE_SHARD", "10000"))
pages = load_pages(PT, ["gid", "resiliparse"])
sysf = {"rescraper": W + "/ours.jsonl"}
for s in ("ultrax", "proxc", "refinedweb_rule", "fineweb_rule"):
    sysf[s] = W + "/sys_%s.jsonl" % s
S = {k: load_by_gid(v) for k, v in sysf.items()}
for k, v in S.items():
    assert sorted(v) == list(range(len(pages))), ("system output does not cover the page table", k)
uniq, index = {}, []
def add(gid, field, t):
    if not nonempty(t): return
    h = md5(t); uniq.setdefault(h, t); index.append({"gid": gid, "field": field, "h": h})
for p in pages:
    g = p["gid"]
    add(g, "pre", p["resiliparse"])
    for k in S:
        add(g, "post:" + k, S[k][g].get("text"))
    o = S["rescraper"][g]
    if o.get("extracted") is not None:
        add(g, "ours_extracted", o["extracted"])
        add(g, "op_before", o["extracted"].strip())
    if o.get("text") is not None:
        add(g, "op_after", o["text"].strip())
hs = sorted(uniq)          # deterministic order
d = W + "/score_in"; os.makedirs(d, exist_ok=True)
for f in os.listdir(d):
    os.remove(os.path.join(d, f))
for i in range(0, len(hs), SHARD):
    write_jsonl_atomic("%s/uniq_%03d.jsonl" % (d, i // SHARD), [{"h": h, "text": uniq[h]} for h in hs[i:i + SHARD]])
write_jsonl_atomic(W + "/texts_uniq.jsonl", [{"h": h, "n_chars": len(uniq[h])} for h in hs])
write_jsonl_atomic(W + "/texts_index.jsonl", index)
c = collections.Counter(x["field"] for x in index)
print(json.dumps({"unique_texts": len(hs), "shards": (len(hs) + SHARD - 1) // SHARD, "refs": dict(c)}, indent=1)); print("GATHER_OK")
