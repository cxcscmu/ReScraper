"""html5k.jsonl: gid -> raw pool html of every held-out page (existing rows from html_old.jsonl by gold_idx, new rows
from html_new.jsonl by cid); asserts md5(html) == in_md5 for every page. The extraction baselines (trafilatura, jusText,
resiliparse fallback) read this table.
usage: s6_html_table.py <heldout5k.jsonl> <workdir with html_old.jsonl, html_new.jsonl> <out html5k.jsonl>"""
import sys, json, hashlib, os
G, WD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
md5 = lambda s: hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()
rows = [json.loads(l) for l in open(G, encoding="utf-8")]
old = {}; new = {}
for l in open(WD + "/html_old.jsonl", encoding="utf-8"):
    o = json.loads(l); old[o["gold_idx"]] = o["html"]
want = set(r["cid"] for r in rows if r["source"] == "new_shard")
for l in open(WD + "/html_new.jsonl", encoding="utf-8"):
    o = json.loads(l)
    if o["cid"] in want: new[o["cid"]] = o["html"]
bad = 0
with open(OUT + ".tmp", "w", encoding="utf-8") as f:
    for r in rows:
        h = old[r["gid"]] if r["source"] != "new_shard" else new[r["cid"]]
        bad += md5(h) != r["in_md5"]
        f.write(json.dumps({"gid": r["gid"], "html": h}, ensure_ascii=False) + "\n")
assert bad == 0, bad
os.replace(OUT + ".tmp", OUT); print("html5k rows", len(rows), "md5 mismatches", bad)
