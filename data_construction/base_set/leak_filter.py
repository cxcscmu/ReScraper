"""Content-collision filter against held-out pages (applied to the FIRST half of the base set only).

Drops every training row whose input, Dripper text or (non-delete) output is byte-identical (md5) to that of a
page in <gold_dir>/render_*.jsonl (the rendered rows of the held-out shards), then re-checks the result.
Held-out disjointness itself is guaranteed by shard stem (select_pages.py, check_stem_disjoint.py); identical
text across different shards is a natural duplicate, not leakage, so this filter was NOT applied to the second
half. It is kept here only because the released base set's first half went through it.
usage: leak_filter.py <gold_dir> <in.jsonl> <out.jsonl>
"""
import sys, json, glob, hashlib, os
h = lambda s: hashlib.md5(s.encode("utf-8", "surrogatepass")).hexdigest()
gold_dir, src, dst = sys.argv[1:4]
black = set()
for f in sorted(glob.glob(gold_dir + "/render_*.jsonl")):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        black.add(h(r["input"])); black.add(h(r.get("drip", "")))
        o = r.get("output", "")
        if o.strip() and o.strip() != "<delete>":
            black.add(h(o))
black.discard(h(""))
print("held-out blacklist hashes:", len(black), flush=True)
n = kept = dropped = 0
tmp = dst + ".tmp"
with open(src, encoding="utf-8") as fi, open(tmp, "w", encoding="utf-8") as fo:
    for l in fi:
        r = json.loads(l); n += 1
        keys = {h(r.get("input", "")), h(r.get("drip", "")), h(r.get("output", ""))}
        if keys & black:
            dropped += 1; continue
        fo.write(l if l.endswith("\n") else l + "\n"); kept += 1
os.replace(tmp, dst)
print("%s: rows %d -> kept %d, dropped %d (held-out collisions)" % (os.path.basename(src), n, kept, dropped), flush=True)
# re-check on the written file
hits = 0
for l in open(dst, encoding="utf-8"):
    r = json.loads(l)
    if {h(r.get("input", "")), h(r.get("drip", "")), h(r.get("output", ""))} & black:
        hits += 1
print("re-check hits on output:", hits, flush=True)
if hits:
    sys.exit("FATAL: output still collides with held-out pages")
print("LEAK_FILTER_DONE", flush=True)
