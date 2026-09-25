"""Attach the pool resiliparse_extract text (the text UltraX, ProX-C and the rule baselines consume) to every held-out
page. Join key: WARC-Record-ID from the raw pool record (exact); fallback: WARC-Target-URI if it is unique in that
shard's resiliparse file. Pages with no match get resiliparse = null and are counted.
usage: s5_resiliparse_join.py <heldout5k_nores.jsonl> <out heldout5k.jsonl>"""
import sys, os, json, gzip, re, collections
P = os.environ.get("RESILIPARSE_EXTRACT_DIR") or os.path.join(os.environ["WORK_DIR"], "dclm_pipeline", "resiliparse",
                                                              "resiliparse_extract", "resiliparse_extract", "processed_data")
WID = re.compile(r"WARC-Record-ID['\"]?\s*:\s*['\"](<urn:uuid:[^>]+>)"); URI = re.compile(r"WARC-Target-URI['\"]?\s*:\s*['\"]([^'\"]*)")
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
idx = {}
for stem in sorted(set(r["stem"] for r in rows)):
    byw, byu = {}, collections.defaultdict(list); n = 0
    with gzip.open(f"{P}/{stem}_processed.jsonl.gz", "rt", encoding="utf-8") as f:
        for k, l in enumerate(f):
            o = json.loads(l); m = o.get("metadata"); ms = m if isinstance(m, str) else json.dumps(m)
            a = WID.search(ms); b = URI.search(ms); n += 1
            if a: byw[a.group(1)] = (k, o.get("text"))
            if b: byu[b.group(1)].append((k, o.get("text")))
    idx[stem] = (byw, byu); print(stem, "resiliparse records", n, flush=True)
C = collections.Counter(); CS = collections.Counter()
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for r in rows:
        byw, byu = idx[r["stem"]]
        hit = byw.get(r["warc_id"]) if r.get("warc_id") else None
        if hit: st = "warc_id"
        elif r.get("url") and len(byu.get(r["url"], [])) == 1: hit = byu[r["url"]][0]; st = "url_unique"
        else: st = "none"
        r["resiliparse"] = hit[1] if hit else None; r["resiliparse_row"] = hit[0] if hit else None; r["resiliparse_join"] = st
        C[st] += 1; CS[(r["source"], st)] += 1
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("join:", dict(C)); print("by source:", {"%s/%s" % k: v for k, v in sorted(CS.items())})
print("empty resiliparse text among joined:", sum(1 for r in rows if r["resiliparse"] is not None and not r["resiliparse"].strip()))
print("RESI_JOIN_OK")
