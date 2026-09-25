"""Map WARC-Record-IDs (the two-stage refiner's 50k-page Dripper SFT source, a superset of its 45,610 training pages)
to pool shards by scanning the metadata of every pool resiliparse_extract shard (10,319 shards, 18.0M pages).
usage: s1b_two_stage_shards.py <two_stage_src_warc.tsv> <out_prefix> [procs]
writes <out_prefix>_hits.tsv (warc id, stem, row) and <out_prefix>_stems.txt (the shards to exclude)"""
import sys, os, gzip, re, json, collections
from multiprocessing import Pool
P = os.environ.get("RESILIPARSE_EXTRACT_DIR") or os.path.join(os.environ["WORK_DIR"], "dclm_pipeline", "resiliparse",
                                                              "resiliparse_extract", "resiliparse_extract", "processed_data")
WID = re.compile(r"WARC-Record-ID['\"]?\s*:\s*['\"](<urn:uuid:[^>]+>)")
WANT = None
def scan(fn):
    global WANT
    if WANT is None:
        WANT = set(l.split("\t")[2] for l in open(sys.argv[1]) if l.split("\t")[2])
    stem = fn[:-len("_processed.jsonl.gz")]; hits = []; n = 0
    try:
        with gzip.open(os.path.join(P, fn), "rt", encoding="utf-8") as f:
            for k, l in enumerate(f):
                n += 1; i = l.find("WARC-Record-ID")
                m = WID.search(l, i - 2) if i >= 0 else None
                if m and m.group(1) in WANT: hits.append((m.group(1), k))
    except Exception as e:
        return stem, hits, n, repr(e)[:100]
    return stem, hits, n, None
if __name__ == "__main__":
    fs = sorted(f for f in os.listdir(P) if f.endswith("_processed.jsonl.gz"))
    tot = 0; errs = []; out = []
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 16) as pool:
        for j, (stem, hits, n, err) in enumerate(pool.imap_unordered(scan, fs, chunksize=16)):
            tot += n
            if err: errs.append((stem, err))
            for w, k in hits: out.append((w, stem, k))
            if (j + 1) % 1000 == 0: print("scanned", j + 1, "records", tot, "hits", len(out), flush=True)
    with open(sys.argv[2] + "_hits.tsv", "w") as f:
        for w, stem, k in sorted(out, key=lambda x: (x[1], x[2])): f.write("%s\t%s\t%d\n" % (w, stem, k))
    c = collections.Counter(s for _, s, _ in out)
    with open(sys.argv[2] + "_stems.txt", "w") as f:
        f.write("".join(s + "\n" for s in sorted(c)))
    want = set(l.split("\t")[2] for l in open(sys.argv[1]) if l.split("\t")[2])
    print(json.dumps({"shards_scanned": len(fs), "records": tot, "errors": errs[:20], "n_errors": len(errs), "want_ids": len(want),
                      "hits": len(out), "distinct_ids_hit": len(set(w for w, _, _ in out)), "stems": len(c)}), flush=True)
    print("TWO_STAGE_SHARDS_OK")
