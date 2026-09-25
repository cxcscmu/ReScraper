"""Completeness gate for the UltraX stages.
For every $UX_WORK/in/<stem>.parquet: inf/<stem>.parquet and post/<stem>.parquet must open and have the same row count
(inference.py writes one row per input row, post one per inf row). With --clean, an unreadable/short inf file (a
write cut by an interrupted task) is deleted together with its post file, and an unreadable/short post file is deleted, so
the next UltraX pass redoes them. ONLY run with --clean when no ux_inf / ux_fix task is running.
Note: a finished inference task does not imply a complete slice (UltraX's inference.py does not propagate a dead
worker into its exit code), so this check is the one that decides whether the pool is done.
Writes $UX_WORK/counts/validate.json and $UX_WORK/counts/missing_inf.txt, missing_post.txt (stems with .parquet).
usage: ux_validate.py [--clean]   env: UX_WORK (or WORK_DIR)
"""
import json, os, sys
from multiprocessing import Pool
import pyarrow.parquet as pq

R = os.environ.get("WORK_DIR", "")
W = os.environ.get("UX_WORK", os.path.join(R, "dclm_pipeline", "resiliparse_raw_ultrax"))
CLEAN = "--clean" in sys.argv


def rows(p):
    if not os.path.exists(p):
        return None
    try:
        return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return -1


def chk(f):
    return f, rows(W + "/in/" + f), rows(W + "/inf/" + f), rows(W + "/post/" + f)


files = sorted(f for f in os.listdir(W + "/in") if f.endswith(".parquet"))
with Pool(16) as pool:
    res = pool.map(chk, files, chunksize=32)
bad_inf = [f for f, a, b, c in res if b is not None and b != a]
bad_post = [f for f, a, b, c in res if c is not None and c != a]
if CLEAN:
    for f in bad_inf:
        for d in ("inf", "post"):
            if os.path.exists("%s/%s/%s" % (W, d, f)):
                os.remove("%s/%s/%s" % (W, d, f))
    for f in bad_post:
        if os.path.exists("%s/post/%s" % (W, f)):
            os.remove("%s/post/%s" % (W, f))
    res = [(f, a, (None if f in set(bad_inf) else b), (None if (f in set(bad_inf) or f in set(bad_post)) else c))
           for f, a, b, c in res]
# a shard with 0 input rows gets no inference output (PerFileParquetWriter skips empty buffers): count it done
miss_inf = [f for f, a, b, c in res if a != 0 and b != a]
miss_post = [f for f, a, b, c in res if a != 0 and c != a]
out = {"in_shards": len(files), "in_rows": sum(max(a or 0, 0) for f, a, b, c in res),
       "in_zero_row_shards": sum(a == 0 for f, a, b, c in res),
       "inf_ok": len(files) - len(miss_inf), "post_ok": len(files) - len(miss_post),
       "bad_inf_found": len(bad_inf), "bad_post_found": len(bad_post), "cleaned": CLEAN,
       "missing_inf": len(miss_inf), "missing_post": len(miss_post)}
os.makedirs(W + "/counts", exist_ok=True)
open(W + "/counts/missing_inf.txt", "w").write("".join(f + "\n" for f in miss_inf))
open(W + "/counts/missing_post.txt", "w").write("".join(f + "\n" for f in miss_post))
json.dump(out, open(W + "/counts/validate.json", "w"), indent=1)
print(json.dumps(out))
print("VALIDATE_COMPLETE" if not miss_inf and not miss_post else "VALIDATE_INCOMPLETE")
