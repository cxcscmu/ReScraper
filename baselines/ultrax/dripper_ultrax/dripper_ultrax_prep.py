#!/usr/bin/env python
"""Dripper -> UltraX, stage 1: convert Dripper step2_text shards -> parquet (column `original`) for
UltraX inference, partitioned into N contiguous parts (one per node job).
Resume-safe: skips shards whose parquet already exists. Rows <10 chars dropped
(dead extractions; they would be dropped downstream anyway).
Corrupt/truncated .gz shards are tolerated: whatever decompressed before the
error is kept, the file is reported as damaged, and a shard that yields nothing
is skipped entirely instead of failing the whole pool (a non-gzip input shard would
otherwise raise BadGzipFile and stop the run).
usage: dripper_ultrax_prep.py --out <work>/in [--src <step2_text>] [--parts 6] [--workers 32]
env: DRIPPER_STEP2_DIR / WORK_DIR (default --src)"""
import argparse, glob, gzip, json, os, zlib
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

SRC = os.environ.get("DRIPPER_STEP2_DIR",
                     os.path.join(os.environ.get("WORK_DIR", ""), "dclm_pipeline", "dripper", "step2_text"))


def convert(job):
    src, dst = job
    if os.path.exists(dst):
        return (0, 0, 0, None)
    texts = []
    n = 0
    damaged = 0
    try:
        with gzip.open(src, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                n += 1
                try:
                    t = json.loads(line).get("text") or ""
                except Exception:
                    continue
                if len(t) >= 10:
                    texts.append(t)
    except (gzip.BadGzipFile, EOFError, OSError, zlib.error) as e:
        damaged = 1
        if not texts:
            return (n, 0, 1, f"{os.path.basename(src)}: {type(e).__name__} (empty, skipped)")
    tmp = dst + ".tmp"
    pq.write_table(pa.table({"original": texts}), tmp)
    os.rename(tmp, dst)
    note = f"{os.path.basename(src)}: partial, kept {len(texts)} rows" if damaged else None
    return (n, len(texts), damaged, note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    files = sorted(glob.glob(a.src + "/*.jsonl.gz"))
    jobs = []
    per = (len(files) + a.parts - 1) // a.parts
    for k in range(a.parts):
        d = os.path.join(a.out, f"part{k}")
        os.makedirs(d, exist_ok=True)
        for f in files[k * per : (k + 1) * per]:
            stem = os.path.basename(f).replace(".jsonl.gz", "")
            jobs.append((f, os.path.join(d, stem + ".parquet")))
    with Pool(a.workers) as p:
        res = p.map(convert, jobs, chunksize=16)
    notes = [r[3] for r in res if r[3]]
    print(f"shards={len(jobs)} src_rows={sum(r[0] for r in res)} kept_rows={sum(r[1] for r in res)} "
          f"damaged_shards={sum(r[2] for r in res)}")
    for s in notes[:50]:
        print("DAMAGED " + s)


if __name__ == "__main__":
    main()
