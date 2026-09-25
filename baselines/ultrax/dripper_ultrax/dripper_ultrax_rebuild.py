#!/usr/bin/env python
"""Dripper -> UltraX, stage 4: post-processed parquets (original/cleaned/processed_functions)
-> corpus shards text/<stem>.jsonl.gz ({"text": cleaned}), dropping docs UltraX
nulled (remove_all) or left <50 chars. Emits corpus stats json (<out>/../rebuild_stats.json).
usage: dripper_ultrax_rebuild.py [--post_glob '<work>/post/part*/*.parquet'] [--out <text dir>] [--workers 32]
env: WORK_DIR (defaults under $WORK_DIR/dclm_pipeline/dripper_ultrax)"""
import argparse, collections, glob, gzip, json, os, re
from multiprocessing import Pool

import pyarrow.parquet as pq

OPS = re.compile(r"\b(keep_all|remove_all|remove_lines|replace_str|add_line)\s*\(")
D = os.path.join(os.environ.get("WORK_DIR", ""), "dclm_pipeline", "dripper_ultrax")


def one(job):
    src, dst = job
    if os.path.exists(dst):
        return None
    t = pq.read_table(src)
    cleaned = t.column("cleaned").to_pylist()
    funcs = t.column("processed_functions").to_pylist()
    orig = t.column("original").to_pylist()
    st = collections.Counter()
    ops = collections.Counter()
    tmp = dst + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for o, c, fn in zip(orig, cleaned, funcs):
            st["docs"] += 1
            st["chars_in"] += len(o)
            for op in OPS.findall(fn or ""):
                ops[op] += 1
            if not (c or "").strip():
                st["drop_empty"] += 1
                continue
            if len(c.strip()) < 50:
                st["drop_short"] += 1
                continue
            st["kept"] += 1
            st["chars_out"] += len(c)
            f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
    os.rename(tmp, dst)
    return (dict(st), dict(ops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post_glob", default=D + "/work/post/part*/*.parquet")
    ap.add_argument("--out", default=D + "/text")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for f in sorted(glob.glob(a.post_glob)):
        stem = os.path.basename(f).replace(".parquet", "")
        jobs.append((f, os.path.join(a.out, stem + ".jsonl.gz")))
    with Pool(a.workers) as p:
        res = [r for r in p.map(one, jobs, chunksize=8) if r]
    tot = collections.Counter()
    ops = collections.Counter()
    for s, o in res:
        tot.update(s)
        ops.update(o)
    summary = {"shards_written": len(res), "stats": dict(tot), "ops": dict(ops),
               "char_retention": round(tot["chars_out"] / max(1, tot["chars_in"]), 4)}
    with open(a.out + "/../rebuild_stats.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
